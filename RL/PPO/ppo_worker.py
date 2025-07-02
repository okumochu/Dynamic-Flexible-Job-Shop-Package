import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from .networks import PolicyNetwork, ValueNetwork
from .buffer import PPOBuffer

class PPOWorker:
    """
    PPO Worker Agent for Flexible Job Shop Scheduling with separate policy and value networks.
    """
    def __init__(self, 
                 obs_shape,
                 action_dim,
                 hidden_dim=256,
                 lr=3e-4,
                 gamma=0.99,
                 gae_lambda=0.95,
                 clip_ratio=0.2,
                 entropy_coeff=1e-3,
                 value_coeff=0.5,
                 max_grad_norm=0.5,
                 device='auto'):
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        print(f"PPO Agent initialized on device: {self.device}")
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.entropy_coeff = entropy_coeff
        self.value_coeff = value_coeff
        self.max_grad_norm = max_grad_norm
        self.policy = PolicyNetwork(obs_shape, action_dim, hidden_dim).to(self.device)
        self.value = ValueNetwork(obs_shape, hidden_dim).to(self.device)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.value_optimizer = optim.Adam(self.value.parameters(), lr=lr)
        self.training_stats = {
            'policy_loss': [],
            'value_loss': [],
            'entropy_loss': [],
            'total_loss': [],
            'clip_fraction': []
        }
    def get_action(self, obs: torch.Tensor, action_mask: torch.Tensor):
        with torch.no_grad():
            obs = obs.unsqueeze(0).to(self.device)
            action_mask = action_mask.to(self.device)
            action_logits = self.policy(obs)
            value = self.value(obs)
            action_logits = action_logits.squeeze(0)
            masked_logits = action_logits.clone()
            masked_logits[~action_mask] = float('-inf')

            dist = Categorical(logits=masked_logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            return int(action.item()), float(log_prob.item()), float(value.item())
    def compute_gae_returns(self, rewards, values, dones):
        returns = torch.zeros_like(rewards)
        advantages = torch.zeros_like(rewards)
        gae = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            not_done = 1.0 - dones[t].float()
            delta = rewards[t] + self.gamma * next_value * not_done - values[t]
            gae = delta + self.gamma * self.gae_lambda * not_done * gae
            advantages[t] = gae
        returns = advantages + values
        return returns
    def update(self, buffer: PPOBuffer, num_epochs: int = 4):
        batch = buffer.get_batch()
        returns = self.compute_gae_returns(batch['rewards'], batch['values'], batch['dones'])
        advantages = returns - batch['values']
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        for epoch in range(num_epochs):
            action_logits = self.policy(batch['observations'])
            values = self.value(batch['observations'])
            masked_logits = action_logits.clone()
            masked_logits[~batch['action_masks']] = float('-inf')
            dist = Categorical(logits=masked_logits)
            new_log_probs = dist.log_prob(batch['actions'])
            ratio = torch.exp(new_log_probs - batch['log_probs'])
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(values.squeeze(), returns)
            entropy_loss = -dist.entropy().mean()
            total_loss = (policy_loss + 
                         self.value_coeff * value_loss + 
                         self.entropy_coeff * entropy_loss)
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy_optimizer.step()
            self.value_optimizer.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.value.parameters(), self.max_grad_norm)
            self.value_optimizer.step()
            clip_fraction = (abs(ratio - 1) > self.clip_ratio).float().mean().item()
            self.training_stats['policy_loss'].append(policy_loss.item())
            self.training_stats['value_loss'].append(value_loss.item())
            self.training_stats['entropy_loss'].append(entropy_loss.item())
            self.training_stats['total_loss'].append(total_loss.item())
            self.training_stats['clip_fraction'].append(clip_fraction)
        stats = {
            'policy_loss': float(np.mean(self.training_stats['policy_loss'][-num_epochs:])),
            'value_loss': float(np.mean(self.training_stats['value_loss'][-num_epochs:])),
            'entropy_loss': float(np.mean(self.training_stats['entropy_loss'][-num_epochs:])),
            'total_loss': float(np.mean(self.training_stats['total_loss'][-num_epochs:])),
            'clip_fraction': float(np.mean(self.training_stats['clip_fraction'][-num_epochs:]))
        }
        return stats
    def save(self, path: str):
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'value_state_dict': self.value.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'value_optimizer_state_dict': self.value_optimizer.state_dict(),
            'training_stats': self.training_stats
        }, path)
    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        try:
            self.policy.load_state_dict(checkpoint['policy_state_dict'])
            self.value.load_state_dict(checkpoint['value_state_dict'])
        except RuntimeError as e:
            print("\n[ERROR] Model architecture mismatch while loading state_dict!")
            print("This usually means your current network architecture (e.g., hidden_dim, obs_dim, action_dim) does not match the one used for training.")
            print("Details:", e)
            print("\nTo fix: use the same hidden_dim and architecture as when you trained the model, or retrain from scratch if you changed the environment or network.")
            raise
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        self.value_optimizer.load_state_dict(checkpoint['value_optimizer_state_dict'])
        self.training_stats = checkpoint['training_stats']
    def get_deterministic_action(self, obs: torch.Tensor, action_mask: torch.Tensor):
        """Selects the action with the highest probability (argmax) among valid actions."""
        with torch.no_grad():
            obs = obs.unsqueeze(0).to(self.device)
            action_mask = action_mask.to(self.device)
            action_logits = self.policy(obs)
            action_logits = action_logits.squeeze(0)
            masked_logits = action_logits.clone()
            masked_logits[~action_mask] = float('-inf')
            if (~action_mask).all():
                raise RuntimeError("All actions are masked in get_deterministic_action. This should not happen.")
            action = torch.argmax(masked_logits).item()
            return int(action) 
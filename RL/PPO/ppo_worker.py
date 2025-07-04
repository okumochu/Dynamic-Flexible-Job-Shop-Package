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
                 hidden_dim,
                 pi_lr,
                 v_lr,
                 gamma,
                 gae_lambda,
                 clip_ratio,
                 entropy_coeff,
                 value_coeff,
                 max_grad_norm,
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
        self.pi_lr = pi_lr
        self.v_lr = v_lr
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=pi_lr)
        self.value_optimizer = optim.Adam(self.value.parameters(), lr=v_lr)
        
        # Store configuration parameters for saving
        self.obs_shape = obs_shape
        self.hidden_dim = hidden_dim
        
        self.training_stats = {
            'policy_loss': [],
            'value_loss': [],
            'entropy_loss': [],
            'total_loss': [],
            'clip_fraction': [],
            'kl': []
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
    def update(self, buffer: PPOBuffer, train_pi_iters: int = 4, train_v_iters: int = 4, target_kl: float = 0.0):
        """
        Update PPO agent using buffer.
        Args:
            buffer: PPOBuffer with experience
            train_pi_iters: Number of policy update iterations
            train_v_iters: Number of value update iterations
            target_kl: Early stopping KL threshold (0.0 disables early stopping)
        Returns:
            stats: dict with losses, clip_fraction, kl
        """
        batch = buffer.get_batch()
        returns = self.compute_gae_returns(batch['rewards'], batch['values'], batch['dones'])
        advantages = returns - batch['values']
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        # Policy update
        for i in range(train_pi_iters):
            action_logits = self.policy(batch['observations'])
            masked_logits = action_logits.clone()
            masked_logits[~batch['action_masks']] = float('-inf')
            dist = Categorical(logits=masked_logits)
            new_log_probs = dist.log_prob(batch['actions'])
            ratio = torch.exp(new_log_probs - batch['log_probs'])
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            entropy_loss = -dist.entropy().mean()
            self.policy_optimizer.zero_grad()
            (policy_loss + self.entropy_coeff * entropy_loss).backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy_optimizer.step()
            # KL divergence for early stopping
            with torch.no_grad():
                approx_kl = (batch['log_probs'] - new_log_probs).mean().item()
            if target_kl > 0.0 and approx_kl > 1.5 * target_kl:
                print(f"Early stopping at iter {i} due to reaching max kl: {approx_kl:.4f}")
                break
        # Value function update
        for _ in range(train_v_iters):
            values = self.value(batch['observations'])
            value_loss = F.mse_loss(values.squeeze(), returns)
            self.value_optimizer.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.value.parameters(), self.max_grad_norm)
            self.value_optimizer.step()
        # Logging
        clip_mask = abs(ratio - 1) > self.clip_ratio
        clip_fraction = torch.as_tensor(clip_mask, dtype=torch.float32).mean().item()
        stats = {
            'policy_loss': float(policy_loss.item()),
            'value_loss': float(value_loss.item()),
            'entropy_loss': float(entropy_loss.item()),
            'total_loss': float(policy_loss.item() + self.value_coeff * value_loss.item() + self.entropy_coeff * entropy_loss.item()),
            'clip_fraction': float(clip_fraction),
            'kl': float(approx_kl)
        }
        for k in stats:
            if k in self.training_stats:
                self.training_stats[k].append(stats[k])
        return stats
    def save(self, path: str):
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'value_state_dict': self.value.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'value_optimizer_state_dict': self.value_optimizer.state_dict(),
            'training_stats': self.training_stats,
            # Save configuration parameters
            'obs_shape': self.obs_shape,
            'action_dim': self.action_dim,
            'hidden_dim': self.hidden_dim,
            'pi_lr': self.pi_lr,
            'v_lr': self.v_lr,
            'gamma': self.gamma,
            'gae_lambda': self.gae_lambda,
            'clip_ratio': self.clip_ratio,
            'entropy_coeff': self.entropy_coeff,
            'value_coeff': self.value_coeff,
            'max_grad_norm': self.max_grad_norm,
            'device': str(self.device)
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
    
    @staticmethod
    def get_saved_config(path: str):
        """Get the configuration parameters from a saved model."""
        checkpoint = torch.load(path, map_location='cpu')
        if 'obs_shape' in checkpoint:
            return {
                'obs_shape': checkpoint['obs_shape'],
                'action_dim': checkpoint['action_dim'],
                'hidden_dim': checkpoint['hidden_dim'],
                'pi_lr': checkpoint['pi_lr'],
                'v_lr': checkpoint['v_lr'],
                'gamma': checkpoint['gamma'],
                'gae_lambda': checkpoint['gae_lambda'],
                'clip_ratio': checkpoint['clip_ratio'],
                'entropy_coeff': checkpoint['entropy_coeff'],
                'value_coeff': checkpoint['value_coeff'],
                'max_grad_norm': checkpoint['max_grad_norm'],
                'device': checkpoint['device']
            }
        else:
            raise ValueError("This checkpoint does not contain configuration parameters. It was likely saved with an older version.") 
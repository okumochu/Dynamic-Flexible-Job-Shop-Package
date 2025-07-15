import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from RL.PPO.networks import PolicyNetwork, ValueNetwork, PPOUpdater
from RL.PPO.buffer import PPOBuffer
from typing import Tuple

class FlatAgent:
    """
    PPO Worker Agent for Flexible Job Shop Scheduling with separate policy and value networks.
    """
    def __init__(self, 
                 input_dim,
                 action_dim,
                 pi_lr,
                 v_lr,
                 gamma,
                 gae_lambda,
                 clip_ratio,
                 entropy_coef=0.01,
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
        self.entropy_coef = entropy_coef
        self.policy = PolicyNetwork(input_dim, action_dim).to(self.device)
        self.value = ValueNetwork(input_dim).to(self.device)
        self.pi_lr = pi_lr
        self.v_lr = v_lr
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=pi_lr)
        self.value_optimizer = optim.Adam(self.value.parameters(), lr=v_lr)
        
        # Store configuration parameters for saving
        self.input_dim = input_dim
        
        self.training_stats = {
            'policy_loss': [],
            'value_loss': [],
            'entropy_loss': [],
            'total_loss': [],
            'clip_fraction': [],
            'kl': []
        }
    
    def take_action(self, obs: torch.Tensor, action_mask: torch.Tensor) -> Tuple[int, float, float]:
        with torch.no_grad():
            obs = obs.unsqueeze(0).to(self.device)
            action_mask = action_mask.to(self.device)
            action_logits = self.policy(obs)
            action_logits = action_logits.squeeze(0)
            masked_logits = action_logits.clone()
            
            # Mask out invalid actions (~ action_mask reverse, e.g. action_logits = [0.3, 0.5, 0.2] -> masked_logits = [0.3, -inf, 0.2])
            masked_logits[~action_mask] = float('-inf')

            # Sample action
            dist = Categorical(logits=masked_logits)
            action = dist.sample()

            # return log_prob, value
            log_prob = dist.log_prob(action)
            value = self.value(obs)
            return int(action.item()), float(log_prob.item()), float(value.item())
    
    def update(self, buffer: PPOBuffer, train_pi_iters: int, train_v_iters: int, target_kl: float):
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
        
        # Compute GAE advantages using PPOUpdater
        advantages, returns = PPOUpdater.compute_gae_advantages(
            batch['rewards'], batch['values'], batch['dones'], 
            self.gamma, self.gae_lambda
        )
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        

        """
        Use same data to train policy and value function with setting ammount of steps.
        """
        # Policy update
        for i in range(train_pi_iters):

            """
            In first iteration, the policy almost the same as the old policy.
            But, in the second iteration, the policy is updated.
            """
            action_logits = self.policy(batch['observations'])
            masked_logits = action_logits.clone()
            masked_logits[~batch['action_masks']] = float('-inf')
            dist = Categorical(logits=masked_logits)
            new_log_probs = dist.log_prob(batch['actions'])
            
            # Compute entropy using PPOUpdater
            entropy = PPOUpdater.compute_entropy(action_logits, batch['action_masks'])

            # PPO policy loss using PPOUpdater
            policy_loss = PPOUpdater.ppo_policy_loss(
                new_log_probs, batch['log_probs'], advantages, self.clip_ratio
            )
            
            # Total policy loss with entropy penalty
            total_policy_loss = policy_loss - self.entropy_coef * entropy
            
            # update policy
            self.policy_optimizer.zero_grad()
            total_policy_loss.backward()
            self.policy_optimizer.step()

            # KL divergence for early stopping (appox_kl = old_policy_log_prob - new_policy_log_prob)
            with torch.no_grad():
                approx_kl = (batch['log_probs'] - new_log_probs).mean().item()
            if target_kl > 0.0 and approx_kl > 1.5 * target_kl:
                print(f"Early stopping at iter {i} due to reaching max kl: {approx_kl:.4f}")
                break
        
        # Value function update (off-policy, no need early stopping. think it as a regression problem)
        for _ in range(train_v_iters):
            values = self.value(batch['observations'])
            value_loss = F.mse_loss(values.squeeze(), returns)
            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()
        
        # Logging
        stats = {
            'policy_loss': float(policy_loss.item()),
            'value_loss': float(value_loss.item()),
            'entropy': float(entropy.item()),
            'total_loss': float(total_policy_loss.item() + value_loss.item()),
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
            'input_dim': self.input_dim,
            'action_dim': self.action_dim,
            'pi_lr': self.pi_lr,
            'v_lr': self.v_lr,
            'gamma': self.gamma,
            'gae_lambda': self.gae_lambda,
            'clip_ratio': self.clip_ratio,
            'entropy_coef': self.entropy_coef,
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
    

    @staticmethod
    def get_saved_config(path: str):
        """Get the configuration parameters from a saved model."""
        checkpoint = torch.load(path, map_location='cpu')
        if 'input_dim' in checkpoint:
            return {
                'input_dim': checkpoint['input_dim'],
                'action_dim': checkpoint['action_dim'],
                'pi_lr': checkpoint['pi_lr'],
                'v_lr': checkpoint['v_lr'],
                'gamma': checkpoint['gamma'],
                'gae_lambda': checkpoint['gae_lambda'],
                'clip_ratio': checkpoint['clip_ratio'],
                'entropy_coef': checkpoint.get('entropy_coef', 0.01),  # Default for backward compatibility
                'device': checkpoint['device']
            }
        else:
            raise ValueError("This checkpoint does not contain configuration parameters. It was likely saved with an older version.") 

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
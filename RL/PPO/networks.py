import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical
from typing import Dict, Tuple

"""
PPO Update Class for code reuse
"""

class PPOUpdater:
    """
    Common PPO update logic for both flat and hierarchical agents.
    """
    
    @staticmethod
    def compute_gae_advantages(rewards, values, dones, gamma, gae_lambda):
        """
        Compute GAE advantages and returns.
        Returns:
            advantages: GAE advantages
            returns: advantages + values (targets for value function)
        """
        advantages = torch.zeros_like(rewards)
        gae = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            not_done = 1.0 - dones[t].float()
            delta = rewards[t] + gamma * next_value * not_done - values[t]
            gae = delta + gamma * gae_lambda * not_done * gae
            advantages[t] = gae
        
        returns = advantages + values
        return advantages, returns
    
    @staticmethod
    def ppo_policy_loss(new_log_probs, old_log_probs, advantages, clip_ratio):
        """
        Compute PPO clipped policy loss.
        """
        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * advantages
        return -torch.min(surr1, surr2).mean()
    
    @staticmethod
    def compute_entropy(logits, action_mask=None):
        """
        Compute entropy for policy regularization.
        """
        if action_mask is not None:
            masked_logits = logits.clone()
            masked_logits[~action_mask] = float('-inf')
            dist = Categorical(logits=masked_logits)
        else:
            dist = Categorical(logits=logits)
        return dist.entropy().mean()


"""
Following is the flat RL network.
"""

class PolicyNetwork(nn.Module):
    """
    Separate policy network for PPO agent.
    Takes observation and outputs action logits.
    """
    def __init__(self, input_dim, action_dim, hidden_dim = 256):
        super().__init__()
        self.policy_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )
        self._init_weights()
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0)
    def forward(self, obs):
        # obs: (batch_size, input_dim)
        return self.policy_network(obs)

class ValueNetwork(nn.Module):
    """
    Separate value network for PPO agent.
    Takes observation and outputs state value.
    """
    def __init__(self, input_dim, hidden_dim = 128):
        super().__init__()
        self.value_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        self._init_weights()
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0)
    def forward(self, obs):
        # obs: (batch_size, input_dim)
        return self.value_network(obs)


"""
Following is the hierarchical RL network.
"""

class PerceptualEncoder(nn.Module):
    
    def __init__(self, input_dim: int, latent_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
    
    def forward(self, x):
        z = self.encoder(x)
        return z


class ManagerPolicy(nn.Module):
    """Manager policy network that emits unit-norm goal vectors"""
    
    def __init__(self, latent_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        
        self.policy_network = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, z):
        """
        Args:
            z: [batch_size, latent_dim] = encoded state
        Returns:
            goals: [batch_size, latent_dim] = unit-norm goal vectors
        """
        # Generate goals
        raw_goals = self.policy_network(z)
        goals = F.normalize(raw_goals, p=2, dim=-1)  # Unit-norm goal vectors
        
        return goals


class WorkerPolicy(nn.Module):
    """Worker policy network that combines PPO with goal-conditioned policy (policy only)"""
    
    def __init__(self, latent_dim: int, action_dim: int, 
                 goal_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.goal_dim = goal_dim
        
        # Goal processing (bias-free as specified)
        self.goal_transform = nn.Linear(latent_dim, goal_dim, bias=False)
        
        # Policy network outputs action-goal compatibility matrix
        self.policy_backbone = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim * goal_dim)  # Output |A| × k logits
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights using orthogonal initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                if module.bias is not None:  # Only initialize bias if it exists
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, z_t, pooled_goals):
        """
        Args:
            z_t: [batch_size, latent_dim] = encoded state
            pooled_goals: [batch_size, latent_dim] = pooled manager goals
        Returns:
            action_probs: [batch_size, action_dim] = action probabilities
        """
        
        # Transform pooled goals to worker space
        w_t = self.goal_transform(pooled_goals)  # [batch_size, goal_dim]
        
        # Policy network outputs action-goal compatibility matrix
        action_goal_logits = self.policy_backbone(z_t)  # [batch_size, action_dim * goal_dim]
        
        # Reshape to [batch_size, action_dim, goal_dim]
        action_goal_matrix = action_goal_logits.view(-1, self.action_dim, self.goal_dim)
        
        # Matrix multiplication: [batch_size, action_dim, goal_dim] @ [batch_size, goal_dim, 1]
        # -> [batch_size, action_dim]
        action_logits = torch.bmm(action_goal_matrix, w_t.unsqueeze(-1)).squeeze(-1)
        
        return action_logits

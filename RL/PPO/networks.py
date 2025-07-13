import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


"""
Following is the flat RL network.
"""

class PolicyNetwork(nn.Module):
    """
    Separate policy network for PPO agent.
    Takes observation and outputs action logits.
    """
    def __init__(self, input_dim, action_dim, hidden_dim = 128):
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
    
    def __init__(self, input_dim: int, latent_dim: int, hidden_dim: int = 128):
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


class MLPManager(nn.Module):
    """Simple MLP Manager that emits unit-norm goal vectors"""
    
    def __init__(self, latent_dim: int, hidden_dim: int = 128):
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
        
        # Value function for manager
        self.value_network = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
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
            values: [batch_size, 1] = value estimates
        """
        # Generate goals
        raw_goals = self.policy_network(z)
        goals = F.normalize(raw_goals, p=2, dim=-1)  # Unit-norm goal vectors
        
        # Generate values
        values = self.value_network(z)  # [batch_size, 1]
        
        return goals, values


class HierarchicalWorker(nn.Module):
    """Worker that combines PPO with goal-conditioned policy"""
    
    def __init__(self, latent_dim: int, action_dim: int, 
                 goal_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.goal_dim = goal_dim
        
        # Goal processing (bias-free as specified)
        self.goal_transform = nn.Linear(latent_dim, goal_dim, bias=False)
        
        # Action embedding matrix U ∈ R^{|A| × k}
        self.action_embedding = nn.Parameter(torch.randn(action_dim, goal_dim))
        nn.init.xavier_uniform_(self.action_embedding)
        
        # Policy network backbone
        input_dim = latent_dim + goal_dim  # z_t + pooled_goals
        self.policy_backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)  # Direct output to action_dim
        )

        self.value_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights using orthogonal initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0)
    
    def forward(self, z_t, pooled_goals):
        """
        Args:
            z_t: [batch_size, latent_dim] = encoded state
            pooled_goals: [batch_size, latent_dim] = pooled manager goals
        """
        
        # Transform pooled goals to worker space
        w_t = self.goal_transform(pooled_goals)  # [batch_size, goal_dim]
        
        # Combine inputs
        policy_input = torch.cat([z_t, w_t], dim=-1)
        
        # Policy backbone for basic logits
        policy_logits = self.policy_backbone(policy_input)  # [batch_size, action_dim]
        
        # Compute action logits using goal-action embedding (simplified)
        action_contributions = torch.matmul(w_t, self.action_embedding.T)  # [batch_size, action_dim]
        
        # Final action logits
        action_logits = policy_logits + action_contributions
        
        # Policy distribution
        action_probs = F.softmax(action_logits, dim=-1)
        
        # Value estimate
        value = self.value_network(policy_input)
        
        return action_probs, value
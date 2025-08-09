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
        # Normalize advantages for stability; returns remain unnormalized targets
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
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

    @staticmethod
    def transition_policy_loss(encoded_states, goals, advantages):
        """
        Compute transition policy gradient.
        """
        # the current goal means the direction of the transition from s_t to s_t+c
        cosine_alignment = F.cosine_similarity(encoded_states[1:] - encoded_states[:-1], goals[:-1])
        # Use advantages[:-1] to match the size of cosine_alignment (one less element)
        adv = advantages[:-1]
        return -torch.mean(adv.detach() * cosine_alignment)



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
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
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
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
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
    
    def __init__(self, input_dim: int, latent_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
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
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        z = self.encoder(x)
        return z

class ManagerEncoder(nn.Module):
    """
    Manager-specific encoder that processes PerceptualEncoder output.
    Updates only from manager's value loss.
    """
    def __init__(self, latent_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            # Output must match latent_dim so it can be consumed by ManagerPolicy (GRU input_size=latent_dim)
            # and the Manager ValueNetwork (which expects latent_dim inputs)
            nn.Linear(hidden_dim, latent_dim),
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        """
        Args:
            x: [batch_size, latent_dim] = output from PerceptualEncoder
        Returns:
            z: [batch_size, latent_dim] = manager-specific encoded state
        """
        z = self.encoder(x)
        return z


class ManagerPolicy(nn.Module):
    """Manager policy network that emits unit-norm goal vectors"""
    
    def __init__(self, latent_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        
        # GRU for temporal dynamics
        self.recurrent_network = nn.GRU(input_size=latent_dim, hidden_size=hidden_dim, batch_first=True)
        
        # Policy network after GRU
        self.policy_network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),  # Output goal in latent space
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
        # Process through GRU with proper batch_first=True
        # z already has batch dimension, add sequence dimension
        z_seq = z.unsqueeze(1)  # [batch_size, 1, latent_dim]
        z_recurrent, _ = self.recurrent_network(z_seq)  # [batch_size, 1, hidden_dim]
        z_recurrent = z_recurrent.squeeze(1)  # [batch_size, hidden_dim]
        
        # Process GRU output through policy network
        raw_goals = self.policy_network(z_recurrent)
        goals = F.normalize(raw_goals, p=2, dim=-1)  # Unit-norm goal vectors
        
        return goals


class WorkerPolicy(nn.Module):
    """Worker policy network that combines PPO with goal-conditioned policy (policy only)"""
    
    def __init__(self, latent_dim: int, action_dim: int, 
                 goal_dim: int, hidden_dim: int = 64):
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


class HybridPolicyNetwork(nn.Module):
    """
    Hybrid policy network for PPO agent, outputting both discrete action logits 
    and continuous action parameters (mean and log_std) for idleness.
    """
    def __init__(self, input_dim, discrete_action_dim, continuous_action_dim, hidden_dim=128):
        super().__init__()
        self.discrete_action_dim = discrete_action_dim
        self.continuous_action_dim = continuous_action_dim
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.discrete_head = nn.Linear(hidden_dim, discrete_action_dim)
        # For continuous action (idleness), output mean and log_std
        self.continuous_mean_head = nn.Linear(hidden_dim, continuous_action_dim)
        self.continuous_log_std_head = nn.Linear(hidden_dim, continuous_action_dim)
        
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0)
                
    def forward(self, obs):
        features = self.feature_extractor(obs)
        discrete_logits = self.discrete_head(features)
        continuous_mean = self.continuous_mean_head(features)
        continuous_log_std = self.continuous_log_std_head(features)
        
        # Clamp log_std to avoid numerical instability (e.g., extremely small std)
        continuous_log_std = torch.clamp(continuous_log_std, min=-20, max=2)
        
        return discrete_logits, continuous_mean, continuous_log_std


class HybridWorkerPolicy(nn.Module):
    """
    Worker policy network that combines PPO with goal-conditioned policy, 
    outputting both discrete action logits and continuous action parameters (mean and log_std).
    """
    def __init__(self, latent_dim: int, discrete_action_dim: int, 
                 continuous_action_dim: int, goal_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.latent_dim = latent_dim
        self.discrete_action_dim = discrete_action_dim
        self.continuous_action_dim = continuous_action_dim
        self.goal_dim = goal_dim
        
        # Goal processing (bias-free as specified)
        self.goal_transform = nn.Linear(latent_dim, goal_dim, bias=False)
        
        # Policy backbone for features (shared for discrete and continuous heads)
        self.policy_backbone = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Discrete action head outputs action-goal compatibility matrix
        self.discrete_action_goal_head = nn.Linear(hidden_dim, discrete_action_dim * goal_dim)
        
        # Continuous action heads for mean and log_std
        self.continuous_mean_head = nn.Linear(hidden_dim, continuous_action_dim)
        self.continuous_log_std_head = nn.Linear(hidden_dim, continuous_action_dim)
        
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
            discrete_logits: [batch_size, discrete_action_dim] = discrete action logits
            continuous_mean: [batch_size, continuous_action_dim] = mean for continuous action
            continuous_log_std: [batch_size, continuous_action_dim] = log_std for continuous action
        """
        
        # Transform pooled goals to worker space
        w_t = self.goal_transform(pooled_goals)  # [batch_size, goal_dim]
        
        # Policy backbone features
        features = self.policy_backbone(z_t)
        
        # Discrete action head
        action_goal_logits = self.discrete_action_goal_head(features)  # [batch_size, discrete_action_dim * goal_dim]
        action_goal_matrix = action_goal_logits.view(-1, self.discrete_action_dim, self.goal_dim)
        discrete_logits = torch.bmm(action_goal_matrix, w_t.unsqueeze(-1)).squeeze(-1)
        
        # Continuous action heads
        continuous_mean = self.continuous_mean_head(features)
        continuous_log_std = self.continuous_log_std_head(features)
        continuous_log_std = torch.clamp(continuous_log_std, min=-20, max=2)
        
        return discrete_logits, continuous_mean, continuous_log_std

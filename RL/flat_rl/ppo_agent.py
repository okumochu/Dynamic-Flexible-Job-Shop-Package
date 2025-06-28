"""
PPO Agent for Flexible Job Shop Scheduling
Implements PPO with multi-objective optimization support
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from typing import Tuple, List, Dict, Any
from collections import deque
import random

class PolicyNetwork(nn.Module):
    """
    Separate policy network for PPO agent.
    Takes observation and outputs action logits.
    """
    
    def __init__(self, obs_shape: Tuple[int, int, int], action_dim: int, hidden_dim: int = 256):
        """
        Initialize policy network.
        
        Args:
            obs_shape: Observation shape (max_jobs, max_machines, channels)
            action_dim: Number of possible actions
            hidden_dim: Hidden layer dimension
        """
        super().__init__()
        
        self.max_jobs, self.max_machines, self.channels = obs_shape
        
        # Layer normalization over channel dimension
        self.layer_norm = nn.LayerNorm(self.channels)
        
        # Calculate flattened input size
        flattened_size = self.max_jobs * self.max_machines * self.channels
        
        # Backbone layers
        self.backbone = nn.Sequential(
            nn.Linear(flattened_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        
        # Actor head (outputs action logits)
        self.actor = nn.Linear(hidden_dim // 2, action_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0)
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the policy network.
        
        Args:
            obs: Observation tensor of shape (batch_size, max_jobs, max_machines, channels)
            
        Returns:
            action_logits: Logits for action selection
        """
        batch_size = obs.shape[0]
        
        # Apply layer normalization over channel dimension
        obs_norm = self.layer_norm(obs.view(batch_size, -1, self.channels))
        
        # Flatten observation
        obs_flat = obs_norm.view(batch_size, -1)
        
        # Pass through backbone
        features = self.backbone(obs_flat)
        
        # Get action logits
        action_logits = self.actor(features)
        
        return action_logits

class ValueNetwork(nn.Module):
    """
    Separate value network for PPO agent.
    Takes observation and outputs state value.
    """
    
    def __init__(self, obs_shape: Tuple[int, int, int], hidden_dim: int = 256):
        """
        Initialize value network.
        
        Args:
            obs_shape: Observation shape (max_jobs, max_machines, channels)
            hidden_dim: Hidden layer dimension
        """
        super().__init__()
        
        self.max_jobs, self.max_machines, self.channels = obs_shape
        
        # Layer normalization over channel dimension
        self.layer_norm = nn.LayerNorm(self.channels)
        
        # Calculate flattened input size
        flattened_size = self.max_jobs * self.max_machines * self.channels
        
        # Backbone layers
        self.backbone = nn.Sequential(
            nn.Linear(flattened_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        
        # Value head (outputs state value)
        self.value = nn.Linear(hidden_dim // 2, 1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0)
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the value network.
        
        Args:
            obs: Observation tensor of shape (batch_size, max_jobs, max_machines, channels)
            
        Returns:
            value: State value estimate
        """
        batch_size = obs.shape[0]
        
        # Apply layer normalization over channel dimension
        obs_norm = self.layer_norm(obs.view(batch_size, -1, self.channels))
        
        # Flatten observation
        obs_flat = obs_norm.view(batch_size, -1)
        
        # Pass through backbone
        features = self.backbone(obs_flat)
        
        # Get value
        value = self.value(features)
        
        return value

class PPOBuffer:
    """
    Buffer for storing PPO training data.
    """
    
    def __init__(self, buffer_size: int, obs_shape: Tuple[int, int, int], action_dim: int, device: torch.device):
        """
        Initialize PPO buffer.
        
        Args:
            buffer_size: Maximum number of transitions to store
            obs_shape: Observation shape
            action_dim: Number of possible actions
            device: Device to store tensors on
        """
        self.buffer_size = buffer_size
        self.device = device
        self.ptr = 0
        self.size = 0
        
        # Initialize storage
        self.observations = torch.zeros((buffer_size, *obs_shape), dtype=torch.float32, device=device)
        self.actions = torch.zeros(buffer_size, dtype=torch.long, device=device)
        self.rewards = torch.zeros(buffer_size, dtype=torch.float32, device=device)
        self.values = torch.zeros(buffer_size, dtype=torch.float32, device=device)
        self.log_probs = torch.zeros(buffer_size, dtype=torch.float32, device=device)
        self.action_masks = torch.zeros((buffer_size, action_dim), dtype=torch.bool, device=device)
        self.dones = torch.zeros(buffer_size, dtype=torch.bool, device=device)
    
    def add(self, obs: torch.Tensor, action: int, reward: float, value: float, 
            log_prob: float, action_mask: torch.Tensor, done: bool):
        """Add a transition to the buffer."""
        self.observations[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value
        self.log_probs[self.ptr] = log_prob
        self.action_masks[self.ptr] = action_mask
        self.dones[self.ptr] = done
        
        self.ptr = (self.ptr + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)
    
    def get_batch(self) -> Dict[str, torch.Tensor]:
        """Get all stored data as a batch."""
        return {
            'observations': self.observations[:self.size],
            'actions': self.actions[:self.size],
            'rewards': self.rewards[:self.size],
            'values': self.values[:self.size],
            'log_probs': self.log_probs[:self.size],
            'action_masks': self.action_masks[:self.size],
            'dones': self.dones[:self.size]
        }
    
    def clear(self):
        """Clear the buffer."""
        self.ptr = 0
        self.size = 0

class PPOAgent:
    """
    PPO Agent for Flexible Job Shop Scheduling with separate policy and value networks.
    """
    
    def __init__(self, 
                 obs_shape: Tuple[int, int, int],
                 action_dim: int,
                 hidden_dim: int = 256,
                 lr: float = 3e-4,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 clip_ratio: float = 0.2,
                 entropy_coeff: float = 1e-3,
                 value_coeff: float = 0.5,
                 max_grad_norm: float = 0.5,
                 device: str = 'auto'):
        """
        Initialize PPO agent.
        
        Args:
            obs_shape: Observation shape
            action_dim: Number of possible actions
            hidden_dim: Hidden layer dimension
            lr: Learning rate
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            clip_ratio: PPO clip ratio
            entropy_coeff: Entropy coefficient for exploration
            value_coeff: Value loss coefficient
            max_grad_norm: Maximum gradient norm for clipping
            device: Device to run on ('auto', 'cpu', 'cuda')
        """
        # Auto-detect device
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
        
        # Create separate policy and value networks
        self.policy = PolicyNetwork(obs_shape, action_dim, hidden_dim).to(self.device)
        self.value = ValueNetwork(obs_shape, hidden_dim).to(self.device)
        
        # Create optimizers
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.value_optimizer = optim.Adam(self.value.parameters(), lr=lr)
        
        # Training statistics
        self.training_stats = {
            'policy_loss': [],
            'value_loss': [],
            'entropy_loss': [],
            'total_loss': [],
            'clip_fraction': []
        }
    
    def get_action(self, obs: torch.Tensor, action_mask: torch.Tensor) -> Tuple[int, float, float]:
        """
        Get action from current policy.
        
        Args:
            obs: Current observation
            action_mask: Boolean mask of valid actions
            
        Returns:
            action: Selected action
            log_prob: Log probability of selected action
            value: State value estimate
        """
        with torch.no_grad():
            obs = obs.unsqueeze(0).to(self.device)  # Add batch dimension
            action_mask = action_mask.to(self.device)
            
            # Get action logits and value from separate networks
            action_logits = self.policy(obs)
            value = self.value(obs)
            
            action_logits = action_logits.squeeze(0)  # Remove batch dim for masking
            
            # Apply action mask (set invalid actions to -inf)
            masked_logits = action_logits.clone()
            masked_logits[~action_mask] = float('-inf')
            
            # Sample action
            dist = Categorical(logits=masked_logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            
            return int(action.item()), float(log_prob.item()), float(value.item())
    
    def compute_gae_returns(self, rewards: torch.Tensor, values: torch.Tensor, 
                           dones: torch.Tensor) -> torch.Tensor:
        """
        Compute GAE returns.
        
        Args:
            rewards: Reward sequence
            values: Value estimates
            dones: Done flags
            
        Returns:
            returns: GAE returns
        """
        returns = torch.zeros_like(rewards)
        advantages = torch.zeros_like(rewards)
        
        # Compute GAE
        gae = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            # Cast dones to float for arithmetic
            not_done = 1.0 - dones[t].float()
            delta = rewards[t] + self.gamma * next_value * not_done - values[t]
            gae = delta + self.gamma * self.gae_lambda * not_done * gae
            advantages[t] = gae
        
        # Compute returns
        returns = advantages + values
        
        return returns
    
    def update(self, buffer: PPOBuffer, num_epochs: int = 4) -> Dict[str, float]:
        """
        Update policy using PPO.
        
        Args:
            buffer: Buffer containing training data
            num_epochs: Number of epochs to train
            
        Returns:
            stats: Training statistics
        """
        batch = buffer.get_batch()
        
        # Compute GAE returns
        returns = self.compute_gae_returns(batch['rewards'], batch['values'], batch['dones'])
        advantages = returns - batch['values']
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Training loop
        for epoch in range(num_epochs):
            # Get current policy outputs
            action_logits = self.policy(batch['observations'])
            values = self.value(batch['observations'])
            
            # Apply action mask
            masked_logits = action_logits.clone()
            masked_logits[~batch['action_masks']] = float('-inf')
            
            # Create distribution
            dist = Categorical(logits=masked_logits)
            new_log_probs = dist.log_prob(batch['actions'])
            
            # Compute ratio
            ratio = torch.exp(new_log_probs - batch['log_probs'])
            
            # Compute clipped surrogate loss
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = F.mse_loss(values.squeeze(), returns)
            
            # Entropy loss
            entropy_loss = -dist.entropy().mean()
            
            # Total loss
            total_loss = (policy_loss + 
                         self.value_coeff * value_loss + 
                         self.entropy_coeff * entropy_loss)
            
            # Optimize policy
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy_optimizer.step()
            
            # Optimize value function
            self.value_optimizer.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.value.parameters(), self.max_grad_norm)
            self.value_optimizer.step()
            
            # Record statistics
            clip_fraction = (abs(ratio - 1) > self.clip_ratio).float().mean().item()
            
            self.training_stats['policy_loss'].append(policy_loss.item())
            self.training_stats['value_loss'].append(value_loss.item())
            self.training_stats['entropy_loss'].append(entropy_loss.item())
            self.training_stats['total_loss'].append(total_loss.item())
            self.training_stats['clip_fraction'].append(clip_fraction)
        
        # Return average statistics
        stats = {
            'policy_loss': float(np.mean(self.training_stats['policy_loss'][-num_epochs:])),
            'value_loss': float(np.mean(self.training_stats['value_loss'][-num_epochs:])),
            'entropy_loss': float(np.mean(self.training_stats['entropy_loss'][-num_epochs:])),
            'total_loss': float(np.mean(self.training_stats['total_loss'][-num_epochs:])),
            'clip_fraction': float(np.mean(self.training_stats['clip_fraction'][-num_epochs:]))
        }
        
        return stats
    
    def save(self, path: str):
        """Save model to file."""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'value_state_dict': self.value.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'value_optimizer_state_dict': self.value_optimizer.state_dict(),
            'training_stats': self.training_stats
        }, path)
    
    def load(self, path: str):
        """Load model from file."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.value.load_state_dict(checkpoint['value_state_dict'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        self.value_optimizer.load_state_dict(checkpoint['value_optimizer_state_dict'])
        self.training_stats = checkpoint['training_stats'] 
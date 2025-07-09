import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PerceptualEncoder(nn.Module):
    """Shared perceptual encoder: 2 × FC 512 + ReLU → latent dim 256"""
    
    def __init__(self, input_dim: int, latent_dim: int = 256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim)
        )
        self.layer_norm = nn.LayerNorm(latent_dim)
    
    def forward(self, x):
        z = self.encoder(x)
        return self.layer_norm(z)

class PolicyNetwork(nn.Module):
    """
    Separate policy network for PPO agent.
    Takes observation and outputs action logits.
    """
    def __init__(self, input_dim, action_dim, hidden_dim):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        self.actor = nn.Linear(hidden_dim // 2, action_dim)
        self._init_weights()
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0)
    def forward(self, obs):
        # obs: (batch_size, input_dim)
        features = self.backbone(obs)
        action_logits = self.actor(features)
        return action_logits

class ValueNetwork(nn.Module):
    """
    Separate value network for PPO agent.
    Takes observation and outputs state value.
    """
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        self.value = nn.Linear(hidden_dim // 2, 1)
        self._init_weights()
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0)
    def forward(self, obs):
        # obs: (batch_size, input_dim)
        features = self.backbone(obs)
        value = self.value(features)
        return value


class DilatedLSTMManager(nn.Module):
    """Manager with dilated-LSTM that emits unit-norm goal vectors"""
    
    def __init__(self, latent_dim: int = 256, hidden_dim: int = 256, dilation: int = 10):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.dilation = dilation
        
        # Dilated LSTM (implemented as regular LSTM with manual dilation handling)
        self.lstm = nn.LSTM(latent_dim, hidden_dim, batch_first=True)
        self.goal_projection = nn.Linear(hidden_dim, latent_dim)
        
        # Value function for manager
        self.value_head = nn.Linear(hidden_dim, 1)
        
        # Initialize goal projection to encourage diverse initial goals
        nn.init.xavier_uniform_(self.goal_projection.weight)
        nn.init.zeros_(self.goal_projection.bias)
    
    def forward(self, z_sequence, hidden=None):
        """
        Args:
            z_sequence: [batch_size, seq_len, latent_dim] - sequence of encoded states
            hidden: LSTM hidden state
        Returns:
            goals: [batch_size, seq_len//dilation, latent_dim] - unit-norm goal vectors
            values: [batch_size, seq_len//dilation] - value estimates
            hidden: updated LSTM hidden state
        """
        batch_size, seq_len, _ = z_sequence.shape
        
        # Extract dilated timesteps
        dilated_indices = torch.arange(0, seq_len, self.dilation)
        if len(dilated_indices) == 0:
            # Handle case where sequence is shorter than dilation
            dilated_indices = torch.tensor([seq_len - 1]) if seq_len > 0 else torch.tensor([0])
        
        dilated_z = z_sequence[:, dilated_indices]  # [batch_size, dilated_len, latent_dim]
        
        # LSTM forward pass
        lstm_out, hidden = self.lstm(dilated_z, hidden)  # [batch_size, dilated_len, hidden_dim]
        
        # Generate goals and values
        raw_goals = self.goal_projection(lstm_out)  # [batch_size, dilated_len, latent_dim]
        goals = F.normalize(raw_goals, p=2, dim=-1)  # Unit-norm goal vectors
        values = self.value_head(lstm_out).squeeze(-1)  # [batch_size, dilated_len]
        
        return goals, values, hidden
    
    def get_goal_at_step(self, z_t, step, hidden=None):
        """Get goal for current step (used during rollout)"""
        if step % self.dilation == 0:
            # Time to emit new goal
            z_input = z_t.unsqueeze(0).unsqueeze(0)  # [1, 1, latent_dim]
            goals, values, hidden = self.forward(z_input, hidden)
            return goals.squeeze(0).squeeze(0), values.squeeze(0).squeeze(0), hidden
        else:
            # Return previous goal (handled externally)
            return None, None, hidden


class HierarchicalWorker(nn.Module):
    """Worker that combines PPO with goal-conditioned policy"""
    
    def __init__(self, latent_dim: int = 256, action_dim: int = 100, 
                 goal_dim: int = 32, hidden_dim: int = 512):
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
        input_dim = latent_dim + goal_dim + 1  # z_t + pooled_goals + prev_r_int
        self.policy_backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Policy and value heads
        self.policy_head = nn.Linear(hidden_dim + goal_dim, action_dim)  # +goal_dim for action embedding
        self.value_head = nn.Linear(hidden_dim, 1)
        
    def forward(self, z_t, pooled_goals, prev_r_int, action_mask=None):
        """
        Args:
            z_t: [batch_size, latent_dim] - encoded state
            pooled_goals: [batch_size, latent_dim] - pooled manager goals
            prev_r_int: [batch_size, 1] - previous intrinsic reward
            action_mask: [batch_size, action_dim] - valid action mask
        """
        batch_size = z_t.shape[0]
        
        # Transform pooled goals to worker space
        w_t = self.goal_transform(pooled_goals)  # [batch_size, goal_dim]
        
        # Combine inputs
        policy_input = torch.cat([z_t, w_t, prev_r_int], dim=-1)
        
        # Policy backbone
        features = self.policy_backbone(policy_input)  # [batch_size, hidden_dim]
        
        # Compute action logits using goal-action embedding
        action_contributions = torch.matmul(w_t, self.action_embedding.T)  # [batch_size, action_dim]
        
        policy_logits = self.policy_head(torch.cat([features, w_t], dim=-1))
        action_logits = policy_logits + action_contributions
        
        # Apply action mask if provided
        if action_mask is not None:
            action_logits = action_logits.masked_fill(~action_mask, float('-inf'))
        
        # Policy distribution
        action_probs = F.softmax(action_logits, dim=-1)
        
        # Value estimate
        value = self.value_head(features)
        
        return action_probs, value
    
    def get_action_and_value(self, z_t, pooled_goals, prev_r_int, action_mask=None):
        """Sample action and get value estimate"""
        action_probs, value = self.forward(z_t, pooled_goals, prev_r_int, action_mask)
        
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        return action, log_prob, value.squeeze(-1)
    
    def get_deterministic_action(self, z_t, pooled_goals, prev_r_int, action_mask=None):
        """Get deterministic action (for evaluation)"""
        action_probs, _ = self.forward(z_t, pooled_goals, prev_r_int, action_mask)
        
        if action_mask is not None:
            # Set invalid actions to 0 probability
            masked_probs = action_probs * action_mask.float()
            action = torch.argmax(masked_probs, dim=-1)
        else:
            action = torch.argmax(action_probs, dim=-1)
        
        return action
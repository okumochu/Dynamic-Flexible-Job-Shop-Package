import torch
import torch.nn as nn
import numpy as np

class PolicyNetwork(nn.Module):
    """
    Separate policy network for PPO agent.
    Takes observation and outputs action logits.
    """
    def __init__(self, obs_shape, action_dim, hidden_dim=256):
        super().__init__()
        input_dim = obs_shape[0]  # 1D observation
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
    def __init__(self, obs_shape, hidden_dim=256):
        super().__init__()
        input_dim = obs_shape[0]  # 1D observation
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
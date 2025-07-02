import torch
from typing import Tuple, Dict

class PPOBuffer:
    """
    Buffer for storing PPO training data.
    """
    def __init__(self, buffer_size: int, obs_shape: Tuple[int], action_dim: int, device: torch.device):
        self.buffer_size = buffer_size
        self.device = device
        self.ptr = 0
        self.size = 0
        self.observations = torch.zeros((buffer_size, *obs_shape), dtype=torch.float32, device=device)
        self.actions = torch.zeros(buffer_size, dtype=torch.long, device=device)
        self.rewards = torch.zeros(buffer_size, dtype=torch.float32, device=device)
        self.values = torch.zeros(buffer_size, dtype=torch.float32, device=device)
        self.log_probs = torch.zeros(buffer_size, dtype=torch.float32, device=device)
        self.action_masks = torch.zeros((buffer_size, action_dim), dtype=torch.bool, device=device)
        self.dones = torch.zeros(buffer_size, dtype=torch.bool, device=device)

    def add(self, obs: torch.Tensor, action: int, reward: float, value: float, 
            log_prob: float, action_mask: torch.Tensor, done: bool):
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
        self.ptr = 0
        self.size = 0 
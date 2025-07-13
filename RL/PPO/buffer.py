import torch
from typing import Tuple, Dict, List

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


class ManagerBuffer:
    """
    Buffer for storing Manager training data.
    """
    def __init__(self, buffer_size: int, latent_dim: int, device: torch.device):
        self.buffer_size = buffer_size
        self.device = device
        self.ptr = 0
        self.size = 0
        self.states = torch.zeros((buffer_size, latent_dim), dtype=torch.float32, device=device)
        self.goals = torch.zeros((buffer_size, latent_dim), dtype=torch.float32, device=device)
        self.values = torch.zeros(buffer_size, dtype=torch.float32, device=device)
        self.rewards = torch.zeros(buffer_size, dtype=torch.float32, device=device)
        self.dones = torch.zeros(buffer_size, dtype=torch.bool, device=device)

    def add(self, state: torch.Tensor, goal: torch.Tensor, value: float, reward: float, done: bool):
        """Add manager experience to buffer"""
        self.states[self.ptr] = state
        self.goals[self.ptr] = goal
        self.values[self.ptr] = value
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)

    def get_batch(self) -> Dict[str, torch.Tensor]:
        """Get all stored manager experiences"""
        return {
            'states': self.states[:self.size],
            'goals': self.goals[:self.size],
            'values': self.values[:self.size],
            'rewards': self.rewards[:self.size],
            'dones': self.dones[:self.size]
        }

    def clear(self):
        """Clear the buffer"""
        self.ptr = 0
        self.size = 0


class HierarchicalBuffer:
    """
    Buffer for storing hierarchical rollout data (encoded states, goals, pooled goals).
    More structured approach for collecting hierarchical data during rollouts.
    """
    def __init__(self, buffer_size: int, latent_dim: int, device: torch.device):
        self.buffer_size = buffer_size
        self.device = device
        self.ptr = 0
        self.size = 0
        
        # Hierarchical data storage
        self.encoded_states = torch.zeros((buffer_size, latent_dim), dtype=torch.float32, device=device)
        self.goals = torch.zeros((buffer_size, latent_dim), dtype=torch.float32, device=device)
        self.pooled_goals = torch.zeros((buffer_size, latent_dim), dtype=torch.float32, device=device)
        self.step_indices = torch.zeros(buffer_size, dtype=torch.long, device=device)
        
        # Episode tracking
        self.episode_starts = []
        self.episode_ends = []
        
    def add(self, encoded_state: torch.Tensor, goal: torch.Tensor, pooled_goal: torch.Tensor, step_idx: int):
        """Add hierarchical step data"""
        self.encoded_states[self.ptr] = encoded_state
        self.goals[self.ptr] = goal if goal is not None else torch.zeros_like(encoded_state)
        self.pooled_goals[self.ptr] = pooled_goal
        self.step_indices[self.ptr] = step_idx
        self.ptr = (self.ptr + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)
        
    def mark_episode_start(self):
        """Mark the start of a new episode"""
        self.episode_starts.append(self.ptr)
        
    def mark_episode_end(self):
        """Mark the end of the current episode"""
        self.episode_ends.append(self.ptr)
        
    def get_data(self) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """Get hierarchical data as lists (for compatibility with existing code)"""
        encoded_states = [self.encoded_states[i] for i in range(self.size)]
        goals = [self.goals[i] for i in range(self.size)]
        pooled_goals = [self.pooled_goals[i] for i in range(self.size)]
        return encoded_states, goals, pooled_goals
        
    def get_episode_data(self, episode_idx: int) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """Get data for a specific episode"""
        if episode_idx >= len(self.episode_starts) or episode_idx >= len(self.episode_ends):
            return [], [], []
            
        start = self.episode_starts[episode_idx]
        end = self.episode_ends[episode_idx]
        
        encoded_states = [self.encoded_states[i] for i in range(start, end)]
        goals = [self.goals[i] for i in range(start, end)]
        pooled_goals = [self.pooled_goals[i] for i in range(start, end)]
        return encoded_states, goals, pooled_goals
        
    def clear(self):
        """Clear the buffer"""
        self.ptr = 0
        self.size = 0
        self.episode_starts.clear()
        self.episode_ends.clear() 
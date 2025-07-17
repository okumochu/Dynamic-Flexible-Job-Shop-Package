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


class HierarchicalPPOBuffer:
    """
    Simplified buffer for storing hierarchical RL data.
    Uses single buffer with proper indexing instead of separate pointers.
    """
    def __init__(self, buffer_size: int, latent_dim: int, device: torch.device):
        self.buffer_size = buffer_size
        self.device = device
        
        # FIX: Use single pointer system to avoid misalignment
        self.ptr = 0
        self.size = 0

        # Manager experience data
        self.manager_states = torch.zeros((buffer_size, latent_dim), dtype=torch.float32, device=device)
        self.manager_goals = torch.zeros((buffer_size, latent_dim), dtype=torch.float32, device=device)
        self.manager_values = torch.zeros(buffer_size, dtype=torch.float32, device=device)
        self.manager_rewards = torch.zeros(buffer_size, dtype=torch.float32, device=device)
        self.manager_dones = torch.zeros(buffer_size, dtype=torch.bool, device=device)
        
        # Hierarchical step data (pooled goals for each step)
        self.pooled_goals = torch.zeros((buffer_size, latent_dim), dtype=torch.float32, device=device)
        
        # Track what data is valid
        self.manager_mask = torch.zeros(buffer_size, dtype=torch.bool, device=device)
        self.step_mask = torch.zeros(buffer_size, dtype=torch.bool, device=device)
        
        # Separate counters for tracking
        self.manager_count = 0
        self.step_count = 0

    def add_manager_transition(self, state: torch.Tensor, goal: torch.Tensor, reward: float, value: float, done: bool):
        """Add manager experience to buffer"""
        # FIX: Find next available slot for manager data
        idx = self.manager_count % self.buffer_size
        
        self.manager_states[idx] = state
        self.manager_goals[idx] = goal
        self.manager_rewards[idx] = reward
        self.manager_values[idx] = value
        self.manager_dones[idx] = done
        self.manager_mask[idx] = True
        
        self.manager_count += 1

    def add_step_data(self, pooled_goal: torch.Tensor):
        """Add hierarchical step data (pooled goals)"""
        # FIX: Find next available slot for step data
        idx = self.step_count % self.buffer_size
        
        self.pooled_goals[idx] = pooled_goal
        self.step_mask[idx] = True
        
        self.step_count += 1

    def get_manager_batch(self) -> Dict[str, torch.Tensor]:
        """Get manager experience batch using mask for valid data"""
        # FIX: Only return valid manager data
        valid_manager_indices = self.manager_mask.nonzero(as_tuple=True)[0]
        
        if len(valid_manager_indices) == 0:
            # Return empty tensors if no manager data
            return {
                'states': torch.empty((0, self.manager_states.shape[1]), device=self.device),
                'goals': torch.empty((0, self.manager_goals.shape[1]), device=self.device),
                'values': torch.empty(0, device=self.device),
                'rewards': torch.empty(0, device=self.device),
                'dones': torch.empty(0, dtype=torch.bool, device=self.device)
            }
        
        return {
            'states': self.manager_states[valid_manager_indices],
            'goals': self.manager_goals[valid_manager_indices],
            'values': self.manager_values[valid_manager_indices],
            'rewards': self.manager_rewards[valid_manager_indices],
            'dones': self.manager_dones[valid_manager_indices]
        }

    def get_hierarchical_data(self) -> torch.Tensor:
        """Get pooled goals for worker updates using mask for valid data"""
        # FIX: Only return valid step data
        valid_step_indices = self.step_mask.nonzero(as_tuple=True)[0]
        
        if len(valid_step_indices) == 0:
            # Return empty tensor if no step data
            return torch.empty((0, self.pooled_goals.shape[1]), device=self.device)
        
        return self.pooled_goals[valid_step_indices]

    def clear(self):
        """Clear all data and reset counters"""
        # FIX: Reset all tracking
        self.manager_mask.fill_(False)
        self.step_mask.fill_(False)
        self.manager_count = 0
        self.step_count = 0
        
        # Optional: Zero out tensors for clean slate (not strictly necessary)
        self.manager_states.zero_()
        self.manager_goals.zero_()
        self.manager_values.zero_()
        self.manager_rewards.zero_()
        self.manager_dones.zero_()
        self.pooled_goals.zero_()
    
    @property
    def manager_size(self) -> int:
        """Get number of valid manager transitions"""
        return int(self.manager_mask.sum().item())
    
    @property
    def step_size(self) -> int:
        """Get number of valid step data entries"""
        return int(self.step_mask.sum().item()) 
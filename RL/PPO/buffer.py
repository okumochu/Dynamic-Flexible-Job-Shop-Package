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
    def __init__(self, buffer_size: int, latent_dim: int, obs_shape: tuple, device: torch.device):
        self.buffer_size = buffer_size
        self.device = device
        self.manager_count = 0
        self.step_count = 0
        self.manager_goals = torch.zeros((buffer_size, latent_dim), dtype=torch.float32, device=device)
        self.manager_values = torch.zeros(buffer_size, dtype=torch.float32, device=device)
        self.manager_rewards = torch.zeros(buffer_size, dtype=torch.float32, device=device)
        self.manager_dones = torch.zeros(buffer_size, dtype=torch.bool, device=device)
        self.pooled_goals = torch.zeros((buffer_size, latent_dim), dtype=torch.float32, device=device)
        # Note: No longer need worker_encoded_states - we can re-encode worker observations
        self.manager_observations = torch.zeros((buffer_size, *obs_shape), dtype=torch.float32, device=device)

    def add_manager_transition(self, goal: torch.Tensor, reward: float, value: float, done: bool, observation: torch.Tensor = None):
        idx = self.manager_count % self.buffer_size
        self.manager_observations[idx] = observation
        self.manager_goals[idx] = goal
        self.manager_rewards[idx] = reward
        self.manager_values[idx] = value
        self.manager_dones[idx] = done
        self.manager_count += 1

    def add_step_data(self, pooled_goal: torch.Tensor):
        idx = self.step_count % self.buffer_size
        self.pooled_goals[idx] = pooled_goal
        # Note: No longer store encoded_state - we'll re-encode worker observations when needed
        self.step_count += 1

    def get_manager_batch(self) -> Dict[str, torch.Tensor]:
        count = min(self.manager_count, self.buffer_size)

        return {
            'goals': self.manager_goals[:count],
            'values': self.manager_values[:count],
            'rewards': self.manager_rewards[:count],
            'dones': self.manager_dones[:count],
            'observations': self.manager_observations[:count]
        }

    def get_step_information(self) -> Dict[str, torch.Tensor]:
        count = min(self.step_count, self.buffer_size)

        return {
            'pooled_goals': self.pooled_goals[:count]
        }

    def clear(self):
        self.manager_count = 0
        self.step_count = 0
        self.manager_goals.zero_()
        self.manager_values.zero_()
        self.manager_rewards.zero_()
        self.manager_dones.zero_()
        self.pooled_goals.zero_()
        self.manager_observations.zero_()


class HybridPPOBuffer(PPOBuffer):
    """
    Buffer for storing PPO training data with hybrid action space (discrete + continuous).
    Inherits from PPOBuffer and adds fields for continuous actions and their log probs.
    """
    def __init__(self, buffer_size: int, obs_shape: Tuple[int], discrete_action_dim: int, 
                 continuous_action_dim: int, device: torch.device):
        super().__init__(buffer_size, obs_shape, discrete_action_dim, device) # Call parent constructor
        self.continuous_actions = torch.zeros((buffer_size, continuous_action_dim), dtype=torch.float32, device=device)
        self.continuous_log_probs = torch.zeros(buffer_size, dtype=torch.float32, device=device)

    def add(self, obs: torch.Tensor, discrete_action: int, continuous_action: float, reward: float, value: float, 
            discrete_log_prob: float, continuous_log_prob: float, action_mask: torch.Tensor, done: bool):
        # Store current ptr before parent increments it
        current_ptr = self.ptr
        
        # Call parent's add method for common fields
        super().add(obs, discrete_action, reward, value, discrete_log_prob, action_mask, done)
        
        # Add hybrid-specific fields using the original ptr position
        self.continuous_actions[current_ptr] = continuous_action
        self.continuous_log_probs[current_ptr] = continuous_log_prob

    def get_batch(self) -> Dict[str, torch.Tensor]:
        # Get batch from parent class
        batch = super().get_batch()
        
        # Add hybrid-specific fields
        batch['continuous_actions'] = self.continuous_actions[:self.size]
        batch['continuous_log_probs'] = self.continuous_log_probs[:self.size]
        return batch
    
    def clear(self):
        super().clear()
        self.continuous_actions.zero_()
        self.continuous_log_probs.zero_()
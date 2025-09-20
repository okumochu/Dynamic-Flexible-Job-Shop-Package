"""
RLEnv: Vanilla Event-driven Flexible Job Shop Scheduling Environment
Implements OpenAI Gym API for makespan optimization
Pure job scheduling environment without idle actions
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional, Any, List
import gymnasium as gym
from gymnasium import spaces
from RL.state import State

class RLEnv(gym.Env):
    """
    Vanilla Event-driven Flexible Job Shop Scheduling Environment.
    
    The environment simulates a flexible job shop where:
    - Each job has multiple operations that must be processed in order
    - Each operation can be processed on multiple compatible machines
    - The agent dispatches operations to machines (no idle actions)
    - Time advances to the next completion event after each dispatch
    - Objectives: minimize makespan
    
    This is a pure job scheduling environment without idle actions.
    """
    
    def __init__(self, data_handler, alpha: float = 0.0, use_reward_shaping: bool = True, max_jobs: Optional[int] = None, 
                 max_machines: Optional[int] = None):
        """
        Initialize the environment.
        
        Args:
            data_handler: FlexibleJobShopDataHandler instance
            alpha: Weight for TWT in reward (kept for compatibility, but always 0)
            use_reward_shaping: Whether to use dense rewards
            max_jobs: Maximum jobs for padding (default: num_jobs)
            max_machines: Maximum machines for padding (default: num_machines)
        """
        super().__init__()
        
        # Initialize state manager
        self.state = State(data_handler, max_jobs, max_machines)
        
        # Access state properties through state manager
        self.data_handler = data_handler
        self.jobs = self.state.jobs
        self.num_jobs = self.state.num_jobs
        self.num_machines = self.state.num_machines
        self.due_dates = self.state.due_dates
        self.weights = self.state.weights
        self.alpha = 0.0  # Always 0 - pure makespan optimization
        self.use_reward_shaping = use_reward_shaping
        self.max_jobs = self.state.job_dim
        self.max_machines = self.state.machine_dim
        
        # Override action dimension to exclude idle actions (only job scheduling)
        self.action_dim = self.state.job_dim * self.state.machine_dim
        
        # Initialize state and get obs_len
        self.state.reset()
        self.obs_len = len(self.state._to_numpy())
        
        self.action_space = spaces.Discrete(self.action_dim)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_len,), dtype=np.float32)
        self.last_step_objective = 0
        self.last_episode_objective = 0
        
        # Initialize machine_schedule for tracking
        self.machine_schedule = {machine_id: [] for machine_id in range(self.num_machines)}
        
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        """Reset the environment to initial state."""
        self.state.reset()
        self.last_step_objective = 0
        
        # Reset machine_schedule
        self.machine_schedule = {machine_id: [] for machine_id in range(self.num_machines)}
        
        obs = self.state._to_numpy()
        return obs, {}
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Take a step in the environment.
        
        Args:
            action: Action index (dispatch operations only)
            
        Returns:
            observation: Current observation tensor
            reward: Reward for this step
            terminated: Whether episode is finished (natural)
            truncated: Whether episode is truncated (never in this env)
            info: Additional information (detailed if requested)
        """
        terminated = False
        objective_info = {}
        info = {}

        # Decode Action - only job scheduling actions
        job_id, machine_id = self.decode_action(action)
        
        # Schedule a job to a machine
        job_states = self.state.readable_state['job_states']
        current_job_state = job_states[job_id]
        op_idx = current_job_state['current_op']
        op = self.jobs[job_id].operations[op_idx]
        
        # Get start time from machine_available_time which already accounts for both
        # machine availability and job precedence constraints
        start_time = current_job_state['machine_available_time'][machine_id]
        
        proc_time = op.get_processing_time(machine_id)
        finish_time = start_time + proc_time
        self.state.schedule_operation(job_id, machine_id, start_time, finish_time)
        
        # Get operation_id and add to machine_schedule
        operation_id = op.operation_id
        self.machine_schedule[machine_id].append((operation_id, start_time))
        
        objective_info = self.get_current_objective()

        # Calculate dense reward
        reward = self.get_reward(use_reward_shaping=self.use_reward_shaping)
        self.last_step_objective = objective_info['objective'] # update for dense reward

        obs = self.state._to_numpy()

        # Check if episode is now done after this action
        if self.state.is_done():
            self.last_episode_objective = objective_info['objective'] # update for sparse reward
            terminated = True

        # Store scheduling information in info
        info['objective_info'] = objective_info
        info['machine_schedule'] = self.machine_schedule
        return obs, reward, terminated, False, info
    
    def get_action_mask(self) -> torch.Tensor:
        """Get boolean mask for valid actions (only job scheduling actions)."""
        mask = torch.zeros(self.action_dim, dtype=torch.bool)
        
        job_states = self.state.readable_state['job_states']
        
        # Handle job scheduling actions (actions 0 to job_dim * machine_dim - 1)
        for job_id in range(self.state.job_dim):
            job_state = job_states[job_id]
            op_idx = job_state['current_op']
            
            # Skip if job is done (no more operations) or if it's a padding job
            if job_state['left_ops'] <= 0:
                continue
            
            # Only allow scheduling the current operation (next operation in sequence)
            # This ensures job precedence constraints are enforced
            for machine_id in range(self.state.machine_dim):
                # Check if this operation is compatible with this machine
                if job_state["operations"]["process_time"][op_idx][machine_id] > 0:
                    idx = job_id * self.state.machine_dim + machine_id
                    mask[idx] = True
        
        return mask
    
    def decode_action(self, action: int) -> Tuple[int, int]:
        """Decode action index to (job_id, machine_id) for job scheduling only."""
        job_id = action // self.state.machine_dim
        machine_id = action % self.state.machine_dim
        return job_id, machine_id

    def get_current_objective(self) -> Dict[str, Any]:
        """Calculate the current objective values."""
        job_states = self.state.readable_state['job_states']
        makespan = 0
        
        for job_id in range(self.num_jobs):
            job_state = job_states[job_id]
            
            # Calculate job completion time (max finish time of all operations)
            job_completion_time = 0
            for op_pos in range(len(job_state['operations']['finish_time'])):
                job_completion_time = max(job_completion_time, job_state['operations']['finish_time'][op_pos])
            
            # Update makespan
            makespan = max(makespan, job_completion_time)

        return {
            'makespan': makespan,
            'objective': makespan  # Since alpha = 0
        }
    
    def get_reward(self, use_reward_shaping:bool) -> float:
        """Get reward for the current state.
        Args:
            use_reward_shaping: Whether to use dense reward
        Returns:
            reward: Reward for the current state
        """
        if use_reward_shaping:
            # dense reward means the difference between the last step's objective and the current step's objective
            return self.last_step_objective - self.get_current_objective()['objective']
        else:
            # sparse reward means the difference between the last episode's objective and the current episode's objective
            if self.state.is_done():
                # log(objective) is the negative reward
                return  - self.get_current_objective()['objective']
            else:
                return 0
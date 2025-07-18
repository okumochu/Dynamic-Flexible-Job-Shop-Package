"""
RLEnv: Unified Event-driven Flexible Job Shop Scheduling Environment
Implements OpenAI Gym API for multi-objective optimization (Makespan + TWT)
Compatible with both flat and hierarchical RL algorithms
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import gymnasium as gym
from gymnasium import spaces
from benchmarks.static_benchmark.data_handler import FlexibleJobShopDataHandler
from RL.state import State

class RLEnv(gym.Env):
    """
    Unified Event-driven Flexible Job Shop Scheduling Environment.
    
    The environment simulates a flexible job shop where:
    - Each job has multiple operations that must be processed in order
    - Each operation can be processed on multiple compatible machines
    - The agent dispatches operations to machines
    - Time advances to the next completion event after each dispatch
    - Objectives: minimize makespan and total weighted tardiness
    
    Compatible with both flat and hierarchical RL algorithms.
    """
    
    def __init__(self, data_handler, alpha: float, use_reward_shaping: bool = True, max_jobs: Optional[int] = None, 
                 max_machines: Optional[int] = None):
        """
        Initialize the environment.
        
        Args:
            data_handler: FlexibleJobShopDataHandler instance
            alpha: Weight for TWT in reward
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
        self.alpha = alpha
        self.use_reward_shaping = use_reward_shaping
        self.max_jobs = self.state.job_dim
        self.max_machines = self.state.machine_dim
        self.action_dim = self.state.action_dim
        
        # Initialize state and get obs_len
        self.state.reset()
        self.obs_len = len(self.state._to_numpy())
        
        self.action_space = spaces.Discrete(self.action_dim)
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.obs_len,), dtype=np.float32)
        self.last_step_objective = 0
        self.last_episode_objective = 0
        
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        """Reset the environment to initial state."""
        self.state.reset()
        self.last_step_objective = 0
        obs = self.state._to_numpy()
        return obs, {}
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Take a step in the environment.
        
        Args:
            action: Action index (dispatch operations)
            
        Returns:
            observation: Current observation tensor
            reward: Reward for this step
            terminated: Whether episode is finished (natural)
            truncated: Whether episode is truncated (never in this env)
            info: Additional information (detailed if requested)
        """
        terminated = False
        objective_info = {}

        # Decode Action
        job_id, machine_id = self.decode_action(action)
        job_states = self.state.readable_state['job_states']
        op_idx = job_states[job_id]['current_op']
        op = self.jobs[job_id].operations[op_idx]
        
        # Calculate correct start time based on constraints
        job_state = job_states[job_id]
        machine_available_time = job_state['machine_available_time'][machine_id]
        prev_op_finish_time = job_state['operations']['finish_time'][op_idx - 1] if op_idx > 0 else 0
        
        start_time = max(machine_available_time, prev_op_finish_time)
        
        proc_time = op.get_processing_time(machine_id)
        finish_time = start_time + proc_time
        self.state.schedule_operation(job_id, machine_id, start_time, finish_time)
        objective_info = self.get_current_objective()

        # Calculate dense reward
        reward = self.get_reward(use_reward_shaping=self.use_reward_shaping)
        self.last_step_objective = objective_info['objective'] # update for dense reward

        obs = self.state._to_numpy()

        # Check if episode is now done after this action
        if self.state.is_done():
            self.last_episode_objective = objective_info['objective'] # update for sparse reward
            terminated = True

        return obs, reward, terminated, False, objective_info
    
    def get_action_mask(self) -> torch.Tensor:
        """Get boolean mask for valid actions."""
        mask = torch.zeros(self.action_dim, dtype=torch.bool)
        
        job_states = self.state.readable_state['job_states']
        for job_id in range(self.num_jobs):
            job_state = job_states[job_id]
            op_idx = job_state['current_op']
            
            if op_idx >= len(self.jobs[job_id].operations):
                continue  # No more ops
            
            op = self.jobs[job_id].operations[op_idx]
            for machine_id in op.compatible_machines:
                idx = job_id * self.num_machines + machine_id
                # Always allow scheduling: agent can always schedule on any compatible machine
                mask[idx] = True
        
        return mask
    
    def decode_action(self, action: int) -> Tuple[int, int]:
        """Decode action index to (job_id, machine_id)."""
        job_id = action // self.num_machines
        machine_id = action % self.num_machines
        return job_id, machine_id

    def get_current_objective(self) -> Dict[str, Any]:
        """Calculate the current objective values."""
        job_states = self.state.readable_state['job_states']
        makespan = 0
        twt = 0
        
        for job_id in range(self.num_jobs):
            job_state = job_states[job_id]
            
            # Calculate job completion time (max finish time of all operations)
            job_completion_time = 0
            for op_pos in range(len(job_state['operations']['finish_time'])):
                job_completion_time = max(job_completion_time, job_state['operations']['finish_time'][op_pos])
            
            # Update makespan
            makespan = max(makespan, job_completion_time)
            
            # Calculate tardiness for this job
            due_date = self.due_dates[job_id] if job_id < len(self.due_dates) else float('inf')
            tardiness = max(0, job_completion_time - due_date)
            weight = self.weights[job_id] if job_id < len(self.weights) else 1.0
            twt += weight * tardiness

        return {
            'makespan': makespan,
            'twt': twt,
            'objective': (1 - self.alpha) * makespan + self.alpha * twt
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
                return self.last_episode_objective - self.get_current_objective()['objective']
            else:
                return 0
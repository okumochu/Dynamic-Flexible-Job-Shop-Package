"""
FlatRLEnv: Event-driven Flexible Job Shop Scheduling Environment
Implements OpenAI Gym API for multi-objective optimization (Makespan + TWT)
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import gymnasium as gym
from gymnasium import spaces
import heapq
from benchmarks.static_benchmark.data_handler import FlexibleJobShopDataHandler
from RL.flat_rl_state import FlatRLState

class FlatRLEnv(gym.Env):
    """
    Event-driven Flexible Job Shop Scheduling Environment.
    
    The environment simulates a flexible job shop where:
    - Each job has multiple operations that must be processed in order
    - Each operation can be processed on multiple compatible machines
    - The agent dispatches operations to machines
    - Time advances to the next completion event after each dispatch
    - Objectives: minimize makespan and total weighted tardiness
    """
    
    def __init__(self, data_handler, alpha: float, max_jobs: Optional[int] = None, max_machines: Optional[int] = None):
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
        self.state = FlatRLState(data_handler, max_jobs, max_machines)
        
        # Access state properties through state manager
        self.data_handler = data_handler
        self.jobs = self.state.jobs
        self.operations = self.state.operations
        self.num_jobs = self.state.num_jobs
        self.num_machines = self.state.num_machines
        self.num_operations = self.state.num_operations
        
        # Extract due dates and weights
        self.due_dates = self.state.due_dates
        self.weights = self.state.weights
        
        self.alpha = alpha  # Weight for TWT
        self.beta = 1 - alpha  # Weight for makespan
        
        # Set padding dimensions
        self.max_jobs = self.state.max_jobs
        self.max_machines = self.state.max_machines
        
        # Access state dimensions through state manager
        self.action_dim = self.state.action_dim
        self.obs_len = self.state.obs_len
        
        # Action space: num_jobs * num_machines
        self.action_space = spaces.Discrete(self.action_dim)
        
        # Observation: [proc_times (JxM), machine_avail_time (JxM), job_remain (J), job_weight (J), ops_left (J)]
        self.observation_space = spaces.Box(low=-1, high=1, shape=(self.obs_len,), dtype=np.float32)
        
        # Initialize state variables
        self.last_objective = None
    
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        """Reset the environment to initial state."""
        self.state.reset()
        self.last_objective = None
        obs = self.state.get_observation()
        return obs.numpy(), {}
    
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
            info: Additional information
        """
        # 1. Check if the environment is done
        terminated = False
        if self.state.is_done():
            terminated = True
        else:
            # 2. Decode action to (job_id, machine_id)
            job_id, machine_id = self.state.decode_action(action)
            op_idx = self.state.job_states[job_id]['current_op']
            op = self.jobs[job_id].operations[op_idx]
            
            # 3. Schedule the operation: update machine available time, job state, and operation status
            #    The operation starts at the max of current time and machine available time
            machine_ready = self.state.machine_states[machine_id]['finish_time']
            if machine_ready is None:
                machine_ready = self.state.current_time
            start_time = max(self.state.current_time, machine_ready)
            proc_time = op.get_processing_time(machine_id)
            finish_time = start_time + proc_time
            
            # Schedule the operation using state manager
            self.state.schedule_operation(job_id, machine_id, start_time, finish_time)
            
            # 4. Advance current time to the finish time of this operation
            self.state.advance_time(finish_time)
            
            # Calculate dense reward: negative change in objective function
            current_objective = self.beta * self.state.current_makespan + self.alpha * self.state.current_twt
            
            if self.last_objective is not None:
                # Reward is negative change in objective (improvement = positive reward)
                reward = self.last_objective - current_objective
            else:
                # First step: no reward (or small initialization reward)
                reward = 0.0
            
            self.last_objective = current_objective
            
            info = {
                'makespan': self.state.current_makespan,
                'twt': self.state.current_twt,
                'current_time': self.state.current_time,
                'action_mask': self.state.get_action_mask()
            }
            
            obs = self.state.get_observation()
        return obs.numpy(), reward, terminated, False, info
    
    def get_action_mask(self) -> torch.Tensor:
        """Get the current action mask."""
        return self.state.get_action_mask()
    
    def get_schedule_info(self) -> Dict[str, Any]:
        """Get detailed schedule information for visualization."""
        return self.state.get_schedule_info()
    
    def get_state_summary(self) -> Dict[str, Any]:
        """Get a summary of the current state."""
        return self.state.get_state_summary() 
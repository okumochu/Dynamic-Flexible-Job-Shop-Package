import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from typing import Dict, List, Tuple, Optional, Any
from torch_geometric.data import HeteroData

from RL.graph_state import GraphState
from benchmarks.static_benchmark.data_handler import FlexibleJobShopDataHandler


class GraphRlEnv(gym.Env):
    """
    Gymnasium environment for Flexible Job Shop Scheduling Problem using graph representations.
    
    The environment uses a heterogeneous graph to represent the FJSP state, where operations 
    and machines are nodes with different features, connected by various edge types.
    """
    
    def __init__(self, problem_data: FlexibleJobShopDataHandler, alpha: float = 0.0, device: str = "cpu"):
        """
        Initialize the FJSP Graph RL Environment.
        
        Args:
            problem_data: FlexibleJobShopDataHandler instance with the problem definition
            alpha: Weight for multi-objective optimization. 
                   0.0 = pure makespan minimization
                   1.0 = pure tardiness minimization  
                   0.5 = balanced optimization
        """
        super().__init__()
        
        self.problem_data = problem_data
        self.graph_state = GraphState(problem_data, device=device)
        self.alpha = alpha  # Multi-objective weight parameter
        
        # Due date information for tardiness calculation
        self.due_dates = problem_data.get_jobs_due_date()
        self.weights = problem_data.get_jobs_weight()
        
        # Episode management
        self.current_step = 0
        self.last_makespan = 0.0  # Start with 0 instead of inf
        self.last_total_weighted_tardiness = 0.0  # Track previous tardiness for reward calculation
        
        # Action space: discrete actions representing (operation, machine) pairs
        # We'll create a mapping from action index to (op_idx, machine_idx) pairs
        self.action_to_pair_map = self._build_action_mapping()
        # Reverse mapping for O(1) lookup from (op, machine) to action index
        self.pair_to_action_map = {pair: idx for idx, pair in self.action_to_pair_map.items()}
        self.action_space = spaces.Discrete(len(self.action_to_pair_map))
        
        # Observation space: We can't easily define the exact structure of HeteroData
        # so we'll use a flexible Dict space that can accommodate the graph structure
        self.observation_space = spaces.Dict({
            'graph': spaces.Box(
                low=-np.inf, high=np.inf, 
                shape=(1,),  # Placeholder - actual graph structure is variable
                dtype=np.float32
            )
        })
        
        # Metrics tracking
        self.episode_rewards = []
        self.episode_makespans = []
        
    def _build_action_mapping(self) -> Dict[int, Tuple[int, int]]:
        """
        Build mapping from discrete action indices to (operation, machine) pairs.
        
        Returns:
            Dictionary mapping action_idx -> (op_idx, machine_idx)
        """
        action_map = {}
        action_idx = 0
        
        for op_idx in range(self.problem_data.num_operations):
            operation = self.problem_data.get_operation(op_idx)
            for machine_idx in operation.compatible_machines:
                action_map[action_idx] = (op_idx, machine_idx)
                action_idx += 1
                
        return action_map
    
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[HeteroData, Dict]:
        """
        Reset the environment to initial state.
        
        Args:
            seed: Random seed for environment
            options: Additional options for reset
            
        Returns:
            observation: Initial HeteroData graph observation
            info: Dictionary with auxiliary information
        """
        super().reset(seed=seed)
        
        # Reset the graph state
        self.graph_state.reset()
        self.current_step = 0
        self.last_makespan = 0.0  # Start with 0 instead of inf
        self.last_total_weighted_tardiness = 0.0  # Reset tardiness tracking
        
        # Get initial observation
        observation = self.graph_state.get_observation()
        
        # Prepare info dict
        info = {
            'valid_actions': self.graph_state.get_valid_actions(),
            'ready_operations': self.graph_state.get_ready_operations(),
            'makespan': self.graph_state.get_makespan(),
            'num_scheduled_operations': np.sum(self.graph_state.operation_status == 1)
        }
        
        return observation, info
    
    def step(self, action: int) -> Tuple[HeteroData, float, bool, bool, Dict]:
        """
        Execute one step in the environment.
        
        Args:
            action: Integer action representing an (operation, machine) pair
            
        Returns:
            observation: Updated HeteroData graph observation
            reward: Reward for the action
            terminated: Whether the episode has ended (all operations scheduled)
            truncated: Whether the episode was truncated (max steps reached)
            info: Dictionary with auxiliary information
        """
        # Validate action
        if action not in self.action_to_pair_map:
            raise ValueError(f"Invalid action: {action}")
        
        op_idx, machine_idx = self.action_to_pair_map[action]
        
        # Check if action is valid (operation is ready and machine is compatible)
        valid_actions = self.graph_state.get_valid_actions()
        if (op_idx, machine_idx) not in valid_actions:
            # Invalid action - return negative reward and continue
            reward = -10.0  # Penalty for invalid action
            observation = self.graph_state.get_observation()
            info = self._get_step_info()
            return observation, reward, False, False, info
        
        # Store previous state for reward calculation
        prev_makespan = self.graph_state.get_makespan()
        
        # Execute the action
        self.graph_state.update_state(op_idx, machine_idx)
        self.current_step += 1
        
        # Calculate reward (multi-objective: makespan + tardiness)
        reward = self._calculate_reward(prev_makespan)
        
        # Check terminal conditions
        terminated = self.graph_state.is_done()
        
        # Get updated observation
        observation = self.graph_state.get_observation()
        
        # Prepare info
        info = self._get_step_info()
        info['action_taken'] = (op_idx, machine_idx)
        
        
        return observation, reward, terminated, None, info
    
    def _calculate_reward(self, prev_makespan: float) -> float:
        """
        Calculate multi-objective reward combining makespan and tardiness minimization.
        
        Args:
            prev_makespan: Makespan before the action
            
        Returns:
            Reward value combining makespan and tardiness objectives
        """
        current_makespan = self.graph_state.get_makespan()
        current_twt = self._calculate_total_weighted_tardiness()
        
        # Makespan improvement reward (like FlatRL dense reward)
        makespan_reward = prev_makespan - current_makespan
        
        # Tardiness improvement reward (decrease in tardiness is good)
        tardiness_reward = self.last_total_weighted_tardiness - current_twt
        
        # Update last tardiness for next step
        self.last_total_weighted_tardiness = current_twt
        
        # ALWAYS return the combined dense reward.
        # The telescoping sum property ensures the agent optimizes the final objective.
        combined_reward = (1 - self.alpha) * makespan_reward + self.alpha * tardiness_reward
        return combined_reward
    
    def _calculate_total_weighted_tardiness(self) -> float:
        """
        Calculate Total Weighted Tardiness (TWT) for completed jobs.
        
        Returns:
            Total weighted tardiness value
        """
        total_weighted_tardiness = 0.0
        
        for job_id in range(self.problem_data.num_jobs):
            # Get all operations for this job
            job_operations = self.problem_data.get_job_operations(job_id)
            
            # Find the completion time of the last operation (job completion time)
            job_completion_time = 0.0
            all_ops_completed = True
            
            for operation in job_operations:
                op_id = operation.operation_id
                if self.graph_state.operation_status[op_id] == 1:  # completed
                    job_completion_time = max(job_completion_time, self.graph_state.operation_completion_times[op_id])
                else:
                    all_ops_completed = False
                    break
            
            # Only calculate tardiness for completed jobs
            if all_ops_completed:
                due_date = self.due_dates[job_id]
                weight = self.weights[job_id]
                tardiness = max(0.0, job_completion_time - due_date)
                total_weighted_tardiness += weight * tardiness
        
        return total_weighted_tardiness
    
    def _get_step_info(self) -> Dict[str, Any]:
        """Get information dictionary for current step."""
        return {
            'valid_actions': self.graph_state.get_valid_actions(),
            'ready_operations': self.graph_state.get_ready_operations(),
            'makespan': self.graph_state.get_makespan(),
            'total_weighted_tardiness': self._calculate_total_weighted_tardiness(),
            'num_scheduled_operations': np.sum(self.graph_state.operation_status == 1),
            'current_step': self.current_step,
            'utilization': self._calculate_machine_utilization(),
            'alpha': self.alpha  # Multi-objective weight
        }
    
    def _calculate_machine_utilization(self) -> Dict[str, float]:
        """Calculate current machine utilization statistics."""
        total_workload = self.graph_state.machine_workloads.sum()
        max_workload = self.graph_state.machine_workloads.max()
        min_workload = self.graph_state.machine_workloads.min()
        avg_workload = total_workload / self.problem_data.num_machines
        
        return {
            'total_workload': float(total_workload),
            'max_workload': float(max_workload),
            'min_workload': float(min_workload),
            'avg_workload': float(avg_workload),
            'workload_std': float(self.graph_state.machine_workloads.std())
        }

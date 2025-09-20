import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Dict, List, Tuple, Optional, Any
from torch_geometric.data import HeteroData

from RL.graph_state import GraphState
from benchmarks.data_handler import FlexibleJobShopDataHandler


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
            alpha: Weight for multi-objective optimization (kept for compatibility, but always 0)
            device: Device to run on
        """
        super().__init__()
        
        self.problem_data = problem_data
        self.graph_state = GraphState(problem_data, device=device)
        self.alpha = 0.0  # Always 0 - pure makespan optimization
        
        # Due date and weight information (kept for potential future use)
        self.due_dates = problem_data.get_jobs_due_date()
        self.weights = problem_data.get_jobs_weight()
        
        # Episode management
        self.current_step = 0
        self.last_makespan = 0.0  # Start with 0 instead of inf
        
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
        
        op_idx, machine_idx = self.action_to_pair_map[action]
        
        
        # Execute the action
        self.graph_state.update_state(op_idx, machine_idx)
        self.current_step += 1
        
        # Calculate reward (multi-objective: makespan + tardiness)
        reward = self._calculate_reward()
        
        # Check terminal conditions
        terminated = self.graph_state.is_done()
        
        # Get updated observation
        observation = self.graph_state.get_observation()
        
        # Prepare info
        info = self._get_step_info()
        info['action_taken'] = (op_idx, machine_idx)
        
        
        return observation, reward, terminated, None, info
    
    def _calculate_reward(self) -> float:
        """
        Calculate reward based on makespan improvement.
        
        Returns:
            Reward value based on makespan improvement
        """
        # Get current objective values
        current_makespan = self.graph_state.get_makespan()
        
        # Calculate improvement reward (last - current, since improvement is good)
        makespan_reward = self.last_makespan - current_makespan
        
        # Update last values for next step
        self.last_makespan = current_makespan
        
        # Return the makespan improvement reward
        return makespan_reward
    
    
    def _get_step_info(self) -> Dict[str, Any]:
        """Get information dictionary for current step."""
        return {
            'valid_actions': self.graph_state.get_valid_actions(),
            'ready_operations': self.graph_state.get_ready_operations(),
            'makespan': self.graph_state.get_makespan(),
            'num_scheduled_operations': np.sum(self.graph_state.operation_status == 1),
            'current_step': self.current_step,
            'alpha': self.alpha  # Always 0
        } 

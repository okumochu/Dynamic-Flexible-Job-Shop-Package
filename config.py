"""
Shared configuration for RL experiments
Reduces code duplication and centralizes parameter management
"""

import os
import time
from typing import Dict, Any

class ExperimentConfig:
    """Centralized configuration for all RL experiments"""
    
    def __init__(self):
        # Common simulation parameters
        self.simulation_params = {
            'num_jobs': 10,
            'num_machines': 5,
            'operation_lb': 5,
            'operation_ub': 5,
            'processing_time_lb': 1,
            'processing_time_ub': 99,
            'compatible_machines_lb': 1,
            'compatible_machines_ub': 5,
            'TF': 0.4,
            'RDD': 0.8,
            'seed': 42,
        }
        
        # Common RL parameters (shared by all RL methods)
        self.common_rl_params = {
            'alpha': 0,  # TWT weight in objective
            'gae_lambda': 0.95,
            'clip_ratio': 0.2,
            'entropy_coef': 0.01, # graph RL no use
            'value_coef': 0.5,  # Value loss coefficient for PPO
            'use_reward_shaping': True,  # Whether to use dense rewards
            'test_interval': 10,  # How often to test generalization (in epochs)
            'device': 'auto',  # Device for training ('auto', 'cpu', 'cuda', etc.)
            'test_envs': 30,  # Number of test environments for generalization runs
            'episodes_per_epoch': 2,  # Number of episodes to collect per epoch
            'epochs': 2000,  # Very quick training run for Gantt chart generation
            'train_per_episode': 2,  # Number of training iterations per episode
            'target_kl': 0.01, # graph RL no use
            'max_grad_norm': 0.5,
            'seed': 42,
        }
        
        # Flat RL specific parameters
        self.flat_rl_params = {
            'gamma': 0.99,  # Discount factor for flat RL
            'pi_lr': 3e-4,  # Policy learning rate
            'v_lr': 3e-4,   # Value function learning rate
        }
        
        # Hierarchical RL specific parameters
        self.hierarchical_rl_params = {
            'goal_duration_ratio': 12,  # Manager horizon ratio (total_ops // ratio), normally 5~20 goals per episode
            'latent_dim': 128,  # Encoded state dimension
            'goal_dim': 16,  # Goal space dimension
            'manager_lr': 2e-4,  # Manager learning rate
            'worker_lr': 3e-4,   # Worker learning rate
            'gamma_manager': 0.9999,  # Discount factor for manager
            'gamma_worker': 0.999,    # Discount factor for worker
            'intrinsic_reward_scale': 0.5,  # Intrinsic:extrinsic reward ratio for hierarchical RL
        }

        # Graph RL specific parameters
        self.graph_rl_params = {
            # Network Architecture
            'hidden_dim': 128,  # Hidden dimension for graph networks (must be divisible by num_heads)
            'num_hgt_layers': 4,  # Number of HGT layers
            'num_heads': 4,  # Number of attention heads in HGT (hidden_dim must be divisible by this)
            'dropout': 0.1,  # Dropout rate for graph networks
            
            # Temporal Encoding (for HGT RTE)
            'temporal_dim': 16,  # Dimension for temporal embeddings in machine_precedes edges
            'max_temporal_freq': 1000.0,  # Maximum frequency for sinusoidal temporal encoding
            
            # Learning Parameters
            'pi_lr': 3e-4,  # Policy learning rate for graph RL
            'v_lr': 3e-4,   # Value function learning rate for graph RL
            'gamma': 1,  # Discount factor for graph RL
            
            # Multi-Objective Optimization
            'alpha': 0.5,  # Multi-objective weight: 0.0=pure makespan, 1.0=pure tardiness, 0.5=balanced
            
            # Feature Engineering (optional overrides - normally auto-detected)
            'op_feature_dim': None,  # Override operation feature dimension (None=auto-detect)
            'machine_feature_dim': None,  # Override machine feature dimension (None=auto-detect)  
            'job_feature_dim': None,  # Override job feature dimension (None=auto-detect)
        }
    
    def create_experiment_result_dir(self, experiment_name: str) -> str:
        """
        Create a standardized result directory structure for experiments.
        
        Args:
            experiment_name: Name of the experiment file (e.g., 'exp_graph_rl')
            
        Returns:
            Path to the timestamp-based directory where results should be stored
        """
        # Base result directory (relative path)
        base_result_dir = "result"
        
        # Create experiment directory
        experiment_dir = os.path.join(base_result_dir, experiment_name)
        
        # Create timestamp directory
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        timestamp_dir = os.path.join(experiment_dir, timestamp)
        
        # Create the directories
        os.makedirs(timestamp_dir, exist_ok=True)
        
        return timestamp_dir
    
    def get_flat_rl_config(self, experiment_name: str = "exp_rl") -> Dict[str, Any]:
        """Get configuration for flat RL experiment"""
        # Create standardized result directory
        result_dir = self.create_experiment_result_dir(experiment_name)
            
        # Combine common and flat-specific RL parameters
        rl_params = {**self.common_rl_params, **self.flat_rl_params}
            
        return {
            'simulation_params': self.simulation_params,
            'rl_params': rl_params,
            'result_dir': result_dir,
            'wandb_project': experiment_name
        }
    
    def get_hierarchical_rl_config(self, experiment_name: str = "exp_hrl") -> Dict[str, Any]:
        """Get configuration for hierarchical RL experiment"""
        # Create standardized result directory
        result_dir = self.create_experiment_result_dir(experiment_name)
            
        # Combine common and hierarchical-specific RL parameters
        rl_params = {**self.common_rl_params, **self.hierarchical_rl_params}
            
        return {
            'simulation_params': self.simulation_params,
            'rl_params': rl_params,
            'result_dir': result_dir,
            'wandb_project': experiment_name
        }
    
    def get_brandimarte_config(self, dataset_name: str, dataset_path: str, experiment_name: str = "exp_graph_rl") -> Dict[str, Any]:
        """Get configuration for Brandimarte dataset experiment"""
        # Use flat RL parameters by default for Brandimarte experiments
        rl_params = {**self.common_rl_params, **self.flat_rl_params}
        
        # Create subdirectory for specific dataset within the experiment
        result_dir = self.create_experiment_result_dir(experiment_name)
        dataset_result_dir = os.path.join(result_dir, dataset_name)
        os.makedirs(dataset_result_dir, exist_ok=True)
        
        return {
            'dataset_info': {
                'name': dataset_name,
                'path': dataset_path,
                'type': 'brandimarte'
            },
            'simulation_params': {
                'TF': self.simulation_params['TF'],
                'RDD': self.simulation_params['RDD'],
                'seed': self.simulation_params['seed']
            },
            'rl_params': rl_params,  # Can be overridden
            'result_dir': dataset_result_dir,
            'wandb_project': experiment_name
        }
    
    def get_graph_rl_config(self, experiment_name: str = "exp_graph_rl", dataset_name: str = None) -> Dict[str, Any]:
        """Get configuration for graph RL experiment"""
        # Create standardized result directory
        result_dir = self.create_experiment_result_dir(experiment_name)
        
        # If dataset_name is provided, create a subdirectory for it
        if dataset_name:
            dataset_result_dir = os.path.join(result_dir, dataset_name)
            os.makedirs(dataset_result_dir, exist_ok=True)
            result_dir = dataset_result_dir
            
        # Combine common and graph-specific RL parameters
        rl_params = {**self.common_rl_params, **self.graph_rl_params}
            
        return {
            'simulation_params': self.simulation_params,
            'rl_params': rl_params,
            'result_dir': result_dir,
            'wandb_project': experiment_name
        }
    

    
    def get_all_brandimarte_datasets(self) -> Dict[str, str]:
        """Get all available Brandimarte dataset paths"""
        return {
            'mk01': 'benchmarks/static_benchmark/datasets/brandimarte/mk01.txt',
            'mk02': 'benchmarks/static_benchmark/datasets/brandimarte/mk02.txt',
            'mk03': 'benchmarks/static_benchmark/datasets/brandimarte/mk03.txt',
            'mk04': 'benchmarks/static_benchmark/datasets/brandimarte/mk04.txt',
            'mk05': 'benchmarks/static_benchmark/datasets/brandimarte/mk05.txt',
            'mk06': 'benchmarks/static_benchmark/datasets/brandimarte/mk06.txt',
            'mk07': 'benchmarks/static_benchmark/datasets/brandimarte/mk07.txt',
            'mk08': 'benchmarks/static_benchmark/datasets/brandimarte/mk08.txt',
            'mk09': 'benchmarks/static_benchmark/datasets/brandimarte/mk09.txt',
            'mk10': 'benchmarks/static_benchmark/datasets/brandimarte/mk10.txt',
            'mk11': 'benchmarks/static_benchmark/datasets/brandimarte/mk11.txt',
            'mk12': 'benchmarks/static_benchmark/datasets/brandimarte/mk12.txt',
            'mk13': 'benchmarks/static_benchmark/datasets/brandimarte/mk13.txt',
            'mk14': 'benchmarks/static_benchmark/datasets/brandimarte/mk14.txt',
            'mk15': 'benchmarks/static_benchmark/datasets/brandimarte/mk15.txt'
        }
    
    def setup_wandb_env(self, result_dir: str, project_name: str) -> None:
        """Setup wandb environment variables"""
        os.environ["WANDB_DIR"] = result_dir
        os.environ["WANDB_PROJECT"] = project_name

# Global config instance
config = ExperimentConfig() 
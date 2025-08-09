"""
Shared configuration for RL experiments
Reduces code duplication and centralizes parameter management
"""

import os
from typing import Dict, Any

class ExperimentConfig:
    """Centralized configuration for all RL experiments"""
    
    def __init__(self):
        # Common simulation parameters
        self.simulation_params = {
            'num_jobs': 16,
            'num_machines': 4,
            'operation_lb': 4,
            'operation_ub': 4,
            'processing_time_lb': 5,
            'processing_time_ub': 10,   
            'compatible_machines_lb': 2,
            'compatible_machines_ub': 2,
            'TF': 0.4,
            'RDD': 0.8,
            'seed': 42,
        }
        
        # Unified RL parameters (combines flat and hierarchical RL parameters)
        self.rl_params = {
            # Common parameters
            'alpha': 0.5,  # TWT weight in objective
            'gae_lambda': 0.95,
            'clip_ratio': 0.2,
            'entropy_coef': 0.01,
            'use_reward_shaping': False,  # Whether to use dense rewards
            'test_interval': 10,  # How often to test generalization (in epochs)
            'device': 'cuda:0',  # Device for training ('auto', 'cpu', 'cuda', etc.)
            'test_envs': 30,  # Number of test environments for generalization runs
            
            # Training parameters
            'episodes_per_epoch': 4,  # Number of episodes to collect per epoch
            'epochs': 900,  # Full training run
            'train_per_episode': 1,  # Number of training iterations per episode
            
            # Flat RL specific parameters
            'gamma': 0.999,
            'pi_lr': 3e-4,
            'v_lr': 3e-4,
            
            # Hierarchical RL specific parameters
            'goal_duration_ratio': 12,  # Manager horizon ratio (total_ops // ratio), normally 5~20 goals per episode
            'latent_dim': 128,  # Encoded state dimension
            'goal_dim': 16,  # Goal space dimension
            'manager_lr': 2e-4,
            'worker_lr': 3e-4,
            'gamma_manager': 0.9999,
            'gamma_worker': 0.999,
            'intrinsic_reward_scale': 0.5,  # Intrinsic:extrinsic reward ratio for hierarchical RL

            # Advanced PPO/optimization controls (mainly for hybrid variants)
            'target_kl': 0.01,
            'max_grad_norm': 0.5,
            'seed': 42,
        }
    
    def get_flat_rl_config(self, project_name: str = "exp_hrl_and_flarl", dataset_name: str = None, exp_name: str = "flat") -> Dict[str, Any]:
        """Get configuration for flat RL experiment"""
        # Build organized result directory structure
        if project_name and dataset_name:
            base_path = f"result/{project_name}/{dataset_name}/{exp_name}"
        elif project_name:
            base_path = f"result/{project_name}/{exp_name}"
        else:
            base_path = "result/flat_rl"
            
        return {
            'simulation_params': self.simulation_params,
            'rl_params': self.rl_params,
            'result_dirs': {
                'training_process': f"{base_path}/training_process",
                'model': f"{base_path}/model"
            },
            'wandb_project': project_name
        }
    
    def get_hierarchical_rl_config(self, project_name: str = "exp_hrl_and_flarl", dataset_name: str = None, exp_name: str = "hrl") -> Dict[str, Any]:
        """Get configuration for hierarchical RL experiment"""
        # Build organized result directory structure
        if project_name and dataset_name:
            base_path = f"result/{project_name}/{dataset_name}/{exp_name}"
        elif project_name:
            base_path = f"result/{project_name}/{exp_name}"
        else:
            base_path = "result/hierarchical_rl"
            
        return {
            'simulation_params': self.simulation_params,
            'rl_params': self.rl_params,
            'result_dirs': {
                'training_process': f"{base_path}/training_process",
                'model': f"{base_path}/model"
            },
            'wandb_project': project_name
        }
    
    def get_brandimarte_config(self, dataset_name: str, dataset_path: str, project_name: str = "brandimarte_experiments") -> Dict[str, Any]:
        """Get configuration for Brandimarte dataset experiment"""
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
            'rl_params': self.rl_params,  # Can be overridden
            'result_dirs': {
                'training_process': f"result/{project_name}/{dataset_name}/flat/training_process",
                'model': f"result/{project_name}/{dataset_name}/flat/model"
            },
            'wandb_project': project_name
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
    
    def setup_directories(self, result_dirs: Dict[str, str]) -> None:
        """Create result directories if they don't exist"""
        for dir_path in result_dirs.values():
            os.makedirs(dir_path, exist_ok=True)
    
    def setup_wandb_env(self, training_dir: str, project_name: str) -> None:
        """Setup wandb environment variables"""
        os.environ["WANDB_DIR"] = training_dir
        os.environ["WANDB_PROJECT"] = project_name

# Global config instance
config = ExperimentConfig() 
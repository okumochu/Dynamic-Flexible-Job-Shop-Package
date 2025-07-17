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
            'num_jobs': 12,
            'num_machines': 4,
            'operation_lb': 4,
            'operation_ub': 4,
            'processing_time_lb': 5,
            'processing_time_ub': 5,   
            'compatible_machines_lb': 2,
            'compatible_machines_ub': 2,
            'seed': 42,
        }
        
        # Calculate common derived values
        self.total_max_steps = (self.simulation_params['num_jobs'] * 
                               self.simulation_params['operation_ub'] * 
                               self.simulation_params['num_machines'])
        
        # Common RL parameters
        self.common_rl_params = {
            'alpha': 0.5,  # TWT weight in objective
            'gae_lambda': 0.95,
            'clip_ratio': 0.2,
            'entropy_coef': 0.01,
        }
        
        # Flat RL specific parameters
        self.flat_rl_params = {
            **self.common_rl_params,
            'gamma': 0.99,
            'steps_per_epoch': self.total_max_steps,
            'epochs': 500,  
            'pi_lr': 1e-5,
            'v_lr': 1e-5,
            'train_pi_iters': self.total_max_steps,
            'train_v_iters': self.total_max_steps,
        }
        
        # Hierarchical RL specific parameters
        self.hierarchical_rl_params = {
            **self.common_rl_params,
            'epochs': 250,  # Fewer epochs as hierarchical might converge faster
            'steps_per_epoch': self.total_max_steps,
            'goal_duration': 20,  # Manager horizon c
            'latent_dim': 128,  # Encoded state dimension
            'goal_dim': 16,  # Goal space dimension
            'manager_lr': 1e-5,
            'worker_lr': 1e-5,
            'gamma_manager': 0.995,
            'gamma_worker': 0.95,
            'train_pi_iters': self.total_max_steps,
            'train_v_iters': self.total_max_steps,
        }
    
    def get_flat_rl_config(self) -> Dict[str, Any]:
        """Get configuration for flat RL experiment"""
        return {
            'simulation_params': self.simulation_params,
            'rl_params': self.flat_rl_params,
            'result_dirs': {
                'training_process': "result/flat_rl/training_process",
                'model': "result/flat_rl/model"
            },
            'wandb_project': "Flexible-Job-Shop-RL"
        }
    
    def get_hierarchical_rl_config(self) -> Dict[str, Any]:
        """Get configuration for hierarchical RL experiment"""
        return {
            'simulation_params': self.simulation_params,
            'rl_params': self.hierarchical_rl_params,
            'result_dirs': {
                'training_process': "result/hierarchical_rl/training_process",
                'model': "result/hierarchical_rl/model"
            },
            'wandb_project': "Hierarchical-Job-Shop-RL"
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
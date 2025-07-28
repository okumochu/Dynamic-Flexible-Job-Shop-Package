"""
Flat RL Trainer for Flexible Job Shop Scheduling
Handles training, evaluation, and model management
"""

import torch
import numpy as np
import time
import os
from typing import Dict, List, Optional
from tqdm import tqdm
import wandb
from RL.PPO.flat_agent import FlatAgent
from RL.PPO.buffer import PPOBuffer
from RL.rl_env import RLEnv

class FlatRLTrainer:
    """
    Trainer class for Flat RL agent on Flexible Job Shop Scheduling.
    Handles training, evaluation, and model management.
    """
    
    def __init__(self, 
                 env: RLEnv,
                 epochs: int, 
                 steps_per_epoch: int,
                 train_pi_iters: int,
                 train_v_iters: int,
                 pi_lr: float,
                 v_lr: float,
                 gamma: float,
                 gae_lambda: float,
                 clip_ratio: float,
                 entropy_coef: float,
                 project_name: Optional[str] = None,
                 run_name: Optional[str] = None,
                 device: str = 'auto',
                 model_save_dir: str = 'result/flat_rl/model'):
        
        self.env = env
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.train_pi_iters = train_pi_iters
        self.train_v_iters = train_v_iters
        self.model_save_dir = model_save_dir
        self.project_name = project_name
        self.run_name = run_name
        
        # Add episode tracking
        self.episode_makespans = []
        self.episode_twts = []
        self.episode_objectives = []
        
        # Create agent
        self.agent = FlatAgent(
            input_dim=env.obs_len,
            action_dim=env.action_dim,
            pi_lr=pi_lr,
            v_lr=v_lr,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_ratio=clip_ratio,
            entropy_coef=entropy_coef,
            device=device
        )
        
        # Create save directory
        os.makedirs(model_save_dir, exist_ok=True)
        
        print(f"Flat RL Trainer initialized")
        print(f"Environment: {env.num_jobs} jobs, {env.num_machines} machines")
    
    def collect_rollout(self, env_or_envs, buffer, epoch: int) -> Dict:
        """
        Collect rollout data for one epoch.
        Supports both single environment and curriculum learning with multiple environments.
        
        Args:
            env_or_envs: Single environment or list of environments for curriculum learning
            buffer: PPOBuffer to store collected experiences
            epoch: Current epoch number (used for curriculum learning)
            
        Returns:
            Dictionary containing episode metrics for this epoch
        """
        # Handle curriculum learning - multiple environments (sequential training)
        if isinstance(env_or_envs, list):
            envs = env_or_envs
            # Sequential training: distribute epochs evenly across environments
            epochs_per_env = self.epochs // len(envs)
            
            # Make the rest of the epochs go to the last environment
            env_idx = epoch // epochs_per_env if epoch < (len(envs) - 1) * epochs_per_env else len(envs) - 1
            if env_idx >= len(envs):
                env_idx = len(envs) - 1
                
            current_env = envs[env_idx]
        else:
            current_env = env_or_envs
        
        # Reset environment and initialize episode tracking
        obs, _ = current_env.reset()
        obs = torch.tensor(obs, dtype=torch.float32, device=self.agent.device)

        # Collect data for steps_per_epoch
        for t in range(self.steps_per_epoch):
            # Take valid action
            action_mask = current_env.get_action_mask()
            action, log_prob, value = self.agent.take_action(obs, action_mask)

            # Step environment
            next_obs, reward, terminated, truncated, info = current_env.step(action)
            done = terminated or truncated
            next_obs = torch.tensor(next_obs, dtype=torch.float32, device=self.agent.device)

            # Store experience in buffer
            buffer.add(obs, action, reward, value, log_prob, action_mask, done)
            obs = next_obs
            
            if done:
                # Track episode metrics
                self.episode_makespans.append(info['makespan'])
                self.episode_twts.append(info['twt'])
                self.episode_objectives.append(info['objective'])
                
                # Log individual episode metrics for real-time monitoring
                if wandb.run is not None:
                    wandb.log({
                        "episode_objective": info['objective'],
                        "episode_makespan": info['makespan'], 
                        "episode_twt": info['twt']
                    })

                # Start new episode
                obs, _ = current_env.reset()
                obs = torch.tensor(obs, dtype=torch.float32, device=self.agent.device)

    def train(self, env_or_envs=None, test_environments=None, test_interval=50):
        """
        Main training loop with support for curriculum learning and generalization testing.
        
        Args:
            env_or_envs: Single environment or list of environments. If None, uses self.env
            test_environments: List of test environments for generalization testing
            test_interval: How often to test generalization (in epochs)
        """
        # Use provided environments or fall back to self.env
        if env_or_envs is None:
            env_or_envs = self.env
        
        # Configure wandb to save in proper directory
        wandb_dir = os.path.join(os.path.dirname(self.model_save_dir), 'training_process')
        os.makedirs(wandb_dir, exist_ok=True)
        
        wandb.init(
            name=self.run_name,
            project=self.project_name,
            dir=wandb_dir,
            config={
                "epochs": self.epochs,
                "steps_per_epoch": self.steps_per_epoch,
                "pi_lr": self.agent.pi_lr,
                "v_lr": self.agent.v_lr,
                "gamma": self.agent.gamma,
                "gae_lambda": self.agent.gae_lambda,
                "clip_ratio": self.agent.clip_ratio,
                "entropy_coef": self.agent.entropy_coef,
                "train_pi_iters": self.train_pi_iters,
                "train_v_iters": self.train_v_iters,
            }
        )

        # Get observation dimensions from first environment (assuming all have same structure)
        if isinstance(env_or_envs, list):
            obs_len = env_or_envs[0].obs_len
            action_dim = env_or_envs[0].action_dim
        else:
            obs_len = env_or_envs.obs_len
            action_dim = env_or_envs.action_dim

        # Buffer
        buffer = PPOBuffer(self.steps_per_epoch, (obs_len,), action_dim, self.agent.device)

        # Training loop
        start_time = time.time()
        pbar = tqdm(range(self.epochs), desc="Flat Training")

        for epoch in pbar:
            # Collect rollout data
            self.collect_rollout(env_or_envs, buffer, epoch)
            
            # Update agent
            stats = self.agent.update(
                buffer,
                train_pi_iters=self.train_pi_iters,
                train_v_iters=self.train_v_iters
            )

            # Log training metrics
            wandb_log = {
                "policy_loss": stats["policy_loss"],
                "value_loss": stats["value_loss"],
                "entropy": stats["entropy"]
            }
            
            # Test generalization periodically
            if test_environments is not None and (epoch + 1) % test_interval == 0:
                generalization_results = self.evaluate_generalization(test_environments)
                
                # Add generalization metrics to wandb log
                wandb_log.update({
                    "generalization/mean_makespan": generalization_results['aggregate_stats']['avg_makespan'],
                    "generalization/mean_twt": generalization_results['aggregate_stats']['avg_twt'],
                    "generalization/mean_objective": generalization_results['aggregate_stats']['avg_objective'],
                })
            
            # Log all metrics
            wandb.log(wandb_log)

            buffer.clear()
        
        pbar.close()
        
        # Save model with timestamp
        timestamp = time.strftime('%Y%m%d_%H%M')
        model_filename = f"model_{timestamp}.pth"
        self.save_model(model_filename)
        wandb.finish()

        return {
            'training_time': time.time() - start_time,
            'model_filename': model_filename,
            'episode_makespans': self.episode_makespans,
            'episode_twts': self.episode_twts,
            'episode_objectives': self.episode_objectives
        }
    
    def save_model(self, filename: str):
        """Save model using agent's save method"""
        filepath = os.path.join(self.model_save_dir, filename)
        self.agent.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filename: str):
        """Load model using agent's load method"""
        filepath = os.path.join(self.model_save_dir, filename)
        if os.path.exists(filepath):
            self.agent.load(filepath)
            print(f"Model loaded from {filepath}")
        else:
            print(f"Model file {filepath} not found")

    def evaluate_generalization(self, data_handlers: List) -> Dict:
        """
        Evaluate model generalization across multiple environments.
        
        Args:
            data_handlers: List of FlexibleJobShopDataHandler instances for different test environments
            temp_model_name: Optional name for temporary model save. If None, uses timestamp
            
        Returns:
            Dictionary containing evaluation results for each environment
        """
        
        # (Removed model saving/loading, not needed for evaluation)
        
        # Evaluate on each environment
        generalization_results = {}
        all_makespans = []
        all_twts = []
        all_objectives = []
        all_rewards = []
        
        for i, data_handler in enumerate(data_handlers):
            
            # Create environment from data handler
            test_env = RLEnv(data_handler, alpha=0.1, use_reward_shaping=True)
            
            # Reset environment
            obs, _ = test_env.reset()
            obs = torch.tensor(obs, dtype=torch.float32, device=self.agent.device)
            
            episode_reward = 0
            step_count = 0
            max_steps = test_env.num_jobs * max(len(job.operations) for job in test_env.jobs.values()) * 2
            
            # Run deterministic policy
            while not test_env.state.is_done() and step_count < max_steps:
                action_mask = test_env.get_action_mask()
                if not action_mask.any():
                    break
                
                # Take deterministic action
                action = self.agent.get_deterministic_action(obs, action_mask)
                
                # Step environment
                next_obs, reward, terminated, truncated, info = test_env.step(action)
                done = terminated or truncated
                
                episode_reward += reward
                obs = torch.tensor(next_obs, dtype=torch.float32, device=self.agent.device)
                step_count += 1
                
                if done:
                    break
            
            # Get final results
            final_info = test_env.get_current_objective()
            
            env_result = {
                'makespan': final_info['makespan'],
                'twt': final_info['twt'],
                'objective': final_info['objective'],
                'episode_reward': episode_reward,
                'steps_taken': step_count,
                'is_valid_completion': test_env.state.is_done(),
                'num_jobs': test_env.num_jobs,
                'num_machines': test_env.num_machines
            }
            
            generalization_results[f'env_{i+1}'] = env_result
            all_makespans.append(final_info['makespan'])
            all_twts.append(final_info['twt'])
            all_objectives.append(final_info['objective'])
            all_rewards.append(episode_reward)
        
        # Calculate aggregate statistics
        aggregate_stats = {
            'avg_makespan': np.mean(all_makespans),
            'std_makespan': np.std(all_makespans),
            'avg_twt': np.mean(all_twts),
            'std_twt': np.std(all_twts),
            'avg_objective': np.mean(all_objectives),
            'std_objective': np.std(all_objectives),
            'avg_reward': np.mean(all_rewards),
            'std_reward': np.std(all_rewards),
            'num_environments': len(data_handlers)
        }
                
        return {
            'individual_results': generalization_results,
            'aggregate_stats': aggregate_stats
        }
    



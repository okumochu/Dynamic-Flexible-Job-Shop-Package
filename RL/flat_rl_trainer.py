"""
Flat RL Trainer for Flexible Job Shop Scheduling
Handles training, evaluation, and model management
"""

import torch
import numpy as np
import time
import os
from typing import Dict, List, Tuple, Optional, cast
from tqdm import tqdm
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces
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
                 steps_per_epoch: int = 4000,
                 train_pi_iters: int = 80,
                 train_v_iters: int = 80,
                 pi_lr: float = 3e-5,
                 v_lr: float = 1e-4,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.97,
                 clip_ratio: float = 0.2,
                 entropy_coef: float = 0.01,
                 device: str = 'auto',
                 model_save_dir: str = 'result/flat_rl/model'):
        """
        Initialize the trainer.
        
        Args:
            env: The environment to train on
            pi_lr: Policy learning rate
            v_lr: Value function learning rate
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            clip_ratio: PPO clip ratio

            target_kl: Early stopping KL threshold (0.0 disables early stopping)
            train_pi_iters: Number of policy update iterations per epoch
            train_v_iters: Number of value update iterations per epoch
            steps_per_epoch: Number of environment interaction steps to collect per training epoch
            epochs: Total number of training epochs to run
            
        Training Flow:
            Each epoch: Collect steps_per_epoch steps → Update policy train_pi_iters times → Update value train_v_iters times
            Total training steps = steps_per_epoch * epochs
            device: Device to run on ('auto', 'cpu', 'cuda')
            model_save_dir: Directory to save models
        """
        self.env = env
        self.model_save_dir = model_save_dir
        self.steps_per_epoch = steps_per_epoch
        self.epochs = epochs
        self.train_pi_iters = train_pi_iters
        self.train_v_iters = train_v_iters
        
        # Create save directory
        os.makedirs(model_save_dir, exist_ok=True)
        
        # Initialize agent
        # Ensure obs_shape is always a 1D tuple (obs_dim,) for flat state vectors
        # obs_len is set in env after env.reset()
        obs_dim = env.obs_len
        obs_shape = (obs_dim,)
        action_space = cast(spaces.Discrete, env.action_space)
        action_dim = int(action_space.n)
        self.agent = FlatAgent(
            input_dim=obs_dim,
            action_dim=action_dim,
            pi_lr=pi_lr,
            v_lr=v_lr,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_ratio=clip_ratio,
            entropy_coef=entropy_coef,
            device=device
        )
        
        # Training statistics
        self.training_history = {
            'episode_rewards': [],
            'episode_makespans': [],
            'episode_twts': [],
            'episode_objectives': [],
            'training_stats': [],
            'evaluation_results': []
        }
        
        print(f"Trainer initialized on device: {self.agent.device}")
        print(f"Environment: {env.num_jobs} jobs, {env.num_machines} machines")
    
    def collect_rollout(self, env_or_envs, buffer, epoch: int = 0) -> Dict:
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
            remaining_epochs = self.epochs % len(envs)
            
            env_idx = epoch // epochs_per_env if epoch < (len(envs) - 1) * epochs_per_env else len(envs) - 1
            if env_idx >= len(envs):
                env_idx = len(envs) - 1
                
            current_env = envs[env_idx]
        else:
            current_env = env_or_envs
        
        # Reset environment and initialize episode tracking
        obs, _ = current_env.reset()
        obs = torch.tensor(obs, dtype=torch.float32, device=self.agent.device)
        ep_reward = 0
        ep_objective = 0
        
        # Track episode metrics for this epoch
        epoch_episodes = []

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
            ep_reward += reward
            ep_objective += info.get('objective', 0)
            obs = next_obs
            
            if done:
                # Collect episode metrics for later aggregation
                epoch_episodes.append({
                    'objective': info['objective'],
                    'makespan': info['makespan'],
                    'twt': info['twt'],
                    'reward': ep_reward
                })
                
                self.training_history['episode_rewards'].append(ep_reward)
                self.training_history['episode_makespans'].append(info['makespan'])
                self.training_history['episode_twts'].append(info['twt'])
                self.training_history['episode_objectives'].append(info['objective'])
                
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
                ep_reward = 0
                ep_objective = 0
        
        # Return aggregated episode metrics for this epoch
        if epoch_episodes:
            return {
                'mean_makespan': np.mean([ep['makespan'] for ep in epoch_episodes]),
                'mean_twt': np.mean([ep['twt'] for ep in epoch_episodes]),
                'mean_objective': np.mean([ep['objective'] for ep in epoch_episodes]),
                'num_episodes': len(epoch_episodes)
            }
        else:
            return {
                'mean_makespan': 0.0,
                'mean_twt': 0.0,
                'mean_objective': 0.0,
                'num_episodes': 0
            }

    def train(self, env_or_envs=None, test_environments=None, test_interval=50) -> Dict:
        """
        Train the agent using epoch-based PPO (Spinning Up style).
        Now supports curriculum learning with multiple environments and generalization testing.
        
        Args:
            env_or_envs: Single environment or list of environments. If None, uses self.env
            test_environments: List of test environments for generalization testing
            test_interval: How often to test generalization (in epochs)
            
        Returns:
            Dictionary containing training results
        """
        # Use provided environments or fall back to self.env
        if env_or_envs is None:
            env_or_envs = self.env
        
        # Get observation dimensions from first environment (assuming all have same structure)
        if isinstance(env_or_envs, list):
            obs_len = env_or_envs[0].obs_len
            action_dim = env_or_envs[0].action_dim
        else:
            obs_len = env_or_envs.obs_len
            action_dim = env_or_envs.action_dim
            
        buffer = PPOBuffer(self.steps_per_epoch, (obs_len,), action_dim, self.agent.device)
        # Configure wandb to save in proper directory
        wandb_dir = os.path.join(os.path.dirname(self.model_save_dir), 'training_process')
        os.makedirs(wandb_dir, exist_ok=True)
        
        wandb.init(
            name=f"Flat_RL_{time.strftime('%Y%m%d_%H%M')}",
            project="dfjs",
            dir=wandb_dir,
            config={
                "steps_per_epoch": self.steps_per_epoch,
                "epochs": self.epochs,
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

        # Training loop
        start_time = time.time()
        pbar = tqdm(range(self.epochs), desc="Training Progress")
        for epoch in pbar:
            # Collect rollout data
            epoch_metrics = self.collect_rollout(env_or_envs, buffer, epoch)
            
            # PPO update
            stats = self.agent.update(
                buffer,
                train_pi_iters=self.train_pi_iters,
                train_v_iters=self.train_v_iters
            )
            self.training_history['training_stats'].append(stats)
            buffer.clear()
            
            # Log training metrics (only loss/entropy, not redundant episode aggregates)
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
        
        training_time = time.time() - start_time
        pbar.close()
        
        # Save model with timestamp
        timestamp = time.strftime('%Y%m%d_%H%M')
        model_filename = f"model_{timestamp}.pth"
        self.agent.save(os.path.join(self.model_save_dir, model_filename))
        wandb.finish()
        
        return {
            'episode_rewards': self.training_history['episode_rewards'],
            'episode_makespans': self.training_history['episode_makespans'],
            'episode_twts': self.training_history['episode_twts'],
            'training_stats': self.training_history['training_stats'],
            'training_time': training_time,
            'model_filename': model_filename
        }
    

    
    def evaluate_generalization(self, data_handlers: List, temp_model_name: Optional[str] = None) -> Dict:
        """
        Evaluate model generalization across multiple environments.
        
        Args:
            data_handlers: List of FlexibleJobShopDataHandler instances for different test environments
            temp_model_name: Optional name for temporary model save. If None, uses timestamp
            
        Returns:
            Dictionary containing evaluation results for each environment
        """
        from RL.rl_env import RLEnv
        
        # Save current model temporarily
        if temp_model_name is None:
            temp_model_name = f"temp_model_{time.strftime('%Y%m%d_%H%M%S')}.pth"
        temp_model_path = os.path.join(self.model_save_dir, temp_model_name)
        self.agent.save(temp_model_path)
        
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
        
        # Note: Generalization metrics are logged by the main training loop
        

        
        # Clean up temporary model
        if os.path.exists(temp_model_path):
            os.remove(temp_model_path)
        
        return {
            'individual_results': generalization_results,
            'aggregate_stats': aggregate_stats
        }
    



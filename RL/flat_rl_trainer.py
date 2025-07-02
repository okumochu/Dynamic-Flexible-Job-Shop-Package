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
import gym
from gym import spaces
import wandb
from RL.PPO.ppo_worker import PPOWorker
from RL.PPO.buffer import PPOBuffer
from RL.flat_rl_env import FlatRLEnv

class FlatRLTrainer:
    """
    Trainer class for Flat RL agent on Flexible Job Shop Scheduling.
    Handles training, evaluation, and model management.
    """
    
    def __init__(self, 
                 env: FlatRLEnv,
                 hidden_dim: int = 128,
                 lr: float = 3e-4,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 clip_ratio: float = 0.2,
                 entropy_coeff: float = 1e-3,
                 value_coeff: float = 0.5,
                 max_grad_norm: float = 0.5,
                 device: str = 'auto',
                 model_save_dir: str = 'result/flat_rl/model'):
        """
        Initialize the trainer.
        
        Args:
            env: The environment to train on
            hidden_dim: Hidden layer dimension for networks
            lr: Learning rate
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            clip_ratio: PPO clip ratio
            entropy_coeff: Entropy coefficient for exploration
            value_coeff: Value loss coefficient
            max_grad_norm: Maximum gradient norm for clipping
            device: Device to run on ('auto', 'cpu', 'cuda')
            model_save_dir: Directory to save models
        """
        self.env = env
        self.model_save_dir = model_save_dir
        
        # Create save directory
        os.makedirs(model_save_dir, exist_ok=True)
        
        # Initialize agent
        # Ensure obs_shape is always a 1D tuple (obs_dim,) for flat state vectors
        
        obs_dim = env.obs_len
        obs_shape = (obs_dim,)
        action_space = cast(spaces.Discrete, env.action_space)
        action_dim = action_space.n
        self.agent = PPOWorker(
            obs_shape=obs_shape,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            lr=lr,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_ratio=clip_ratio,
            entropy_coeff=entropy_coeff,
            value_coeff=value_coeff,
            max_grad_norm=max_grad_norm,
            device=device
        )
        
        # Training statistics
        self.training_history = {
            'episode_rewards': [],
            'episode_makespans': [],
            'episode_twts': [],
            'training_stats': [],
            'evaluation_results': []
        }
        
        print(f"Trainer initialized on device: {self.agent.device}")
        print(f"Environment: {env.num_jobs} jobs, {env.num_machines} machines")
        print(f"Action space: {action_dim} actions")
        print(f"Observation space: {obs_shape}")
    
    def train(self, 
              num_episodes: int = 500, 
              buffer_size: int = 1000, 
              update_frequency: int = 10,
              eval_frequency: int = 50,
              save_frequency: int = 100) -> Dict:
        """
        Train the agent.
        
        Args:
            num_episodes: Number of episodes to train
            buffer_size: Size of the replay buffer
            update_frequency: How often to update the agent
            eval_frequency: How often to evaluate the agent
            save_frequency: How often to save the model
            
        Returns:
            Dictionary containing training results
        """
        # Initialize buffer
        obs_dim = self.env.obs_len
        obs_shape = (obs_dim,)
        action_space = cast(spaces.Discrete, self.env.action_space)
        action_dim = action_space.n
        buffer = PPOBuffer(buffer_size, obs_shape, action_dim, self.agent.device)
        
        # Initialize wandb
        wandb.init(
            project="Flexible-Job-Shop-RL",
            config={
                "num_episodes": num_episodes,
                "buffer_size": buffer_size,
                "update_frequency": update_frequency,
                "eval_frequency": eval_frequency,
                "save_frequency": save_frequency,
                "hidden_dim": self.agent.policy.backbone[0].out_features,
                "lr": self.agent.policy_optimizer.param_groups[0]['lr'],
                "gamma": self.agent.gamma,
                "gae_lambda": self.agent.gae_lambda,
                "clip_ratio": self.agent.clip_ratio,
                "entropy_coeff": self.agent.entropy_coeff,
                "value_coeff": self.agent.value_coeff,
                "max_grad_norm": self.agent.max_grad_norm,
            }
        )
        
        start_time = time.time()
        
        # Training loop with progress bar
        pbar = tqdm(range(num_episodes), desc="Training Progress")
        for episode in pbar:
            obs = self.env.reset()
            total_reward = 0
            step_count = 0
            while True:
                action_mask = self.env.get_action_mask()
                if not action_mask.any():
                    break
                action, log_prob, value = self.agent.get_action(obs, action_mask)
                next_obs, reward, done, info = self.env.step(action)
                buffer.add(obs, action, reward, value, log_prob, action_mask, done)
                total_reward += reward
                obs = next_obs
                step_count += 1
                if done:
                    break
            # Only log necessary episode stats to wandb
            makespan = info.get('makespan', 0)
            twt = info.get('twt', 0)
            alpha = getattr(self.env, 'alpha', 0.5)
            objective = alpha * makespan + (1 - alpha) * twt
            wandb.log({
                "episode": episode,
                "reward": total_reward,
                "makespan": makespan,
                "twt": twt,
                "objective": objective
            })
            self.training_history['episode_rewards'].append(total_reward)
            self.training_history['episode_makespans'].append(makespan)
            self.training_history['episode_twts'].append(twt)
            if buffer.size >= update_frequency and episode % update_frequency == 0:
                stats = self.agent.update(buffer, num_epochs=4)
                self.training_history['training_stats'].append(stats)
                buffer.clear()
                wandb.log({
                    "policy_loss": stats["policy_loss"],
                    "value_loss": stats["value_loss"],
                    "entropy_loss": stats["entropy_loss"],
                    "total_loss": stats["total_loss"],
                    "clip_fraction": stats["clip_fraction"]
                })
        training_time = time.time() - start_time
        pbar.close()
        self.save_model("final_model.pth")
        wandb.finish()
        return {
            'episode_rewards': self.training_history['episode_rewards'],
            'episode_makespans': self.training_history['episode_makespans'],
            'episode_twts': self.training_history['episode_twts'],
            'training_stats': self.training_history['training_stats'],
            'training_time': training_time
        }
    
    def evaluate(self, num_episodes: int = 10) -> Dict:
        """
        Evaluate the trained agent.
        
        Args:
            num_episodes: Number of episodes to evaluate
            
        Returns:
            Dictionary containing evaluation results
        """
        makespans = []
        twts = []
        schedules = []
        
        # Evaluation loop with progress bar
        pbar = tqdm(range(num_episodes), desc="Evaluation Progress", leave=False)
        
        for episode in pbar:
            obs = self.env.reset()
            total_reward = 0
            
            while True:
                action_mask = self.env.get_action_mask()
                action, _, _ = self.agent.get_action(obs, action_mask)
                obs, reward, done, info = self.env.step(action)
                total_reward += reward
                
                if done:
                    break
            
            makespans.append(info['makespan'])
            twts.append(info['twt'])
            schedules.append(self.env.get_schedule_info())
        
        pbar.close()
        
        avg_makespan = np.mean(makespans)
        avg_twt = np.mean(twts)
        std_makespan = np.std(makespans)
        std_twt = np.std(twts)
        
        return {
            'makespans': makespans,
            'twts': twts,
            'schedules': schedules,
            'avg_makespan': avg_makespan,
            'avg_twt': avg_twt,
            'std_makespan': std_makespan,
            'std_twt': std_twt
        }
    
    def save_model(self, filename: str):
        """Save the model to file."""
        filepath = os.path.join(self.model_save_dir, filename)
        self.agent.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filename: str):
        """Load the model from file."""
        filepath = os.path.join(self.model_save_dir, filename)
        if os.path.exists(filepath):
            self.agent.load(filepath)
            print(f"Model loaded from {filepath}")
        else:
            print(f"Model file {filepath} not found")


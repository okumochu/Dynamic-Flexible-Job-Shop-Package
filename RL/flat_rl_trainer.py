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
                 pi_lr: float = 3e-5, # default 3e-4
                 v_lr: float = 1e-4, # default 1e-3
                 gamma: float = 0.99,
                 gae_lambda: float = 0.97,
                 clip_ratio: float = 0.2,
                 target_kl: float = 0.1,  # default 0.01
                 train_pi_iters: int = 80,
                 train_v_iters: int = 80,
                 steps_per_epoch: int = 4000, # default 4000
                 epochs: int = 50, # default 50
                 device: str = 'auto',
                 model_save_dir: str = 'result/flat_rl/model'):
        """
        Initialize the trainer.
        
        Args:
            env: The environment to train on
            hidden_dim: Hidden layer dimension for networks
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
        self.target_kl = target_kl
        
        # Create save directory
        os.makedirs(model_save_dir, exist_ok=True)
        
        # Initialize agent
        # Ensure obs_shape is always a 1D tuple (obs_dim,) for flat state vectors
        
        obs_dim = env.obs_len
        obs_shape = (obs_dim,)
        action_space = cast(spaces.Discrete, env.action_space)
        action_dim = int(action_space.n)
        self.agent = PPOWorker(
            obs_shape=obs_shape,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            pi_lr=pi_lr,
            v_lr=v_lr,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_ratio=clip_ratio,
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
    
    def train(self) -> Dict:
        """
        Train the agent using epoch-based PPO (Spinning Up style).
        Returns:
            Dictionary containing training results
        """
        obs_dim = self.env.obs_len
        obs_shape = (obs_dim,)
        action_space = cast(spaces.Discrete, self.env.action_space)
        action_dim = int(action_space.n)
        buffer = PPOBuffer(self.steps_per_epoch, obs_shape, action_dim, self.agent.device)
        wandb.init(
            project="Flexible-Job-Shop-RL",
            config={
                "steps_per_epoch": self.steps_per_epoch,
                "epochs": self.epochs,
                "hidden_dim": self.agent.policy.backbone[0].out_features,
                "pi_lr": self.agent.pi_lr,
                "v_lr": self.agent.v_lr,
                "gamma": self.agent.gamma,
                "gae_lambda": self.agent.gae_lambda,
                "clip_ratio": self.agent.clip_ratio,
                "target_kl": self.target_kl,
                "train_pi_iters": self.train_pi_iters,
                "train_v_iters": self.train_v_iters,
            }
        )
        # Dummy test metric to verify wandb logging
        start_time = time.time()
        pbar = tqdm(range(self.epochs), desc="Training Progress")
        for epoch, _ in enumerate(pbar):
            obs, _ = self.env.reset()
            obs = torch.tensor(obs, dtype=torch.float32, device=self.agent.device)
            ep_reward, ep_makespan, ep_twt = 0, 0, 0
            update_step = 0
            for t in range(self.steps_per_epoch):
                action_mask = self.env.get_action_mask()
                if not action_mask.any():
                    break
                action, log_prob, value = self.agent.get_action(obs, action_mask)
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                next_obs = torch.tensor(next_obs, dtype=torch.float32, device=self.agent.device)
                buffer.add(obs, action, reward, value, log_prob, action_mask, done)
                ep_reward += reward
                obs = next_obs
                if done or (t == self.steps_per_epoch - 1):
                    print(f"Epoch {epoch}, Update Step {update_step}")
                    update_step += 1
                    makespan = float(info.get('makespan', 0))
                    twt = float(info.get('twt', 0))
                    alpha = float(getattr(self.env, 'alpha', 0.5))
                    objective = alpha * makespan + (1 - alpha) * twt
                    self.training_history['episode_rewards'].append(ep_reward)
                    self.training_history['episode_makespans'].append(makespan)
                    self.training_history['episode_twts'].append(twt)
                    wandb.log({
                        "makespan": makespan,
                        "twt": twt,
                        "objective": objective
                    },step=update_step)  # Log without step argument
                    obs, _ = self.env.reset()
                    obs = torch.tensor(obs, dtype=torch.float32, device=self.agent.device)
                    ep_reward, ep_makespan, ep_twt, ep_len = 0, 0, 0, 0
            # PPO update
            stats = self.agent.update(
                buffer,
                train_pi_iters=self.train_pi_iters,
                train_v_iters=self.train_v_iters,
                target_kl=self.target_kl
            )
            self.training_history['training_stats'].append(stats)
            buffer.clear()
            wandb.log({
                "policy_loss": stats["policy_loss"],
                "value_loss": stats["value_loss"],
                "entropy_loss": stats["entropy_loss"],
                "clip_fraction": stats["clip_fraction"]
            },step=epoch)
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


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
                 pi_lr: float = 3e-4,
                 v_lr: float = 1e-3,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.97,
                 clip_ratio: float = 0.2,
                 entropy_coeff: float = 1e-3,
                 value_coeff: float = 0.5,
                 max_grad_norm: float = 0.5,
                 target_kl: float = 0.01,
                 train_pi_iters: int = 80,
                 train_v_iters: int = 80,
                 steps_per_epoch: int = 4000,
                 epochs: int = 50,
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
            entropy_coeff: Entropy coefficient for exploration
            value_coeff: Value loss coefficient
            max_grad_norm: Maximum gradient norm for clipping
            target_kl: Early stopping KL threshold
            train_pi_iters: Policy update iterations per epoch
            train_v_iters: Value update iterations per epoch
            steps_per_epoch
                Number of environment interaction steps (timesteps) to collect per training epoch. 
                This controls how much experience is gathered before each round of policy/value updates.
            epochs
                Total number of training epochs to run. Each epoch consists of collecting `steps_per_epoch` steps,
                followed by policy and value network updates. The total training steps will be `steps_per_epoch * epochs`.
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
        action_dim = action_space.n
        self.agent = PPOWorker(
            obs_shape=obs_shape,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            pi_lr=pi_lr,
            v_lr=v_lr,
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
    
    def train(self) -> Dict:
        """
        Train the agent using epoch-based PPO (Spinning Up style).
        Returns:
            Dictionary containing training results
        """
        obs_dim = self.env.obs_len
        obs_shape = (obs_dim,)
        action_space = cast(spaces.Discrete, self.env.action_space)
        action_dim = action_space.n
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
                "entropy_coeff": self.agent.entropy_coeff,
                "value_coeff": self.agent.value_coeff,
                "max_grad_norm": self.agent.max_grad_norm,
                "target_kl": self.target_kl,
                "train_pi_iters": self.train_pi_iters,
                "train_v_iters": self.train_v_iters,
            }
        )
        start_time = time.time()
        pbar = tqdm(range(self.epochs), desc="Training Progress")
        for epoch in pbar:
            obs = self.env.reset()
            ep_reward, ep_makespan, ep_twt = 0, 0, 0
            ep_len = 0
            for t in range(self.steps_per_epoch):
                action_mask = self.env.get_action_mask()
                if not action_mask.any():
                    break
                action, log_prob, value = self.agent.get_action(obs, action_mask)
                next_obs, reward, done, info = self.env.step(action)
                buffer.add(obs, action, reward, value, log_prob, action_mask, done)
                ep_reward += reward
                obs = next_obs
                ep_len += 1
                if done or (t == self.steps_per_epoch - 1):
                    makespan = info.get('makespan', 0)
                    twt = info.get('twt', 0)
                    alpha = getattr(self.env, 'alpha', 0.5)
                    objective = alpha * makespan + (1 - alpha) * twt
                    self.training_history['episode_rewards'].append(ep_reward)
                    self.training_history['episode_makespans'].append(makespan)
                    self.training_history['episode_twts'].append(twt)
                    wandb.log({
                        "epoch": epoch,
                        "reward": ep_reward,
                        "makespan": makespan,
                        "twt": twt,
                        "objective": objective,
                        "ep_len": ep_len
                    })
                    obs = self.env.reset()
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
                "total_loss": stats["total_loss"],
                "clip_fraction": stats["clip_fraction"],
                "kl": stats.get("kl", 0.0)
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


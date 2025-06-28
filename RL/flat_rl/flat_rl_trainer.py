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

from .flat_rl_env import FlatRLEnv
from .ppo_agent import PPOAgent, PPOBuffer

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
                 model_save_dir: str = 'result/flat_rl_model'):
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
        obs_shape = cast(Tuple[int, int, int], env.observation_space.shape)
        action_space = cast(spaces.Discrete, env.action_space)
        action_dim = action_space.n
        
        self.agent = PPOAgent(
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
        obs_shape = cast(Tuple[int, int, int], self.env.observation_space.shape)
        action_space = cast(spaces.Discrete, self.env.action_space)
        action_dim = action_space.n
        buffer = PPOBuffer(buffer_size, obs_shape, action_dim, self.agent.device)
        
        start_time = time.time()
        
        # Training loop with progress bar
        pbar = tqdm(range(num_episodes), desc="Training Progress")
        
        for episode in pbar:
            obs = self.env.reset()
            total_reward = 0
            step_count = 0
            
            # Episode loop
            while True:
                # Get action mask
                action_mask = self.env.get_action_mask()
                
                # Get action from agent
                action, log_prob, value = self.agent.get_action(obs, action_mask)
                
                # Take step in environment
                next_obs, reward, done, info = self.env.step(action)
                
                # Store transition in buffer
                buffer.add(obs, action, reward, value, log_prob, action_mask, done)
                
                total_reward += reward
                obs = next_obs
                step_count += 1
                
                if done:
                    break
            
            # Update agent if buffer is full enough
            if buffer.size >= update_frequency and episode % update_frequency == 0:
                stats = self.agent.update(buffer, num_epochs=4)
                self.training_history['training_stats'].append(stats)
                buffer.clear()
            
            # Record episode statistics
            self.training_history['episode_rewards'].append(total_reward)
            self.training_history['episode_makespans'].append(info['makespan'])
            self.training_history['episode_twts'].append(info['twt'])
            
            # Update progress bar
            if len(self.training_history['episode_rewards']) >= 50:
                avg_reward = np.mean(self.training_history['episode_rewards'][-50:])
                avg_makespan = np.mean(self.training_history['episode_makespans'][-50:])
                avg_twt = np.mean(self.training_history['episode_twts'][-50:])
                pbar.set_postfix({
                    'Avg Reward': f'{avg_reward:.3f}',
                    'Avg Makespan': f'{avg_makespan:.1f}',
                    'Avg TWT': f'{avg_twt:.1f}'
                })
            
            # Evaluate periodically
            if (episode + 1) % eval_frequency == 0:
                eval_results = self.evaluate(num_episodes=5)
                self.training_history['evaluation_results'].append({
                    'episode': episode + 1,
                    'results': eval_results
                })
            
            # Save model periodically
            if (episode + 1) % save_frequency == 0:
                self.save_model(f"checkpoint_episode_{episode + 1}.pth")
        
        training_time = time.time() - start_time
        pbar.close()
        
        # Save final model
        self.save_model("final_model.pth")
        
        print(f"Training completed in {training_time:.2f} seconds")
        
        return {
            'episode_rewards': self.training_history['episode_rewards'],
            'episode_makespans': self.training_history['episode_makespans'],
            'episode_twts': self.training_history['episode_twts'],
            'training_stats': self.training_history['training_stats'],
            'evaluation_results': self.training_history['evaluation_results'],
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
    
    def plot_training_curves(self, save_path: Optional[str] = None):
        """Plot training curves."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Episode rewards
        axes[0, 0].plot(self.training_history['episode_rewards'])
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].grid(True)
        
        # Episode makespans
        axes[0, 1].plot(self.training_history['episode_makespans'])
        axes[0, 1].set_title('Episode Makespans')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Makespan')
        axes[0, 1].grid(True)
        
        # Episode TWT
        axes[1, 0].plot(self.training_history['episode_twts'])
        axes[1, 0].set_title('Episode Total Weighted Tardiness')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('TWT')
        axes[1, 0].grid(True)
        
        # Training losses
        if self.training_history['training_stats']:
            policy_losses = [stats['policy_loss'] for stats in self.training_history['training_stats']]
            value_losses = [stats['value_loss'] for stats in self.training_history['training_stats']]
            
            axes[1, 1].plot(policy_losses, label='Policy Loss')
            axes[1, 1].plot(value_losses, label='Value Loss')
            axes[1, 1].set_title('Training Losses')
            axes[1, 1].set_xlabel('Update Step')
            axes[1, 1].set_ylabel('Loss')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training curves saved to {save_path}")
        
        plt.show()
    
    def plot_evaluation_progress(self, save_path: Optional[str] = None):
        """Plot evaluation progress over training."""
        if not self.training_history['evaluation_results']:
            print("No evaluation results to plot")
            return
        
        episodes = [result['episode'] for result in self.training_history['evaluation_results']]
        avg_makespans = [result['results']['avg_makespan'] for result in self.training_history['evaluation_results']]
        avg_twts = [result['results']['avg_twt'] for result in self.training_history['evaluation_results']]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Makespan progress
        ax1.plot(episodes, avg_makespans, 'b-o')
        ax1.set_title('Average Makespan Progress')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Makespan')
        ax1.grid(True)
        
        # TWT progress
        ax2.plot(episodes, avg_twts, 'r-o')
        ax2.set_title('Average TWT Progress')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('TWT')
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Evaluation progress saved to {save_path}")
        
        plt.show()
    
    def get_best_schedule(self, num_episodes: int = 10) -> Tuple[Dict, float, float]:
        """
        Get the best schedule from multiple evaluations.
        
        Args:
            num_episodes: Number of episodes to evaluate
            
        Returns:
            Tuple of (best_schedule, best_makespan, best_twt)
        """
        eval_results = self.evaluate(num_episodes)
        
        # Find best schedule (lowest makespan)
        best_idx = np.argmin(eval_results['makespans'])
        best_schedule = eval_results['schedules'][best_idx]
        best_makespan = eval_results['makespans'][best_idx]
        best_twt = eval_results['twts'][best_idx]
        
        return best_schedule, best_makespan, best_twt

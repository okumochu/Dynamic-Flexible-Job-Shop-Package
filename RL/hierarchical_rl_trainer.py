"""
Hierarchical RL Trainer for Flexible Job Shop Scheduling
Feudal-style Manager + Worker for Dynamic Job-Shop Scheduler

Now uses the restructured HierarchicalAgent and HierarchicalRLEnv
following the same pattern as flat RL implementation.
"""

import torch
import torch.nn.functional as F
import numpy as np
import time
import os
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import matplotlib.pyplot as plt
import wandb

# Import from restructured modules
from RL.PPO.buffer import PPOBuffer, HierarchicalBuffer
from RL.PPO.hierarichical_agent import HierarchicalAgent
from RL.hierarchical_rl_env import HierarchicalRLEnv


class HierarchicalRLTrainer:
    """Trainer for Hierarchical RL with Manager-Worker architecture"""
    
    def __init__(self, 
                 env: HierarchicalRLEnv,
                 epochs: int = 100,
                 steps_per_epoch: int = 400,  # ~one Manager cycle as specified
                 dilation: int = 10,  # Manager horizon c
                 latent_dim: int = 256,
                 goal_dim: int = 32,
                 manager_lr: float = 3e-4,
                 worker_lr: float = 3e-4,
                 alpha_start: float = 1.0,  # Intrinsic reward weight
                 alpha_end: float = 0.1,
                 gamma_manager: float = 0.995,  # Manager discount
                 gamma_worker: float = 0.95,   # Worker discount
                 gae_lambda: float = 0.95,
                 clip_ratio: float = 0.2,
                 entropy_coef: float = 0.01,
                 epsilon_greedy: float = 0.1,  # Manager exploration
                 device: str = 'auto',
                 model_save_dir: str = 'result/hierarchical_rl/model'):
        
        self.env = env
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.dilation = dilation
        self.alpha_start = alpha_start
        self.alpha_end = alpha_end
        self.model_save_dir = model_save_dir
        
        # Create hierarchical agent
        self.agent = HierarchicalAgent(
            input_dim=env.obs_len,
            action_dim=env.action_dim,
            latent_dim=latent_dim,
            goal_dim=goal_dim,
            goal_duration=dilation,  # Use dilation as goal_duration (c parameter)
            manager_lr=manager_lr,
            worker_lr=worker_lr,
            gamma_manager=gamma_manager,
            gamma_worker=gamma_worker,
            gae_lambda=gae_lambda,
            clip_ratio=clip_ratio,
            entropy_coef=entropy_coef,
            epsilon_greedy=epsilon_greedy,
            device=device
        )
        
        # Create save directory
        os.makedirs(model_save_dir, exist_ok=True)
        
        # Training history
        self.training_history = {
            'episode_rewards': [],
            'episode_makespans': [],
            'episode_twts': [],
            'manager_stats': [],
            'worker_stats': []
        }
        
        print(f"Hierarchical RL Trainer initialized")
        print(f"Manager dilation: {dilation}, Latent dim: {latent_dim}, Goal dim: {goal_dim}")
    
    def train(self):
        """Main training loop"""
        wandb.init(
            project="Hierarchical-Job-Shop-RL",
            config={
                "epochs": self.epochs,
                "steps_per_epoch": self.steps_per_epoch,
                "dilation": self.dilation,
                "latent_dim": self.agent.latent_dim,
                "goal_dim": self.agent.goal_dim,
                "manager_lr": self.agent.manager_lr,
                "worker_lr": self.agent.worker_lr,
                "alpha_start": self.alpha_start,
                "alpha_end": self.alpha_end,
                "gamma_manager": self.agent.gamma_manager,
                "gamma_worker": self.agent.gamma_worker,
            }
        )
        
        # Use standard PPO buffer for worker experiences
        buffer = PPOBuffer(
            self.steps_per_epoch, (self.env.obs_len,), self.env.action_dim, self.agent.device
        )
        
        # Use HierarchicalBuffer for hierarchical data collection
        hierarchical_buffer = HierarchicalBuffer(
            self.steps_per_epoch, self.agent.latent_dim, self.agent.device
        )
        
        start_time = time.time()
        pbar = tqdm(range(self.epochs), desc="Hierarchical Training")
        
        for epoch in pbar:
            # Anneal alpha (intrinsic reward weight)
            alpha = self.alpha_start + (self.alpha_end - self.alpha_start) * (epoch / self.epochs)
            
            # Collect rollouts
            self.collect_rollouts(buffer, hierarchical_buffer, alpha)
            
            # Get hierarchical data from buffer
            encoded_states, goals, pooled_goals = hierarchical_buffer.get_data()
            
            # Update both manager and worker using agent methods
            manager_stats = self.agent.update_manager()
            worker_stats = self.agent.update_worker(buffer, encoded_states, goals, pooled_goals)
            
            # Log statistics
            self.training_history['manager_stats'].append(manager_stats)
            self.training_history['worker_stats'].append(worker_stats)
            
            wandb.log({
                "epoch": epoch,
                "alpha": alpha,
                "manager_loss": manager_stats.get('loss', 0),
                "worker_policy_loss": worker_stats.get('policy_loss', 0),
                "worker_value_loss": worker_stats.get('value_loss', 0),
                "avg_goal_norm": manager_stats.get('avg_goal_norm', 0),
                "avg_cosine_alignment": manager_stats.get('avg_cosine_alignment', 0),
            })
            
            buffer.clear()
            hierarchical_buffer.clear()
            self.agent.clear_manager_data()
        
        training_time = time.time() - start_time
        pbar.close()
        self.save_model("final_model.pth")
        wandb.finish()
        
        return {
            'training_time': training_time,
            'training_history': self.training_history
        }
    
    def collect_rollouts(self, buffer, hierarchical_buffer, alpha):
        """Collect rollouts with hierarchical policy using HierarchicalAgent and HierarchicalBuffer"""
        obs, _ = self.env.reset()
        obs = torch.tensor(obs, dtype=torch.float32, device=self.agent.device)
        hierarchical_buffer.mark_episode_start()
        
        # Manager state
        goals_history = []
        encoded_states_history = []
        pooled_goals_history = []
        
        episode_reward = 0
        step = 0
        
        while step < self.steps_per_epoch:
            # Encode state using agent
            z_t = self.agent.encode_state(obs)
            encoded_states_history.append(z_t)
            
            # Manager decision (every dilation steps)
            if step % self.dilation == 0:
                # Manager emits new goal
                goal, manager_value = self.agent.get_manager_goal(z_t)
                
                # Add epsilon-greedy exploration to goals
                if goal is not None and torch.rand(1).item() < self.agent.epsilon_greedy:
                    goal = torch.randn_like(goal)
                    goal = F.normalize(goal, p=2, dim=0)
                
                current_goal = goal
                goals_history.append(goal)
                
                # Store manager experience for later processing
                if len(goals_history) > 1 and manager_value is not None:  # Not first goal and valid value
                    prev_goal_idx = len(goals_history) - 2
                    manager_reward = self.agent.compute_manager_reward(
                        encoded_states_history, goals_history, prev_goal_idx, step - self.dilation
                    )
                    
                    # Add manager experience to agent
                    manager_state_idx = max(0, step - self.dilation)
                    if manager_state_idx < len(encoded_states_history):
                        self.agent.add_manager_experience(
                            encoded_states_history[manager_state_idx],
                            goals_history[prev_goal_idx],
                            manager_value,
                            manager_reward,
                            False
                        )
            
            # Pool goals for worker using agent
            pooled_goal = self.agent.pool_goals(goals_history, step, self.dilation)
            pooled_goals_history.append(pooled_goal)
            
            # Add to hierarchical buffer
            current_goal = goals_history[-1] if goals_history else None
            hierarchical_buffer.add(z_t, current_goal, pooled_goal, step)
            
            # Worker action using agent
            action_mask = self.env.get_action_mask()
            
            action, log_prob, worker_value = self.agent.take_action(
                obs, action_mask, pooled_goal
            )
            
            # Environment step
            next_obs, reward_ext, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            next_obs = torch.tensor(next_obs, dtype=torch.float32, device=self.agent.device)
            
            # Compute intrinsic reward using agent
            reward_int = self.agent.compute_intrinsic_reward(
                encoded_states_history, goals_history, step, self.dilation
            )
            
            # Mixed reward for worker
            reward_mixed = reward_ext + alpha * reward_int
            
            # Store worker experience in standard PPO buffer
            buffer.add(obs, action, reward_mixed, worker_value, log_prob, action_mask, done)
            
            episode_reward += reward_ext
            # Update episode reward in agent
            self.agent.update_episode_reward(reward_ext)
            obs = next_obs
            step += 1
            
            if done:
                # Add final manager reward for the last goal when episode ends
                if len(goals_history) > 0:
                    final_goal_idx = len(goals_history) - 1
                    final_manager_reward = self.agent.compute_manager_reward(
                        encoded_states_history, goals_history, final_goal_idx, step - self.dilation
                    )
                    
                    # Add final manager experience
                    self.agent.add_manager_experience(
                        encoded_states_history[-self.dilation] if len(encoded_states_history) >= self.dilation else encoded_states_history[0],
                        goals_history[final_goal_idx],
                        0.0,  # Value will be computed during update
                        final_manager_reward,
                        True  # Episode done
                    )
                
                # Log episode info
                wandb.log({
                    "episode_reward": episode_reward,
                    "episode_makespan": info['makespan'],
                    "episode_twt": info['twt']
                })
                self.training_history['episode_rewards'].append(episode_reward)
                self.training_history['episode_makespans'].append(info['makespan'])
                self.training_history['episode_twts'].append(info['twt'])
                
                # Mark episode end in hierarchical buffer
                hierarchical_buffer.mark_episode_end()
                
                # Reset for new episode
                obs, _ = self.env.reset()
                obs = torch.tensor(obs, dtype=torch.float32, device=self.agent.device)
                hierarchical_buffer.mark_episode_start()
                manager_hidden = None
                current_goal = torch.zeros(self.agent.latent_dim, device=self.agent.device)
                goals_history = []
                encoded_states_history = []
                pooled_goals_history = []
                episode_reward = 0
    
    def save_model(self, filename: str):
        """Save model using agent's save method"""
        filepath = os.path.join(self.model_save_dir, filename)
        self.agent.save(filepath)
        print(f"Models saved to {filepath}")
    
    def load_model(self, filename: str):
        """Load model using agent's load method"""
        filepath = os.path.join(self.model_save_dir, filename)
        if os.path.exists(filepath):
            self.agent.load(filepath)
            print(f"Models loaded from {filepath}")
        else:
            print(f"Model file {filepath} not found")
    
    def evaluate(self, num_episodes: int = 10):
        """Evaluate the hierarchical policy"""
        total_rewards = []
        total_makespans = []
        total_twts = []
        
        for episode in range(num_episodes):
            obs, _ = self.env.reset()
            obs = torch.tensor(obs, dtype=torch.float32, device=self.agent.device)
            
            manager_hidden = None
            current_goal = torch.zeros(self.agent.latent_dim, device=self.agent.device)
            goals_history = []
            encoded_states_history = []
            
            episode_reward = 0
            step = 0
            max_steps = self.env.num_jobs * max(len(job.operations) for job in self.env.jobs.values()) * 2
            
            while not self.env.state.is_done() and step < max_steps:
                # Encode state
                z_t = self.agent.encode_state(obs)
                encoded_states_history.append(z_t)
                
                # Manager decision
                if step % self.dilation == 0:
                    goal, _ = self.agent.get_manager_goal(z_t)
                    current_goal = goal
                    goals_history.append(goal)
                
                # Pool goals for worker
                pooled_goal = self.agent.pool_goals(goals_history, step, self.dilation)
                
                # Worker action (deterministic)
                action_mask = self.env.get_action_mask()
                if not action_mask.any():
                    break
                
                action = self.agent.get_deterministic_action(
                    obs, action_mask, pooled_goal
                )
                
                # Environment step
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                
                episode_reward += reward
                next_obs = torch.tensor(next_obs, dtype=torch.float32, device=self.agent.device)
                obs = next_obs
                step += 1
                
                if done:
                    break
            
            total_rewards.append(episode_reward)
            total_makespans.append(info['makespan'])
            total_twts.append(info['twt'])
        
        return {
            'avg_reward': np.mean(total_rewards),
            'avg_makespan': np.mean(total_makespans),
            'avg_twt': np.mean(total_twts),
            'std_reward': np.std(total_rewards),
            'std_makespan': np.std(total_makespans),
            'std_twt': np.std(total_twts),
            'all_rewards': total_rewards,
            'all_makespans': total_makespans,
            'all_twts': total_twts
        } 
"""
Hierarchical RL Trainer for Flexible Job Shop Scheduling
Feudal-style Manager + Worker for Dynamic Job-Shop Scheduler

Now uses the restructured HierarchicalAgent and unified RLEnv
following the same pattern as flat RL implementation.
"""

import torch
import torch.nn.functional as F
import numpy as np
import time
import os
from tqdm import tqdm
import wandb
from RL.PPO.buffer import PPOBuffer, HierarchicalPPOBuffer
from RL.PPO.hierarchical_agent import HierarchicalAgent
from RL.rl_env import RLEnv
from typing import Dict, Optional, List


class HierarchicalRLTrainer:
    """Trainer for Hierarchical RL with Manager-Worker architecture"""
    
    def __init__(self, 
                 env: RLEnv,
                 epochs: int,
                 steps_per_epoch: int,
                 goal_duration: int,
                 latent_dim: int,
                 goal_dim: int,
                 manager_lr: float,
                 worker_lr: float,
                 gamma_manager: float,
                 gamma_worker: float,
                 clip_ratio: float,
                 entropy_coef: float,
                 gae_lambda: float,
                 train_pi_iters: int,
                 train_v_iters: int,
                 alpha: float,
                 device: str = 'auto',
                 model_save_dir: str = 'result/hierarchical_rl/model'):
        
        self.env = env
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.goal_duration = goal_duration
        self.alpha = alpha
        self.train_pi_iters = train_pi_iters
        self.train_v_iters = train_v_iters
        self.model_save_dir = model_save_dir
        
        # Create hierarchical agent
        self.agent = HierarchicalAgent(
            input_dim=env.obs_len,
            action_dim=env.action_dim,
            latent_dim=latent_dim,
            goal_dim=goal_dim,
            goal_duration=goal_duration,  # Use goal_duration as c parameter
            manager_lr=manager_lr,
            worker_lr=worker_lr,
            gamma_manager=gamma_manager,
            gamma_worker=gamma_worker,
            gae_lambda=gae_lambda,
            clip_ratio=clip_ratio,
            entropy_coef=entropy_coef,
            device=device
        )
        
        # Create save directory
        os.makedirs(model_save_dir, exist_ok=True)
        
        # Training history
        self.training_history = {
            'episode_rewards': [],
            'episode_makespans': [],
            'episode_twts': [],
            'episode_objectives': [],
            'manager_stats': [],
            'worker_stats': []
        }
        
        print(f"Hierarchical RL Trainer initialized")
        print(f"Manager goal duration: {goal_duration}, Latent dim: {latent_dim}, Goal dim: {goal_dim}")
    
    def collect_rollout(self, env_or_envs, worker_buffer, manager_buffer, epoch: int = 0) -> Dict:
        """
        Collect rollout data for one epoch with hierarchical structure.
        Supports both single environment and curriculum learning with multiple environments.
        
        Args:
            env_or_envs: Single environment or list of environments for curriculum learning
            worker_buffer: PPOBuffer for worker experiences
            manager_buffer: HierarchicalPPOBuffer for manager experiences
            epoch: Current epoch number (used for curriculum learning)
            
        Returns:
            Dictionary containing episode metrics for this epoch
        """
        # Handle curriculum learning - multiple environments (sequential training)
        if isinstance(env_or_envs, list):
            envs = env_or_envs
            # Sequential training: distribute epochs evenly across environments
            epochs_per_env = self.epochs // len(envs)
            
            env_idx = epoch // epochs_per_env if epoch < (len(envs) - 1) * epochs_per_env else len(envs) - 1
            if env_idx >= len(envs):
                env_idx = len(envs) - 1
                
            current_env = envs[env_idx]
        else:
            current_env = env_or_envs
        
        # Reset environment and episode tracking
        obs, _ = current_env.reset()
        obs = torch.tensor(obs, dtype=torch.float32, device=self.agent.device)
        
        # Reset episode-specific data at the start of each epoch
        episode_steps = 0
        goals_history = []
        encoded_states_history = []
        ep_reward = 0
        
        # Track episode metrics for this epoch
        epoch_episodes = []

        for t in range(self.steps_per_epoch):
            
            # Encode state
            z_t = self.agent.encode_state(obs)
            encoded_states_history.append(z_t)
            
            # FIX: Manager decision every goal_duration steps (starting from step 0)
            if episode_steps % self.goal_duration == 0:
                goal = self.agent.get_manager_goal(z_t)
                if goal is not None:
                    goals_history.append(goal)
            
            # FIX: Compute manager reward and add transition after we have enough history
            # Only add manager transitions after we have completed at least one goal period
            if (episode_steps > 0 and 
                episode_steps % self.goal_duration == 0 and 
                len(goals_history) >= 2 and 
                len(encoded_states_history) >= self.goal_duration):
                
                # Get the goal that was active in the previous period
                prev_goal_idx = len(goals_history) - 2  # Previous goal
                prev_goal = goals_history[prev_goal_idx]
                
                # Get states from the previous goal period
                start_idx = max(0, episode_steps - self.goal_duration)
                end_idx = episode_steps
                
                # FIX: Ensure we have valid indices
                if start_idx < len(encoded_states_history) and end_idx <= len(encoded_states_history):
                    states_over_period = encoded_states_history[start_idx:end_idx]
                    
                    if len(states_over_period) >= 2:  # Need at least start and end state
                        manager_reward = self.agent.compute_manager_reward(states_over_period, prev_goal)
                        
                        # Get manager value for the start state of this period
                        with torch.no_grad():
                            start_state = states_over_period[0]
                            manager_value = self.agent.manager_value(start_state.unsqueeze(0)).item()
                        
                        # Add manager transition
                        manager_buffer.add_manager_transition(
                            start_state,
                            prev_goal,
                            manager_reward,
                            manager_value, 
                            False  # Episode not done
                        )
            
            # Pool goals for worker (this happens every step)
            pooled_goal = self.agent.pool_goals(goals_history, episode_steps, self.goal_duration)
            
            # Add pooled goal to buffer for worker updates
            manager_buffer.add_step_data(pooled_goal)
            
            # Worker action
            action_mask = current_env.get_action_mask()
            action, log_prob, worker_value = self.agent.take_action(obs, action_mask, pooled_goal)
            
            # Environment step
            next_obs, reward_ext, terminated, truncated, info = current_env.step(action)
            done = terminated or truncated
            next_obs = torch.tensor(next_obs, dtype=torch.float32, device=self.agent.device)

            # Intrinsic reward (with better bounds checking from agent)
            reward_int = self.agent.compute_intrinsic_reward(
                encoded_states_history, goals_history, episode_steps, self.goal_duration
            )
            reward_mixed = reward_ext + self.alpha * reward_int
            
            # Store worker experience
            worker_buffer.add(obs, action, reward_mixed, worker_value, log_prob, action_mask, done)
            ep_reward += reward_ext
                            
            obs = next_obs
            episode_steps += 1
            
            if done:
                # FIX: Handle final manager transition if we have a complete goal period
                if (len(goals_history) >= 1 and 
                    len(encoded_states_history) >= self.goal_duration):
                    
                    # Find the last complete goal period
                    last_goal_start = (len(goals_history) - 1) * self.goal_duration
                    if last_goal_start < len(encoded_states_history):
                        final_states = encoded_states_history[last_goal_start:]
                        if len(final_states) >= 2:
                            final_goal = goals_history[-1]
                            final_reward = self.agent.compute_manager_reward(final_states, final_goal)
                            
                            with torch.no_grad():
                                final_value = self.agent.manager_value(final_states[0].unsqueeze(0)).item()
                            
                            manager_buffer.add_manager_transition(
                                final_states[0],
                                final_goal,
                                final_reward,
                                final_value,
                                True  # Episode done
                            )
                
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

                # Reset for new episode
                obs, _ = current_env.reset()
                obs = torch.tensor(obs, dtype=torch.float32, device=self.agent.device)
                ep_reward = 0
                episode_steps = 0
                goals_history = []
                encoded_states_history = []
        
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
            name=f"Hierarchical_RL_{time.strftime('%Y%m%d_%H%M')}",
            project="dfjs",
            dir=wandb_dir,
            config={
                "epochs": self.epochs,
                "steps_per_epoch": self.steps_per_epoch,
                "goal_duration": self.goal_duration,
                "latent_dim": self.agent.latent_dim,
                "goal_dim": self.agent.goal_dim,
                "manager_lr": self.agent.manager_lr,
                "worker_lr": self.agent.worker_lr,
                "alpha": self.alpha,
                "gamma_manager": self.agent.gamma_manager,
                "gamma_worker": self.agent.gamma_worker,
                "entropy_coef": self.agent.entropy_coef,
            }
        )

        # Get observation dimensions from first environment (assuming all have same structure)
        if isinstance(env_or_envs, list):
            obs_len = env_or_envs[0].obs_len
            action_dim = env_or_envs[0].action_dim
        else:
            obs_len = env_or_envs.obs_len
            action_dim = env_or_envs.action_dim

        # Buffers
        worker_buffer = PPOBuffer(self.steps_per_epoch, (obs_len,), action_dim, self.agent.device)
        manager_buffer = HierarchicalPPOBuffer(self.steps_per_epoch, self.agent.latent_dim, self.agent.device)

        # Training loop
        start_time = time.time()
        pbar = tqdm(range(self.epochs), desc="Hierarchical Training")

        for epoch in pbar:
            # Collect rollout data
            epoch_metrics = self.collect_rollout(env_or_envs, worker_buffer, manager_buffer, epoch)
            
            # --- End of inner loop (data collection) ---
            
            # Update after collecting steps_per_epoch data
            pooled_goals = manager_buffer.get_hierarchical_data()
            manager_stats = self.agent.update_manager(manager_buffer)
            worker_stats = self.agent.update_worker(
                worker_buffer, pooled_goals,
                train_pi_iters=self.train_pi_iters, train_v_iters=self.train_v_iters
            )

            # Log training metrics (only worker loss/entropy to align with flat RL)
            wandb_log = {
                "policy_loss": worker_stats.get('policy_loss', 0),
                "value_loss": worker_stats.get('value_loss', 0),
                "entropy": worker_stats.get('entropy', 0),
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

            worker_buffer.clear()
            manager_buffer.clear()
        
        pbar.close()
        
        # Save model with timestamp
        timestamp = time.strftime('%Y%m%d_%H%M')
        model_filename = f"model_{timestamp}.pth"
        self.save_model(model_filename)
        wandb.finish()

        return {
            'training_time': time.time() - start_time,
            'training_history': self.training_history,
            'model_filename': model_filename
        }
    
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
    


    def evaluate_generalization(self, data_handlers: List, temp_model_name: Optional[str] = None) -> Dict:
        """
        Evaluate hierarchical model generalization across multiple environments.
        
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
            
            # Initialize hierarchical state tracking
            goals_history = []
            episode_reward = 0
            step = 0
            manager_decisions = 0
            max_steps = test_env.num_jobs * max(len(job.operations) for job in test_env.jobs.values()) * 2
            
            # Run deterministic policy
            while not test_env.state.is_done() and step < max_steps:
                z_t = self.agent.encode_state(obs)
                
                # Manager decision every goal_duration steps
                if step % self.goal_duration == 0:
                    goal = self.agent.get_manager_goal(z_t, deterministic=True)
                    if goal is not None:
                        goals_history.append(goal)
                    manager_decisions += 1
                
                # Pool goals for worker
                pooled_goal = self.agent.pool_goals(goals_history, step, self.goal_duration)
                
                action_mask = test_env.get_action_mask()
                if not action_mask.any():
                    break
                
                # Take deterministic action
                action = self.agent.get_deterministic_action(obs, action_mask, pooled_goal)
                
                # Step environment
                next_obs, reward, terminated, truncated, info = test_env.step(action)
                done = terminated or truncated
                
                episode_reward += reward
                next_obs = torch.tensor(next_obs, dtype=torch.float32, device=self.agent.device)
                obs = next_obs
                step += 1
                
                if done:
                    break
            
            # Get final results
            final_info = test_env.get_current_objective()
            
            env_result = {
                'makespan': final_info['makespan'],
                'twt': final_info['twt'],
                'objective': final_info['objective'],
                'episode_reward': episode_reward,
                'steps_taken': step,
                'manager_decisions': manager_decisions,
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

 
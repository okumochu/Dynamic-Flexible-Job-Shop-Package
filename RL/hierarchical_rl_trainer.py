"""
Hierarchical RL Trainer for Flexible Job Shop Scheduling
Feudal-style Manager + Worker for Dynamic Job-Shop Scheduler

Now uses the restructured HierarchicalAgent and unified RLEnv
following the same pattern as flat RL implementation.
"""

import torch
import numpy as np
import time
import os
from tqdm import tqdm
import wandb
from RL.PPO.buffer import PPOBuffer, HierarchicalPPOBuffer
from RL.PPO.hierarchical_agent import HierarchicalAgent
from RL.rl_env import RLEnv
from typing import Dict, Optional, List
from collections import deque



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
                 intrinsic_reward_scale: float,
                 project_name: Optional[str] = None,
                 run_name: Optional[str] = None,
                 device: str = 'auto',
                 model_save_dir: str = 'result/hierarchical_rl/model'):
        
        self.env = env
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.goal_duration = goal_duration
        self.intrinsic_reward_scale = intrinsic_reward_scale
        self.train_pi_iters = train_pi_iters
        self.train_v_iters = train_v_iters
        self.model_save_dir = model_save_dir
        self.project_name = project_name
        self.run_name = run_name
        
        # Add episode tracking  
        self.episode_makespans = []
        self.episode_twts = []
        self.episode_objectives = []
        
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
        
        print(f"Hierarchical RL Trainer initialized")
        print(f"Manager goal duration: {goal_duration}, Latent dim: {latent_dim}, Goal dim: {goal_dim}")
    
    def collect_rollout(self, env_or_envs, worker_buffer, manager_buffer, epoch: int) -> Dict:
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
            
            # Make the rest of the epochs go to the last environment
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
        last_goal = None
        current_goal = None
        pooled_goal_history = deque(maxlen=self.goal_duration)  # queue with c slots
        encoded_state_history = deque(maxlen=self.goal_duration)  # queue with c slots
        last_encoded_state = None
        current_encoded_state = None        

        for t in range(self.steps_per_epoch):
            
            # Encode state
            z_t = self.agent.encode_state(obs)
            current_encoded_state = z_t
            encoded_state_history.append(current_encoded_state)
            
            # Manager decision every goal_duration steps (starting from step 0)
            if episode_steps % self.goal_duration == 0:
                current_goal = self.agent.get_manager_goal(z_t)

                # if it is first step then reward is 0
                manager_reward = self.agent.compute_manager_reward(s_t = last_encoded_state, s_t_plus_c = current_encoded_state, g_t = last_goal)
                            
                # Get manager value for the start state of the finished period 
                manager_value = self.agent.manager_value(current_encoded_state.unsqueeze(0)).item()
                            
                # Add manager transition for the completed period
                manager_buffer.add_manager_transition(
                    current_goal,
                    manager_reward,
                    manager_value, 
                    False,  # Episode not done
                    obs  # Raw observation for encoder gradient updates
                )
                
                # Update last goal
                last_goal = current_goal
            
            # Pool goals for worker every step
            pooled_goal = self.agent.pool_goals(last_goal, current_goal, episode_steps, self.goal_duration)
            pooled_goal_history.append(pooled_goal)
            
            # Add pooled goal to buffer for worker updates
            manager_buffer.add_step_data(pooled_goal)
            
            # Worker action
            action_mask = current_env.get_action_mask()
            action, log_prob, worker_value = self.agent.take_action(obs, action_mask, pooled_goal)
            
            # Environment step
            next_obs, reward_ext, terminated, truncated, info = current_env.step(action)
            done = terminated or truncated
            next_obs = torch.tensor(next_obs, dtype=torch.float32, device=self.agent.device)

            # Intrinsic reward
            reward_int = self.agent.compute_intrinsic_reward(
                encoded_state_history, pooled_goal_history, episode_steps, self.goal_duration
            )
            reward_mixed = reward_ext + self.intrinsic_reward_scale * reward_int
            
            # Store worker experience
            worker_buffer.add(obs, action, reward_mixed, worker_value, log_prob, action_mask, done)                            
            obs = next_obs
            last_encoded_state = current_encoded_state
            episode_steps += 1
            
            if done:
                final_reward = self.agent.compute_manager_reward(last_encoded_state, current_encoded_state, last_goal)
                
                with torch.no_grad():
                    final_value = self.agent.manager_value(current_encoded_state.unsqueeze(0)).item()
                
                manager_buffer.add_manager_transition(
                    current_goal,
                    final_reward,
                    final_value,
                    True,  # Episode done
                    obs  # Raw observation for encoder gradient updates
                )
                
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

                # Reset for new episode
                obs, _ = current_env.reset()
                obs = torch.tensor(obs, dtype=torch.float32, device=self.agent.device)
                episode_steps = 0
                pooled_goal_history.clear()
                encoded_state_history.clear()

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
                "goal_duration": self.goal_duration,
                "latent_dim": self.agent.latent_dim,
                "goal_dim": self.agent.goal_dim,
                "manager_lr": self.agent.manager_lr,
                "worker_lr": self.agent.worker_lr,
                "intrinsic_reward_scale": self.intrinsic_reward_scale,
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
        manager_buffer = HierarchicalPPOBuffer(self.steps_per_epoch, self.agent.latent_dim, (obs_len,), self.agent.device)

        # Training loop
        start_time = time.time()
        pbar = tqdm(range(self.epochs), desc="Hierarchical Training")

        for epoch in pbar:
            # Collect rollout data
            self.collect_rollout(env_or_envs, worker_buffer, manager_buffer, epoch)
                        
            # Update after collecting steps_per_epoch data
            manager_stats = self.agent.update_manager(manager_buffer)
            worker_stats = self.agent.update_worker(
                worker_buffer, manager_buffer,
                train_pi_iters=self.train_pi_iters, train_v_iters=self.train_v_iters
            )

            # Log training metrics (only worker loss/entropy to align with flat RL)
            wandb_log = {
                "policy_loss": worker_stats.get('policy_loss', 0),
                "value_loss": worker_stats.get('value_loss', 0),
                "entropy": worker_stats.get('entropy', 0),
                "manager_policy_loss": manager_stats.get('manager_policy_loss', 0),
                "manager_value_loss": manager_stats.get('manager_value_loss', 0)
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
            'model_filename': model_filename,
            'training_history': {
                'episode_makespans': self.episode_makespans,
                'episode_twts': self.episode_twts,
                'episode_objectives': self.episode_objectives
            }
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
    


    def evaluate_generalization(self, data_handlers: List) -> Dict:
        """
        Evaluate hierarchical model generalization across multiple environments.
        
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
            
            # Initialize hierarchical state tracking
            last_goal = None
            current_goal = None
            episode_reward = 0
            step = 0
            manager_decisions = 0
            max_steps = test_env.num_jobs * max(len(job.operations) for job in test_env.jobs.values()) * 2
            
            # Run deterministic policy
            while not test_env.state.is_done() and step < max_steps:
                z_t = self.agent.encode_state(obs)
                
                # Manager decision every goal_duration steps
                if step % self.goal_duration == 0:
                    current_goal = self.agent.get_manager_goal(z_t)
                    manager_decisions += 1
                
                # Pool goals for worker
                pooled_goal = self.agent.pool_goals(last_goal, current_goal, step, self.goal_duration)
                
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
                
                # Update last_goal for next iteration
                if step % self.goal_duration == 0:
                    last_goal = current_goal
                
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
        
        return {
            'individual_results': generalization_results,
            'aggregate_stats': aggregate_stats
        }

 
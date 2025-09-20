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
from RL.PPO.buffer import PPOBuffer, HierarchicalPPOBuffer, HybridPPOBuffer
from RL.PPO.hierarchical_agent import HierarchicalAgent, HybridHierarchicalAgent
from RL.rl_env import RLEnv
from RL.rl_env_continuius_idleness import RLEnvContinuousIdleness
from typing import Dict, Optional, List
from collections import deque
from config import config


class HierarchicalRLTrainer:
    """Trainer for Hierarchical RL with Manager-Worker architecture"""
    
    def __init__(self, 
                 env: RLEnv,
                 epochs: int,
                 episodes_per_epoch: int,
                 goal_duration_ratio: int,  # Changed from goal_duration to goal_duration_ratio
                 latent_dim: int,
                 goal_dim: int,
                 manager_lr: float,
                 worker_lr: float,
                 gamma_manager: float,
                 gamma_worker: float,
                 clip_ratio: float,
                 entropy_coef: float,
                 gae_lambda: float,
                 train_per_episode: int,
                 intrinsic_reward_scale: float,
                 project_name: Optional[str] = None,
                 run_name: Optional[str] = None,
                 device: str = 'auto',
                 model_save_dir: str = 'result/hierarchical_rl/model',
                 seed: Optional[int] = None):
        
        self.env = env
        self.epochs = epochs
        self.episodes_per_epoch = episodes_per_epoch
        self.goal_duration_ratio = goal_duration_ratio
        self.intrinsic_reward_scale = intrinsic_reward_scale
        self.train_per_episode = train_per_episode
        self.model_save_dir = model_save_dir
        self.project_name = project_name
        self.run_name = run_name
        self.seed = seed
        
        # Calculate goal_duration dynamically based on total operations
        total_operations = self.env.num_jobs * max(len(job.operations) for job in self.env.jobs.values())
        self.goal_duration = total_operations // goal_duration_ratio  # Ensure at least 1
        
        # Add episode tracking for current epoch only
        self.episode_makespans = []
        self.episode_objectives = []
        self.episode_rewards = []
        
        # Create hierarchical agent
        self.agent = HierarchicalAgent(
            input_dim=env.obs_len,
            action_dim=env.action_dim,
            latent_dim=latent_dim,
            goal_dim=goal_dim,
            goal_duration=self.goal_duration,  # Use calculated goal_duration as c parameter
            manager_lr=manager_lr,
            worker_lr=worker_lr,
            gamma_manager=gamma_manager,
            gamma_worker=gamma_worker,
            gae_lambda=gae_lambda,
            clip_ratio=clip_ratio,
            entropy_coef=entropy_coef,
            device=device,
            seed=seed
        )
        
        # Create save directory
        os.makedirs(model_save_dir, exist_ok=True)
        
        print(f"Hierarchical RL Trainer initialized")
        print(f"Total operations: {total_operations}, Goal duration ratio: {goal_duration_ratio}")
        print(f"Calculated goal duration: {self.goal_duration}, Latent dim: {latent_dim}, Goal dim: {goal_dim}")


    
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
        
        # Calculate goal_duration dynamically for current environment
        total_operations = current_env.num_jobs * max(len(job.operations) for job in current_env.jobs.values())
        current_goal_duration = max(1, total_operations // self.goal_duration_ratio)
        
        # Collect data for episodes_per_epoch
        for episode in range(self.episodes_per_epoch):
            # Reset environment for new episode
            obs, _ = current_env.reset()
            obs = torch.tensor(obs, dtype=torch.float32, device=self.agent.device)
            
            # Reset episode-specific data at the start of each episode
            episode_steps = 0
            last_goal = None
            current_goal = None
            pooled_goal_history = deque(maxlen=current_goal_duration)  # queue with c slots
            encoded_state_history = deque(maxlen=current_goal_duration)  # queue with c slots
            last_encoded_state = None
            current_encoded_state = None
            episode_reward = 0

            # Run episode until completion
            while not current_env.state.is_done():
                # Encode state
                z_t = self.agent.encode_state(obs)
                current_encoded_state = z_t
                encoded_state_history.append(current_encoded_state)
                
                # Manager decision every goal_duration steps (starting from step 0)
                if episode_steps % current_goal_duration == 0:
                    current_goal = self.agent.get_manager_goal(z_t)

                    # if it is first step then reward is 0
                    manager_reward = self.agent.compute_manager_reward(s_t = last_encoded_state, s_t_plus_c = current_encoded_state, g_t = last_goal)
                                
                    # Get manager value for the start state of the finished period using manager-encoded state
                    with torch.no_grad():
                        manager_encoded = self.agent.manager_encoder(current_encoded_state.unsqueeze(0))
                        manager_value = self.agent.manager_value(manager_encoded).item()
                                
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
                pooled_goal = self.agent.pool_goals(last_goal, current_goal, episode_steps, current_goal_duration)
                pooled_goal_history.append(pooled_goal)
                
                # Add pooled goal to buffer for worker updates
                manager_buffer.add_step_data(pooled_goal)
                
                # Worker action
                action_mask = current_env.get_action_mask()
                if not action_mask.any():
                    break
                    
                action, log_prob, worker_value = self.agent.take_action(obs, action_mask, pooled_goal)
                
                # Environment step
                next_obs, reward_ext, terminated, truncated, info = current_env.step(action)
                done = terminated or truncated
                next_obs = torch.tensor(next_obs, dtype=torch.float32, device=self.agent.device)

                # Intrinsic reward
                reward_int = self.agent.compute_intrinsic_reward(
                    encoded_state_history, pooled_goal_history
                )
                reward_mixed = reward_ext + self.intrinsic_reward_scale * reward_int
                
                # Store worker experience
                worker_buffer.add(obs, action, reward_mixed, worker_value, log_prob, action_mask, done)                            
                obs = next_obs
                last_encoded_state = current_encoded_state
                episode_steps += 1
                episode_reward += reward_mixed
                
                if done:
                    break
            
            # Final manager transition for completed episode
            if episode_steps > 0:
                final_reward = self.agent.compute_manager_reward(last_encoded_state, current_encoded_state, last_goal)
                
                with torch.no_grad():
                    manager_encoded = self.agent.manager_encoder(current_encoded_state.unsqueeze(0))
                    final_value = self.agent.manager_value(manager_encoded).item()
                
                manager_buffer.add_manager_transition(
                    current_goal,
                    final_reward,
                    final_value,
                    True,  # Episode done
                    obs  # Raw observation for encoder gradient updates
                )
            
            # Track episode metrics
            final_info = current_env.get_current_objective()
            self.episode_makespans.append(final_info['makespan'])
            self.episode_objectives.append(final_info['objective'])
            self.episode_rewards.append(episode_reward)
        
        return {
            'episodes_completed': self.episodes_per_epoch,
            'makespan_mean': float(np.mean(self.episode_makespans)),
            'objective_mean': float(np.mean(self.episode_objectives)),
            'reward_mean': float(np.mean(self.episode_rewards))
        }

    def train(self, env_or_envs=None, test_environments=None, test_interval=50, seed: Optional[int] = None):
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
        
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        wandb.init(
            name=self.run_name,
            project=self.project_name,
            dir=wandb_dir,
            config={
                "epochs": self.epochs,
                "episodes_per_epoch": self.episodes_per_epoch,
                "goal_duration_ratio": self.goal_duration_ratio,
                "goal_duration": self.goal_duration,  # Calculated goal duration for reference
                "latent_dim": self.agent.latent_dim,
                "goal_dim": self.agent.goal_dim,
                "manager_lr": self.agent.manager_lr,
                "worker_lr": self.agent.worker_lr,
                "intrinsic_reward_scale": self.intrinsic_reward_scale,
                "gamma_manager": self.agent.gamma_manager,
                "gamma_worker": self.agent.gamma_worker,
                "entropy_coef": self.agent.entropy_coef,
                "train_per_episode": self.train_per_episode,
            }
        )

        # Get observation dimensions from first environment (assuming all have same structure)
        if isinstance(env_or_envs, list):
            obs_len = env_or_envs[0].obs_len
            action_dim = env_or_envs[0].action_dim
        else:
            obs_len = env_or_envs.obs_len
            action_dim = env_or_envs.action_dim

        # Buffers - estimate buffer size based on episodes and average steps per episode
        estimated_steps_per_episode = env_or_envs[0].num_jobs * env_or_envs[0].num_machines * 2 if isinstance(env_or_envs, list) else env_or_envs.num_jobs * env_or_envs.num_machines * 2
        buffer_size = self.episodes_per_epoch * estimated_steps_per_episode
        worker_buffer = PPOBuffer(buffer_size, (obs_len,), action_dim, self.agent.device)
        manager_buffer = HierarchicalPPOBuffer(buffer_size, self.agent.latent_dim, (obs_len,), self.agent.device)

        # Training loop
        start_time = time.time()
        pbar = tqdm(range(self.epochs), desc="Hierarchical Training")

        for epoch in pbar:
            # Collect rollout data
            collection_stats = self.collect_rollout(env_or_envs, worker_buffer, manager_buffer, epoch)
                        
            # Update after collecting episodes_per_epoch data
            manager_stats = self.agent.update_manager(manager_buffer)
            worker_stats = self.agent.update_worker(
                worker_buffer, manager_buffer,
                train_pi_iters=self.train_per_episode * self.episodes_per_epoch, 
                train_v_iters=self.train_per_episode * self.episodes_per_epoch
            )

            # Log all metrics together
            wandb_log = {
                "policy_loss": worker_stats.get('policy_loss', 0),
                "value_loss": worker_stats.get('value_loss', 0),
                "entropy": worker_stats.get('entropy', 0),
                "manager_policy_loss": manager_stats.get('manager_policy_loss', 0),
                "manager_value_loss": manager_stats.get('manager_value_loss', 0),
                "total_epochs": epoch + 1,
                "learning_rate": (self.agent.worker_lr + self.agent.manager_lr) / 2,  # Average of worker and manager learning rates
                "performance/makespan_mean": collection_stats["makespan_mean"],
                "performance/objective_mean": collection_stats["objective_mean"],
                "performance/reward_mean": collection_stats["reward_mean"]
            }
            
            # Test generalization periodically
            if test_environments is not None and (epoch + 1) % test_interval == 0:
                generalization_results = self.evaluate_generalization(test_environments)
                
                # Add generalization metrics to wandb log
                wandb_log.update({
                    "generalization/mean_makespan": generalization_results['aggregate_stats']['avg_makespan'],
                    "generalization/mean_objective": generalization_results['aggregate_stats']['avg_objective'],
                })
            
            # Log all metrics
            wandb.log(wandb_log)

            worker_buffer.clear()
            manager_buffer.clear()
            self.episode_makespans.clear()
            self.episode_objectives.clear()
            self.episode_rewards.clear()
        
        pbar.close()
        
        # Save model with timestamp
        model_filename = config.create_model_filename()
        self.save_model(model_filename)
        wandb.finish()

        return {
            'training_time': time.time() - start_time,
            'model_filename': model_filename,
            'training_history': {
                'episode_makespans': self.episode_makespans,
                'episode_objectives': self.episode_objectives,
                'episode_rewards': self.episode_rewards
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
        all_objectives = []
        all_rewards = []
        
        for i, data_handler in enumerate(data_handlers):
            
            # Create environment from data handler using same alpha and reward shaping as training env
            test_env = RLEnv(
                data_handler,
                alpha=getattr(self.env, 'alpha', 0.5),
                use_reward_shaping=getattr(self.env, 'use_reward_shaping', False)
            )
            
            # Calculate goal_duration dynamically for this test environment
            total_operations = test_env.num_jobs * max(len(job.operations) for job in test_env.jobs.values())
            current_goal_duration = max(1, total_operations // self.goal_duration_ratio)
            
            # Reset environment
            obs, _ = test_env.reset()
            obs = torch.tensor(obs, dtype=torch.float32, device=self.agent.device)
            
            # Initialize hierarchical state tracking
            last_goal = None
            current_goal = None
            episode_reward = 0
            step = 0
            max_steps = test_env.num_jobs * max(len(job.operations) for job in test_env.jobs.values()) * 2
            
            # Run deterministic policy
            while not test_env.state.is_done() and step < max_steps:
                z_t = self.agent.encode_state(obs)
                
                # Manager decision every goal_duration steps
                if step % current_goal_duration == 0:
                    current_goal = self.agent.get_manager_goal(z_t)
                
                # Pool goals for worker
                pooled_goal = self.agent.pool_goals(last_goal, current_goal, step, current_goal_duration)
                
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
                if step % current_goal_duration == 0:
                    last_goal = current_goal
                
                if done:
                    break
            
            # Get final results
            final_info = test_env.get_current_objective()
            
            env_result = {
                'makespan': final_info['makespan'],
                'objective': final_info['objective'],
                'episode_reward': episode_reward,
                'steps_taken': step,
                'is_valid_completion': test_env.state.is_done(),
                'num_jobs': test_env.num_jobs,
                'num_machines': test_env.num_machines
            }
            
            generalization_results[f'env_{i+1}'] = env_result
            all_makespans.append(final_info['makespan'])
            all_objectives.append(final_info['objective'])
            all_rewards.append(episode_reward)
        
        # Calculate aggregate statistics
        aggregate_stats = {
            'avg_makespan': np.mean(all_makespans),
            'std_makespan': np.std(all_makespans),
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

class HybridHierarchicalRLTrainer:
    """Trainer for Hybrid Hierarchical RL (discrete scheduling + continuous idleness)."""

    def __init__(self,
                 env: RLEnvContinuousIdleness,
                 epochs: int,
                 episodes_per_epoch: int,
                 goal_duration_ratio: int,
                 latent_dim: int,
                 goal_dim: int,
                 manager_lr: float,
                 worker_lr: float,
                 gamma_manager: float,
                 gamma_worker: float,
                 clip_ratio: float,
                 entropy_coef: float,
                 gae_lambda: float,
                 train_per_episode: int,
                 intrinsic_reward_scale: float,
                 project_name: Optional[str] = None,
                 run_name: Optional[str] = None,
                 device: str = 'auto',
                 model_save_dir: str = 'result/hybrid_hierarchical_rl/model',
                 target_kl: float = 0.01,
                 max_grad_norm: float = 0.5,
                 seed: Optional[int] = None):
        self.env = env
        self.epochs = epochs
        self.episodes_per_epoch = episodes_per_epoch
        self.goal_duration_ratio = goal_duration_ratio
        self.intrinsic_reward_scale = intrinsic_reward_scale
        self.train_per_episode = train_per_episode
        self.model_save_dir = model_save_dir
        self.project_name = project_name
        self.run_name = run_name

        total_operations = self.env.num_jobs * max(len(job.operations) for job in self.env.jobs.values())
        self.goal_duration = max(1, total_operations // goal_duration_ratio)

        self.episode_makespans: List[float] = []
        self.episode_objectives: List[float] = []
        self.episode_rewards: List[float] = []

        self.agent = HybridHierarchicalAgent(
            input_dim=env.obs_len,
            discrete_action_dim=env.discrete_action_dim,
            continuous_action_dim=env.continuous_action_dim,
            latent_dim=latent_dim,
            goal_dim=goal_dim,
            goal_duration=self.goal_duration,
            manager_lr=manager_lr,
            worker_lr=worker_lr,
            gamma_manager=gamma_manager,
            gamma_worker=gamma_worker,
            gae_lambda=gae_lambda,
            clip_ratio=clip_ratio,
            entropy_coef=entropy_coef,
            device=device,
            max_grad_norm=max_grad_norm,
            target_kl=target_kl,
            seed=seed,
        )

        os.makedirs(model_save_dir, exist_ok=True)
        print("Hybrid Hierarchical RL Trainer initialized")
        print(f"Total operations: {total_operations}, Goal duration ratio: {goal_duration_ratio}")
        print(f"Calculated goal duration: {self.goal_duration}, Latent dim: {latent_dim}, Goal dim: {goal_dim}")

    def collect_rollout(self, env_or_envs, worker_buffer: HybridPPOBuffer, manager_buffer: HierarchicalPPOBuffer, epoch: int) -> Dict:
        if isinstance(env_or_envs, list):
            envs = env_or_envs
            epochs_per_env = self.epochs // len(envs)
            env_idx = epoch // epochs_per_env if epoch < (len(envs) - 1) * epochs_per_env else len(envs) - 1
            if env_idx >= len(envs):
                env_idx = len(envs) - 1
            current_env = envs[env_idx]
        else:
            current_env = env_or_envs

        total_operations = current_env.num_jobs * max(len(job.operations) for job in current_env.jobs.values())
        current_goal_duration = max(1, total_operations // self.goal_duration_ratio)

        for episode in range(self.episodes_per_epoch):
            obs, _ = current_env.reset()
            obs = torch.tensor(obs, dtype=torch.float32, device=self.agent.device)

            episode_steps = 0
            last_goal = None
            current_goal = None
            pooled_goal_history = deque(maxlen=current_goal_duration)
            encoded_state_history = deque(maxlen=current_goal_duration)
            last_encoded_state = None
            episode_reward = 0.0

            while not current_env.state.is_done():
                z_t = self.agent.encode_state(obs)
                encoded_state_history.append(z_t)

                if episode_steps % current_goal_duration == 0:
                    current_goal = self.agent.get_manager_goal(z_t)
                    manager_reward = self.agent.compute_manager_reward(s_t=last_encoded_state, s_t_plus_c=z_t, g_t=last_goal)
                    with torch.no_grad():
                        manager_encoded = self.agent.manager_encoder(z_t.unsqueeze(0))
                        manager_value = self.agent.manager_value(manager_encoded).item()
                    manager_buffer.add_manager_transition(current_goal, manager_reward, manager_value, False, obs)
                    last_goal = current_goal

                pooled_goal = self.agent.pool_goals(last_goal, current_goal, episode_steps, current_goal_duration)
                pooled_goal_history.append(pooled_goal)
                manager_buffer.add_step_data(pooled_goal)

                action_mask = current_env.get_action_mask()
                if not action_mask.any():
                    break
                d_action, c_action, d_logp, c_logp, worker_value = self.agent.take_action(obs, action_mask, pooled_goal)
                next_obs, reward_ext, terminated, truncated, info = current_env.step((d_action, c_action))
                done = terminated or truncated
                next_obs = torch.tensor(next_obs, dtype=torch.float32, device=self.agent.device)

                reward_int = self.agent.compute_intrinsic_reward(encoded_state_history, pooled_goal_history)
                reward_mixed = reward_ext + self.intrinsic_reward_scale * reward_int

                worker_buffer.add(obs, d_action, c_action, reward_mixed, worker_value, d_logp, c_logp, action_mask, done)
                obs = next_obs
                last_encoded_state = z_t
                episode_steps += 1
                episode_reward += reward_mixed
                if done:
                    break

            if episode_steps > 0:
                final_reward = self.agent.compute_manager_reward(last_encoded_state, z_t, last_goal)
                with torch.no_grad():
                    manager_encoded = self.agent.manager_encoder(z_t.unsqueeze(0))
                    final_value = self.agent.manager_value(manager_encoded).item()
                manager_buffer.add_manager_transition(current_goal, final_reward, final_value, True, obs)

            final_info = current_env.get_current_objective()
            self.episode_makespans.append(final_info['makespan'])
            self.episode_objectives.append(final_info['objective'])
            self.episode_rewards.append(episode_reward)

            if wandb.run is not None:
                wandb.log({
                    "episode_performance/episode_objective": final_info['objective'],
                    "episode_performance/episode_makespan": final_info['makespan'],
                    "episode_performance/episode_reward": episode_reward,
                })
        
        return {
            'episodes_completed': self.episodes_per_epoch,
            'makespan_mean': float(np.mean(self.episode_makespans)),
            'objective_mean': float(np.mean(self.episode_objectives)),
            'reward_mean': float(np.mean(self.episode_rewards))
        }

    def train(self, env_or_envs=None, test_environments=None, test_interval=50, seed: Optional[int] = None):
        if env_or_envs is None:
            env_or_envs = self.env

        wandb_dir = os.path.join(os.path.dirname(self.model_save_dir), 'training_process')
        os.makedirs(wandb_dir, exist_ok=True)
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        wandb.init(
            name=self.run_name,
            project=self.project_name,
            dir=wandb_dir,
            config={
                "epochs": self.epochs,
                "episodes_per_epoch": self.episodes_per_epoch,
                "goal_duration_ratio": self.goal_duration_ratio,
                "goal_duration": self.goal_duration,
                "latent_dim": self.agent.latent_dim,
                "goal_dim": self.agent.goal_dim,
                "manager_lr": self.agent.manager_lr,
                "worker_lr": self.agent.worker_lr,
                "intrinsic_reward_scale": self.intrinsic_reward_scale,
                "gamma_manager": self.agent.gamma_manager,
                "gamma_worker": self.agent.gamma_worker,
                "entropy_coef": self.agent.entropy_coef,
                "train_per_episode": self.train_per_episode,
                "target_kl": self.agent.target_kl,
            }
        )

        if isinstance(env_or_envs, list):
            obs_len = env_or_envs[0].obs_len
            discrete_action_dim = env_or_envs[0].discrete_action_dim
            continuous_action_dim = env_or_envs[0].continuous_action_dim
        else:
            obs_len = env_or_envs.obs_len
            discrete_action_dim = env_or_envs.discrete_action_dim
            continuous_action_dim = env_or_envs.continuous_action_dim

        estimated_steps_per_episode = env_or_envs[0].num_jobs * env_or_envs[0].num_machines * 2 if isinstance(env_or_envs, list) else env_or_envs.num_jobs * env_or_envs.num_machines * 2
        buffer_size = self.episodes_per_epoch * estimated_steps_per_episode
        worker_buffer = HybridPPOBuffer(buffer_size, (obs_len,), discrete_action_dim, continuous_action_dim, self.agent.device)
        manager_buffer = HierarchicalPPOBuffer(buffer_size, self.agent.latent_dim, (obs_len,), self.agent.device)

        start_time = time.time()
        pbar = tqdm(range(self.epochs), desc="Hybrid Hierarchical Training")
        for epoch in pbar:
            collection_stats = self.collect_rollout(env_or_envs, worker_buffer, manager_buffer, epoch)
            manager_stats = self.agent.update_manager(manager_buffer)
            worker_stats = self.agent.update_worker(worker_buffer, manager_buffer,
                                                    train_pi_iters=self.train_per_episode * self.episodes_per_epoch,
                                                    train_v_iters=self.train_per_episode * self.episodes_per_epoch)
            
            # Log all metrics together
            wandb_log = {
                "policy_loss": worker_stats.get('policy_loss', 0),
                "value_loss": worker_stats.get('value_loss', 0),
                "entropy": worker_stats.get('entropy', 0),
                "manager_policy_loss": manager_stats.get('manager_policy_loss', 0),
                "manager_value_loss": manager_stats.get('manager_value_loss', 0),
                "total_epochs": epoch + 1,
                "learning_rate": (self.agent.worker_lr + self.agent.manager_lr) / 2,  # Average of worker and manager learning rates
                "performance/makespan_mean": collection_stats["makespan_mean"],
                "performance/objective_mean": collection_stats["objective_mean"],
                "performance/reward_mean": collection_stats["reward_mean"]
            }

            wandb.log(wandb_log)
            worker_buffer.clear()
            manager_buffer.clear()
            self.episode_makespans.clear()
            self.episode_objectives.clear()
            self.episode_rewards.clear()

        pbar.close()
        model_filename = config.create_model_filename()
        self.save_model(model_filename)
        wandb.finish()
        return {
            'training_time': time.time() - start_time,
            'model_filename': model_filename,
            'training_history': {
                'episode_makespans': self.episode_makespans,
                'episode_objectives': self.episode_objectives,
                'episode_rewards': self.episode_rewards,
            }
        }

    def save_model(self, filename: str):
        filepath = os.path.join(self.model_save_dir, filename)
        self.agent.save(filepath)
        print(f"Hybrid hierarchical models saved to {filepath}")

    def load_model(self, filename: str):
        filepath = os.path.join(self.model_save_dir, filename)
        if os.path.exists(filepath):
            self.agent.load(filepath)
            print(f"Hybrid hierarchical models loaded from {filepath}")
        else:
            print(f"Model file {filepath} not found")
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
from typing import Dict, Optional


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
                 gae_lambda: float = 0.95,
                 train_pi_iters: int = 10,
                 train_v_iters: int = 10,
                 alpha: float = 0.1,
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
    
    def train(self):
        """Main training loop"""
        wandb.init(
            name=f"{time.strftime('%Y%m%d_%H%M')}",
            project="Hierarchical-RL",
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

        # Buffers
        worker_buffer = PPOBuffer(self.steps_per_epoch, (self.env.obs_len,), self.env.action_dim, self.agent.device)
        manager_buffer = HierarchicalPPOBuffer(self.steps_per_epoch, self.agent.latent_dim, self.agent.device)

        # Training loop
        start_time = time.time()
        pbar = tqdm(range(self.epochs), desc="Hierarchical Training")

        for epoch in pbar:
            # Reset environment and episode tracking
            obs, _ = self.env.reset()
            obs = torch.tensor(obs, dtype=torch.float32, device=self.agent.device)
            ep_reward = 0
            ep_objective = 0
            
            # Reset episode-specific data at the start of each epoch
            episode_steps = 0
            goals_history = []
            encoded_states_history = []

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
                action_mask = self.env.get_action_mask()
                action, log_prob, worker_value = self.agent.take_action(obs, action_mask, pooled_goal)
                
                # Environment step
                next_obs, reward_ext, terminated, truncated, info = self.env.step(action)
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
                ep_objective += info.get('objective', 0)
                
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
                    
                    # Log episode stats
                    wandb.log({
                        "episode_objective": info['objective'],
                        "episode_makespan": info['makespan'],
                        "episode_TWT": info['twt']
                    })
                    self.training_history['episode_rewards'].append(ep_reward)
                    self.training_history['episode_makespans'].append(info['makespan'])
                    self.training_history['episode_twts'].append(info['twt'])
                    self.training_history['episode_objectives'].append(info['objective'])

                    # Reset for new episode
                    obs, _ = self.env.reset()
                    obs = torch.tensor(obs, dtype=torch.float32, device=self.agent.device)
                    ep_reward = 0
                    ep_objective = 0
                    episode_steps = 0
                    goals_history = []
                    encoded_states_history = []
            
            # --- End of inner loop (data collection) ---
            
            # Update after collecting steps_per_epoch data
            pooled_goals = manager_buffer.get_hierarchical_data()
            manager_stats = self.agent.update_manager(manager_buffer)
            worker_stats = self.agent.update_worker(
                worker_buffer, pooled_goals,
                train_pi_iters=self.train_pi_iters, train_v_iters=self.train_v_iters
            )

            # Log stats at the end of the epoch
            wandb.log({
                "epoch": epoch + 1,
                "manager_policy_loss": manager_stats.get('policy_loss', 0),
                "manager_value_loss": manager_stats.get('value_loss', 0),
                "worker_policy_loss": worker_stats.get('policy_loss', 0),
                "worker_value_loss": worker_stats.get('value_loss', 0),
                "worker_entropy": worker_stats.get('entropy', 0),
            })

            worker_buffer.clear()
            manager_buffer.clear()
        
        pbar.close()
        self.save_model("final_model.pth")
        wandb.finish()

        return {
            'training_time': time.time() - start_time,
            'training_history': self.training_history
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
    
    def evaluate(self, num_episodes: int = 1) -> Dict:
        """
        Evaluate the trained agent by running deterministic policy and collecting schedule information.
        """
        all_episode_results = []

        for _ in range(num_episodes):
            obs, _ = self.env.reset()
            obs = torch.tensor(obs, dtype=torch.float32, device=self.agent.device)
            
            machine_schedule = {m: [] for m in range(self.env.num_machines)}
            operation_schedules = []
            
            goals_history = []
            episode_reward = 0
            step = 0
            max_steps = self.env.num_jobs * max(len(j.operations) for j in self.env.jobs.values()) * 2

            while not self.env.state.is_done() and step < max_steps:
                z_t = self.agent.encode_state(obs)
                
                if step % self.goal_duration == 0:
                    goal = self.agent.get_manager_goal(z_t, deterministic=True)
                    if goal is not None:
                        goals_history.append(goal)
                
                pooled_goal = self.agent.pool_goals(goals_history, step, self.agent.goal_duration)
                
                action_mask = self.env.get_action_mask()
                if not action_mask.any():
                    break
                
                action = self.agent.get_deterministic_action(obs, action_mask, pooled_goal)
                
                # Decode action for logging
                job_id, machine_id = self.env.decode_action(action)
                op_idx = self.env.state.readable_state['job_states'][job_id]['current_op']
                operation_id = sum(len(self.env.jobs[j].operations) for j in range(job_id)) + op_idx

                # Step environment
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                
                # Log schedule info
                updated_job_state = self.env.state.readable_state['job_states'][job_id]
                start_time = updated_job_state['operations']['operation_start_time'][op_idx][machine_id]
                finish_time = updated_job_state['operations']['finish_time'][op_idx]
                
                machine_schedule[machine_id].append((operation_id, start_time, finish_time))
                operation_schedules.append({
                    'operation_id': operation_id, 'job_id': job_id, 'op_idx': op_idx, 'machine_id': machine_id,
                    'start_time': start_time, 'finish_time': finish_time,
                    'processing_time': self.env.jobs[job_id].operations[op_idx].get_processing_time(machine_id)
                })

                episode_reward += reward
                obs = torch.tensor(next_obs, dtype=torch.float32, device=self.agent.device)
                step += 1
                
                if done:
                    break
            
            # Sort machine schedules by start time
            for m in machine_schedule:
                machine_schedule[m].sort(key=lambda x: x[1])

            final_info = self.env.get_current_objective()
            all_episode_results.append({
                'machine_schedule': machine_schedule, 'operation_schedules': operation_schedules,
                'episode_reward': episode_reward, 'makespan': final_info['makespan'], 'twt': final_info['twt'],
                'objective': final_info['objective'], 'steps_taken': step, 'is_valid_completion': self.env.state.is_done()
            })

        # Aggregate results
        rewards = [res['episode_reward'] for res in all_episode_results]
        makespans = [res['makespan'] for res in all_episode_results]
        twts = [res['twt'] for res in all_episode_results]
        objectives = [res['objective'] for res in all_episode_results]

        avg_results = {
            'avg_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'avg_makespan': np.mean(makespans),
            'std_makespan': np.std(makespans),
            'avg_twt': np.mean(twts),
            'std_twt': np.std(twts),
            'avg_objective': np.mean(objectives),
            'std_objective': np.std(objectives),
        }
        # Return the first episode's detailed schedule for visualization
        final_result = {**avg_results, **all_episode_results[0]}
        return final_result

    def visualize_schedule(self, evaluation_result: Optional[Dict] = None, save_path: Optional[str] = None):
        """Visualize the schedule from evaluation results."""
        if evaluation_result is None:
            evaluation_result = self.evaluate()
            
        try:
            from utils.solution_utils import SolutionUtils
            
            machine_schedule = evaluation_result['machine_schedule']
            data_handler = self.env.data_handler
            
            solution_utils = SolutionUtils(data_handler, machine_schedule)
            validation_result = solution_utils.validate_solution()
            
            print(f"Solution validation: {'VALID' if validation_result['is_valid'] else 'INVALID'}")
            if not validation_result['is_valid']:
                for violation in validation_result['violations']:
                    print(f"  - {violation}")

            fig = solution_utils.draw_gantt(show_validation=True, show_due_dates=True)
            if save_path and fig:
                fig.write_image(save_path)
                print(f"Gantt chart saved to {save_path}")
            
            return fig
            
        except ImportError as e:
            print(f"Could not import SolutionUtils: {e}. Visualization skipped.")
            return None
        except Exception as e:
            print(f"An error occurred during visualization: {e}")
            return None 
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
from RL.flat_rl_env import FlatRLEnv

class FlatRLTrainer:
    """
    Trainer class for Flat RL agent on Flexible Job Shop Scheduling.
    Handles training, evaluation, and model management.
    """
    
    def __init__(self, 
                 env: FlatRLEnv,
                 epochs: int, 
                 steps_per_epoch: int = 4000, # default 4000
                 train_pi_iters: int = 80, # default 80
                 train_v_iters: int = 80, # default 80
                 target_kl: float = 0.1,  # default 0.01
                 hidden_dim: int = 128,
                 pi_lr: float = 3e-5, # default 3e-4
                 v_lr: float = 1e-4, # default 1e-3
                 gamma: float = 0.99,
                 gae_lambda: float = 0.97,
                 clip_ratio: float = 0.2,
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
        # obs_len is set in env after env.reset()
        obs_dim = env.obs_len
        obs_shape = (obs_dim,)
        action_space = cast(spaces.Discrete, env.action_space)
        action_dim = int(action_space.n)
        self.agent = FlatAgent(
            input_dim=obs_dim,
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
    
    def train(self) -> Dict:
        """
        Train the agent using epoch-based PPO (Spinning Up style).
        Returns:
            Dictionary containing training results
        """
        buffer = PPOBuffer(self.steps_per_epoch, (self.env.obs_len,), self.env.action_dim, self.agent.device)
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

        # Training loop
        start_time = time.time()
        pbar = tqdm(range(self.epochs), desc="Training Progress")
        for epoch in pbar:

            # reset env and initialize episode reward
            obs, _ = self.env.reset()
            obs = torch.tensor(obs, dtype=torch.float32, device=self.agent.device)
            ep_reward = 0

            # collect data
            for t in range(self.steps_per_epoch):

                # take valid action
                action_mask = self.env.get_action_mask()
                action, log_prob, value = self.agent.take_action(obs, action_mask)

                # step, get next obs, reward, done
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                next_obs = torch.tensor(next_obs, dtype=torch.float32, device=self.agent.device)

                # one action is selected, add to buffer
                buffer.add(obs, action, reward, value, log_prob, action_mask, done)
                ep_reward += reward
                obs = next_obs
                
                if done:
                    # Episode finished - log episode info
                    wandb.log({
                        "episode_reward": ep_reward,
                        "episode_makespan": info['makespan'],
                        "episode_twt": info['twt']
                    })
                    self.training_history['episode_rewards'].append(ep_reward)
                    self.training_history['episode_makespans'].append(info['makespan'])
                    self.training_history['episode_twts'].append(info['twt'])

                    # start new episode
                    obs, _ = self.env.reset()
                    obs = torch.tensor(obs, dtype=torch.float32, device=self.agent.device)
                    ep_reward = 0
            
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
                "total_loss": stats["total_loss"],
                "kl": stats["kl"]
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
    
    def evaluate(self) -> Dict:
        """
        Evaluate the trained agent by running deterministic policy and collecting schedule information.
        Returns data needed for SolutionUtils to draw Gantt charts.
        """
        # Reset environment
        obs, _ = self.env.reset()
        obs = torch.tensor(obs, dtype=torch.float32, device=self.agent.device)
        
        # Track schedule information
        machine_schedule = {machine_id: [] for machine_id in range(self.env.num_machines)}
        operation_schedules = []
        
        episode_reward = 0
        step_count = 0
        max_steps = self.env.num_jobs * max(len(job.operations) for job in self.env.jobs.values()) * 2  # Safety limit
        
        while not self.env.state.is_done() and step_count < max_steps:
            # Get valid actions
            action_mask = self.env.get_action_mask()
            
            if not action_mask.any():
                print("Warning: No valid actions available, but episode not done")
                break
            
            # Take deterministic action
            action = self.agent.get_deterministic_action(obs, action_mask)
            
            # Decode action to get job and machine
            job_id, machine_id = self.env.decode_action(action)
            
            # Get operation info before step
            job_states = self.env.state.readable_state['job_states']
            op_idx = job_states[job_id]['current_op']
            
            # Check if this is a valid operation
            if op_idx >= len(self.env.jobs[job_id].operations):
                print(f"Warning: Invalid operation index {op_idx} for job {job_id}")
                break
            
            # Step environment
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
            # Calculate operation_id (assuming sequential numbering)
            operation_id = sum(len(self.env.jobs[j].operations) for j in range(job_id)) + op_idx
            
            # Get the start time that was used (from the updated state)
            updated_job_state = self.env.state.readable_state['job_states'][job_id]
            start_time = updated_job_state['operations']['operation_start_time'][op_idx][machine_id]
            
            # Add to machine schedule
            machine_schedule[machine_id].append((operation_id, start_time))
            
            # Add to operation schedules for detailed tracking
            operation_schedules.append({
                'operation_id': operation_id,
                'job_id': job_id,
                'operation_index': op_idx,
                'machine_id': machine_id,
                'start_time': start_time,
                'finish_time': updated_job_state['operations']['finish_time'][op_idx],
                'processing_time': self.env.jobs[job_id].operations[op_idx].get_processing_time(machine_id)
            })
            
            episode_reward += reward
            obs = torch.tensor(next_obs, dtype=torch.float32, device=self.agent.device)
            step_count += 1
            
            if done:
                break
        
        # Get final objective values
        final_info = self.env.get_current_objective()
        
        # Sort machine schedules by start time for better visualization
        for machine_id in machine_schedule:
            machine_schedule[machine_id].sort(key=lambda x: x[1])  # Sort by start_time
        
        evaluation_result = {
            'machine_schedule': machine_schedule,
            'operation_schedules': operation_schedules,
            'episode_reward': episode_reward,
            'makespan': final_info['makespan'],
            'twt': final_info['twt'],
            'objective': final_info['objective'],
            'steps_taken': step_count,
            'is_valid_completion': self.env.state.is_done()
        }
        
        print(f"Evaluation completed:")
        print(f"  Makespan: {final_info['makespan']:.2f}")
        print(f"  Total Weighted Tardiness: {final_info['twt']:.2f}")
        print(f"  Episode Reward: {episode_reward:.2f}")
        print(f"  Steps Taken: {step_count}")
        print(f"  Valid Completion: {self.env.state.is_done()}")
        
        return evaluation_result
    
    def visualize_schedule(self, evaluation_result: Optional[Dict] = None, save_path: Optional[str] = None):
        """
        Visualize the schedule using SolutionUtils.
        
        Args:
            evaluation_result: Result from evaluate() method. If None, will run evaluate() first.
            save_path: Optional path to save the Gantt chart image
        """
        if evaluation_result is None:
            print("No evaluation result provided, running evaluation...")
            evaluation_result = self.evaluate()
        
        try:
            from utils.solution_utils import SolutionUtils
            
            machine_schedule = evaluation_result['machine_schedule']
            data_handler = self.env.data_handler
            
            # Create SolutionUtils instance
            solution_utils = SolutionUtils(data_handler, machine_schedule)
            
            # Validate the solution
            validation_result = solution_utils.validate_solution()
            print(f"Solution validation: {'VALID' if validation_result['is_valid'] else 'INVALID'}")
            if not validation_result['is_valid']:
                print("Validation violations:")
                for violation in validation_result['violations']:
                    print(f"  - {violation}")
            
            # Draw Gantt chart
            fig = solution_utils.draw_gantt(show_validation=True, show_due_dates=True)
            
            if save_path and fig is not None:
                try:
                    fig.write_image(save_path)
                    print(f"Gantt chart saved to {save_path}")
                except Exception as e:
                    print(f"Could not save Gantt chart: {e}")
                    print("Note: You may need to install kaleido: pip install kaleido")
            
            return fig
            
        except ImportError as e:
            print(f"Could not import SolutionUtils: {e}")
            print("Make sure the utils module is in your Python path")
            return None
        except Exception as e:
            print(f"Error creating visualization: {e}")
            return None


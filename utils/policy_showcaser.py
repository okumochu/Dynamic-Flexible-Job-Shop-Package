import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from RL.PPO.ppo_worker import PPOWorker
from utils.solution_utils import SolutionUtils

class PolicyShowcaser:
    def __init__(self, model_dir, env):
        self.env = env
        # Find the model file (assume final_model.pth)
        model_path = os.path.join(model_dir, "final_model.pth")
        
        # Try to load configuration from saved model
        try:
            config = PPOWorker.get_saved_config(model_path)
            print(f"Loaded configuration from saved model:")
            for key, value in config.items():
                print(f"  {key}: {value}")
            
            # Create agent with saved configuration
            self.agent = PPOWorker(
                obs_shape=config['obs_shape'],
                action_dim=config['action_dim'],
                hidden_dim=config['hidden_dim'],
                pi_lr=config.get('pi_lr', 3e-4),
                v_lr=config.get('v_lr', 3e-4),
                gamma=config['gamma'],
                gae_lambda=config['gae_lambda'],
                clip_ratio=config['clip_ratio'],
                device=config['device']
            )
        except Exception as e:
            print(f"Could not load configuration from model: {e}")
            print("Falling back to environment-based configuration...")
            # Fallback to environment-based configuration
            obs_shape = env.observation_space.shape
            action_dim = int(env.action_space.n)
            self.agent = PPOWorker(
                obs_shape=obs_shape,
                action_dim=action_dim,
                hidden_dim=128,  # Default hidden dimension
                pi_lr=3e-4,  # Default policy learning rate
                v_lr=3e-4,   # Default value learning rate
                gamma=0.99,  # Default gamma
                gae_lambda=0.95,  # Default GAE lambda
                clip_ratio=0.2,  # Default clip ratio
                device='auto'  # Auto-detect device
            )
        
        # Load the trained model
        self.agent.load(model_path)

    def showcase(self, render_gantt=True, gantt_save_path=None):
        obs, _ = self.env.reset()
        obs = torch.tensor(obs, dtype=torch.float32, device=self.agent.device)
        done = False
        total_reward = 0
        makespan = None
        twt = None
        schedule_history = []
        while not done:
            action_mask = self.env.get_action_mask()
            if not action_mask.any():
                # No valid actions, break out of loop
                break
            action = self.agent.get_deterministic_action(obs, action_mask)
            obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            obs = torch.tensor(obs, dtype=torch.float32, device=self.agent.device)
            total_reward += reward
            schedule_history.append(self.env.get_schedule_info())
            makespan = info.get('makespan', None)
            twt = info.get('twt', None)
        if render_gantt and schedule_history:
            self.plot_gantt(schedule_history[-1], save_path=gantt_save_path)
        return {"makespan": makespan, "twt": twt, "total_reward": total_reward}

    def plot_gantt(self, schedule, save_path=None):
        operation_schedules = schedule.get('operation_schedules', [])
        if not operation_schedules:
            print("No operation schedule data for Gantt chart.")
            return
        # Convert operation_schedules to machine_schedule dict for SolutionUtils
        machine_schedule = {}
        for op in operation_schedules:
            m = op['machine_id']
            if m not in machine_schedule:
                machine_schedule[m] = []
            machine_schedule[m].append((op['operation_id'], op['start_time']))
        # Use SolutionUtils for validation and Gantt plotting
        data_handler = getattr(self.env, 'data_handler', None)
        if data_handler is None:
            print("No data_handler found in environment for SolutionUtils.")
            return
        solution_utils = SolutionUtils(data_handler, machine_schedule)
        # Draw Gantt chart using SolutionUtils (uses Plotly)
        fig = solution_utils.draw_gantt(show_validation=True, show_due_dates=True)
        if save_path and fig is not None:
            # Save Plotly figure as static image (requires kaleido)
            try:
                fig.write_image(save_path)
                print(f"Gantt chart saved to {save_path}")
            except Exception as e:
                print(f"Could not save Gantt chart as image: {e}")

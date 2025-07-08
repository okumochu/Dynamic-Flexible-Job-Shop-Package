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
                input_dim=config['input_dim'],  # Fixed: use input_dim instead of obs_shape
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
            input_dim = env.obs_len  # Fixed: use obs_len from environment
            action_dim = int(env.action_space.n)
            self.agent = PPOWorker(
                input_dim=input_dim,  # Fixed: use input_dim
                action_dim=action_dim,
                hidden_dim=128,  # Default hidden dimension
                pi_lr=3e-4,  # Default policy learning rate
                v_lr=3e-4,   # Default value learning rate
                gamma=0.99,  # Default gamma
                gae_lambda=0.97,  # Default GAE lambda
                clip_ratio=0.2,  # Default clip ratio
                device='auto'  # Auto-detect device
            )
        
        # Load the trained model
        self.agent.load(model_path)

    def showcase(self, render_gantt=True, gantt_save_path=None):
        """
        Run the trained agent and showcase its performance.
        Returns evaluation metrics and optionally renders Gantt chart.
        """
        # Reset environment
        obs, _ = self.env.reset()
        obs = torch.tensor(obs, dtype=torch.float32, device=self.agent.device)
        
        # Track schedule information
        machine_schedule = {machine_id: [] for machine_id in range(self.env.num_machines)}
        operation_schedules = []
        
        total_reward = 0
        step_count = 0
        max_steps = self.env.num_jobs * max(len(job.operations) for job in self.env.jobs.values()) * 2  # Safety limit
        
        while not self.env.state.is_done() and step_count < max_steps:
            action_mask = self.env.get_action_mask()
            if not action_mask.any():
                print("Warning: No valid actions available")
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
            obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            obs = torch.tensor(obs, dtype=torch.float32, device=self.agent.device)
            total_reward += reward
            step_count += 1
            
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
            
            if done:
                break
        
        # Get final metrics
        final_info = self.env.get_current_objective()
        makespan = final_info['makespan']
        twt = final_info['twt']
        
        # Sort machine schedules by start time for better visualization
        for machine_id in machine_schedule:
            machine_schedule[machine_id].sort(key=lambda x: x[1])  # Sort by start_time
        
        # Create schedule info dictionary for compatibility
        schedule_info = {
            'operation_schedules': operation_schedules,
            'machine_schedule': machine_schedule,
            'makespan': makespan,
            'twt': twt
        }
        
        if render_gantt and operation_schedules:
            self.plot_gantt(schedule_info, save_path=gantt_save_path)
        
        result = {
            "makespan": makespan,
            "twt": twt,
            "total_reward": total_reward,
            "steps_taken": step_count,
            "schedule_info": schedule_info,
            "is_valid_completion": self.env.state.is_done()
        }
        
        print(f"Showcase completed:")
        print(f"  Makespan: {makespan:.2f}")
        print(f"  Total Weighted Tardiness: {twt:.2f}")
        print(f"  Total Reward: {total_reward:.2f}")
        print(f"  Steps Taken: {step_count}")
        print(f"  Valid Completion: {self.env.state.is_done()}")
        
        return result

    def plot_gantt(self, schedule_info, save_path=None):
        """
        Plot Gantt chart using SolutionUtils.
        """
        machine_schedule = schedule_info.get('machine_schedule', {})
        
        if not machine_schedule or not any(machine_schedule.values()):
            print("No machine schedule data for Gantt chart.")
            return None
        
        # Get data handler from environment
        data_handler = getattr(self.env, 'data_handler', None)
        if data_handler is None:
            print("No data_handler found in environment for SolutionUtils.")
            return None
        
        try:
            # Use SolutionUtils for validation and Gantt plotting
            solution_utils = SolutionUtils(data_handler, machine_schedule)
            
            # Validate the solution first
            validation_result = solution_utils.validate_solution()
            print(f"Solution validation: {'VALID' if validation_result['is_valid'] else 'INVALID'}")
            if not validation_result['is_valid']:
                print("Validation violations:")
                for violation in validation_result['violations']:
                    print(f"  - {violation}")
            
            # Draw Gantt chart using SolutionUtils (uses Plotly)
            fig = solution_utils.draw_gantt(show_validation=True, show_due_dates=True)
            
            if save_path and fig is not None:
                # Save Plotly figure as static image (requires kaleido)
                try:
                    fig.write_image(save_path)
                    print(f"Gantt chart saved to {save_path}")
                except Exception as e:
                    print(f"Could not save Gantt chart as image: {e}")
                    print("Note: You may need to install kaleido: pip install kaleido")
            
            return fig
            
        except Exception as e:
            print(f"Error creating Gantt chart: {e}")
            return None

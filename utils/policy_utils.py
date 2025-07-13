import os
import torch
import numpy as np
from RL.PPO.flat_agent import FlatAgent
from RL.PPO.hierarichical_agent import HierarchicalAgent


def load_flat_agent(model_dir, env):
    """
    Load a flat agent from the given model directory.
    Returns the loaded agent.
    """
    model_path = os.path.join(model_dir, "final_model.pth")
    
    # Try to load configuration from saved model
    try:
        config = FlatAgent.get_saved_config(model_path)
        print(f"Loaded configuration from saved model:")
        for key, value in config.items():
            print(f"  {key}: {value}")
        
        # Create agent with saved configuration
        agent = FlatAgent(
            input_dim=config['input_dim'],
            action_dim=config['action_dim'],
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
        input_dim = env.obs_len
        action_dim = int(env.action_space.n)
        agent = FlatAgent(
            input_dim=input_dim,
            action_dim=action_dim,
            pi_lr=3e-4,
            v_lr=3e-4,
            gamma=0.99,
            gae_lambda=0.97,
            clip_ratio=0.2,
            device='auto'
        )
    
    # Load the trained model
    agent.load(model_path)
    return agent


def load_hierarchical_agent(model_dir, env):
    """
    Load a hierarchical agent from the given model directory.
    Returns the loaded agent.
    """
    model_path = os.path.join(model_dir, "final_model.pth")
    
    # Try to load configuration from saved model
    try:
        config = HierarchicalAgent.get_saved_config(model_path)
        print(f"Loaded hierarchical configuration from saved model:")
        for key, value in config.items():
            print(f"  {key}: {value}")
        
        # Create hierarchical agent with saved configuration
        agent = HierarchicalAgent(
            input_dim=config['input_dim'],
            action_dim=config['action_dim'],
            latent_dim=config['latent_dim'],
            goal_dim=config['goal_dim'],
            goal_duration=config['dilation'],  # Map dilation to goal_duration
            manager_lr=config['manager_lr'],
            worker_lr=config['worker_lr'],
            gamma_manager=config['gamma_manager'],
            gamma_worker=config['gamma_worker'],
            gae_lambda=config['gae_lambda'],
            clip_ratio=config['clip_ratio'],
            entropy_coef=config['entropy_coef'],
            epsilon_greedy=config['epsilon_greedy'],
            device=config['device']
        )
    except Exception as e:
        print(f"Could not load configuration from model: {e}")
        print("Falling back to environment-based configuration...")
        # Fallback to environment-based configuration
        input_dim = env.obs_len
        action_dim = int(env.action_space.n)
        agent = HierarchicalAgent(
            input_dim=input_dim,
            action_dim=action_dim,
            latent_dim=256,
            goal_dim=32,
            goal_duration=10,  # Use goal_duration instead of dilation
            manager_lr=3e-4,
            worker_lr=3e-4,
            gamma_manager=0.995,
            gamma_worker=0.95,
            gae_lambda=0.95,
            clip_ratio=0.2,
            entropy_coef=0.01,
            epsilon_greedy=0.1,
            device='auto'
        )
    
    # Load the trained model
    agent.load(model_path)
    return agent


def showcase_flat_policy(model_dir, env):
    """
    Run a trained flat agent and showcase its performance.
    Returns evaluation metrics and data needed for Gantt chart creation.
    """
    # Load the agent
    agent = load_flat_agent(model_dir, env)
    
    # Reset environment
    obs, _ = env.reset()
    obs = torch.tensor(obs, dtype=torch.float32, device=agent.device)
    
    # Track schedule information
    machine_schedule = {machine_id: [] for machine_id in range(env.num_machines)}
    operation_schedules = []
    
    total_reward = 0
    step_count = 0
    max_steps = env.num_jobs * max(len(job.operations) for job in env.jobs.values()) * 2  # Safety limit
    
    while not env.state.is_done() and step_count < max_steps:
        action_mask = env.get_action_mask()
        if not action_mask.any():
            print("Warning: No valid actions available")
            break
            
        # Take deterministic action
        action = agent.get_deterministic_action(obs, action_mask)
        
        # Decode action to get job and machine
        job_id, machine_id = env.decode_action(action)
        
        # Get operation info before step
        job_states = env.state.readable_state['job_states']
        op_idx = job_states[job_id]['current_op']
        
        # Check if this is a valid operation
        if op_idx >= len(env.jobs[job_id].operations):
            print(f"Warning: Invalid operation index {op_idx} for job {job_id}")
            break
        
        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        obs = torch.tensor(obs, dtype=torch.float32, device=agent.device)
        total_reward += reward
        step_count += 1
        
        # Calculate operation_id (assuming sequential numbering)
        operation_id = sum(len(env.jobs[j].operations) for j in range(job_id)) + op_idx
        
        # Get the start time that was used (from the updated state)
        updated_job_state = env.state.readable_state['job_states'][job_id]
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
            'processing_time': env.jobs[job_id].operations[op_idx].get_processing_time(machine_id)
        })
        
        if done:
            break
    
    # Get final metrics
    final_info = env.get_current_objective()
    makespan = final_info['makespan']
    twt = final_info['twt']
    
    # Sort machine schedules by start time for better visualization
    for machine_id in machine_schedule:
        machine_schedule[machine_id].sort(key=lambda x: x[1])  # Sort by start_time
    
    result = {
        "makespan": makespan,
        "twt": twt,
        "total_reward": total_reward,
        "steps_taken": step_count,
        "is_valid_completion": env.state.is_done(),
        # Data for SolutionUtils
        "data_handler": env.data_handler,
        "machine_schedule": machine_schedule,
        "operation_schedules": operation_schedules,
    }
    
    print(f"Showcase completed:")
    print(f"  Makespan: {makespan:.2f}")
    print(f"  Total Weighted Tardiness: {twt:.2f}")
    print(f"  Total Reward: {total_reward:.2f}")
    print(f"  Steps Taken: {step_count}")
    print(f"  Valid Completion: {env.state.is_done()}")
    
    return result


def showcase_hierarchical_policy(model_dir, env):
    """
    Run a trained hierarchical agent and showcase its performance.
    Returns evaluation metrics and data needed for Gantt chart creation.
    """
    # Load the agent
    agent = load_hierarchical_agent(model_dir, env)
    
    # Reset environment
    obs, _ = env.reset()
    obs = torch.tensor(obs, dtype=torch.float32, device=agent.device)
    
    # Track schedule information
    machine_schedule = {machine_id: [] for machine_id in range(env.num_machines)}
    operation_schedules = []
    
    # Hierarchical RL specific tracking
    manager_hidden = None
    goals_history = []
    encoded_states_history = []
    
    total_reward = 0
    step_count = 0
    max_steps = env.num_jobs * max(len(job.operations) for job in env.jobs.values()) * 2
    
    while not env.state.is_done() and step_count < max_steps:
        # Encode state
        z_t = agent.encode_state(obs)
        encoded_states_history.append(z_t)
        
        # Manager decision (every goal_duration steps)
        if step_count % agent.goal_duration == 0:
            goal, _ = agent.get_manager_goal(z_t)
            if goal is not None:
                goals_history.append(goal)
        
        # Pool goals for worker
        pooled_goal = agent.pool_goals(goals_history, step_count, agent.goal_duration)
        
        # Worker action (deterministic)
        action_mask = env.get_action_mask()
        if not action_mask.any():
            print("Warning: No valid actions available")
            break
        
        action = agent.get_deterministic_action(
            obs, action_mask, pooled_goal
        )
        
        # Decode action to get job and machine
        job_id, machine_id = env.decode_action(action)
        
        # Get operation info before step
        job_states = env.state.readable_state['job_states']
        op_idx = job_states[job_id]['current_op']
        
        # Check if this is a valid operation
        if op_idx >= len(env.jobs[job_id].operations):
            print(f"Warning: Invalid operation index {op_idx} for job {job_id}")
            break
        
        # Step environment
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        next_obs = torch.tensor(next_obs, dtype=torch.float32, device=agent.device)
        total_reward += reward
        step_count += 1
        
        # Compute intrinsic reward for next step (no longer used)
        # if len(encoded_states_history) > 1:
        #     prev_r_int = agent.compute_intrinsic_reward(
        #         encoded_states_history, goals_history, step_count - 1
        #     )
        
        # Calculate operation_id (assuming sequential numbering)
        operation_id = sum(len(env.jobs[j].operations) for j in range(job_id)) + op_idx
        
        # Get the start time that was used (from the updated state)
        updated_job_state = env.state.readable_state['job_states'][job_id]
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
            'processing_time': env.jobs[job_id].operations[op_idx].get_processing_time(machine_id)
        })
        
        obs = next_obs
        
        if done:
            break
    
    # Get final metrics
    final_info = env.get_current_objective()
    makespan = final_info['makespan']
    twt = final_info['twt']
    
    # Sort machine schedules by start time for better visualization
    for machine_id in machine_schedule:
        machine_schedule[machine_id].sort(key=lambda x: x[1])  # Sort by start_time
    
    result = {
        "makespan": makespan,
        "twt": twt,
        "total_reward": total_reward,
        "steps_taken": step_count,
        "is_valid_completion": env.state.is_done(),
        "goals_used": len(goals_history),
        "manager_decisions": len(goals_history),
        # Data for SolutionUtils
        "data_handler": env.data_handler,
        "machine_schedule": machine_schedule,
        "operation_schedules": operation_schedules,
    }
    
    print(f"Hierarchical showcase completed:")
    print(f"  Makespan: {makespan:.2f}")
    print(f"  Total Weighted Tardiness: {twt:.2f}")
    print(f"  Total Reward: {total_reward:.2f}")
    print(f"  Steps Taken: {step_count}")
    print(f"  Manager Decisions: {len(goals_history)}")
    print(f"  Valid Completion: {env.state.is_done()}")
    
    return result


def create_gantt_chart(result, save_path=None, title_suffix=""):
    """
    Create a Gantt chart using SolutionUtils from the result of a policy showcase.
    
    Args:
        result: Result dictionary from showcase_flat_policy or showcase_hierarchical_policy
        save_path: Optional path to save the chart
        title_suffix: Optional suffix to add to the chart title (e.g., "Hierarchical RL")
    
    Returns:
        Plotly figure object or None if creation failed
    """
    from utils.solution_utils import SolutionUtils
    
    data_handler = result.get('data_handler')
    machine_schedule = result.get('machine_schedule')
    
    if not data_handler or not machine_schedule:
        print("Missing data_handler or machine_schedule in result.")
        return None
    
    if not any(machine_schedule.values()):
        print("No machine schedule data for Gantt chart.")
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
        
        # Draw Gantt chart using SolutionUtils
        fig = solution_utils.draw_gantt(show_validation=True, show_due_dates=True)
        
        # Update title if suffix provided
        if title_suffix and fig is not None:
            makespan = result.get('makespan', 0)
            twt = result.get('twt', 0)
            goals_used = result.get('goals_used', 0)
            
            if 'hierarchical' in title_suffix.lower():
                new_title = f"{title_suffix} Job Shop Schedule<br>Makespan: {makespan:.2f}, TWT: {twt:.2f}, Manager Goals Used: {goals_used}"
            else:
                new_title = f"{title_suffix} Job Shop Schedule<br>Makespan: {makespan:.2f}, TWT: {twt:.2f}"
            
            fig.update_layout(title=new_title)
        
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


# Backward compatibility classes (can be removed later)
class PolicyShowcaser:
    def __init__(self, model_dir, env):
        self.model_dir = model_dir
        self.env = env
    
    def showcase(self, render_gantt=True, gantt_save_path=None):
        result = showcase_flat_policy(self.model_dir, self.env)
        
        if render_gantt:
            create_gantt_chart(result, save_path=gantt_save_path, title_suffix="Flat RL")
        
        return result
    
    def plot_gantt(self, schedule_info, save_path=None):
        # This method is deprecated, use create_gantt_chart instead
        return create_gantt_chart(schedule_info, save_path=save_path)


class HierarchicalPolicyShowcaser:
    def __init__(self, model_dir, env):
        self.model_dir = model_dir
        self.env = env
    
    def showcase(self, render_gantt=True, gantt_save_path=None):
        result = showcase_hierarchical_policy(self.model_dir, self.env)
        
        if render_gantt:
            create_gantt_chart(result, save_path=gantt_save_path, title_suffix="Hierarchical RL")
        
        return result
    
    def plot_gantt(self, schedule_info, save_path=None):
        # This method is deprecated, use create_gantt_chart instead
        return create_gantt_chart(schedule_info, save_path=save_path, title_suffix="Hierarchical RL")

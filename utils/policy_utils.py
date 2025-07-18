import torch
import os
import numpy as np
from RL.rl_env import RLEnv
from RL.PPO.flat_agent import FlatAgent
from RL.PPO.hierarchical_agent import HierarchicalAgent
from typing import Dict, Optional
import plotly.graph_objects as go
from benchmarks.static_benchmark.data_handler import FlexibleJobShopDataHandler

def showcase_flat_policy(model_path: str, env: RLEnv) -> Dict:
    """Load a trained flat PPO agent and run it for one episode."""
    
    config = FlatAgent.get_saved_config(model_path)
    
    agent = FlatAgent(
        input_dim=config['input_dim'],
        action_dim=config['action_dim'],
        pi_lr=config['pi_lr'],
        v_lr=config['v_lr'],
        gamma=config['gamma'],
        gae_lambda=config['gae_lambda'],
        clip_ratio=config['clip_ratio'],
        entropy_coef=config['entropy_coef'],
        device=config['device']
    )
    
    agent.load(model_path)
    
    print(f"\nEvaluating flat policy from {model_path}...")
    
    obs, _ = env.reset()
    obs = torch.tensor(obs, dtype=torch.float32)
    
    # Initialize schedule tracking
    machine_schedule = {m: [] for m in range(env.num_machines)}
    operation_schedules = []
    
    episode_reward = 0
    step_count = 0
    
    while True:
        action_mask = env.get_action_mask()
        if not action_mask.any():
            break
        
        action = agent.get_deterministic_action(obs, action_mask)
        
        # Decode action for logging before environment step
        job_id, machine_id = env.decode_action(action)
        op_idx = env.state.readable_state['job_states'][job_id]['current_op']
        operation_id = sum(len(env.jobs[j].operations) for j in range(job_id)) + op_idx
        
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Log schedule info after environment step
        updated_job_state = env.state.readable_state['job_states'][job_id]
        start_time = updated_job_state['operations']['operation_start_time'][op_idx][machine_id]
        finish_time = updated_job_state['operations']['finish_time'][op_idx]
        
        machine_schedule[machine_id].append((operation_id, start_time))
        operation_schedules.append({
            'operation_id': operation_id, 'job_id': job_id, 'op_idx': op_idx, 'machine_id': machine_id,
            'start_time': start_time, 'finish_time': finish_time,
            'processing_time': env.jobs[job_id].operations[op_idx].get_processing_time(machine_id)
        })
        
        episode_reward += reward
        obs = torch.tensor(next_obs, dtype=torch.float32)
        step_count += 1
        
        if done:
            break
    
    # Sort machine schedules by start time
    for m in machine_schedule:
        machine_schedule[m].sort(key=lambda x: x[1])
            
    final_info = env.get_current_objective()
    
    # Create comprehensive schedule_info including machine_schedule
    schedule_info = {
        'makespan': final_info['makespan'],
        'twt': final_info['twt'],
        'objective': final_info['objective'],
        'machine_schedule': machine_schedule,
        'operation_schedules': operation_schedules
    }
    
    return {
        'makespan': final_info['makespan'],
        'twt': final_info['twt'],
        'total_reward': episode_reward,
        'steps_taken': step_count,
        'is_valid_completion': env.state.is_done(),
        'schedule_info': schedule_info,
        'data_handler': env.data_handler
    }

def showcase_hierarchical_policy(model_path: str, env: RLEnv) -> Dict:
    """Load a trained hierarchical agent and run it for one episode."""
    
    # If model_path is a directory, find the most recent model file
    if os.path.isdir(model_path):
        model_files = [f for f in os.listdir(model_path) if f.endswith('.pth')]
        if not model_files:
            raise FileNotFoundError(f"No .pth model files found in {model_path}")
        # Get the most recent model file
        model_files.sort(reverse=True)  # Sort by name (timestamp-based names)
        model_path = os.path.join(model_path, model_files[0])
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")

    config = HierarchicalAgent.get_saved_config(model_path)
    
    agent = HierarchicalAgent(
        input_dim=config['input_dim'],
        action_dim=config['action_dim'],
        latent_dim=config['latent_dim'],
        goal_dim=config['goal_dim'],
        goal_duration=config.get('goal_duration', 10),
        manager_lr=config['manager_lr'],
        worker_lr=config['worker_lr'],
        gamma_manager=config['gamma_manager'],
        gamma_worker=config['gamma_worker'],
        gae_lambda=config['gae_lambda'],
        clip_ratio=config['clip_ratio'],
        entropy_coef=config['entropy_coef'],
        device=config['device']
    )
    agent.load(model_path)
    
    print(f"\nEvaluating hierarchical policy from {model_path}...")
    
    obs, _ = env.reset()
    obs = torch.tensor(obs, dtype=torch.float32, device=agent.device)
    
    # Initialize schedule tracking
    machine_schedule = {m: [] for m in range(env.num_machines)}
    operation_schedules = []
    
    goals_history = []
    episode_reward = 0
    step = 0
    manager_decisions = 0
    max_steps = env.num_jobs * max(len(job.operations) for job in env.jobs.values()) * 2
    
    while not env.state.is_done() and step < max_steps:
        z_t = agent.encode_state(obs)
        
        if step % agent.goal_duration == 0:
            goal = agent.get_manager_goal(z_t, deterministic=True)
            if goal is not None:
                goals_history.append(goal)
            manager_decisions += 1
        
        pooled_goal = agent.pool_goals(goals_history, step, agent.goal_duration)
        
        action_mask = env.get_action_mask()
        if not action_mask.any():
            break
        
        action = agent.get_deterministic_action(obs, action_mask, pooled_goal)
        
        # Decode action for logging before environment step
        job_id, machine_id = env.decode_action(action)
        op_idx = env.state.readable_state['job_states'][job_id]['current_op']
        operation_id = sum(len(env.jobs[j].operations) for j in range(job_id)) + op_idx
        
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Log schedule info after environment step
        updated_job_state = env.state.readable_state['job_states'][job_id]
        start_time = updated_job_state['operations']['operation_start_time'][op_idx][machine_id]
        finish_time = updated_job_state['operations']['finish_time'][op_idx]
        
        machine_schedule[machine_id].append((operation_id, start_time))
        operation_schedules.append({
            'operation_id': operation_id, 'job_id': job_id, 'op_idx': op_idx, 'machine_id': machine_id,
            'start_time': start_time, 'finish_time': finish_time,
            'processing_time': env.jobs[job_id].operations[op_idx].get_processing_time(machine_id)
        })
        
        episode_reward += reward
        next_obs = torch.tensor(next_obs, dtype=torch.float32, device=agent.device)
        obs = next_obs
        step += 1
        
        if done:
            break

    # Sort machine schedules by start time
    for m in machine_schedule:
        machine_schedule[m].sort(key=lambda x: x[1])

    final_info = env.get_current_objective()
    
    # Create comprehensive schedule_info including machine_schedule
    schedule_info = {
        'makespan': final_info['makespan'],
        'twt': final_info['twt'],
        'objective': final_info['objective'],
        'machine_schedule': machine_schedule,
        'operation_schedules': operation_schedules
    }
    
    return {
        'makespan': final_info['makespan'],
        'twt': final_info['twt'],
        'total_reward': episode_reward,
        'steps_taken': step,
        'manager_decisions': manager_decisions,
        'is_valid_completion': env.state.is_done(),
        'schedule_info': schedule_info,
        "data_handler": env.data_handler
    }

def create_gantt_chart(evaluation_result: Dict, save_path: str, title_suffix: str = ""):
    data_handler = evaluation_result.get('data_handler')
    if not data_handler:
        print("Warning: `data_handler` not found in evaluation result. Gantt chart cannot be created.")
        return

    schedule_info = evaluation_result.get('schedule_info')
    if not schedule_info:
        print("Warning: `schedule_info` not found in evaluation result. Gantt chart cannot be created.")
        return

    machine_schedule = schedule_info.get('machine_schedule', {})
    if not machine_schedule:
        print("Warning: `machine_schedule` not found in `schedule_info`. Gantt chart cannot be created.")
        return

    try:
        from utils.solution_utils import SolutionUtils
        solution_utils = SolutionUtils(data_handler, machine_schedule)
        fig = solution_utils.draw_gantt(show_validation=True, show_due_dates=True)

        if fig:
            if title_suffix:
                fig.update_layout(title=f"Gantt Chart - {title_suffix}")
            fig.write_image(save_path)
            print(f"Gantt chart saved to {save_path}")
        else:
            print("Gantt chart could not be created.")
    except ImportError:
        print("Warning: SolutionUtils could not be imported. Using basic Gantt chart.")
        fig = go.Figure()
        for machine_id, tasks in machine_schedule.items():
            for task in tasks:
                op_id, start_time = task
                # Estimate processing time for basic chart (this is a simplified fallback)
                processing_time = 10  # Default processing time
                finish_time = start_time + processing_time
                fig.add_trace(go.Bar(
                    x=[processing_time],
                    y=[f"Machine {machine_id}"],
                    base=[start_time],
                    name=f"Op {op_id}",
                    orientation='h'
                ))
        if title_suffix:
            fig.update_layout(title=f"Gantt Chart - {title_suffix}")
        fig.show()
        if save_path:
            fig.write_image(save_path)
            print(f"Gantt chart saved to {save_path}")
    except Exception as e:
        print(f"Error creating Gantt chart: {e}")


def evaluate_flat_policy(model_path: str, env: RLEnv, num_episodes: int = 1) -> Dict:
    """
    Evaluate a trained flat PPO agent by running deterministic policy and collecting schedule information.
    
    Args:
        model_path: Path to the saved model
        env: Environment to evaluate on
        num_episodes: Number of episodes to run (default: 1)
        
    Returns:
        Dictionary containing evaluation results
    """
    # Load agent configuration and model
    config = FlatAgent.get_saved_config(model_path)
    
    agent = FlatAgent(
        input_dim=config['input_dim'],
        action_dim=config['action_dim'],
        pi_lr=config['pi_lr'],
        v_lr=config['v_lr'],
        gamma=config['gamma'],
        gae_lambda=config['gae_lambda'],
        clip_ratio=config['clip_ratio'],
        entropy_coef=config['entropy_coef'],
        device=config['device']
    )
    
    agent.load(model_path)
    
    print(f"\nEvaluating flat policy from {model_path} for {num_episodes} episodes...")
    
    all_episode_results = []

    for episode in range(num_episodes):
        # Reset environment
        obs, _ = env.reset()
        obs = torch.tensor(obs, dtype=torch.float32, device=agent.device)
        
        # Track schedule information
        machine_schedule = {machine_id: [] for machine_id in range(env.num_machines)}
        operation_schedules = []
        
        episode_reward = 0
        step_count = 0
        max_steps = env.num_jobs * max(len(job.operations) for job in env.jobs.values()) * 2  # Safety limit
        
        while not env.state.is_done() and step_count < max_steps:
            # Get valid actions
            action_mask = env.get_action_mask()
            
            if not action_mask.any():
                print("Warning: No valid actions available, but episode not done")
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
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
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
            
            episode_reward += reward
            obs = torch.tensor(next_obs, dtype=torch.float32, device=agent.device)
            step_count += 1
            
            if done:
                break
        
        # Get final objective values
        final_info = env.get_current_objective()
        
        # Sort machine schedules by start time for better visualization
        for machine_id in machine_schedule:
            machine_schedule[machine_id].sort(key=lambda x: x[1])  # Sort by start_time
        
        episode_result = {
            'machine_schedule': machine_schedule,
            'operation_schedules': operation_schedules,
            'episode_reward': episode_reward,
            'makespan': final_info['makespan'],
            'twt': final_info['twt'],
            'objective': final_info['objective'],
            'steps_taken': step_count,
            'is_valid_completion': env.state.is_done()
        }
        
        all_episode_results.append(episode_result)
    
    if num_episodes > 1:
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
        # Return the first episode's detailed schedule for visualization plus aggregates
        final_result = {**avg_results, **all_episode_results[0]}
        return final_result
    else:
        # Single episode - just return the result
        return all_episode_results[0]


def evaluate_hierarchical_policy(model_path: str, env: RLEnv, num_episodes: int = 1) -> Dict:
    """
    Evaluate a trained hierarchical agent by running deterministic policy and collecting schedule information.
    
    Args:
        model_path: Path to the saved model or directory containing model
        env: Environment to evaluate on
        num_episodes: Number of episodes to run (default: 1)
        
    Returns:
        Dictionary containing evaluation results
    """
    # If model_path is a directory, find the most recent model file
    if os.path.isdir(model_path):
        model_files = [f for f in os.listdir(model_path) if f.endswith('.pth')]
        if not model_files:
            raise FileNotFoundError(f"No .pth model files found in {model_path}")
        # Get the most recent model file
        model_files.sort(reverse=True)  # Sort by name (timestamp-based names)
        model_path = os.path.join(model_path, model_files[0])
    
    # Load agent configuration and model
    config = HierarchicalAgent.get_saved_config(model_path)
    
    agent = HierarchicalAgent(
        input_dim=config['input_dim'],
        action_dim=config['action_dim'],
        latent_dim=config['latent_dim'],
        goal_dim=config['goal_dim'],
        goal_duration=config.get('goal_duration', 10),
        manager_lr=config['manager_lr'],
        worker_lr=config['worker_lr'],
        gamma_manager=config['gamma_manager'],
        gamma_worker=config['gamma_worker'],
        gae_lambda=config['gae_lambda'],
        clip_ratio=config['clip_ratio'],
        entropy_coef=config['entropy_coef'],
        device=config['device']
    )
    agent.load(model_path)
    
    print(f"\nEvaluating hierarchical policy from {model_path} for {num_episodes} episodes...")
    
    all_episode_results = []

    for episode in range(num_episodes):
        obs, _ = env.reset()
        obs = torch.tensor(obs, dtype=torch.float32, device=agent.device)
        
        # Initialize schedule tracking
        machine_schedule = {m: [] for m in range(env.num_machines)}
        operation_schedules = []
        
        goals_history = []
        episode_reward = 0
        step = 0
        manager_decisions = 0
        max_steps = env.num_jobs * max(len(job.operations) for job in env.jobs.values()) * 2

        while not env.state.is_done() and step < max_steps:
            z_t = agent.encode_state(obs)
            
            if step % agent.goal_duration == 0:
                goal = agent.get_manager_goal(z_t, deterministic=True)
                if goal is not None:
                    goals_history.append(goal)
                manager_decisions += 1
            
            pooled_goal = agent.pool_goals(goals_history, step, agent.goal_duration)
            
            action_mask = env.get_action_mask()
            if not action_mask.any():
                break
            
            action = agent.get_deterministic_action(obs, action_mask, pooled_goal)
            
            # Decode action for logging
            job_id, machine_id = env.decode_action(action)
            op_idx = env.state.readable_state['job_states'][job_id]['current_op']
            operation_id = sum(len(env.jobs[j].operations) for j in range(job_id)) + op_idx

            # Step environment
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Log schedule info
            updated_job_state = env.state.readable_state['job_states'][job_id]
            start_time = updated_job_state['operations']['operation_start_time'][op_idx][machine_id]
            finish_time = updated_job_state['operations']['finish_time'][op_idx]
            
            machine_schedule[machine_id].append((operation_id, start_time, finish_time))
            operation_schedules.append({
                'operation_id': operation_id, 'job_id': job_id, 'op_idx': op_idx, 'machine_id': machine_id,
                'start_time': start_time, 'finish_time': finish_time,
                'processing_time': env.jobs[job_id].operations[op_idx].get_processing_time(machine_id)
            })

            episode_reward += reward
            obs = torch.tensor(next_obs, dtype=torch.float32, device=agent.device)
            step += 1
            
            if done:
                break

        # Sort machine schedules by start time
        for m in machine_schedule:
            machine_schedule[m].sort(key=lambda x: x[1])

        final_info = env.get_current_objective()
        episode_result = {
            'machine_schedule': machine_schedule, 'operation_schedules': operation_schedules,
            'episode_reward': episode_reward, 'makespan': final_info['makespan'], 'twt': final_info['twt'],
            'objective': final_info['objective'], 'steps_taken': step, 'manager_decisions': manager_decisions,
            'is_valid_completion': env.state.is_done()
        }
        
        all_episode_results.append(episode_result)

    if num_episodes > 1:
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
        # Return the first episode's detailed schedule for visualization plus aggregates
        final_result = {**avg_results, **all_episode_results[0]}
        return final_result
    else:
        # Single episode - just return the result
        return all_episode_results[0]


def visualize_policy_schedule(evaluation_result: Dict, env: RLEnv, save_path: Optional[str] = None):
    """
    Visualize the schedule from evaluation results using SolutionUtils.
    
    Args:
        evaluation_result: Result from evaluate_flat_policy or evaluate_hierarchical_policy
        env: Environment instance (for data_handler access)
        save_path: Optional path to save the Gantt chart image
    """
    try:
        from utils.solution_utils import SolutionUtils
        
        machine_schedule = evaluation_result['machine_schedule']
        data_handler = env.data_handler
        
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
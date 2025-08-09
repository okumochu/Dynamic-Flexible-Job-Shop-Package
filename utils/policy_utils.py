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
    
    episode_reward = 0
    step_count = 0
    final_info = None
    final_machine_schedule = None
    
    while True:
        action_mask = env.get_action_mask()
        if not action_mask.any():
            break
        
        action = agent.get_deterministic_action(obs, action_mask)
        
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        episode_reward += reward
        obs = torch.tensor(next_obs, dtype=torch.float32)
        step_count += 1
        
        # Store the final info and machine_schedule
        final_info = info.get('objective_info', {})
        final_machine_schedule = info.get('machine_schedule', {})
        
        if done:
            break
    
    # Create comprehensive schedule_info including machine_schedule
    schedule_info = {
        'makespan': final_info.get('makespan', 0),
        'twt': final_info.get('twt', 0),
        'objective': final_info.get('objective', 0),
        'machine_schedule': final_machine_schedule
    }
    
    return {
        'makespan': final_info.get('makespan', 0),
        'twt': final_info.get('twt', 0),
        'total_reward': episode_reward,
        'steps_taken': step_count,
        'is_valid_completion': env.state.is_done(),
        'schedule_info': schedule_info,
        'data_handler': env.data_handler
    }

def showcase_hierarchical_policy(model_path: str, env: RLEnv) -> Dict:
    """
    Showcase a trained hierarchical agent by running deterministic policy and collecting schedule information.
    
    Args:
        model_path: Path to the saved model or directory containing model
        env: Environment to showcase on
        
    Returns:
        Dictionary containing showcase results
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
    
    # Calculate goal_duration dynamically based on environment
    total_operations = env.num_jobs * max(len(job.operations) for job in env.jobs.values())
    goal_duration_ratio = config.get('goal_duration_ratio', 12)  # Default ratio if not in config
    goal_duration = max(1, total_operations // goal_duration_ratio)
    
    agent = HierarchicalAgent(
        input_dim=config['input_dim'],
        action_dim=config['action_dim'],
        latent_dim=config['latent_dim'],
        goal_dim=config['goal_dim'],
        goal_duration=goal_duration,  # Use calculated goal_duration
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
    
    print(f"\nShowcasing hierarchical policy from {model_path}...")
    print(f"Environment: {env.num_jobs} jobs, {env.num_machines} machines")
    print(f"Total operations: {total_operations}, Goal duration ratio: {goal_duration_ratio}")
    print(f"Calculated goal duration: {goal_duration}")
    
    obs, _ = env.reset()
    obs = torch.tensor(obs, dtype=torch.float32, device=agent.device)
    
    goals_history = []
    episode_reward = 0
    step = 0
    manager_decisions = 0
    max_steps = env.num_jobs * max(len(job.operations) for job in env.jobs.values()) * 2
    final_info = None
    final_machine_schedule = None
    
    while not env.state.is_done() and step < max_steps:
        z_t = agent.encode_state(obs)
        
        if step % goal_duration == 0:  # Use calculated goal_duration
            goal = agent.get_manager_goal(z_t)
            if goal is not None:
                goals_history.append(goal)
            manager_decisions += 1
        
        # Use the last goal and current goal for pooling
        last_goal = goals_history[-2] if len(goals_history) > 1 else None
        current_goal = goals_history[-1] if goals_history else None
        
        if current_goal is not None:
            pooled_goal = agent.pool_goals(last_goal, current_goal, step, goal_duration)  # Use calculated goal_duration
        else:
            # If no goals available, create a zero tensor as fallback
            pooled_goal = torch.zeros(agent.latent_dim, device=agent.device)
        
        action_mask = env.get_action_mask()
        if not action_mask.any():
            break
        
        action = agent.get_deterministic_action(obs, action_mask, pooled_goal)
        
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        episode_reward += reward
        next_obs = torch.tensor(next_obs, dtype=torch.float32, device=agent.device)
        obs = next_obs
        step += 1
        
        # Store the final info and machine_schedule
        final_info = info.get('objective_info', {})
        final_machine_schedule = info.get('machine_schedule', {})
        
        if done:
            break

    # Create comprehensive schedule_info including machine_schedule
    schedule_info = {
        'makespan': final_info.get('makespan', 0),
        'twt': final_info.get('twt', 0),
        'objective': final_info.get('objective', 0),
        'machine_schedule': final_machine_schedule
    }
    
    return {
        'makespan': final_info.get('makespan', 0),
        'twt': final_info.get('twt', 0),
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

    from utils.solution_utils import SolutionUtils
    solution_utils = SolutionUtils(data_handler, machine_schedule)
    fig = solution_utils.draw_gantt(show_validation=True, show_due_dates=True)

    if fig:
        if title_suffix:
            fig.update_layout(title=f"Gantt Chart - {title_suffix}")
        # Save directly to HTML (no try/except)
        alt_path = os.path.splitext(save_path)[0] + ".html" if save_path else None
        if alt_path:
            fig.write_html(alt_path, include_plotlyjs="cdn")
            print(f"Gantt chart saved to {alt_path}")
    else:
        print("Gantt chart could not be created.")


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
        
        episode_reward = 0
        step_count = 0
        max_steps = env.num_jobs * max(len(job.operations) for job in env.jobs.values()) * 2  # Safety limit
        final_info = None
        final_machine_schedule = None
        
        while not env.state.is_done() and step_count < max_steps:
            # Get valid actions
            action_mask = env.get_action_mask()
            
            if not action_mask.any():
                print("Warning: No valid actions available, but episode not done")
                break
            
            # Take deterministic action
            action = agent.get_deterministic_action(obs, action_mask)
            
            # Step environment
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            obs = torch.tensor(next_obs, dtype=torch.float32, device=agent.device)
            step_count += 1
            
            # Store the final info and machine_schedule
            final_info = info.get('objective_info', {})
            final_machine_schedule = info.get('machine_schedule', {})
            
            if done:
                break
        
        episode_result = {
            'machine_schedule': final_machine_schedule,
            'episode_reward': episode_reward,
            'makespan': final_info.get('makespan', 0),
            'twt': final_info.get('twt', 0),
            'objective': final_info.get('objective', 0),
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
    
    # Calculate goal_duration dynamically based on environment
    total_operations = env.num_jobs * max(len(job.operations) for job in env.jobs.values())
    goal_duration_ratio = config.get('goal_duration_ratio', 12)  # Default ratio if not in config
    goal_duration = max(1, total_operations // goal_duration_ratio)
    
    agent = HierarchicalAgent(
        input_dim=config['input_dim'],
        action_dim=config['action_dim'],
        latent_dim=config['latent_dim'],
        goal_dim=config['goal_dim'],
        goal_duration=goal_duration,  # Use calculated goal_duration
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
    print(f"Environment: {env.num_jobs} jobs, {env.num_machines} machines")
    print(f"Total operations: {total_operations}, Goal duration ratio: {goal_duration_ratio}")
    print(f"Calculated goal duration: {goal_duration}")
    
    all_episode_results = []

    for episode in range(num_episodes):
        obs, _ = env.reset()
        obs = torch.tensor(obs, dtype=torch.float32, device=agent.device)
        
        goals_history = []
        episode_reward = 0
        step = 0
        manager_decisions = 0
        max_steps = env.num_jobs * max(len(job.operations) for job in env.jobs.values()) * 2
        final_info = None
        final_machine_schedule = None

        while not env.state.is_done() and step < max_steps:
            z_t = agent.encode_state(obs)
            
            if step % goal_duration == 0:  # Use calculated goal_duration
                goal = agent.get_manager_goal(z_t)
                if goal is not None:
                    goals_history.append(goal)
                manager_decisions += 1
            
            # Use the last goal and current goal for pooling
            last_goal = goals_history[-2] if len(goals_history) > 1 else None
            current_goal = goals_history[-1] if goals_history else None
            
            if current_goal is not None:
                pooled_goal = agent.pool_goals(last_goal, current_goal, step, goal_duration)  # Use calculated goal_duration
            else:
                # If no goals available, create a zero tensor as fallback
                pooled_goal = torch.zeros(agent.latent_dim, device=agent.device)
            
            action_mask = env.get_action_mask()
            if not action_mask.any():
                break
            
            action = agent.get_deterministic_action(obs, action_mask, pooled_goal)
            
            # Step environment
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            episode_reward += reward
            obs = torch.tensor(next_obs, dtype=torch.float32, device=agent.device)
            step += 1
            
            # Store the final info and machine_schedule
            final_info = info.get('objective_info', {})
            final_machine_schedule = info.get('machine_schedule', {})
            
            if done:
                break
        
        episode_result = {
            'machine_schedule': final_machine_schedule,
            'episode_reward': episode_reward,
            'makespan': final_info.get('makespan', 0),
            'twt': final_info.get('twt', 0),
            'objective': final_info.get('objective', 0),
            'steps_taken': step,
            'manager_decisions': manager_decisions,
            'is_valid_completion': env.state.is_done()
        }
        
        all_episode_results.append(episode_result)
    
    if num_episodes > 1:
        # Aggregate results
        rewards = [r['episode_reward'] for r in all_episode_results]
        makespans = [r['makespan'] for r in all_episode_results]
        twts = [r['twt'] for r in all_episode_results]
        objectives = [r['objective'] for r in all_episode_results]
        
        return {
            'avg_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'avg_makespan': np.mean(makespans),
            'std_makespan': np.std(makespans),
            'avg_twt': np.mean(twts),
            'std_twt': np.std(twts),
            'avg_objective': np.mean(objectives),
            'std_objective': np.std(objectives),
            'num_episodes': num_episodes,
            'individual_results': all_episode_results
        }
    else:
        # Single episode result
        result = all_episode_results[0]
        return {
            'episode_reward': result['episode_reward'],
            'makespan': result['makespan'],
            'twt': result['twt'],
            'objective': result['objective'],
            'steps_taken': result['steps_taken'],
            'manager_decisions': result['manager_decisions'],
            'is_valid_completion': result['is_valid_completion']
        }


def visualize_policy_schedule(evaluation_result: Dict, env: RLEnv, save_path: Optional[str] = None):
    """
    Visualize the schedule from evaluation results using SolutionUtils.
    
    Args:
        evaluation_result: Result from evaluate_flat_policy or evaluate_hierarchical_policy
        env: Environment instance (for data_handler access)
        save_path: Optional path to save the Gantt chart image
    """
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
        # Save directly to HTML (no try/except)
        alt_path = os.path.splitext(save_path)[0] + ".html"
        fig.write_html(alt_path, include_plotlyjs="cdn")
        print(f"Gantt chart saved to {alt_path}")
    
    return fig
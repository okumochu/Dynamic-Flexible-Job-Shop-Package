import torch
import os
import numpy as np
from RL.rl_env import RLEnv
from RL.PPO.flat_agent import FlatAgent
from RL.PPO.hierarchical_agent import HierarchicalAgent
from typing import Dict, Optional, List, Tuple
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
        # Save to PNG with fallback to HTML
        if save_path:
            try:
                png_path = os.path.splitext(save_path)[0] + ".png"
                fig.write_image(png_path, width=1400, height=800, scale=2)
                print(f"Gantt chart saved to {png_path}")
            except Exception as e:
                print(f"Failed to save PNG (may need kaleido): {e}")
                # Fallback to HTML
                html_path = os.path.splitext(save_path)[0] + ".html"
                fig.write_html(html_path, include_plotlyjs="cdn")
                print(f"Gantt chart saved to {html_path} (fallback)")
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
        # Save to PNG with fallback to HTML
        try:
            png_path = os.path.splitext(save_path)[0] + ".png"
            fig.write_image(png_path, width=1400, height=800, scale=2)
            print(f"Gantt chart saved to {png_path}")
        except Exception as e:
            print(f"Failed to save PNG (may need kaleido): {e}")
            # Fallback to HTML
            html_path = os.path.splitext(save_path)[0] + ".html"
            fig.write_html(html_path, include_plotlyjs="cdn")
            print(f"Gantt chart saved to {html_path} (fallback)")
    
    return fig


def convert_graph_schedule_to_machine_schedule(graph_env) -> Dict[int, List[Tuple[int, float]]]:
    """
    Convert GraphRlEnv schedule data to SolutionUtils expected format.
    
    Args:
        graph_env: GraphRlEnv instance after episode completion
        
    Returns:
        Dictionary mapping machine_id to list of (operation_id, start_time) tuples
    """
    machine_schedule = {}
    
    # Initialize empty lists for each machine
    for machine_id in range(graph_env.problem_data.num_machines):
        machine_schedule[machine_id] = []
    
    # Process each scheduled operation
    for op_id in range(graph_env.problem_data.num_operations):
        if graph_env.graph_state.operation_status[op_id] == 1:  # scheduled
            # Get assigned machine and completion time
            assigned_machine = graph_env.graph_state.operation_machine_assignments.get(op_id)
            completion_time = graph_env.graph_state.operation_completion_times[op_id]
            
            if assigned_machine is not None:
                # Calculate start time
                operation = graph_env.problem_data.get_operation(op_id)
                processing_time = operation.get_processing_time(assigned_machine)
                start_time = completion_time - processing_time
                
                # Add to machine schedule
                machine_schedule[assigned_machine].append((op_id, start_time))
    
    # Sort operations by start time for each machine
    for machine_id in machine_schedule:
        machine_schedule[machine_id].sort(key=lambda x: x[1])
    
    return machine_schedule


def evaluate_graph_policy_with_visualization(model_path: str, graph_env, trainer, save_path: str = None) -> Dict:
    """
    Evaluate a trained graph RL policy and create Gantt chart using SolutionUtils.
    
    Args:
        model_path: Path to the saved model
        graph_env: Graph RL environment
        trainer: Graph trainer instance
        save_path: Optional path to save the Gantt chart
        
    Returns:
        Dictionary with evaluation metrics and visualization
    """
    if os.path.exists(model_path):
        trainer.load_model(model_path)
        print(f"Loaded model from {model_path}")
    else:
        print(f"Model file not found: {model_path}")
        return {}
    
    trainer.policy.eval()
    
    # Run one episode to get final state
    obs, info = graph_env.reset()
    done = False
    episode_reward = 0
    episode_length = 0
    
    while not done:
        obs = obs.to(trainer.device)
        
        with torch.no_grad():
            action_logits, value, action_mask, valid_pairs = trainer.policy(obs)
            
            if len(action_logits) == 0:
                print(f"Warning: No valid actions available")
                break
            
            # Take deterministic action (argmax)
            action_idx = torch.argmax(action_logits).item()
            
            # Convert to environment action
            if action_idx < len(valid_pairs):
                target_pair = valid_pairs[action_idx]
                env_action = None
                for env_action_idx, pair in graph_env.action_to_pair_map.items():
                    if pair == target_pair:
                        env_action = env_action_idx
                        break
                
                if env_action is None:
                    print(f"Warning: Could not find environment action for pair {target_pair}")
                    break
            else:
                print(f"Warning: Action index {action_idx} out of range")
                break
        
        next_obs, reward, terminated, truncated, next_info = graph_env.step(env_action)
        
        episode_reward += reward
        episode_length += 1
        done = terminated or truncated
        
        if not done:
            obs = next_obs
    
    # Extract final metrics
    final_makespan = graph_env.graph_state.get_makespan()
    
    # Convert schedule to SolutionUtils format
    machine_schedule = convert_graph_schedule_to_machine_schedule(graph_env)
    
    # Create SolutionUtils instance and visualize
    from utils.solution_utils import SolutionUtils
    solution_utils = SolutionUtils(graph_env.problem_data, machine_schedule)
    
    # Validate the solution
    validation_result = solution_utils.validate_solution()
    print(f"Solution validation: {'VALID' if validation_result['is_valid'] else 'INVALID'}")
    if not validation_result['is_valid']:
        print("Validation violations:")
        for violation in validation_result['violations']:
            print(f"  - {violation}")
    
    # Create Gantt chart
    fig = solution_utils.draw_gantt(show_validation=True, show_due_dates=True)
    
    if save_path and fig is not None:
        # Save to PNG
        try:
            png_path = os.path.splitext(save_path)[0] + ".png"
            fig.write_image(png_path, width=1400, height=800, scale=2)
            print(f"Gantt chart saved to {png_path}")
        except Exception as e:
            print(f"Failed to save PNG (may need kaleido): {e}")
            # Fallback to HTML
            html_path = os.path.splitext(save_path)[0] + ".html"
            fig.write_html(html_path, include_plotlyjs="cdn")
            print(f"Gantt chart saved to {html_path} (fallback)")
    
    # Calculate total weighted tardiness from validation result
    total_twt = 0.0
    due_dates = graph_env.problem_data.get_jobs_due_date()
    weights = graph_env.problem_data.get_jobs_weight()
    
    for job_id in range(graph_env.problem_data.num_jobs):
        job_operations = graph_env.problem_data.get_job_operations(job_id)
        last_op = job_operations[-1]
        if last_op.operation_id in validation_result["operation_end_times"]:
            completion_time = validation_result["operation_end_times"][last_op.operation_id]
            tardiness = max(0, completion_time - due_dates[job_id])
            total_twt += weights[job_id] * tardiness
    
    # For single objective optimization: objective = makespan
    objective = final_makespan
    
    results = {
        'episode_reward': episode_reward,
        'makespan': final_makespan,
        'twt': total_twt,
        'objective': objective,
        'episode_length': episode_length,
        'is_valid_completion': graph_env.graph_state.is_done(),
        'validation_result': validation_result,
        'machine_schedule': machine_schedule,
        'gantt_figure': fig
    }
    
    print(f"Evaluation complete: "
          f"Reward={episode_reward:.2f}, "
          f"Makespan={final_makespan:.2f}, "
          f"TWT={total_twt:.2f}, "
          f"Objective={objective:.2f}")
    
    return results


def create_baseline_gantt_chart(data_handler: FlexibleJobShopDataHandler, save_path: str = None, method: str = "fifo") -> str:
    """
    Create a baseline Gantt chart using simple dispatching rules.
    
    Args:
        data_handler: FlexibleJobShopDataHandler instance
        save_path: Optional path to save the chart
        method: Dispatching method ('fifo' or 'spt' for shortest processing time)
        
    Returns:
        Path to the saved Gantt chart
    """
    from utils.solution_utils import SolutionUtils
    
    machine_schedule = {}
    for machine_id in range(data_handler.num_machines):
        machine_schedule[machine_id] = []
    
    if method.lower() == "fifo":
        # FIFO: Process operations in job order
        current_time = [0.0] * data_handler.num_machines
        
        for job_id in range(data_handler.num_jobs):
            job_start_time = 0.0
            job_operations = data_handler.get_job_operations(job_id)
            
            for operation in job_operations:
                # Find best machine (shortest processing time)
                best_machine = None
                best_time = float('inf')
                
                for machine_id, proc_time in operation.machine_processing_times.items():
                    if proc_time < best_time:
                        best_time = proc_time
                        best_machine = machine_id
                
                if best_machine is not None:
                    start_time = max(job_start_time, current_time[best_machine])
                    machine_schedule[best_machine].append((operation.operation_id, start_time))
                    current_time[best_machine] = start_time + best_time
                    job_start_time = start_time + best_time
    
    elif method.lower() == "spt":
        # SPT: Shortest Processing Time first (global priority)
        operations_list = []
        for job_id in range(data_handler.num_jobs):
            job_operations = data_handler.get_job_operations(job_id)
            for op_idx, operation in enumerate(job_operations):
                min_time = operation.min_processing_time
                operations_list.append((operation.operation_id, job_id, op_idx, min_time))
        
        # Sort by processing time
        operations_list.sort(key=lambda x: x[3])
        
        current_time = [0.0] * data_handler.num_machines
        job_progress = [0] * data_handler.num_jobs  # Track progress of each job
        
        for op_id, job_id, op_idx, _ in operations_list:
            # Check if this operation can be scheduled (precedence constraint)
            if op_idx != job_progress[job_id]:
                continue  # Skip if not the next operation for this job
            
            operation = data_handler.get_operation(op_id)
            
            # Find best machine for this operation
            best_machine = None
            best_time = float('inf')
            
            for machine_id, proc_time in operation.machine_processing_times.items():
                if proc_time < best_time:
                    best_time = proc_time
                    best_machine = machine_id
            
            if best_machine is not None:
                start_time = current_time[best_machine]
                machine_schedule[best_machine].append((op_id, start_time))
                current_time[best_machine] = start_time + best_time
                job_progress[job_id] += 1
    
    # Create Gantt chart
    solution_utils = SolutionUtils(data_handler, machine_schedule)
    validation_result = solution_utils.validate_solution()
    
    print(f"Baseline ({method.upper()}) Schedule:")
    print(f"  Makespan: {validation_result['makespan']:.1f}")
    print(f"  Valid: {'✅' if validation_result['is_valid'] else '❌'}")
    
    fig = solution_utils.draw_gantt(show_validation=True, show_due_dates=True)
    fig.update_layout(title=f"Baseline Schedule - {method.upper()} Dispatching Rule")
    
    if save_path and fig is not None:
        try:
            png_path = save_path if save_path.endswith('.png') else save_path + '.png'
            fig.write_image(png_path, width=1400, height=800, scale=2)
            print(f"Baseline Gantt chart saved to: {png_path}")
            return png_path
        except Exception as e:
            html_path = save_path.replace('.png', '.html') if save_path.endswith('.png') else save_path + '.html'
            fig.write_html(html_path, include_plotlyjs="cdn")
            print(f"Baseline Gantt chart saved to: {html_path}")
            return html_path
    
    return None
"""
Unified Policy Utilities

This module contains utilities for working with different types of RL policies
(flat, graph, hierarchical) for Flexible Job Shop Scheduling problems.
Each policy type has one main evaluation function that returns data
compatible with SolutionUtils for visualization.
"""

import torch
import os
import numpy as np
from typing import Dict, Optional, List, Tuple, Any
import logging

# Import policy-specific modules
from RL.rl_env import RLEnv
from RL.PPO.flat_agent import FlatAgent
from RL.PPO.hierarchical_agent import HierarchicalAgent
from RL.PPO.graph_network import HGTPolicy
from benchmarks.data_handler import FlexibleJobShopDataHandler

logger = logging.getLogger(__name__)


def prepare_solution_utils_data(machine_schedule: Dict[int, List[Tuple[int, float, float]]], 
                               data_handler: FlexibleJobShopDataHandler) -> Dict[str, Any]:
    """
    Convert machine_schedule and data_handler to SolutionUtils format.
    
    Args:
        machine_schedule: Dictionary from policy evaluation
        data_handler: FlexibleJobShopDataHandler instance
        
    Returns:
        Dictionary with data needed for SolutionUtils
    """
    # Create job assignments (operation_id -> job_id)
    job_assignments = {}
    for job_id in range(data_handler.num_jobs):
        job_operations = data_handler.get_job_operations(job_id)
        for operation in job_operations:
            job_assignments[operation.operation_id] = job_id
    
    # Create job due dates (job_id -> due_date)
    job_due_dates = {}
    due_dates = data_handler.get_jobs_due_date()
    for job_id in range(data_handler.num_jobs):
        job_due_dates[job_id] = due_dates[job_id]
    
    # Create machine assignments (operation_id -> list of valid machine_ids)
    machine_assignments = {}
    for job_id in range(data_handler.num_jobs):
        job_operations = data_handler.get_job_operations(job_id)
        for operation in job_operations:
            machine_assignments[operation.operation_id] = list(operation.compatible_machines)
    
    return {
        'job_assignments': job_assignments,
        'job_due_dates': job_due_dates,
        'machine_assignments': machine_assignments
    }


def evaluate_flat_policy(model_path: str, env: RLEnv) -> Dict:
    """
    Evaluate a trained flat PPO policy and return data compatible with SolutionUtils.
    
    Args:
        model_path: Path to the saved model
        env: RLEnv instance to evaluate on
        
    Returns:
        Dictionary containing evaluation results with machine_schedule for SolutionUtils
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
    
    print(f"\nEvaluating flat policy from {model_path}...")
    
    obs, _ = env.reset()
    obs = torch.tensor(obs, dtype=torch.float32, device=agent.device)
    
    episode_reward = 0
    step_count = 0
    max_steps = env.num_jobs * max(len(job.operations) for job in env.jobs.values()) * 2
    final_info = None
    final_machine_schedule = None
    
    while not env.state.is_done() and step_count < max_steps:
        action_mask = env.get_action_mask()
        if not action_mask.any():
            break
        
        action = agent.get_deterministic_action(obs, action_mask)
        
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
    
    # Prepare solution data for SolutionUtils
    solution_data = prepare_solution_utils_data(final_machine_schedule, env.data_handler)
    
    return {
        'makespan': final_info.get('makespan', 0),
        'objective': final_info.get('objective', 0),
        'episode_reward': episode_reward,
        'steps_taken': step_count,
        'is_valid_completion': env.state.is_done(),
        'machine_schedule': final_machine_schedule,
        'data_handler': env.data_handler,
        'solution_data': solution_data
    }


def evaluate_hierarchical_policy(model_path: str, env: RLEnv) -> Dict:
    """
    Evaluate a trained hierarchical PPO policy and return data compatible with SolutionUtils.
    
    Args:
        model_path: Path to the saved model or directory containing model
        env: RLEnv instance to evaluate on
        
    Returns:
        Dictionary containing evaluation results with machine_schedule for SolutionUtils
    """
    # If model_path is a directory, find the most recent model file
    if os.path.isdir(model_path):
        model_files = [f for f in os.listdir(model_path) if f.endswith('.pth')]
        if not model_files:
            raise FileNotFoundError(f"No .pth model files found in {model_path}")
        model_files.sort(reverse=True)
        model_path = os.path.join(model_path, model_files[0])
    
    # Load agent configuration and model
    config = HierarchicalAgent.get_saved_config(model_path)
    
    # Calculate goal_duration dynamically based on environment
    total_operations = env.num_jobs * max(len(job.operations) for job in env.jobs.values())
    goal_duration_ratio = config.get('goal_duration_ratio', 12)
    goal_duration = max(1, total_operations // goal_duration_ratio)
    
    agent = HierarchicalAgent(
        input_dim=config['input_dim'],
        action_dim=config['action_dim'],
        latent_dim=config['latent_dim'],
        goal_dim=config['goal_dim'],
        goal_duration=goal_duration,
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
    
    goals_history = []
    episode_reward = 0
    step = 0
    manager_decisions = 0
    max_steps = env.num_jobs * max(len(job.operations) for job in env.jobs.values()) * 2
    final_info = None
    final_machine_schedule = None
    
    while not env.state.is_done() and step < max_steps:
        z_t = agent.encode_state(obs)
        
        if step % goal_duration == 0:
            goal = agent.get_manager_goal(z_t)
            if goal is not None:
                goals_history.append(goal)
            manager_decisions += 1
        
        # Use the last goal and current goal for pooling
        last_goal = goals_history[-2] if len(goals_history) > 1 else None
        current_goal = goals_history[-1] if goals_history else None
        
        if current_goal is not None:
            pooled_goal = agent.pool_goals(last_goal, current_goal, step, goal_duration)
        else:
            pooled_goal = torch.zeros(agent.latent_dim, device=agent.device)
        
        action_mask = env.get_action_mask()
        if not action_mask.any():
            break
        
        action = agent.get_deterministic_action(obs, action_mask, pooled_goal)
        
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
    
    # Prepare solution data for SolutionUtils
    solution_data = prepare_solution_utils_data(final_machine_schedule, env.data_handler)
    
    return {
        'makespan': final_info.get('makespan', 0),
        'objective': final_info.get('objective', 0),
        'episode_reward': episode_reward,
        'steps_taken': step,
        'manager_decisions': manager_decisions,
        'is_valid_completion': env.state.is_done(),
        'machine_schedule': final_machine_schedule,
        'data_handler': env.data_handler,
        'solution_data': solution_data
    }


def evaluate_graph_policy(model_path: str, graph_env, trainer) -> Dict:
    """
    Evaluate a trained graph RL policy and return data compatible with SolutionUtils.
    
    Args:
        model_path: Path to the saved model
        graph_env: Graph RL environment instance
        trainer: Graph trainer instance
        
    Returns:
        Dictionary containing evaluation results with machine_schedule for SolutionUtils
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    trainer.load_model(model_path)
    trainer.policy.eval()
    
    print(f"\nEvaluating graph policy from {model_path}...")
    
    # Run one episode to get final state
    obs, info = graph_env.reset()
    done = False
    episode_reward = 0
    episode_length = 0
    
    while not done:
        obs = obs.to(trainer.device)
        
        # Get valid actions from the environment
        valid_actions = graph_env.graph_state.get_valid_actions()
        
        if not valid_actions:
            print("Warning: No valid actions available")
            break
        
        with torch.no_grad():
            action_logits, value = trainer.policy(obs, valid_actions)
            
            if len(action_logits) == 0:
                print("Warning: No valid actions available")
                break
            
            # Take deterministic action (argmax)
            action_idx = torch.argmax(action_logits).item()
            
            # Convert to environment action
            if action_idx < len(valid_actions):
                target_pair = valid_actions[action_idx]
                env_action = graph_env.pair_to_action_map.get(tuple(target_pair))
                
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
    machine_schedule = _convert_graph_schedule_to_machine_schedule(graph_env)
    
    # TWT calculation removed - only makespan optimization
    
    # For single objective optimization: objective = makespan
    objective = final_makespan
    
    # Prepare solution data for SolutionUtils
    solution_data = prepare_solution_utils_data(machine_schedule, graph_env.problem_data)
    
    return {
        'episode_reward': episode_reward,
        'makespan': final_makespan,
        'objective': objective,
        'episode_length': episode_length,
        'is_valid_completion': graph_env.graph_state.is_done(),
        'machine_schedule': machine_schedule,
        'data_handler': graph_env.problem_data,
        'solution_data': solution_data
    }


def _convert_graph_schedule_to_machine_schedule(graph_env) -> Dict[int, List[Tuple[int, float, float]]]:
    """
    Convert GraphRlEnv schedule data to SolutionUtils expected format.
    
    Args:
        graph_env: GraphRlEnv instance after episode completion
        
    Returns:
        Dictionary mapping machine_id to list of (operation_id, start_time, end_time) tuples
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
                machine_schedule[assigned_machine].append((op_id, start_time, completion_time))
    
    # Sort operations by start time for each machine
    for machine_id in machine_schedule:
        machine_schedule[machine_id].sort(key=lambda x: x[1])
    
    return machine_schedule


def create_baseline_schedule(data_handler: FlexibleJobShopDataHandler, method: str = "fifo") -> Dict[str, Any]:
    """
    Create a baseline schedule using simple dispatching rules.
    
    Args:
        data_handler: FlexibleJobShopDataHandler instance
        method: Dispatching method ('fifo' or 'spt' for shortest processing time)
        
    Returns:
        Dictionary containing machine_schedule and solution_data for SolutionUtils
    """
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
                    end_time = start_time + best_time
                    machine_schedule[best_machine].append((operation.operation_id, start_time, end_time))
                    current_time[best_machine] = end_time
                    job_start_time = end_time
    
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
                end_time = start_time + best_time
                machine_schedule[best_machine].append((op_id, start_time, end_time))
                current_time[best_machine] = end_time
                job_progress[job_id] += 1
    
    # Prepare solution data for SolutionUtils
    solution_data = prepare_solution_utils_data(machine_schedule, data_handler)
    
    return {
        'machine_schedule': machine_schedule,
        'solution_data': solution_data
    }

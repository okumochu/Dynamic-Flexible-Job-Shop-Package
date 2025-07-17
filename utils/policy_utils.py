import torch
import os
import numpy as np
from RL.rl_env import RLEnv
from RL.PPO.flat_agent import FlatAgent
from RL.PPO.hierarchical_agent import HierarchicalAgent
from typing import Dict
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

def showcase_hierarchical_policy(model_dir: str, env: RLEnv) -> Dict:
    """Load a trained hierarchical agent and run it for one episode."""
    
    model_path = os.path.join(model_dir, "final_model.pth")
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
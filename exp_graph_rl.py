#!/usr/bin/env python3

"""
Graph-based Reinforcement Learning Experiment for FJSP

This script runs experiments using Heterogeneous Graph Transformer (HGT) networks
for solving the Flexible Job Shop Scheduling Problem. It follows the same patterns
as other experiment files in the project.
"""

import os
import time
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Optional
import wandb

# Import project modules
from benchmarks.static_benchmark.data_handler import FlexibleJobShopDataHandler
from RL.graph_rl_env import GraphRlEnv
from RL.graph_rl_trainer import GraphPPOTrainer
from config import config


def evaluate_graph_policy(model_path: str, env: GraphRlEnv, num_episodes: int = 5) -> Dict[str, Any]:
    """
    Evaluate a trained graph RL policy.
    
    Args:
        model_path: Path to the saved model
        env: Graph RL environment
        num_episodes: Number of episodes to evaluate
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Load the trainer and model
    trainer = GraphPPOTrainer(
        problem_data=env.problem_data,
        epochs=1,  # Dummy values for evaluation
        episodes_per_epoch=1,
        train_per_episode=1,
        device='auto'
    )
    
    if os.path.exists(model_path):
        trainer.load_model(model_path)
        print(f"Loaded model from {model_path}")
    else:
        print(f"Model file not found: {model_path}")
        return {}
    
    trainer.policy.eval()
    
    # Collect evaluation metrics
    episode_rewards = []
    episode_makespans = []
    episode_twts = []
    episode_lengths = []
    episode_objectives = []
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        
        while not done:
            obs = obs.to(trainer.device)
            
            with torch.no_grad():
                action_logits, value, action_mask, valid_pairs = trainer.policy(obs)
                
                if len(action_logits) == 0:
                    print(f"Warning: No valid actions in episode {episode}, step {episode_length}")
                    break
                
                # Take deterministic action (argmax)
                action_idx = torch.argmax(action_logits).item()
                
                # Convert to environment action
                if action_idx < len(valid_pairs):
                    target_pair = valid_pairs[action_idx]
                    env_action = None
                    for env_action_idx, pair in env.action_to_pair_map.items():
                        if pair == target_pair:
                            env_action = env_action_idx
                            break
                    
                    if env_action is None:
                        print(f"Warning: Could not find environment action for pair {target_pair}")
                        break
                else:
                    print(f"Warning: Action index {action_idx} out of range")
                    break
            
            next_obs, reward, terminated, truncated, next_info = env.step(env_action)
            
            episode_reward += reward
            episode_length += 1
            done = terminated or truncated
            
            if not done:
                obs = next_obs
        
        # Extract final metrics
        final_makespan = env.graph_state.get_makespan()
        
        # For single objective optimization: objective = makespan
        total_twt = 0.0  # Not used for makespan-only optimization
        objective = final_makespan
        
        episode_rewards.append(episode_reward)
        episode_makespans.append(final_makespan)
        episode_twts.append(total_twt)
        episode_lengths.append(episode_length)
        episode_objectives.append(objective)
        
        print(f"Episode {episode + 1}/{num_episodes}: "
              f"Reward={episode_reward:.2f}, "
              f"Makespan={final_makespan:.2f}, "
              f"TWT={total_twt:.2f}, "
              f"Objective={objective:.2f}")
    
    # Calculate statistics
    results = {
        'avg_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'avg_makespan': np.mean(episode_makespans),
        'std_makespan': np.std(episode_makespans),
        'avg_twt': np.mean(episode_twts),
        'std_twt': np.std(episode_twts),
        'avg_length': np.mean(episode_lengths),
        'std_length': np.std(episode_lengths),
        'avg_objective': np.mean(episode_objectives),
        'std_objective': np.std(episode_objectives),
        'episode_rewards': episode_rewards,
        'episode_makespans': episode_makespans,
        'episode_twts': episode_twts,
        'episode_objectives': episode_objectives,
        'episode_lengths': episode_lengths,
        # Single episode metrics for compatibility
        'episode_reward': episode_rewards[0] if episode_rewards else 0,
        'makespan': episode_makespans[0] if episode_makespans else 0,
        'twt': episode_twts[0] if episode_twts else 0,
        'objective': episode_objectives[0] if episode_objectives else 0
    }
    
    return results


def visualize_graph_schedule(evaluation_result: Dict, env: GraphRlEnv, save_path: str) -> Optional[plt.Figure]:
    """
    Create a Gantt chart visualization of the schedule.
    
    Args:
        evaluation_result: Results from policy evaluation
        env: Graph RL environment 
        save_path: Path to save the visualization
        
    Returns:
        Matplotlib figure object
    """
    try:
        # Get the current state from the environment (assumes environment is in final state)
        operation_completion_times = env.graph_state.operation_completion_times
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Colors for different jobs
        colors = plt.cm.Set3(np.linspace(0, 1, env.problem_data.num_jobs))
        
        # Create Gantt chart
        machine_schedules = [[] for _ in range(env.problem_data.num_machines)]
        
        # Collect all operations and their assignments
        for op_id in range(env.problem_data.num_operations):
            if env.graph_state.operation_status[op_id] == 1:  # scheduled
                operation = env.problem_data.get_operation(op_id)
                job_id, _ = env.problem_data.get_operation_info(op_id)
                
                # Find which machine this operation was assigned to
                completion_time = operation_completion_times[op_id]
                
                # Find the machine with the processing time that matches
                assigned_machine = None
                for machine_id in operation.compatible_machines:
                    processing_time = operation.get_processing_time(machine_id)
                    if env.graph_state.machine_available_times[machine_id] == completion_time:
                        assigned_machine = machine_id
                        break
                
                if assigned_machine is not None:
                    processing_time = operation.get_processing_time(assigned_machine)
                    start_time = completion_time - processing_time
                    
                    machine_schedules[assigned_machine].append({
                        'job_id': job_id,
                        'op_id': op_id,
                        'start': start_time,
                        'duration': processing_time,
                        'end': completion_time
                    })
        
        # Sort operations by start time for each machine
        for machine_id in range(env.problem_data.num_machines):
            machine_schedules[machine_id].sort(key=lambda x: x['start'])
        
        # Plot the Gantt chart
        for machine_id in range(env.problem_data.num_machines):
            for op_info in machine_schedules[machine_id]:
                job_id = op_info['job_id']
                ax.barh(machine_id, op_info['duration'], left=op_info['start'],
                       height=0.6, color=colors[job_id], alpha=0.8,
                       edgecolor='black', linewidth=0.5)
                
                # Add operation label
                ax.text(op_info['start'] + op_info['duration']/2, machine_id,
                       f'J{job_id}O{op_info["op_id"]}',
                       ha='center', va='center', fontsize=8)
        
        # Formatting
        ax.set_xlabel('Time')
        ax.set_ylabel('Machine')
        ax.set_yticks(range(env.problem_data.num_machines))
        ax.set_yticklabels([f'M{i}' for i in range(env.problem_data.num_machines)])
        ax.grid(True, alpha=0.3)
        
        # Title with metrics
        makespan = evaluation_result.get('makespan', 0)
        twt = evaluation_result.get('twt', 0)
        objective = evaluation_result.get('objective', 0)
        
        ax.set_title(f'Graph RL Schedule\n'
                    f'Makespan: {makespan:.2f}, TWT: {twt:.2f}, Objective: {objective:.2f}')
        
        plt.tight_layout()
        
        # Save figure
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Gantt chart saved to {save_path}")
        
        return fig
        
    except Exception as e:
        print(f"Error creating Gantt chart: {e}")
        return None


def get_graph_rl_config(project_name: str = "exp_graph_rl", dataset_name: str = None, exp_name: str = "graph") -> Dict[str, Any]:
    """Get configuration for graph RL experiment."""
    return config.get_graph_rl_config(project_name, dataset_name, exp_name)


def run_single_graph_rl_experiment():
    """Run a single graph RL experiment with simulation data."""
    
    print("="*60)
    print("🔗  GRAPH-BASED REINFORCEMENT LEARNING EXPERIMENT")
    print("📊 Heterogeneous Graph Transformer for Job Shop Scheduling")
    print("="*60)
    
    # Get configuration
    exp_config = get_graph_rl_config()
    simulation_params = exp_config['simulation_params']
    rl_params = exp_config['rl_params']
    result_dirs = exp_config['result_dirs']
    
    # Setup directories
    config.setup_directories(result_dirs)
    config.setup_wandb_env(result_dirs['training_process'], exp_config['wandb_project'])
    
    print("Creating data handler and graph environment...")
    data_handler = FlexibleJobShopDataHandler(data_source=simulation_params, data_type="simulation")
    env = GraphRlEnv(data_handler)
    
    print(f"Graph environment created: {env.problem_data.num_jobs} jobs, {env.problem_data.num_machines} machines")
    print(f"Total operations: {env.problem_data.num_operations}")
    print(f"Action space size: {env.action_space.n}")
    print(f"Graph features - Operations: 8, Machines: 2")
    
    # Create graph trainer using epoch/episode structure like flat_rl_trainer
    trainer = GraphPPOTrainer(
        problem_data=data_handler,
        epochs=rl_params['epochs'],
        episodes_per_epoch=rl_params['episodes_per_epoch'],
        train_per_episode=rl_params['train_per_episode'],
        hidden_dim=rl_params['hidden_dim'],
        num_hgt_layers=rl_params['num_hgt_layers'],
        num_heads=rl_params['num_heads'],
        pi_lr=rl_params['pi_lr'],
        v_lr=rl_params['v_lr'],
        gamma=rl_params['gamma'],
        gae_lambda=rl_params['gae_lambda'],
        clip_ratio=rl_params['clip_ratio'],
        entropy_coef=rl_params['entropy_coef'],
        device=rl_params['device'],
        project_name=exp_config['wandb_project'],
        run_name="graph_rl_single_experiment",
        model_save_dir=result_dirs['model'],
        seed=rl_params['seed']
    )
    
    print(f"Starting graph RL training for {rl_params['epochs']} epochs...")
    start_time = time.time()
    
    # Train the model
    training_results = trainer.train(seed=rl_params['seed'])
    
    training_time = training_results['training_time']
    print(f"Graph RL training complete. Training time: {training_time:.2f} seconds")
    
    # Evaluate the trained policy
    print("\nEvaluating trained graph RL policy...")
    model_filename = training_results['model_filename']
    model_path = os.path.join(result_dirs['model'], model_filename)
    evaluation_result = evaluate_graph_policy(model_path, env, num_episodes=5)
    
    print(f"Final objective: {evaluation_result.get('avg_objective', 0):.2f} ± {evaluation_result.get('std_objective', 0):.2f}")
    print(f"Final makespan: {evaluation_result.get('avg_makespan', 0):.2f} ± {evaluation_result.get('std_makespan', 0):.2f}")
    print(f"Final TWT: {evaluation_result.get('avg_twt', 0):.2f} ± {evaluation_result.get('std_twt', 0):.2f}")
    
    # Save evaluation results
    df = pd.DataFrame([{
        'method': 'Graph RL (HGT)',
        'objective': evaluation_result.get('avg_objective', 0.0),
        'makespan': evaluation_result.get('avg_makespan', 0.0),
        'twt': evaluation_result.get('avg_twt', 0.0),
        'objective_std': evaluation_result.get('std_objective', 0.0),
        'makespan_std': evaluation_result.get('std_makespan', 0.0),
        'twt_std': evaluation_result.get('std_twt', 0.0),
        'training_time': training_time
    }])
    
    csv_path = os.path.join(result_dirs['training_process'], 'graph_rl_results.csv')
    df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")
    
    # Create visualization
    print("Creating Gantt chart visualization...")
    # Run one more episode to get final state for visualization
    obs, _ = env.reset()
    done = False
    while not done:
        obs = obs.to(trainer.device)
        with torch.no_grad():
            action_logits, _, _, valid_pairs = trainer.policy(obs)
            if len(action_logits) == 0:
                break
            action_idx = torch.argmax(action_logits).item()
            if action_idx < len(valid_pairs):
                target_pair = valid_pairs[action_idx]
                env_action = None
                for env_action_idx, pair in env.action_to_pair_map.items():
                    if pair == target_pair:
                        env_action = env_action_idx
                        break
                if env_action is None:
                    break
            else:
                break
        next_obs, _, terminated, truncated, _ = env.step(env_action)
        done = terminated or truncated
        if not done:
            obs = next_obs
    
    gantt_save_path = os.path.join(result_dirs['training_process'], "graph_rl_gantt.png")
    visualize_graph_schedule(evaluation_result, env, gantt_save_path)
    
    print(f"\n" + "="*50)
    print("GRAPH RL EXPERIMENT SUMMARY")
    print("="*50)
    print(f"Training Configuration:")
    print(f"  Epochs: {rl_params['epochs']}")
    print(f"  Episodes per Epoch: {rl_params['episodes_per_epoch']}")
    print(f"  Hidden Dimension: {rl_params['hidden_dim']}")
    print(f"  HGT Layers: {rl_params['num_hgt_layers']}")
    print(f"  Attention Heads: {rl_params['num_heads']}")
    print(f"  Learning Rate: {rl_params['pi_lr']}")
    print(f"  Device: {rl_params['device']}")
    print(f"  Training Time: {training_time:.2f} seconds")
    
    return {
        'training_time': training_time,
        'evaluation_result': evaluation_result,
        'model_path': model_path,
        'training_results': training_results
    }


def run_brandimarte_graph_rl_experiment():
    """Run graph RL experiments on Brandimarte datasets."""
    print("="*70)
    print("🔗  GRAPH RL ON BRANDIMARTE DATASETS")
    print("📊 Testing HGT networks on real-world FJSP instances")
    print("="*70)
    
    # Datasets to test
    datasets = {
        'mk01': 'benchmarks/static_benchmark/datasets/brandimarte/mk01.txt',
        'mk06': 'benchmarks/static_benchmark/datasets/brandimarte/mk06.txt',
        'mk08': 'benchmarks/static_benchmark/datasets/brandimarte/mk08.txt'
    }
    
    project_name = "exp_graph_rl_brandimarte"
    all_results = {}
    
    for dataset_name, dataset_path in datasets.items():
        print(f"\n{'='*70}")
        print(f"EXPERIMENT: {dataset_name.upper()}")
        print(f"{'='*70}")
        
        # Check if dataset exists
        if not os.path.exists(dataset_path):
            print(f"Dataset not found: {dataset_path}")
            continue
        
        try:
            # Get configuration
            exp_config = get_graph_rl_config(project_name, dataset_name, "graph")
            result_dirs = exp_config['result_dirs']
            rl_params = exp_config['rl_params']
            
            # Setup directories
            config.setup_directories(result_dirs)
            
            # Load dataset
            print(f"Loading dataset: {dataset_path}")
            data_handler = FlexibleJobShopDataHandler(
                data_source=dataset_path,
                data_type="dataset",
                TF=config.simulation_params['TF'],
                RDD=config.simulation_params['RDD'],
                seed=config.simulation_params['seed']
            )
            
            print(f"Dataset loaded - Jobs: {data_handler.num_jobs}, "
                  f"Machines: {data_handler.num_machines}, "
                  f"Operations: {data_handler.num_operations}")
            
            # Create environment
            env = GraphRlEnv(data_handler)
            
            # Adjust training parameters for dataset size (reduce epochs for larger datasets)
            adjusted_epochs = min(rl_params['epochs'], max(100, 1000 // (data_handler.num_operations // 10)))
            
            # Create trainer
            trainer = GraphPPOTrainer(
                problem_data=data_handler,
                epochs=adjusted_epochs,
                episodes_per_epoch=rl_params['episodes_per_epoch'],
                train_per_episode=rl_params['train_per_episode'],
                hidden_dim=rl_params['hidden_dim'],
                num_hgt_layers=rl_params['num_hgt_layers'],
                num_heads=rl_params['num_heads'],
                pi_lr=rl_params['pi_lr'],
                gamma=rl_params['gamma'],
                gae_lambda=rl_params['gae_lambda'],
                clip_ratio=rl_params['clip_ratio'],
                entropy_coef=rl_params['entropy_coef'],
                device=rl_params['device'],
                seed=rl_params['seed'],
                model_save_dir=result_dirs['model'],
                project_name=project_name,
                run_name=f"graph_rl_{dataset_name}"
            )
            
            print(f"Training graph RL for {adjusted_epochs} epochs...")
            start_time = time.time()
            
            # Train
            training_results = trainer.train(seed=rl_params['seed'])
            
            training_time = training_results['training_time']
            
            # Evaluate
            model_filename = training_results['model_filename']
            model_path = os.path.join(result_dirs['model'], model_filename)
            evaluation_result = evaluate_graph_policy(model_path, env, num_episodes=3)
            
            all_results[dataset_name] = {
                'training_time': training_time,
                'evaluation_result': evaluation_result,
                'model_path': model_path
            }
            
            print(f"Dataset {dataset_name} completed:")
            print(f"  Objective: {evaluation_result.get('avg_objective', 0):.2f}")
            print(f"  Makespan: {evaluation_result.get('avg_makespan', 0):.2f}")
            print(f"  Training time: {training_time:.2f}s")
            
            # Save results for this dataset
            df = pd.DataFrame([{
                'dataset': dataset_name,
                'method': 'Graph RL (HGT)',
                'objective': evaluation_result.get('avg_objective', 0.0),
                'makespan': evaluation_result.get('avg_makespan', 0.0),
                'twt': evaluation_result.get('avg_twt', 0.0),
                'training_time': training_time
            }])
            
            csv_path = os.path.join(result_dirs['training_process'], f'{dataset_name}_results.csv')
            df.to_csv(csv_path, index=False)
            
        except Exception as e:
            print(f"Error processing dataset {dataset_name}: {e}")
            continue
    
    # Create summary results
    summary_data = []
    for dataset_name, results in all_results.items():
        eval_result = results['evaluation_result']
        summary_data.append({
            'dataset': dataset_name,
            'method': 'Graph RL (HGT)',
            'objective': eval_result.get('avg_objective', 0.0),
            'makespan': eval_result.get('avg_makespan', 0.0),
            'twt': eval_result.get('avg_twt', 0.0),
            'training_time': results['training_time']
        })
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_path = os.path.join('result', project_name, 'brandimarte_summary.csv')
        os.makedirs(os.path.dirname(summary_path), exist_ok=True)
        summary_df.to_csv(summary_path, index=False)
        print(f"\nSummary results saved to: {summary_path}")
        
        print("\n" + "="*50)
        print("BRANDIMARTE EXPERIMENT SUMMARY")
        print("="*50)
        for _, row in summary_df.iterrows():
            print(f"{row['dataset']}: Objective={row['objective']:.2f}, "
                  f"Makespan={row['makespan']:.2f}, Time={row['training_time']:.1f}s")
    
    return all_results


def main():
    """Main function to run graph RL experiments."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run Graph RL experiments on FJSP')
    parser.add_argument('--experiment', type=str, default='single',
                       choices=['single', 'brandimarte'],
                       help='Type of experiment to run')
    args = parser.parse_args()
    
    if args.experiment == 'single':
        print("Running single graph RL experiment...")
        results = run_single_graph_rl_experiment()
    elif args.experiment == 'brandimarte':
        print("Running graph RL experiments on Brandimarte datasets...")
        results = run_brandimarte_graph_rl_experiment()
    
    print("\nGraph RL experiment completed successfully! 🎉")
    return results


if __name__ == "__main__":
    main()

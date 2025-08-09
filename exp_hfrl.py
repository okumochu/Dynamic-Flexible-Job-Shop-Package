"""
Hierarchical RL vs Flat RL Experiment on Brandimarte Datasets
Tests hierarchical RL against flat RL baseline on real-world datasets
"""

import os
import pandas as pd
from typing import List, Dict, Any
from config import config
import numpy as np

# Import necessary modules
from benchmarks.static_benchmark.data_handler import FlexibleJobShopDataHandler
from RL.rl_env import RLEnv
from RL.flat_rl_trainer import FlatRLTrainer
from RL.hierarchical_rl_trainer import HierarchicalRLTrainer
from utils.policy_utils import evaluate_flat_policy, evaluate_hierarchical_policy, visualize_policy_schedule

def create_training_environment(dataset_path: str) -> FlexibleJobShopDataHandler:
    """Create training environment from Brandimarte dataset"""
    # Use TF and RDD from config for due date generation
    tf = config.simulation_params.get('TF', 0.4)
    rdd = config.simulation_params.get('RDD', 0.8)
    seed = config.simulation_params.get('seed', 42)
    
    data_handler = FlexibleJobShopDataHandler(
        data_source=dataset_path,
        data_type="dataset",
        TF=tf,
        RDD=rdd,
        seed=seed
    )
    return data_handler

def train_flat_rl_baseline(training_rl_env: RLEnv, config_params: Dict, 
                          project_name: str, run_name: str) -> Dict[str, Any]:
    """Train flat RL as baseline"""
    
    flat_trainer = FlatRLTrainer(
        env=training_rl_env,
        epochs=config_params['rl_params']['epochs'],
        episodes_per_epoch=config_params['rl_params']['episodes_per_epoch'],
        train_per_episode=config_params['rl_params']['train_per_episode'],
        pi_lr=config_params['rl_params']['pi_lr'],
        v_lr=config_params['rl_params']['v_lr'],
        gamma=config_params['rl_params']['gamma'],
        gae_lambda=config_params['rl_params']['gae_lambda'],
        clip_ratio=config_params['rl_params']['clip_ratio'],
        entropy_coef=config_params['rl_params']['entropy_coef'],
        device=config_params['rl_params']['device'],
        model_save_dir=config_params['result_dirs']['model'],
        project_name=project_name,
        run_name=run_name
    )
    
    training_results = flat_trainer.train(env_or_envs=training_rl_env)

    # Evaluate deterministically on the training environment
    model_filename = training_results.get('model_filename', '')
    model_path = os.path.join(config_params['result_dirs']['model'], model_filename)
    evaluation_result = None
    if os.path.exists(model_path):
        evaluation_result = evaluate_flat_policy(model_path, training_rl_env, num_episodes=1)
    else:
        print(f"Warning: Model file not found at {model_path}")

    # Use evaluation metrics if available; otherwise fallback to training history
    if evaluation_result is not None:
        final_objective = evaluation_result.get('objective', 0.0)
        final_makespan = evaluation_result.get('makespan', 0.0)
        final_twt = evaluation_result.get('twt', 0.0)
    elif training_results['training_history']['episode_objectives']:
        final_objective = np.mean(training_results['training_history']['episode_objectives'][-10:])
        final_makespan = np.mean(training_results['training_history']['episode_makespans'][-10:])
        final_twt = np.mean(training_results['training_history']['episode_twts'][-10:])
    else:
        final_objective = 0.0
        final_makespan = 0.0
        final_twt = 0.0
    
    return {
        'training_results': training_results,
        'final_objective': final_objective,
        'final_makespan': final_makespan,
        'final_twt': final_twt,
        'evaluation_result': evaluation_result,
        'model_path': model_path
    }

def train_hierarchical_rl(training_rl_env: RLEnv, config_params: Dict, 
                         project_name: str, run_name: str) -> Dict[str, Any]:
    """Train hierarchical RL"""
    
    hierarchical_trainer = HierarchicalRLTrainer(
        env=training_rl_env,
        epochs=config_params['rl_params']['epochs'],
        episodes_per_epoch=config_params['rl_params']['episodes_per_epoch'],
        goal_duration_ratio=config_params['rl_params']['goal_duration_ratio'],
        latent_dim=config_params['rl_params']['latent_dim'],
        goal_dim=config_params['rl_params']['goal_dim'],
        manager_lr=config_params['rl_params']['manager_lr'],
        worker_lr=config_params['rl_params']['worker_lr'],
        gamma_manager=config_params['rl_params']['gamma_manager'],
        gamma_worker=config_params['rl_params']['gamma_worker'],
        clip_ratio=config_params['rl_params']['clip_ratio'],
        entropy_coef=config_params['rl_params']['entropy_coef'],
        gae_lambda=config_params['rl_params']['gae_lambda'],
        train_per_episode=config_params['rl_params']['train_per_episode'],
        intrinsic_reward_scale=config_params['rl_params']['intrinsic_reward_scale'],
        device=config_params['rl_params']['device'],
        model_save_dir=config_params['result_dirs']['model'],
        project_name=project_name,
        run_name=run_name
    )
    
    training_results = hierarchical_trainer.train(env_or_envs=training_rl_env)

    # Evaluate deterministically on the training environment
    model_filename = training_results.get('model_filename', '')
    model_path = os.path.join(config_params['result_dirs']['model'], model_filename)
    evaluation_result = None
    if os.path.exists(model_path):
        evaluation_result = evaluate_hierarchical_policy(model_path, training_rl_env, num_episodes=1)
    else:
        print(f"Warning: Model file not found at {model_path}")

    # Use evaluation metrics if available; otherwise fallback to training history
    if evaluation_result is not None and 'objective' in evaluation_result:
        final_objective = evaluation_result.get('objective', 0.0)
        final_makespan = evaluation_result.get('makespan', 0.0)
        final_twt = evaluation_result.get('twt', 0.0)
    elif training_results['training_history']['episode_objectives']:
        final_objective = np.mean(training_results['training_history']['episode_objectives'][-10:])
        final_makespan = np.mean(training_results['training_history']['episode_makespans'][-10:])
        final_twt = np.mean(training_results['training_history']['episode_twts'][-10:])
    else:
        final_objective = 0.0
        final_makespan = 0.0
        final_twt = 0.0
    
    return {
        'training_results': training_results,
        'final_objective': final_objective,
        'final_makespan': final_makespan,
        'final_twt': final_twt,
        'evaluation_result': evaluation_result,
        'model_path': model_path
    }

def run_brandimarte_experiment():
    """Main experiment comparing HRL vs Flat RL on Brandimarte datasets"""
    print("Starting Brandimarte Dataset Experiment...")
    print("Testing: Flat RL vs Hierarchical RL on real-world datasets")
    print("="*70)
    
    # Datasets to evaluate
    datasets = {
        'mk06': 'benchmarks/static_benchmark/datasets/brandimarte/mk06.txt',
        'mk08': 'benchmarks/static_benchmark/datasets/brandimarte/mk08.txt'
    }
    
    # Use config parameters
    project_name = "exp_hrl_and_flarl"
    epochs = config.rl_params['epochs']
    use_reward_shaping = config.rl_params['use_reward_shaping']
    
    print(f"Testing datasets: {list(datasets.keys())}")
    print(f"Training epochs: {epochs}")
    print(f"Use reward shaping: {use_reward_shaping}")
    
    # Results storage
    all_results = {}
    
    # Main loop: test each dataset
    for dataset_name, dataset_path in datasets.items():
        print(f"\n{'='*70}")
        print(f"EXPERIMENT: {dataset_name.upper()}")
        print(f"{'='*70}")
        
        try:
            # Prepare configs with organized directory structure
            flat_config = config.get_flat_rl_config(project_name=project_name, dataset_name=dataset_name, exp_name="flat")
            hrl_config = config.get_hierarchical_rl_config(project_name=project_name, dataset_name=dataset_name, exp_name="hrl")
            
            # Set epochs and reward shaping
            for cfg in [flat_config, hrl_config]:
                cfg['rl_params']['epochs'] = epochs
                cfg['rl_params']['use_reward_shaping'] = use_reward_shaping
            
            # Setup directories
            config.setup_directories(flat_config['result_dirs'])
            config.setup_directories(hrl_config['result_dirs'])
            
            # Create training environment from dataset
            print(f"Loading dataset: {dataset_path}")
            training_data_handler = create_training_environment(dataset_path)
            print(f"Dataset loaded - Jobs: {training_data_handler.num_jobs}, Machines: {training_data_handler.num_machines}")
            
            # Create RL environment
            training_rl_env = RLEnv(
                training_data_handler, 
                alpha=config.rl_params['alpha'], 
                use_reward_shaping=use_reward_shaping
            )
            
            # Train flat RL baseline
            print(f"\nTraining Flat RL baseline...")
            flat_run_name = f"flat_{dataset_name}"
            flat_results = train_flat_rl_baseline(
                training_rl_env, flat_config, project_name, flat_run_name
            )
            all_results[f'{dataset_name}_flat'] = flat_results
            
            print(f"Flat RL baseline completed - Final objective: {flat_results['final_objective']:.2f}")
            
            # Train hierarchical RL
            print(f"\nTraining Hierarchical RL...")
            hrl_run_name = f"hrl_{dataset_name}"
            hrl_results = train_hierarchical_rl(
                training_rl_env, hrl_config, project_name, hrl_run_name
            )
            all_results[f'{dataset_name}_hrl'] = hrl_results
            
            print(f"Hierarchical RL completed - Final objective: {hrl_results['final_objective']:.2f}")
            
            # Create results DataFrame
            results_data = [
                {
                    'method': 'Flat RL',
                    'objective': flat_results['final_objective'],
                    'makespan': flat_results['final_makespan'],
                    'twt': flat_results['final_twt'],
                    'run_name': flat_run_name
                },
                {
                    'method': 'Hierarchical RL',
                    'objective': hrl_results['final_objective'],
                    'makespan': hrl_results['final_makespan'],
                    'twt': hrl_results['final_twt'],
                    'run_name': hrl_run_name
                }
            ]
            
            results_df = pd.DataFrame(results_data)
            
            # Save results
            result_project_dir = os.path.join('result', project_name, dataset_name)
            os.makedirs(result_project_dir, exist_ok=True)
            csv_path = os.path.join(result_project_dir, f'{dataset_name}_results.csv')
            results_df.to_csv(csv_path, index=False)
            
            print(f"\n📊 {dataset_name.upper()} RESULTS")
            print("="*70)
            print(f"Flat RL objective: {flat_results['final_objective']:.2f}")
            print(f"Hierarchical RL objective: {hrl_results['final_objective']:.2f}")
            
            # Determine winner
            if flat_results['final_objective'] < hrl_results['final_objective']:
                winner = "Flat RL"
                if flat_results['final_objective'] > 0:
                    improvement = ((hrl_results['final_objective'] - flat_results['final_objective']) / flat_results['final_objective']) * 100
                else:
                    improvement = 0.0
            elif hrl_results['final_objective'] < flat_results['final_objective']:
                winner = "Hierarchical RL"
                if hrl_results['final_objective'] > 0:
                    improvement = ((flat_results['final_objective'] - hrl_results['final_objective']) / hrl_results['final_objective']) * 100
                else:
                    improvement = 0.0
            else:
                # Both objectives are equal (including both being 0)
                winner = "Tie"
                improvement = 0.0
            
            print(f"Winner: {winner} (improvement: {improvement:.1f}%)")
            print(f"\n💾 Results saved to: {csv_path}")
            
            # Generate and save Gantt charts for both methods
            print(f"\n📊 Generating Gantt charts...")
            
            # Always generate Gantt charts
            # Create visualization directory
            viz_dir = os.path.join(result_project_dir, 'gantt_charts')
            os.makedirs(viz_dir, exist_ok=True)
            
            # Generate Gantt chart for Flat RL
            if flat_results['evaluation_result']:
                flat_gantt_path = os.path.join(viz_dir, f'{dataset_name}_flat_rl_gantt.png')
                flat_fig = visualize_policy_schedule(
                    flat_results['evaluation_result'], 
                    training_rl_env, 
                    save_path=flat_gantt_path
                )
                if flat_fig:
                    print(f"✅ Flat RL Gantt chart saved to: {flat_gantt_path}")
                else:
                    print("❌ Could not generate Flat RL Gantt chart")
            else:
                print("❌ No evaluation result available for Flat RL Gantt chart")
            
            # Generate Gantt chart for Hierarchical RL
            if hrl_results['evaluation_result']:
                hrl_gantt_path = os.path.join(viz_dir, f'{dataset_name}_hierarchical_rl_gantt.png')
                hrl_fig = visualize_policy_schedule(
                    hrl_results['evaluation_result'], 
                    training_rl_env, 
                    save_path=hrl_gantt_path
                )
                if hrl_fig:
                    print(f"✅ Hierarchical RL Gantt chart saved to: {hrl_gantt_path}")
                else:
                    print("❌ Could not generate Hierarchical RL Gantt chart")
            else:
                print("❌ No evaluation result available for Hierarchical RL Gantt chart")
            
            print(f"\n📁 Gantt charts saved in: {viz_dir}")
            
            # Add a small delay to ensure all files are written
            import time
            time.sleep(1)
            
        except Exception as e:
            raise
    
    print(f"\n{'='*70}")
    print("EXPERIMENT COMPLETED SUCCESSFULLY!")
    print(f"{'='*70}")
    
    # Print summary of generated files
    print(f"\n📁 EXPERIMENT OUTPUT SUMMARY:")
    print(f"{'='*70}")
    for dataset_name in datasets.keys():
        result_project_dir = os.path.join('result', project_name, dataset_name)
        csv_path = os.path.join(result_project_dir, f'{dataset_name}_results.csv')
        viz_dir = os.path.join(result_project_dir, 'gantt_charts')
        
        print(f"\n📊 {dataset_name.upper()} DATASET:")
        print(f"  📄 Results CSV: {csv_path}")
        print(f"  📈 Gantt Charts: {viz_dir}")
        if os.path.exists(viz_dir):
            gantt_files = [f for f in os.listdir(viz_dir) if f.endswith('.png')]
            for gantt_file in gantt_files:
                print(f"    - {gantt_file}")
    
    print(f"\n🎯 All results and visualizations have been saved!")
    print(f"💡 You can view the Gantt charts to compare the scheduling strategies.")
    
    return all_results

if __name__ == "__main__":
    print("Remember to activate conda environment: conda activate dfjs")
    all_results = run_brandimarte_experiment()

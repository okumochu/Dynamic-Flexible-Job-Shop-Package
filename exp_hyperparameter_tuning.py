"""
Hyperparameter Tuning Experiment for Hierarchical RL
Tests different hyperparameter combinations for hierarchical RL against flat RL baseline
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

def create_training_environment() -> FlexibleJobShopDataHandler:
    """Create training environment"""
    sim_params = config.simulation_params
    simulation_params = {
        'num_jobs': sim_params['num_jobs'],
        'num_machines': sim_params['num_machines'],
        'operation_lb': sim_params['operation_lb'],
        'operation_ub': sim_params['operation_ub'],
        'processing_time_lb': sim_params['processing_time_lb'],
        'processing_time_ub': sim_params['processing_time_ub'],
        'compatible_machines_lb': sim_params['compatible_machines_lb'],
        'compatible_machines_ub': sim_params['compatible_machines_ub'],
        'seed': sim_params['seed']
    }
    
    data_handler = FlexibleJobShopDataHandler(
        data_source=simulation_params,
        data_type="simulation"
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
    
    # Deterministic evaluation on training environment
    from utils.policy_utils import evaluate_flat_policy
    model_filename = training_results.get('model_filename', '')
    model_path = os.path.join(config_params['result_dirs']['model'], model_filename)
    if os.path.exists(model_path):
        eval_res = evaluate_flat_policy(model_path, training_rl_env, num_episodes=1)
        final_objective = eval_res.get('objective', 0.0)
        final_makespan = eval_res.get('makespan', 0.0)
        final_twt = eval_res.get('twt', 0.0)
    else:
        final_objective = 0.0
        final_makespan = 0.0
        final_twt = 0.0
    
    return {
        'training_results': training_results,
        'final_objective': final_objective,
        'final_makespan': final_makespan,
        'final_twt': final_twt
    }

def train_hierarchical_rl_grid(training_rl_env: RLEnv, config_params: Dict, 
                              project_name: str, run_name: str,
                              intrinsic_reward_scale: float, goal_duration_ratio: float) -> Dict[str, Any]:
    """Train hierarchical RL with specific hyperparameters"""
    
    # Update config with grid search parameters
    grid_config = config_params.copy()
    grid_config['rl_params'] = grid_config['rl_params'].copy()
    grid_config['rl_params']['intrinsic_reward_scale'] = intrinsic_reward_scale
    grid_config['rl_params']['goal_duration_ratio'] = goal_duration_ratio
    
    hierarchical_trainer = HierarchicalRLTrainer(
        env=training_rl_env,
        epochs=grid_config['rl_params']['epochs'],
        episodes_per_epoch=grid_config['rl_params']['episodes_per_epoch'],
        goal_duration_ratio=grid_config['rl_params']['goal_duration_ratio'],
        latent_dim=grid_config['rl_params']['latent_dim'],
        goal_dim=grid_config['rl_params']['goal_dim'],
        manager_lr=grid_config['rl_params']['manager_lr'],
        worker_lr=grid_config['rl_params']['worker_lr'],
        gamma_manager=grid_config['rl_params']['gamma_manager'],
        gamma_worker=grid_config['rl_params']['gamma_worker'],
        clip_ratio=grid_config['rl_params']['clip_ratio'],
        entropy_coef=grid_config['rl_params']['entropy_coef'],
        gae_lambda=grid_config['rl_params']['gae_lambda'],
        train_per_episode=grid_config['rl_params']['train_per_episode'],
        intrinsic_reward_scale=grid_config['rl_params']['intrinsic_reward_scale'],
        device=grid_config['rl_params']['device'],
        model_save_dir=grid_config['result_dirs']['model'],
        project_name=project_name,
        run_name=run_name
    )
    
    training_results = hierarchical_trainer.train(env_or_envs=training_rl_env)
    
    # Deterministic evaluation on training environment
    from utils.policy_utils import evaluate_hierarchical_policy
    model_filename = training_results.get('model_filename', '')
    model_path = os.path.join(grid_config['result_dirs']['model'], model_filename)
    if os.path.exists(model_path):
        eval_res = evaluate_hierarchical_policy(model_path, training_rl_env, num_episodes=1)
        final_objective = eval_res.get('objective', 0.0)
        final_makespan = eval_res.get('makespan', 0.0)
        final_twt = eval_res.get('twt', 0.0)
    else:
        final_objective = 0.0
        final_makespan = 0.0
        final_twt = 0.0
    
    return {
        'training_results': training_results,
        'final_objective': final_objective,
        'final_makespan': final_makespan,
        'final_twt': final_twt,
        'intrinsic_reward_scale': intrinsic_reward_scale,
        'goal_duration_ratio': goal_duration_ratio
    }

def run_hyperparameter_tuning():
    """Main hyperparameter tuning experiment"""
    print("Starting Hyperparameter Tuning Experiment...")
    print("Testing: Flat RL baseline vs Hierarchical RL grid search")
    print("="*70)
    
    # Use device from config
    device = config.rl_params['device']
    
    # Create training environment
    print("Creating training environment...")
    training_data_handler = create_training_environment()
    print("Created training environment")
    
    # Define grid search parameters
    intrinsic_reward_scales = [0.1, 0.5, 1.0]
    goal_duration_ratios = [5, 10, 15, 20]  # These are now ratios (total_ops // ratio)
    
    print(f"Grid search parameters:")
    print(f"  Intrinsic reward scales: {intrinsic_reward_scales}")
    print(f"  Goal duration ratios: {goal_duration_ratios}")
    print(f"  Total combinations: {len(intrinsic_reward_scales) * len(goal_duration_ratios)}")
    
    # Results storage
    all_results = {}
    
    # Main loop: sparse and dense rewards
    for reward_type in ['sparse', 'dense']:
        print(f"\n{'='*70}")
        print(f"EXPERIMENT: {reward_type.upper()} REWARD")
        print(f"{'='*70}")
        
        # Set project name
        project_name = f"16jobs_tuning_{reward_type}"
        
        # Prepare configs with organized directory structure
        flat_config = config.get_flat_rl_config(project_name=project_name, exp_name="flat")
        hrl_config = config.get_hierarchical_rl_config(project_name=project_name, exp_name="hrl_grid")
        
        # Set reward shaping and epochs
        use_reward_shaping = (reward_type == 'dense')
        flat_config['rl_params']['use_reward_shaping'] = use_reward_shaping
        hrl_config['rl_params']['use_reward_shaping'] = use_reward_shaping
        
        for cfg in [flat_config, hrl_config]:
            cfg['rl_params']['epochs'] = config.rl_params['epochs']
        
        # Setup directories
        config.setup_directories(flat_config['result_dirs'])
        config.setup_directories(hrl_config['result_dirs'])
        
        # Create RL environment
        training_rl_env = RLEnv(
            training_data_handler, 
            alpha=config.rl_params['alpha'], 
            use_reward_shaping=use_reward_shaping
        )
        
        # Train flat RL baseline
        print(f"\nTraining Flat RL baseline...")
        flat_results = train_flat_rl_baseline(
            training_rl_env, flat_config, project_name, 'flat'
        )
        all_results[f'{reward_type}_flat'] = flat_results
        
        print(f"Flat RL baseline completed - Final objective: {flat_results['final_objective']:.2f}")
        
        # Grid search for hierarchical RL
        print(f"\nStarting Hierarchical RL grid search...")
        grid_results = []
        
        for intrinsic_scale in intrinsic_reward_scales:
            for goal_dur_ratio in goal_duration_ratios:
                run_name = f"reward_{intrinsic_scale}_duration_{int(goal_dur_ratio)}"
                
                print(f"Training HRL with intrinsic_scale={intrinsic_scale}, goal_duration={int(goal_dur_ratio)}...")
                
                hrl_result = train_hierarchical_rl_grid(
                    training_rl_env, hrl_config, project_name, run_name,
                    intrinsic_scale, goal_dur_ratio
                )
                
                grid_results.append({
                    'intrinsic_reward_scale': intrinsic_scale,
                    'goal_duration_ratio': goal_dur_ratio,
                    'objective': hrl_result['final_objective'],
                    'makespan': hrl_result['final_makespan'],
                    'twt': hrl_result['final_twt'],
                    'run_name': run_name
                })
                
                print(f"  Completed - Objective: {hrl_result['final_objective']:.2f}")
        
        all_results[f'{reward_type}_grid'] = grid_results
        
        # Create results DataFrame
        results_df = pd.DataFrame(grid_results)
        
        # Add baseline to results
        baseline_row = pd.DataFrame([{
            'intrinsic_reward_scale': 'N/A',
            'goal_duration_ratio': 'N/A',
            'objective': flat_results['final_objective'],
            'makespan': flat_results['final_makespan'],
            'twt': flat_results['final_twt'],
            'run_name': 'flat_baseline'
        }])
        
        results_df = pd.concat([baseline_row, results_df], ignore_index=True)
        
        # Save results
        result_project_dir = os.path.join('result', project_name)
        os.makedirs(result_project_dir, exist_ok=True)
        csv_path = os.path.join(result_project_dir, f'{reward_type}_h_tuning.csv')
        results_df.to_csv(csv_path, index=False)
        
        print(f"\n📊 {reward_type.upper()} REWARD RESULTS")
        print("="*70)
        print(f"Flat RL baseline objective: {flat_results['final_objective']:.2f}")
        
        # Find best grid result
        best_grid = results_df[results_df['run_name'] != 'flat_baseline'].loc[
            results_df[results_df['run_name'] != 'flat_baseline']['objective'].idxmin()
        ]
        print(f"Best HRL grid result:")
        print(f"  Intrinsic scale: {best_grid['intrinsic_reward_scale']}")
        print(f"  Goal duration: {best_grid['goal_duration_ratio']}")
        print(f"  Objective: {best_grid['objective']:.2f}")
        
        print(f"\n💾 Results saved to: {csv_path}")
    
    print(f"\n{'='*70}")
    print("EXPERIMENT COMPLETED SUCCESSFULLY!")
    print(f"{'='*70}")
    
    return all_results

if __name__ == "__main__":
    print("Remember to activate conda environment: conda activate dfjs")
    all_results = run_hyperparameter_tuning()

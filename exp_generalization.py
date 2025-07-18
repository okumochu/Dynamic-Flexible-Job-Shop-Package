"""
Generalization Experiment for Flat vs Hierarchical RL with/without Curriculum Learning
Tests the generalization capability and effectiveness of curriculum learning on Dynamic Flexible Job Shop Scheduling

This script compares:
1. Flat RL with Curriculum Learning vs Single Environment
2. Hierarchical RL with Curriculum Learning vs Single Environment
3. Reports results in a comparison table
"""

import os
import time
import random
import numpy as np
import pandas as pd
from typing import List, Dict, Any
from config import config

# Import necessary modules
from benchmarks.static_benchmark.data_handler import FlexibleJobShopDataHandler
from RL.rl_env import RLEnv
from RL.flat_rl_trainer import FlatRLTrainer
from RL.hierarchical_rl_trainer import HierarchicalRLTrainer

def create_random_environments(num_envs: int, seed_base: int = 42) -> List[FlexibleJobShopDataHandler]:
    """Create random environments for training/testing"""
    environments = []
    sim_params = config.simulation_params
    
    for i in range(num_envs):
        # Use different seeds for each environment
        env_seed = seed_base + i * 1000
        
        simulation_params = {
            'num_jobs': sim_params['num_jobs'],
            'num_machines': sim_params['num_machines'],
            'operation_lb': sim_params['operation_lb'],
            'operation_ub': sim_params['operation_ub'],
            'processing_time_lb': sim_params['processing_time_lb'],
            'processing_time_ub': sim_params['processing_time_ub'],
            'compatible_machines_lb': sim_params['compatible_machines_lb'],
            'compatible_machines_ub': sim_params['compatible_machines_ub'],
            'seed': env_seed
        }
        
        data_handler = FlexibleJobShopDataHandler(
            data_source=simulation_params,
            data_type="simulation"
        )
        environments.append(data_handler)
    
    return environments

def train_flat_rl(training_rl_envs: List[RLEnv], test_environments: List[FlexibleJobShopDataHandler], 
                  config_params: Dict, use_curriculum: bool = True) -> Dict[str, Any]:
    """Train flat RL with or without curriculum learning"""
    
    flat_trainer = FlatRLTrainer(
        env=training_rl_envs[0],
        epochs=config_params['rl_params']['epochs'],
        steps_per_epoch=config_params['rl_params']['steps_per_epoch'],
        train_pi_iters=config_params['rl_params']['train_pi_iters'],
        train_v_iters=config_params['rl_params']['train_v_iters'],
        pi_lr=config_params['rl_params']['pi_lr'],
        v_lr=config_params['rl_params']['v_lr'],
        gamma=config_params['rl_params']['gamma'],
        gae_lambda=config_params['rl_params']['gae_lambda'],
        clip_ratio=config_params['rl_params']['clip_ratio'],
        entropy_coef=config_params['rl_params']['entropy_coef'],
        device='auto',
        model_save_dir=config_params['result_dirs']['model']
    )
    
    if use_curriculum:
        # Train with curriculum learning (multiple environments)
        training_results = flat_trainer.train(
            env_or_envs=training_rl_envs,
            test_environments=test_environments,
            test_interval=config_params['rl_params']['test_interval']
        )
    else:
        # Train on single random environment
        single_env = random.choice(training_rl_envs)
        training_results = flat_trainer.train(
            env_or_envs=single_env,
            test_environments=test_environments,
            test_interval=config_params['rl_params']['test_interval']
        )
    
    # Final generalization evaluation
    generalization_results = flat_trainer.evaluate_generalization(test_environments)
    
    return {
        'training_results': training_results,
        'generalization_results': generalization_results
    }

def train_hierarchical_rl(training_rl_envs: List[RLEnv], test_environments: List[FlexibleJobShopDataHandler], 
                         config_params: Dict, use_curriculum: bool = True) -> Dict[str, Any]:
    """Train hierarchical RL with or without curriculum learning"""
    
    hierarchical_trainer = HierarchicalRLTrainer(
        env=training_rl_envs[0],
        epochs=config_params['rl_params']['epochs'],
        steps_per_epoch=config_params['rl_params']['steps_per_epoch'],
        goal_duration=config_params['rl_params']['goal_duration'],
        latent_dim=config_params['rl_params']['latent_dim'],
        goal_dim=config_params['rl_params']['goal_dim'],
        manager_lr=config_params['rl_params']['manager_lr'],
        worker_lr=config_params['rl_params']['worker_lr'],
        gamma_manager=config_params['rl_params']['gamma_manager'],
        gamma_worker=config_params['rl_params']['gamma_worker'],
        clip_ratio=config_params['rl_params']['clip_ratio'],
        entropy_coef=config_params['rl_params']['entropy_coef'],
        gae_lambda=config_params['rl_params']['gae_lambda'],
        train_pi_iters=config_params['rl_params']['train_pi_iters'],
        train_v_iters=config_params['rl_params']['train_v_iters'],
        alpha=config_params['rl_params']['alpha'],
        device='auto',
        model_save_dir=config_params['result_dirs']['model']
    )
    
    if use_curriculum:
        # Train with curriculum learning (multiple environments)
        training_results = hierarchical_trainer.train(
            env_or_envs=training_rl_envs,
            test_environments=test_environments,
            test_interval=config_params['rl_params']['test_interval']
        )
    else:
        # Train on single random environment
        single_env = random.choice(training_rl_envs)
        training_results = hierarchical_trainer.train(
            env_or_envs=single_env,
            test_environments=test_environments,
            test_interval=config_params['rl_params']['test_interval']
        )
    
    # Final generalization evaluation
    generalization_results = hierarchical_trainer.evaluate_generalization(test_environments)
    
    return {
        'training_results': training_results,
        'generalization_results': generalization_results
    }

def create_training_performance_table(all_results: Dict[str, Dict]) -> pd.DataFrame:
    """Create training performance table in 2x2 format"""
    
    # Extract training performance from each approach
    # For flat RL, we need to access the trainer's training_history directly
    # since episode_objectives is not returned in the results
    
    # Calculate average performance for each approach
    def get_flat_performance(trainer_result):
        # For flat RL, episode_objectives is missing from return, but episode_twts and makespans are there
        # We'll need to calculate objectives from makespan and twt
        makespans = trainer_result['episode_makespans']
        twts = trainer_result['episode_twts']
        
        # Calculate objectives (assuming objective = makespan + alpha * twt, alpha typically 0.1)
        objectives = [ms + 0.1 * twt for ms, twt in zip(makespans, twts)]
        
        return {
            'avg_objective': np.mean(objectives) if objectives else 0,
            'avg_makespan': np.mean(makespans) if makespans else 0,
            'avg_twt': np.mean(twts) if twts else 0
        }
    
    def get_hierarchical_performance(trainer_result):
        episodes_data = trainer_result['training_history']
        return {
            'avg_objective': np.mean(episodes_data['episode_objectives']) if episodes_data['episode_objectives'] else 0,
            'avg_makespan': np.mean(episodes_data['episode_makespans']) if episodes_data['episode_makespans'] else 0,
            'avg_twt': np.mean(episodes_data['episode_twts']) if episodes_data['episode_twts'] else 0
        }
    
    # Get performance for each approach
    flat_cl = get_flat_performance(all_results['Flat RL (Curriculum)']['training_results'])
    flat_single = get_flat_performance(all_results['Flat RL (Single Env)']['training_results'])
    hier_cl = get_hierarchical_performance(all_results['Hierarchical RL (Curriculum)']['training_results'])
    hier_single = get_hierarchical_performance(all_results['Hierarchical RL (Single Env)']['training_results'])
    
    # Create the 2x2 table
    training_table = pd.DataFrame({
        'Method': ['Flat RL', 'Hierarchical RL'],
        'With CL': [
            f"Obj: {flat_cl['avg_objective']:.2f}, MS: {flat_cl['avg_makespan']:.2f}, TWT: {flat_cl['avg_twt']:.2f}",
            f"Obj: {hier_cl['avg_objective']:.2f}, MS: {hier_cl['avg_makespan']:.2f}, TWT: {hier_cl['avg_twt']:.2f}"
        ],
        'Without CL': [
            f"Obj: {flat_single['avg_objective']:.2f}, MS: {flat_single['avg_makespan']:.2f}, TWT: {flat_single['avg_twt']:.2f}",
            f"Obj: {hier_single['avg_objective']:.2f}, MS: {hier_single['avg_makespan']:.2f}, TWT: {hier_single['avg_twt']:.2f}"
        ]
    })
    
    return training_table

def create_testing_performance_table(all_results: Dict[str, Dict]) -> pd.DataFrame:
    """Create testing performance table in 2x2 format"""
    
    # Extract testing performance from each approach
    flat_cl = all_results['Flat RL (Curriculum)']['generalization_results']['aggregate_stats']
    flat_single = all_results['Flat RL (Single Env)']['generalization_results']['aggregate_stats']
    hier_cl = all_results['Hierarchical RL (Curriculum)']['generalization_results']['aggregate_stats']
    hier_single = all_results['Hierarchical RL (Single Env)']['generalization_results']['aggregate_stats']
    
    # Create the 2x2 table
    testing_table = pd.DataFrame({
        'Method': ['Flat RL', 'Hierarchical RL'],
        'With CL': [
            f"Obj: {flat_cl['avg_objective']:.2f}, MS: {flat_cl['avg_makespan']:.2f}, TWT: {flat_cl['avg_twt']:.2f}",
            f"Obj: {hier_cl['avg_objective']:.2f}, MS: {hier_cl['avg_makespan']:.2f}, TWT: {hier_cl['avg_twt']:.2f}"
        ],
        'Without CL': [
            f"Obj: {flat_single['avg_objective']:.2f}, MS: {flat_single['avg_makespan']:.2f}, TWT: {flat_single['avg_twt']:.2f}",
            f"Obj: {hier_single['avg_objective']:.2f}, MS: {hier_single['avg_makespan']:.2f}, TWT: {hier_single['avg_twt']:.2f}"
        ]
    })
    
    return testing_table

def run_generalization_experiment(training_envs: int, test_envs: int, epochs: int = 20):
    """Main experiment function testing both curriculum and non-curriculum approaches"""
    print("Starting Comprehensive Generalization Experiment...")
    print("Testing: Curriculum Learning vs Single Environment Training")
    print("="*70)
    
    # Create environments
    print("Creating training and test environments...")
    training_environments = create_random_environments(training_envs, seed_base=42)
    test_environments = create_random_environments(test_envs, seed_base=10000)
    
    print(f"Created {len(training_environments)} training environments")
    print(f"Created {len(test_environments)} test environments")
    
    # Convert to RL environments
    training_rl_envs = []
    for data_handler in training_environments:
        rl_env = RLEnv(data_handler, alpha=config.common_rl_params['alpha'], 
                      use_reward_shaping=config.common_rl_params['use_reward_shaping'])
        training_rl_envs.append(rl_env)
    
    # Setup directories
    flat_config = config.get_flat_rl_config()
    hierarchical_config = config.get_hierarchical_rl_config()
    
    # Override epochs with the provided value
    flat_config['rl_params']['epochs'] = epochs
    hierarchical_config['rl_params']['epochs'] = epochs
    
    config.setup_directories(flat_config['result_dirs'])
    config.setup_directories(hierarchical_config['result_dirs'])
    
    # Results storage
    all_results = {}
    
    # ============= EXPERIMENT 1: FLAT RL WITH CURRICULUM LEARNING =============
    print("\n" + "="*70)
    print("EXPERIMENT 1: FLAT RL WITH CURRICULUM LEARNING")
    print("="*70)
    
    # Modify model save directory to avoid conflicts
    flat_config_cl = flat_config.copy()
    flat_config_cl['result_dirs'] = flat_config['result_dirs'].copy()
    flat_config_cl['result_dirs']['model'] = flat_config['result_dirs']['model'].replace('model', 'model_curriculum')
    config.setup_directories(flat_config_cl['result_dirs'])
    
    all_results['Flat RL (Curriculum)'] = train_flat_rl(
        training_rl_envs, test_environments, flat_config_cl, use_curriculum=True
    )
    
    # ============= EXPERIMENT 2: FLAT RL WITH SINGLE ENVIRONMENT =============
    print("\n" + "="*70)
    print("EXPERIMENT 2: FLAT RL WITH SINGLE ENVIRONMENT")
    print("="*70)
    
    # Modify model save directory to avoid conflicts
    flat_config_single = flat_config.copy()
    flat_config_single['result_dirs'] = flat_config['result_dirs'].copy()
    flat_config_single['result_dirs']['model'] = flat_config['result_dirs']['model'].replace('model', 'model_single')
    config.setup_directories(flat_config_single['result_dirs'])
    
    all_results['Flat RL (Single Env)'] = train_flat_rl(
        training_rl_envs, test_environments, flat_config_single, use_curriculum=False
    )
    
    # ============= EXPERIMENT 3: HIERARCHICAL RL WITH CURRICULUM LEARNING =============
    print("\n" + "="*70)
    print("EXPERIMENT 3: HIERARCHICAL RL WITH CURRICULUM LEARNING")
    print("="*70)
    
    # Modify model save directory to avoid conflicts
    hierarchical_config_cl = hierarchical_config.copy()
    hierarchical_config_cl['result_dirs'] = hierarchical_config['result_dirs'].copy()
    hierarchical_config_cl['result_dirs']['model'] = hierarchical_config['result_dirs']['model'].replace('model', 'model_curriculum')
    config.setup_directories(hierarchical_config_cl['result_dirs'])
    
    all_results['Hierarchical RL (Curriculum)'] = train_hierarchical_rl(
        training_rl_envs, test_environments, hierarchical_config_cl, use_curriculum=True
    )
    
    # ============= EXPERIMENT 4: HIERARCHICAL RL WITH SINGLE ENVIRONMENT =============
    print("\n" + "="*70)
    print("EXPERIMENT 4: HIERARCHICAL RL WITH SINGLE ENVIRONMENT")
    print("="*70)
    
    # Modify model save directory to avoid conflicts
    hierarchical_config_single = hierarchical_config.copy()
    hierarchical_config_single['result_dirs'] = hierarchical_config['result_dirs'].copy()
    hierarchical_config_single['result_dirs']['model'] = hierarchical_config['result_dirs']['model'].replace('model', 'model_single')
    config.setup_directories(hierarchical_config_single['result_dirs'])
    
    all_results['Hierarchical RL (Single Env)'] = train_hierarchical_rl(
        training_rl_envs, test_environments, hierarchical_config_single, use_curriculum=False
    )
    
    # ============= RESULTS ANALYSIS =============
    print("\n" + "="*70)
    print("RESULTS ANALYSIS")
    print("="*70)
    
    # Create the two requested tables
    training_table = create_training_performance_table(all_results)
    testing_table = create_testing_performance_table(all_results)
    
    print("\n📊 TRAINING PERFORMANCE (2x2 Table)")
    print("="*70)
    print(training_table.to_string(index=False))
    
    print("\n📊 TESTING PERFORMANCE (2x2 Table)")
    print("="*70)
    print(testing_table.to_string(index=False))
    
    # Save results to CSV
    training_csv_path = 'training_performance_results.csv'
    testing_csv_path = 'testing_performance_results.csv'
    
    training_table.to_csv(training_csv_path, index=False)
    testing_table.to_csv(testing_csv_path, index=False)
    
    print(f"\n💾 Results saved to:")
    print(f"   Training performance: {training_csv_path}")
    print(f"   Testing performance: {testing_csv_path}")
    
    print("\n✅ Experiment completed successfully!")
    
    return all_results, training_table, testing_table

if __name__ == "__main__":
    # Activate conda environment (as per user rules)
    print("Remember to activate conda environment: conda activate dfjs")
    
    # Run the experiment with 800 epochs for full training
    all_results, training_table, testing_table = run_generalization_experiment(training_envs=5, test_envs=30, epochs=800)

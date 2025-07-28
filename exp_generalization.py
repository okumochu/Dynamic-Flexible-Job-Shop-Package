"""
Generalization Experiment for Flat vs Hierarchical RL (Sparse vs Dense Reward)
Tests the generalization capability of different RL approaches on Dynamic Flexible Job Shop Scheduling
"""

import os
import pandas as pd
from typing import List, Dict, Any
from config import config

# Import necessary modules
from benchmarks.static_benchmark.data_handler import FlexibleJobShopDataHandler
from RL.rl_env import RLEnv
from RL.flat_rl_trainer import FlatRLTrainer
from RL.hierarchical_rl_trainer import HierarchicalRLTrainer

def create_random_environments(num_envs: int, seed_base: int = 42) -> List[FlexibleJobShopDataHandler]:
    """Create random environments for testing"""
    environments = []
    sim_params = config.simulation_params
    
    for i in range(num_envs):
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

def train_flat_rl(training_rl_env: RLEnv, test_environments: List[FlexibleJobShopDataHandler], 
                  config_params: Dict, project_name: str = None, run_name: str = None) -> Dict[str, Any]:
    """Train flat RL"""
    
    flat_trainer = FlatRLTrainer(
        env=training_rl_env,
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
        device=config_params['rl_params']['device'],
        model_save_dir=config_params['result_dirs']['model'],
        project_name=project_name or config_params.get('wandb_project', None),
        run_name=run_name
    )
    
    training_results = flat_trainer.train(
        env_or_envs=training_rl_env,
        test_environments=test_environments,
        test_interval=config_params['rl_params']['test_interval']
    )
    
    # Post-training evaluation
    train_generalization_results = flat_trainer.evaluate_generalization([training_rl_env.data_handler])
    test_generalization_results = flat_trainer.evaluate_generalization(test_environments)
    
    return {
        'training_results': training_results,
        'train_generalization_results': train_generalization_results,
        'test_generalization_results': test_generalization_results
    }

def train_hierarchical_rl(training_rl_env: RLEnv, test_environments: List[FlexibleJobShopDataHandler], 
                         config_params: Dict, project_name: str = None, run_name: str = None) -> Dict[str, Any]:
    """Train hierarchical RL"""
    
    hierarchical_trainer = HierarchicalRLTrainer(
        env=training_rl_env,
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
        intrinsic_reward_scale=config_params['rl_params']['intrinsic_reward_scale'],
        device=config_params['rl_params']['device'],
        model_save_dir=config_params['result_dirs']['model'],
        project_name=project_name or config_params.get('wandb_project', None),
        run_name=run_name
    )
    
    training_results = hierarchical_trainer.train(
        env_or_envs=training_rl_env,
        test_environments=test_environments,
        test_interval=config_params['rl_params']['test_interval']
    )
    
    # Post-training evaluation
    train_generalization_results = hierarchical_trainer.evaluate_generalization([training_rl_env.data_handler])
    test_generalization_results = hierarchical_trainer.evaluate_generalization(test_environments)
    
    return {
        'training_results': training_results,
        'train_generalization_results': train_generalization_results,
        'test_generalization_results': test_generalization_results
    }

def run_generalization_experiment(test_envs: int, epochs: int = 20, project_name: str = None, device: str = None):
    """Main experiment function testing sparse/dense reward for flat and hierarchical RL"""
    print("Starting Generalization Experiment (Sparse vs Dense Reward)...")
    print("Testing: Flat vs Hierarchical RL, Sparse vs Dense Reward")
    print("="*70)
    
    # Override device if specified
    if device is not None:
        config.common_rl_params['device'] = device
        print(f"Using device: {device}")
    
    # Create training environment
    print("Creating training and test environments...")
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
    training_data_handler = FlexibleJobShopDataHandler(
        data_source=simulation_params,
        data_type="simulation"
    )
    print("Created 1 training environment")
    
    test_environments = create_random_environments(test_envs, seed_base=10000)
    print(f"Created {len(test_environments)} test environments")
    
    # Set project name
    if project_name is None:
        project_name = f"{config.simulation_params['num_jobs']}jobs"
    
    # Prepare configs
    flat_config_sparse = config.get_flat_rl_config()
    flat_config_dense = config.get_flat_rl_config()
    hrl_config_sparse = config.get_hierarchical_rl_config()
    hrl_config_dense = config.get_hierarchical_rl_config()
    
    # Set reward shaping and epochs
    flat_config_sparse['rl_params']['use_reward_shaping'] = False
    flat_config_dense['rl_params']['use_reward_shaping'] = True
    hrl_config_sparse['rl_params']['use_reward_shaping'] = False
    hrl_config_dense['rl_params']['use_reward_shaping'] = True
    
    for cfg in [flat_config_sparse, flat_config_dense, hrl_config_sparse, hrl_config_dense]:
        cfg['rl_params']['epochs'] = epochs
    
    # Set up result directories
    for cfg, exp_name in [
        (flat_config_sparse, 'flat_sparse'),
        (flat_config_dense, 'flat_dense'),
        (hrl_config_sparse, 'hrl_sparse'),
        (hrl_config_dense, 'hrl_dense')
    ]:
        for k in cfg['result_dirs']:
            rel_path = cfg['result_dirs'][k]
            if rel_path.startswith('result/'):
                rel_path = rel_path[len('result/'):]
            cfg['result_dirs'][k] = os.path.join('result', project_name, exp_name, rel_path)
        config.setup_directories(cfg['result_dirs'])
    
    # Create RL environments
    training_rl_env_sparse = RLEnv(training_data_handler, alpha=config.common_rl_params['alpha'], use_reward_shaping=False)
    training_rl_env_dense = RLEnv(training_data_handler, alpha=config.common_rl_params['alpha'], use_reward_shaping=True)
    
    # Results storage
    all_results = {}
    
    # Run experiments
    experiments = [
        ('flat_sparse', 'FLAT RL (SPARSE REWARD)', training_rl_env_sparse, flat_config_sparse, 'flat_sparse'),
        ('flat_dense', 'FLAT RL (DENSE REWARD)', training_rl_env_dense, flat_config_dense, 'flat_dense'),
        ('hrl_sparse', 'HIERARCHICAL RL (SPARSE REWARD)', training_rl_env_sparse, hrl_config_sparse, 'hrl_sparse'),
        ('hrl_dense', 'HIERARCHICAL RL (DENSE REWARD)', training_rl_env_dense, hrl_config_dense, 'hrl_dense')
    ]
    
    for exp_key, exp_name, env, cfg, run_name in experiments:
        print(f"\n{'='*70}")
        print(f"EXPERIMENT {experiments.index((exp_key, exp_name, env, cfg, run_name)) + 1}: {exp_name}")
        print(f"{'='*70}")
        
        if 'flat' in exp_key:
            all_results[exp_key] = train_flat_rl(env, test_environments, cfg, project_name=project_name, run_name=run_name)
        else:
            all_results[exp_key] = train_hierarchical_rl(env, test_environments, cfg, project_name=project_name, run_name=run_name)
    
    # Results analysis
    print(f"\n{'='*70}")
    print("RESULTS ANALYSIS")
    print(f"{'='*70}")
    
    def get_perf(result):
        stats = result['aggregate_stats']
        return f"Obj: {stats['avg_objective']:.2f}, MS: {stats['avg_makespan']:.2f}, TWT: {stats['avg_twt']:.2f}"
    
    training_table = pd.DataFrame({
        'Method': ['Flat RL', 'Flat RL', 'Hierarchical RL', 'Hierarchical RL'],
        'Reward': ['Sparse', 'Dense', 'Sparse', 'Dense'],
        'Train': [
            get_perf(all_results['flat_sparse']['train_generalization_results']),
            get_perf(all_results['flat_dense']['train_generalization_results']),
            get_perf(all_results['hrl_sparse']['train_generalization_results']),
            get_perf(all_results['hrl_dense']['train_generalization_results'])
        ]
    })
    
    testing_table = pd.DataFrame({
        'Method': ['Flat RL', 'Flat RL', 'Hierarchical RL', 'Hierarchical RL'],
        'Reward': ['Sparse', 'Dense', 'Sparse', 'Dense'],
        'Test': [
            get_perf(all_results['flat_sparse']['test_generalization_results']),
            get_perf(all_results['flat_dense']['test_generalization_results']),
            get_perf(all_results['hrl_sparse']['test_generalization_results']),
            get_perf(all_results['hrl_dense']['test_generalization_results'])
        ]
    })
    
    print("\n📊 TRAINING PERFORMANCE")
    print("="*70)
    print(training_table.to_string(index=False))
    
    print("\n📊 TESTING PERFORMANCE")
    print("="*70)
    print(testing_table.to_string(index=False))
    
    # Save results
    result_project_dir = os.path.join('result', project_name)
    os.makedirs(result_project_dir, exist_ok=True)
    training_csv_path = os.path.join(result_project_dir, 'training_performance_results.csv')
    testing_csv_path = os.path.join(result_project_dir, 'testing_performance_results.csv')
    
    training_table.to_csv(training_csv_path, index=False)
    testing_table.to_csv(testing_csv_path, index=False)
    
    print(f"\n💾 Results saved to:")
    print(f"   Training performance: {training_csv_path}")
    print(f"   Testing performance: {testing_csv_path}")
    
    print("\n✅ Experiment completed successfully!")
    
    return all_results, training_table, testing_table

if __name__ == "__main__":
    print("Remember to activate conda environment: conda activate dfjs")

    import argparse
    parser = argparse.ArgumentParser(description="Run RL generalization experiments (sparse/dense, flat/HRL)")
    parser.add_argument('--project_name', type=str, default=None, help='Project name for results and wandb (default: <num_jobs>jobs)')
    parser.add_argument('--epochs', type=int, default=400, help='Number of epochs for each experiment (default: 500)')
    parser.add_argument('--test_envs', type=int, default=30, help='Number of test environments (default: 30)')
    parser.add_argument('--device', type=str, default=None, help='Device for training (auto, cpu, cuda, etc.)')
    args = parser.parse_args()

    all_results, training_table, testing_table = run_generalization_experiment(
        test_envs=args.test_envs,
        epochs=args.epochs,
        project_name=args.project_name,
        device=args.device
    )

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from RL.rl_env import RLEnv
from RL.hierarchical_rl_trainer import HierarchicalRLTrainer
from benchmarks.static_benchmark.data_handler import FlexibleJobShopDataHandler
import wandb
from utils.policy_utils import showcase_hierarchical_policy, create_gantt_chart, evaluate_hierarchical_policy, visualize_policy_schedule
from config import config


def run_hierarchical_rl_experiment():
    """Run hierarchical RL experiment with centralized configuration"""
    
    print("="*60)
    print("🏗️  HIERARCHICAL REINFORCEMENT LEARNING EXPERIMENT")
    print("📋 Manager-Worker Architecture for Job Shop Scheduling")
    print("="*60)
    
    # Use device from config (used via trainer params)
        
    print("\n" + "="*50)
    print("Starting hierarchical RL experiment...")
    print("="*50)
    
    # Get configuration from config
    exp_config = config.get_hierarchical_rl_config()
    simulation_params = exp_config['simulation_params']
    hrl_params = exp_config['rl_params']
    result_dirs = exp_config['result_dirs']
    
    # Setup directories and wandb
    config.setup_directories(result_dirs)
    config.setup_wandb_env(result_dirs['training_process'], exp_config['wandb_project'])
    
    print("Creating data handler and hierarchical environment...")
    data_handler = FlexibleJobShopDataHandler(data_source=simulation_params, data_type="simulation")
    env = RLEnv(data_handler, alpha=hrl_params['alpha'], use_reward_shaping=hrl_params['use_reward_shaping'])
    
    print(f"Hierarchical environment created: {env.num_jobs} jobs, {env.num_machines} machines")
    print(f"Observation dimension: {env.obs_len}")
    print(f"Action dimension: {env.action_dim}")
    print(f"Goal duration ratio: {hrl_params['goal_duration_ratio']}")
    print(f"Latent dimension: {hrl_params['latent_dim']}")
    print(f"Goal dimension: {hrl_params['goal_dim']}")
    
    # Create hierarchical trainer with all parameters from config
    trainer = HierarchicalRLTrainer(
        env=env,
        epochs=hrl_params['epochs'],
        episodes_per_epoch=hrl_params['episodes_per_epoch'],
        goal_duration_ratio=hrl_params['goal_duration_ratio'],
        latent_dim=hrl_params['latent_dim'],
        goal_dim=hrl_params['goal_dim'],
        manager_lr=hrl_params['manager_lr'],
        worker_lr=hrl_params['worker_lr'],
        intrinsic_reward_scale=hrl_params['intrinsic_reward_scale'],
        gamma_manager=hrl_params['gamma_manager'],
        gamma_worker=hrl_params['gamma_worker'],
        gae_lambda=hrl_params['gae_lambda'],
        clip_ratio=hrl_params['clip_ratio'],
        entropy_coef=hrl_params['entropy_coef'],
        train_per_episode=hrl_params['train_per_episode'],
        device=hrl_params['device'],
        project_name=exp_config['wandb_project'],
        model_save_dir=result_dirs['model']
    )
    
    print(f"Starting hierarchical training for {hrl_params['epochs']} epochs...")
    results = trainer.train()
    print("Hierarchical training complete.")

    print(f"Training time: {results['training_time']:.2f} seconds")
    print(f"Model saved in {result_dirs['model']}, wandb logs in {result_dirs['training_process']}.")

    # Deterministic evaluation on training environment
    print("\nEvaluating trained hierarchical policy (deterministic) on training environment...")
    model_path = os.path.join(result_dirs['model'], results['model_filename'])
    evaluation_result = evaluate_hierarchical_policy(model_path, env, num_episodes=1)
    print(f"Final objective: {evaluation_result.get('objective', 0):.2f}")
    print(f"Final makespan: {evaluation_result.get('makespan', 0):.2f}")
    print(f"Final TWT: {evaluation_result.get('twt', 0):.2f}")

    # Save evaluation summary CSV similar format
    import pandas as pd
    os.makedirs(result_dirs['training_process'], exist_ok=True)
    reward_label = "Dense" if hrl_params['use_reward_shaping'] else "Sparse"
    run_name_hrl = trainer.run_name if hasattr(trainer, 'run_name') and trainer.run_name else 'hrl'
    df = pd.DataFrame([
        {
            'method': f"Hierarchical RL ({reward_label})",
            'objective': evaluation_result.get('objective', 0.0),
            'makespan': evaluation_result.get('makespan', 0.0),
            'twt': evaluation_result.get('twt', 0.0),
            'run_name': run_name_hrl
        }
    ])
    csv_path = os.path.join(result_dirs['training_process'], 'hrl_results.csv')
    df.to_csv(csv_path, index=False)
    print(f"Saved evaluation CSV to {csv_path}")
    
    print("\nHierarchical Evaluation Results:")
    if 'avg_reward' in evaluation_result:
        print(f"  Average Reward: {evaluation_result['avg_reward']:.2f} ± {evaluation_result['std_reward']:.2f}")
        print(f"  Average Makespan: {evaluation_result['avg_makespan']:.2f} ± {evaluation_result['std_makespan']:.2f}")
        print(f"  Average TWT: {evaluation_result['avg_twt']:.2f} ± {evaluation_result['std_twt']:.2f}")
    else:
        print(f"  Single Episode - Reward: {evaluation_result['episode_reward']:.2f}")
        print(f"  Single Episode - Makespan: {evaluation_result['makespan']:.2f}")
        print(f"  Single Episode - TWT: {evaluation_result['twt']:.2f}")
    
    # Visualize using policy_utils
    print("Creating Gantt chart using policy_utils...")
    gantt_save_path_evaluation = os.path.join(result_dirs['training_process'], "hierarchical_gantt_evaluation.png")
    visualize_policy_schedule(evaluation_result, env, save_path=gantt_save_path_evaluation)
    
    # Also showcase using hierarchical policy showcase function
    print("Creating Gantt chart using hierarchical showcase function...")
    gantt_save_path_showcaser = os.path.join(result_dirs['training_process'], "hierarchical_gantt_showcase.png")
    showcaser_result = showcase_hierarchical_policy(model_path=result_dirs['model'], env=env)
    
    # Create Gantt chart separately
    create_gantt_chart(showcaser_result, save_path=gantt_save_path_showcaser, title_suffix="Hierarchical RL")
    
    print("\nHierarchical Showcase Results:")
    print(f"  Makespan: {showcaser_result['makespan']:.2f}")
    print(f"  TWT: {showcaser_result['twt']:.2f}")
    print(f"  Total Reward: {showcaser_result['total_reward']:.2f}")
    print(f"  Steps Taken: {showcaser_result['steps_taken']}")
    print(f"  Valid Completion: {showcaser_result['is_valid_completion']}")
    
    print(f"\nHierarchical RL experiment completed successfully!")
    print(f"Results saved in: {result_dirs['training_process']}")
    
    print(f"\n" + "="*50)
    print("HIERARCHICAL RL EXPERIMENT SUMMARY")
    print("="*50)
    print(f"Training Configuration:")
    print(f"  Epochs: {hrl_params['epochs']}")
    print(f"  Episodes per Epoch: {hrl_params['episodes_per_epoch']}")
    print(f"  Goal Duration Ratio: {hrl_params['goal_duration_ratio']}")
    print(f"  Latent Dimension: {hrl_params['latent_dim']}")
    print(f"  Goal Dimension: {hrl_params['goal_dim']}")
    print(f"  Manager LR: {hrl_params['manager_lr']}")
    print(f"  Worker LR: {hrl_params['worker_lr']}")
    print(f"  Intrinsic Reward Scale: {hrl_params['intrinsic_reward_scale']}")
    print(f"  Gamma Manager: {hrl_params['gamma_manager']}")
    print(f"  Gamma Worker: {hrl_params['gamma_worker']}")
    print(f"  GAE Lambda: {hrl_params['gae_lambda']}")
    print(f"  Clip Ratio: {hrl_params['clip_ratio']}")
    print(f"  Entropy Coef: {hrl_params['entropy_coef']}")
    print(f"  Train per Episode: {hrl_params['train_per_episode']}")
    print(f"  Device: {hrl_params['device']}")
    print(f"  Use Reward Shaping: {hrl_params['use_reward_shaping']}")
    print(f"  Alpha (TWT Weight): {hrl_params['alpha']}")
    
    return results
        
    
    return results


def main():
    """Main function to run hierarchical RL experiment"""
    run_hierarchical_rl_experiment()


if __name__ == "__main__":
    main()

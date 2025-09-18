import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from RL.rl_env import RLEnv
from RL.hierarchical_rl_trainer import HierarchicalRLTrainer
from benchmarks.data_handler import FlexibleJobShopDataHandler
import wandb
# Evaluation utilities removed - only training code kept
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
    result_dir = exp_config['result_dir']
    
    # Setup wandb
    config.setup_wandb_env(result_dir, exp_config['wandb_project'])
    
    print("Creating data handler and hierarchical environment...")
    data_handler = FlexibleJobShopDataHandler(
        data_source=simulation_params, 
        data_type="simulation",
        TF=simulation_params['TF'],
        RDD=simulation_params['RDD'],
        seed=simulation_params['seed']
    )
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
        run_name=hrl_params['wandb_run_name'],
        model_save_dir=result_dir,
        seed=hrl_params['seed']
    )
    
    print(f"Starting hierarchical training for {hrl_params['epochs']} epochs...")
    results = trainer.train(seed=hrl_params['seed'])
    print("Hierarchical training complete.")

    print(f"Training time: {results['training_time']:.2f} seconds")
    print(f"Model saved in {result_dir}, wandb logs in {result_dir}.")
    
    print(f"\nHierarchical RL experiment completed successfully!")
    print(f"Results saved in: {result_dir}")
    
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


def main():
    """Main function to run hierarchical RL experiment"""
    run_hierarchical_rl_experiment()


if __name__ == "__main__":
    main()

import os
import torch
from RL.rl_env import RLEnv
from RL.flat_rl_trainer import FlatRLTrainer
from benchmarks.data_handler import FlexibleJobShopDataHandler
import wandb
# Evaluation utilities removed - only training code kept
from config import config


def run_flat_rl_experiment():
    """Run flat RL experiment with centralized configuration"""
    
    print("\n" + "="*50)
    print("🤖 FLAT REINFORCEMENT LEARNING EXPERIMENT")
    print("📋 Single-Agent PPO for Job Shop Scheduling")
    print("="*50)
    
    # Use device from config (used via trainer params)
    
    # Get configuration from config
    exp_config = config.get_flat_rl_config()
    simulation_params = exp_config['simulation_params']
    rl_params = exp_config['rl_params']
    result_dir = exp_config['result_dir']
    
    # Setup wandb
    config.setup_wandb_env(result_dir, exp_config['wandb_project'])
    
    print("Creating data handler and environment...")
    data_handler = FlexibleJobShopDataHandler(
        data_source=simulation_params, 
        data_type="simulation",
        TF=simulation_params['TF'],
        RDD=simulation_params['RDD'],
        seed=simulation_params['seed']
    )
    env = RLEnv(data_handler, alpha=rl_params['alpha'], use_reward_shaping=rl_params['use_reward_shaping'])
    
    print(f"Environment created: {env.num_jobs} jobs, {env.num_machines} machines")
    print(f"Observation dimension: {env.obs_len}")
    print(f"Action dimension: {env.action_dim}")
    
    # Create trainer with all parameters from config
    trainer = FlatRLTrainer(
        env=env,
        epochs=rl_params['epochs'],
        episodes_per_epoch=rl_params['episodes_per_epoch'],
        train_per_episode=rl_params['train_per_episode'],
        pi_lr=rl_params['pi_lr'],
        v_lr=rl_params['v_lr'],
        gamma=rl_params['gamma'],
        gae_lambda=rl_params['gae_lambda'],
        clip_ratio=rl_params['clip_ratio'],
        entropy_coef=rl_params['entropy_coef'],
        device=rl_params['device'],
        project_name=exp_config['wandb_project'],
        run_name=rl_params['wandb_run_name'],
        model_save_dir=result_dir,
        seed=rl_params['seed']
    )
    
    print(f"Starting training for {rl_params['epochs']} epochs...")
    results = trainer.train(seed=rl_params['seed'])
    print("Training complete.")
        
    print(f"Model saved in {result_dir}, wandb logs in {result_dir}.")
    
    print(f"\nFlat RL experiment completed successfully!")
    print(f"Results saved in: {result_dir}")
    
    print(f"\n" + "="*50)
    print("FLAT RL EXPERIMENT SUMMARY")
    print("="*50)
    print(f"Training Configuration:")
    print(f"  Epochs: {rl_params['epochs']}")
    print(f"  Episodes per Epoch: {rl_params['episodes_per_epoch']}")
    print(f"  Train per Episode: {rl_params['train_per_episode']}")
    print(f"  Policy LR: {rl_params['pi_lr']}")
    print(f"  Value LR: {rl_params['v_lr']}")
    print(f"  Gamma: {rl_params['gamma']}")
    print(f"  GAE Lambda: {rl_params['gae_lambda']}")
    print(f"  Clip Ratio: {rl_params['clip_ratio']}")
    print(f"  Entropy Coef: {rl_params['entropy_coef']}")
    print(f"  Device: {rl_params['device']}")
    print(f"  Use Reward Shaping: {rl_params['use_reward_shaping']}")
    print(f"  Alpha (TWT Weight): {rl_params['alpha']}")
    
    return results


def main():
    """Main function to run flat RL experiment"""
    run_flat_rl_experiment()


if __name__ == "__main__":
    main()

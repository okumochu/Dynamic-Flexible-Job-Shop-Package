import os
import torch
from RL.rl_env import RLEnv
from RL.flat_rl_trainer import FlatRLTrainer
from benchmarks.static_benchmark.data_handler import FlexibleJobShopDataHandler
import wandb
from utils.policy_utils import showcase_flat_policy, create_gantt_chart, evaluate_flat_policy, visualize_policy_schedule
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
    result_dirs = exp_config['result_dirs']
    
    # Setup directories and wandb
    config.setup_directories(result_dirs)
    config.setup_wandb_env(result_dirs['training_process'], exp_config['wandb_project'])
    
    print("Creating data handler and environment...")
    data_handler = FlexibleJobShopDataHandler(data_source=simulation_params, data_type="simulation")
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
        model_save_dir=result_dirs['model']
    )
    
    print(f"Starting training for {rl_params['epochs']} epochs...")
    results = trainer.train()
    print("Training complete.")
        
    print(f"Model saved in {result_dirs['model']}, wandb logs in {result_dirs['training_process']}.")
    
    # Deterministic evaluation on training environment
    print("\nEvaluating trained policy (deterministic) on training environment...")
    model_path = os.path.join(result_dirs['model'], results['model_filename'])
    evaluation_result = evaluate_flat_policy(model_path, env, num_episodes=1)
    print(f"Final objective: {evaluation_result.get('objective', 0):.2f}")
    print(f"Final makespan: {evaluation_result.get('makespan', 0):.2f}")
    print(f"Final TWT: {evaluation_result.get('twt', 0):.2f}")

    # Save evaluation summary CSV similar to mk06_results_dense.csv
    import pandas as pd
    os.makedirs(result_dirs['training_process'], exist_ok=True)
    reward_label = "Dense" if rl_params['use_reward_shaping'] else "Sparse"
    run_name_flat = trainer.run_name if hasattr(trainer, 'run_name') and trainer.run_name else 'flat'
    df = pd.DataFrame([
        {
            'method': f"Flat RL ({reward_label})",
            'objective': evaluation_result.get('objective', 0.0),
            'makespan': evaluation_result.get('makespan', 0.0),
            'twt': evaluation_result.get('twt', 0.0),
            'run_name': run_name_flat
        }
    ])
    csv_path = os.path.join(result_dirs['training_process'], 'flat_results.csv')
    df.to_csv(csv_path, index=False)
    print(f"Saved evaluation CSV to {csv_path}")
    
    # Visualize using policy_utils
    print("Creating Gantt chart using policy_utils...")
    gantt_save_path_trainer = os.path.join(result_dirs['training_process'], "gantt_evaluation.png")
    visualize_policy_schedule(evaluation_result, env, save_path=gantt_save_path_trainer)
    
    # Also showcase using flat policy showcase function
    print("Creating Gantt chart using showcase function...")
    gantt_save_path_showcaser = os.path.join(result_dirs['training_process'], "gantt_showcase.png")
    model_path = os.path.join(result_dirs['model'], "final_model.pth")
    showcaser_result = showcase_flat_policy(model_path=model_path, env=env)
    
    # Create Gantt chart separately
    create_gantt_chart(showcaser_result, save_path=gantt_save_path_showcaser, title_suffix="Flat RL")
    
    print("\nShowcase Results:")
    print(f"  Makespan: {showcaser_result['makespan']:.2f}")
    print(f"  TWT: {showcaser_result['twt']:.2f}")
    print(f"  Total Reward: {showcaser_result['total_reward']:.2f}")
    print(f"  Steps Taken: {showcaser_result['steps_taken']}")
    print(f"  Valid Completion: {showcaser_result['is_valid_completion']}")
    
    print(f"\nFlat RL experiment completed successfully!")
    print(f"Results saved in: {result_dirs['training_process']}")
    
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

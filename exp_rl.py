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
    
    # Get configuration
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
    
    # Create trainer with all parameters
    trainer = FlatRLTrainer(
        env=env,
        epochs=rl_params['epochs'],
        steps_per_epoch=rl_params['steps_per_epoch'],
        train_pi_iters=rl_params['train_pi_iters'],
        train_v_iters=rl_params['train_v_iters'],
        pi_lr=rl_params['pi_lr'],
        v_lr=rl_params['v_lr'],
        gamma=rl_params['gamma'],
        gae_lambda=rl_params['gae_lambda'],
        model_save_dir=result_dirs['model']
    )
    
    print(f"Starting training for {rl_params['epochs']} epochs...")
    try:
        results = trainer.train()
        print("Training complete.")
        
        # Print training statistics
        if results['episode_rewards']:
            recent_rewards = results['episode_rewards'][-10:]
            print(f"Final average reward: {sum(recent_rewards) / len(recent_rewards):.2f}")
        if results['episode_makespans']:
            recent_makespans = results['episode_makespans'][-10:]
            print(f"Final average makespan: {sum(recent_makespans) / len(recent_makespans):.2f}")
        if results['episode_twts']:
            recent_twts = results['episode_twts'][-10:]
            print(f"Final average TWT: {sum(recent_twts) / len(recent_twts):.2f}")
        
        print(f"Model saved in {result_dirs['model']}, wandb logs in {result_dirs['training_process']}.")
        
        # Evaluate the trained policy using policy_utils
        print("\nEvaluating trained policy...")
        model_path = os.path.join(result_dirs['model'], results['model_filename'])
        evaluation_result = evaluate_flat_policy(model_path, env, num_episodes=1)
        
        # Visualize using policy_utils
        print("Creating Gantt chart using policy_utils...")
        gantt_save_path_trainer = os.path.join(result_dirs['training_process'], "gantt_evaluation.png")
        visualize_policy_schedule(evaluation_result, env, save_path=gantt_save_path_trainer)
        
        # Also showcase using flat policy showcase function
        print("Creating Gantt chart using showcase function...")
        try:
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
            
        except Exception as e:
            print(f"Showcase function failed: {e}")
            print("Using trainer evaluation result instead.")
        
        print(f"\nFlat RL experiment completed successfully!")
        print(f"Results saved in: {result_dirs['training_process']}")
        
        print(f"\n" + "="*50)
        print("FLAT RL EXPERIMENT SUMMARY")
        print("="*50)
        print(f"Training Configuration:")
        print(f"  Epochs: {rl_params['epochs']}")
        print(f"  Steps per Epoch: {rl_params['steps_per_epoch']}")
        print(f"  Policy LR: {rl_params['pi_lr']}")
        print(f"  Value LR: {rl_params['v_lr']}")
        print(f"  Gamma: {rl_params['gamma']}")
        print(f"Final Performance:")
        print(f"  Makespan: {evaluation_result['makespan']:.2f}")
        print(f"  TWT: {evaluation_result['twt']:.2f}")
        print(f"  Total Reward: {evaluation_result['episode_reward']:.2f}")
        print(f"Training Time: {results['training_time']:.2f}s")
        
        return results, evaluation_result
        
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def main():
    """Main function to run flat RL experiment"""
    run_flat_rl_experiment()


if __name__ == "__main__":
    main()

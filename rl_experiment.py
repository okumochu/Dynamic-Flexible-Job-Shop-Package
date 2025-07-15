import os
import torch
from RL.rl_env import RLEnv
from RL.flat_rl_trainer import FlatRLTrainer
from benchmarks.static_benchmark.data_handler import FlexibleJobShopDataHandler
import wandb
from utils.policy_utils import showcase_flat_policy, create_gantt_chart

# Set wandb output directory
training_process_dir = "result/flat_rl/training_process"
model_dir = "result/flat_rl/model"
os.environ["WANDB_DIR"] = training_process_dir


def main():
    
    print("\n" + "="*50)
    print("Starting main experiment...")
    print("="*50)
    
    # Create directories if they don't exist
    os.makedirs(training_process_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    # Example: generate a synthetic problem instance with correct params
    simulation_params = {
        'num_jobs': 12,
        'num_machines': 4,
        'operation_lb': 4,
        'operation_ub': 4,
        'processing_time_lb': 5,
        'processing_time_ub': 5,   
        'compatible_machines_lb': 2,
        'compatible_machines_ub': 2,
        'seed': 42,
    }
    total_max_steps = simulation_params['num_jobs'] * simulation_params['operation_ub'] * simulation_params['num_machines']
    rl_params = {
        'alpha': 0.5,
        'gamma': 0.99,
        'gae_lambda': 0.9,
        'steps_per_epoch': total_max_steps,
        'epochs': 500,  
        'pi_lr': 1e-5,  # Reduced from 3e-5
        'v_lr': 1e-5,   # Reduced from 1e-4
        'target_kl': 0.5,  # Increased from 0.1
        'train_pi_iters': total_max_steps,
        'train_v_iters': total_max_steps,
    }
    
    print("Creating data handler and environment...")
    data_handler = FlexibleJobShopDataHandler(data_source=simulation_params, data_type="simulation")
    env = RLEnv(data_handler, alpha=rl_params['alpha'])
    
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
        target_kl=rl_params['target_kl'],
        pi_lr=rl_params['pi_lr'],
        v_lr=rl_params['v_lr'],
        gamma=rl_params['gamma'],
        gae_lambda=rl_params['gae_lambda'],
        model_save_dir=model_dir
    )
    
    print(f"Starting training for {rl_params['epochs']} epochs...")
    try:
        results = trainer.train()
        print("Training complete.")
        
        # Print training statistics
        if results['episode_rewards']:
            print(f"Final average reward: {sum(results['episode_rewards'][-10:]) / min(10, len(results['episode_rewards'])):.2f}")
        if results['episode_makespans']:
            print(f"Final average makespan: {sum(results['episode_makespans'][-10:]) / min(10, len(results['episode_makespans'])):.2f}")
        if results['episode_twts']:
            print(f"Final average TWT: {sum(results['episode_twts'][-10:]) / min(10, len(results['episode_twts'])):.2f}")
        
        print(f"Model saved in {model_dir}, wandb logs in {training_process_dir}.")
        
        # Evaluate the trained policy
        print("\nEvaluating trained policy...")
        evaluation_result = trainer.evaluate()
        
        # Visualize using trainer's built-in method
        print("Creating Gantt chart using trainer...")
        gantt_save_path_trainer = os.path.join(training_process_dir, "gantt_trainer.png")
        trainer.visualize_schedule(evaluation_result, save_path=gantt_save_path_trainer)
        
        # Also showcase using flat policy showcase function
        print("Creating Gantt chart using showcase function...")
        try:
            gantt_save_path_showcaser = os.path.join(training_process_dir, "gantt_showcase.png")
            showcaser_result = showcase_flat_policy(model_dir=model_dir, env=env)
            
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
        
        print(f"\nExperiment completed successfully!")
        print(f"Results saved in: {training_process_dir}")
        
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return

if __name__ == "__main__":
    main()

import os
import torch
import wandb
import itertools
from RL.flat_rl_env import FlatRLEnv
from RL.flat_rl_trainer import FlatRLTrainer
from benchmarks.static_benchmark.data_handler import FlexibleJobShopDataHandler

# Set wandb output directory
training_process_dir = "result/flat_rl/hyperparameter_tuning"
model_dir = "result/flat_rl/hyperparameter_models"
os.environ["WANDB_DIR"] = training_process_dir

def test_environment_setup():
    """Test basic environment setup."""
    print("\nTesting environment setup...")
    
    try:
        # Create a minimal test environment
        test_params = {
            'num_jobs': 2, 
            'num_machines': 2,
            'operation_lb': 1,
            'operation_ub': 2,
            'processing_time_lb': 1,
            'processing_time_ub': 3,   
            'compatible_machines_lb': 1,
            'compatible_machines_ub': 2,
            'seed': 42,
        }
        
        data_handler = FlexibleJobShopDataHandler(data_source=test_params, data_type="simulation")
        env = FlatRLEnv(data_handler, alpha=0.0)
        
        # Test basic environment functions
        obs, _ = env.reset()
        print(f"✓ Environment reset successful, obs shape: {obs.shape}")
        
        action_mask = env.get_action_mask()
        print(f"✓ Action mask generated, valid actions: {action_mask.sum().item()}")
        
        if action_mask.any():
            valid_action = int(torch.where(action_mask)[0][0].item())
            obs, reward, terminated, truncated, info = env.step(valid_action)
            print(f"✓ Environment step successful, reward: {reward:.2f}")
        
        print("✓ Environment test passed")
        return True
        
    except Exception as e:
        print(f"✗ Environment test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_single_experiment(config, epochs=50, test_run=True):
    """Run a single hyperparameter experiment."""
    
    # Fixed simulation parameters as requested
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
    
    # Base RL parameters
    rl_params = {
        'alpha': 0.5,
        'gamma': 0.99,
        'gae_lambda': config['gae_lambda'],
        'steps_per_epoch': 50,
        'epochs': epochs,  
        'pi_lr': config['pi_lr'],
        'v_lr': 1e-5,
        'target_kl': 0.5,
        'train_pi_iters': config['train_pi_iters'],
        'train_v_iters': 50,
    }
    
    # Initialize wandb
    run_name = f"pi_lr_{config['pi_lr']}_train_pi_{config['train_pi_iters']}_gae_{config['gae_lambda']}"
    if test_run:
        run_name = f"TEST_{run_name}"
    
    wandb.init(
        project="fjsp-hyperparameter-tuning",
        name=run_name,
        config={**simulation_params, **rl_params, **config},
        reinit=True
    )
    
    try:
        print(f"\nRunning experiment: {run_name}")
        print(f"Config: pi_lr={config['pi_lr']}, train_pi_iters={config['train_pi_iters']}, gae_lambda={config['gae_lambda']}")
        
        # Create data handler and environment
        data_handler = FlexibleJobShopDataHandler(data_source=simulation_params, data_type="simulation")
        env = FlatRLEnv(data_handler, alpha=rl_params['alpha'])
        
        # Create model save directory for this specific config
        config_model_dir = os.path.join(model_dir, run_name)
        os.makedirs(config_model_dir, exist_ok=True)
        
        # Create trainer
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
            model_save_dir=config_model_dir
        )
        
        # Train the model
        results = trainer.train()
        
        # Calculate final metrics (last 10 episodes average)
        final_reward = sum(results['episode_rewards'][-10:]) / min(10, len(results['episode_rewards'])) if results['episode_rewards'] else 0
        final_makespan = sum(results['episode_makespans'][-10:]) / min(10, len(results['episode_makespans'])) if results['episode_makespans'] else 0
        final_twt = sum(results['episode_twts'][-10:]) / min(10, len(results['episode_twts'])) if results['episode_twts'] else 0
        
        # Evaluate the trained policy
        evaluation_result = trainer.evaluate()
        
        # Log final results to wandb
        wandb.log({
            "final_avg_reward": final_reward,
            "final_avg_makespan": final_makespan,
            "final_avg_twt": final_twt,
            "eval_makespan": evaluation_result['makespan'],
            "eval_twt": evaluation_result['twt'],
            "eval_episode_reward": evaluation_result['episode_reward'],
            "eval_steps": evaluation_result['steps_taken'],
            "eval_valid_completion": evaluation_result['is_valid_completion']
        })
        
        print(f"✓ Experiment completed successfully!")
        print(f"  Final avg reward: {final_reward:.2f}")
        print(f"  Final avg makespan: {final_makespan:.2f}")
        print(f"  Final avg TWT: {final_twt:.2f}")
        print(f"  Eval makespan: {evaluation_result['makespan']:.2f}")
        
        return {
            'config': config,
            'final_reward': final_reward,
            'final_makespan': final_makespan,
            'final_twt': final_twt,
            'eval_makespan': evaluation_result['makespan'],
            'eval_twt': evaluation_result['twt'],
            'eval_episode_reward': evaluation_result['episode_reward'],
            'success': True
        }
        
    except Exception as e:
        print(f"✗ Experiment failed: {e}")
        # Only log to wandb if wandb is still active
        try:
            wandb.log({"error": str(e), "success": False})
        except:
            pass  # wandb might not be active
        import traceback
        traceback.print_exc()
        return {
            'config': config,
            'error': str(e),
            'success': False
        }
    finally:
        try:
            wandb.finish()
        except:
            pass  # wandb might not be active

def main():
    # Test environment setup first
    if not test_environment_setup():
        print("\nEnvironment test failed! Exiting.")
        return
    
    print("\n" + "="*50)
    print("Starting Hyperparameter Tuning Experiment")
    print("="*50)
    
    # Create directories
    os.makedirs(training_process_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    # Define hyperparameter grid
    hyperparameters = {
        'pi_lr': [1e-5, 2e-5],
        'train_pi_iters': [25, 50],
        'gae_lambda': [0.9, 0.95, 0.97]
    }
    
    # Generate all combinations
    keys = hyperparameters.keys()
    values = hyperparameters.values()
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    print(f"Total combinations to test: {len(combinations)}")
    print("Combinations:")
    for i, combo in enumerate(combinations, 1):
        print(f"  {i:2d}. pi_lr={combo['pi_lr']}, train_pi_iters={combo['train_pi_iters']}, gae_lambda={combo['gae_lambda']}")
    
    # Ask user whether to run test first
    print(f"\nConfiguration:")
    print(f"  - Will run with 50 epochs first for testing")
    print(f"  - Then can run with 1000 epochs for full training")
    
    # Run test experiments first (50 epochs)
    print(f"\n{'='*20} PHASE 1: TEST RUNS (50 epochs) {'='*20}")
    test_results = []
    
    for i, config in enumerate(combinations, 1):
        print(f"\nRunning test {i}/{len(combinations)}")
        result = run_single_experiment(config, epochs=50, test_run=True)
        test_results.append(result)
        
        if not result['success']:
            print(f"Test {i} failed, continuing with next configuration...")
            continue
    
    # Summary of test results
    successful_tests = [r for r in test_results if r['success']]
    print(f"\n{'='*20} TEST PHASE SUMMARY {'='*20}")
    print(f"Successful tests: {len(successful_tests)}/{len(combinations)}")
    
    if len(successful_tests) == 0:
        print("No tests were successful! Please check the configuration.")
        return
    
    # Ask user if they want to proceed with full training
    print(f"\nTest phase completed successfully!")
    print(f"Do you want to proceed with full training (1000 epochs)?")
    user_input = input("Enter 'y' or 'yes' to continue with full training: ").lower().strip()
    
    if user_input in ['y', 'yes']:
        print(f"\n{'='*20} PHASE 2: FULL TRAINING (1000 epochs) {'='*20}")
        final_results = []
        
        for i, config in enumerate(combinations, 1):
            print(f"\nRunning full training {i}/{len(combinations)}")
            result = run_single_experiment(config, epochs=1000, test_run=False)
            final_results.append(result)
            
            if not result['success']:
                print(f"Training {i} failed, continuing with next configuration...")
                continue
        
        # Final summary
        successful_final = [r for r in final_results if r['success']]
        print(f"\n{'='*20} FINAL RESULTS SUMMARY {'='*20}")
        print(f"Successful trainings: {len(successful_final)}/{len(combinations)}")
        
        if successful_final:
            # Find best configuration by evaluation makespan (lower is better)
            best_result = min(successful_final, key=lambda x: x['eval_makespan'])
            print(f"\nBest configuration (by eval makespan):")
            print(f"  pi_lr: {best_result['config']['pi_lr']}")
            print(f"  train_pi_iters: {best_result['config']['train_pi_iters']}")
            print(f"  gae_lambda: {best_result['config']['gae_lambda']}")
            print(f"  Eval makespan: {best_result['eval_makespan']:.2f}")
            print(f"  Eval TWT: {best_result['eval_twt']:.2f}")
        
        print(f"\nHyperparameter tuning completed!")
        print(f"Results and models saved in: {model_dir}")
        print(f"Wandb logs saved in: {training_process_dir}")
    else:
        print("Full training skipped. Test results are available in wandb.")

if __name__ == "__main__":
    main() 
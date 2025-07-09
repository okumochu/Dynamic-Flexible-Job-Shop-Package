import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from RL.hierarchical_rl_env import HierarchicalRLEnv
from RL.hierarchical_rl_trainer import HierarchicalRLTrainer
from benchmarks.static_benchmark.data_handler import FlexibleJobShopDataHandler
import wandb
from utils.policy_utils import showcase_hierarchical_policy, create_gantt_chart

# Set wandb output directory and configuration
training_process_dir = "result/hierarchical_rl/training_process"
model_dir = "result/hierarchical_rl/model"
os.environ["WANDB_DIR"] = training_process_dir

# Configure wandb settings
os.environ["WANDB_PROJECT"] = "Hierarchical-Job-Shop-RL"
os.environ["WANDB_SILENT"] = "true"  # Reduce wandb output verbosity


def test_environment_setup():
    """Test basic hierarchical environment setup."""
    print("\nTesting hierarchical environment setup...")
    
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
        env = HierarchicalRLEnv(data_handler, alpha=0.0)
        
        # Test basic environment functions
        obs, _ = env.reset()
        print(f"✓ Environment reset successful, obs shape: {obs.shape}")
        
        action_mask = env.get_action_mask()
        print(f"✓ Action mask generated, valid actions: {action_mask.sum().item()}")
        
        if action_mask.any():
            valid_action = int(torch.where(action_mask)[0][0].item())
            obs, reward, terminated, truncated, info = env.step(valid_action)
            print(f"✓ Environment step successful, reward: {reward:.2f}")
        
        
        print("✓ Hierarchical environment test passed")
        return True
        
    except Exception as e:
        print(f"✗ Hierarchical environment test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    
    print("="*60)
    print("🏗️  HIERARCHICAL REINFORCEMENT LEARNING EXPERIMENT")
    print("📋 Manager-Worker Architecture for Job Shop Scheduling")
    print("="*60)
    
    if not test_environment_setup():
        print("\nHierarchical environment test failed! Exiting.")
        return
    
    print("\n" + "="*50)
    print("Starting hierarchical RL experiment...")
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
    
    # Hierarchical RL specific parameters
    total_max_steps = simulation_params['num_jobs'] * simulation_params['operation_ub'] * simulation_params['num_machines']
    hrl_params = {
        'alpha': 0.5,
        'epochs': 300,  # Fewer epochs than flat RL as hierarchical might converge faster
        'steps_per_epoch': total_max_steps,
        'dilation': 10,  # Manager horizon c
        'latent_dim': 256,  # Encoded state dimension
        'goal_dim': 32,  # Goal space dimension
        'hidden_dim': 512,  # Network hidden dimension
        'manager_lr': 3e-4,  # Manager learning rate
        'worker_lr': 3e-4,  # Worker learning rate
        'alpha_start': 1.0,  # Initial intrinsic reward weight
        'alpha_end': 0.1,  # Final intrinsic reward weight
        'gamma_manager': 0.995,  # Manager discount factor
        'gamma_worker': 0.95,   # Worker discount factor
        'gae_lambda': 0.95,
        'clip_ratio': 0.2,
        'entropy_coef': 0.01,
        'epsilon_greedy': 0.1,  # Manager exploration
    }
    
    print("Creating data handler and hierarchical environment...")
    data_handler = FlexibleJobShopDataHandler(data_source=simulation_params, data_type="simulation")
    env = HierarchicalRLEnv(data_handler, alpha=hrl_params['alpha'])
    
    print(f"Hierarchical environment created: {env.num_jobs} jobs, {env.num_machines} machines")
    print(f"Observation dimension: {env.obs_len}")
    print(f"Action dimension: {env.action_dim}")
    print(f"Manager dilation: {hrl_params['dilation']}")
    print(f"Latent dimension: {hrl_params['latent_dim']}")
    print(f"Goal dimension: {hrl_params['goal_dim']}")
    
    # Create hierarchical trainer with all parameters
    trainer = HierarchicalRLTrainer(
        env=env,
        epochs=hrl_params['epochs'],
        steps_per_epoch=hrl_params['steps_per_epoch'],
        dilation=hrl_params['dilation'],
        latent_dim=hrl_params['latent_dim'],
        goal_dim=hrl_params['goal_dim'],
        hidden_dim=hrl_params['hidden_dim'],
        manager_lr=hrl_params['manager_lr'],
        worker_lr=hrl_params['worker_lr'],
        alpha_start=hrl_params['alpha_start'],
        alpha_end=hrl_params['alpha_end'],
        gamma_manager=hrl_params['gamma_manager'],
        gamma_worker=hrl_params['gamma_worker'],
        gae_lambda=hrl_params['gae_lambda'],
        clip_ratio=hrl_params['clip_ratio'],
        entropy_coef=hrl_params['entropy_coef'],
        epsilon_greedy=hrl_params['epsilon_greedy'],
        model_save_dir=model_dir
    )
    
    print(f"Starting hierarchical training for {hrl_params['epochs']} epochs...")
    try:
        results = trainer.train()
        print("Hierarchical training complete.")
        
        # Print training statistics
        if results['training_history']['episode_rewards']:
            recent_rewards = results['training_history']['episode_rewards'][-10:]
            print(f"Final average reward: {sum(recent_rewards) / len(recent_rewards):.2f}")
        if results['training_history']['episode_makespans']:
            recent_makespans = results['training_history']['episode_makespans'][-10:]
            print(f"Final average makespan: {sum(recent_makespans) / len(recent_makespans):.2f}")
        if results['training_history']['episode_twts']:
            recent_twts = results['training_history']['episode_twts'][-10:]
            print(f"Final average TWT: {sum(recent_twts) / len(recent_twts):.2f}")
        
        print(f"Training time: {results['training_time']:.2f} seconds")
        print(f"Model saved in {model_dir}, wandb logs in {training_process_dir}.")
        
        # Evaluate the trained hierarchical policy
        print("\nEvaluating trained hierarchical policy...")
        evaluation_result = trainer.evaluate(num_episodes=10)
        
        print("\nHierarchical Evaluation Results:")
        print(f"  Average Reward: {evaluation_result['avg_reward']:.2f} ± {evaluation_result['std_reward']:.2f}")
        print(f"  Average Makespan: {evaluation_result['avg_makespan']:.2f} ± {evaluation_result['std_makespan']:.2f}")
        print(f"  Average TWT: {evaluation_result['avg_twt']:.2f} ± {evaluation_result['std_twt']:.2f}")
        
        # Visualize using trainer's built-in method (if available)
        print("Creating Gantt chart using hierarchical trainer...")
        try:
            # Run a single episode to get schedule information for trainer visualization
            obs, _ = env.reset()
            obs = torch.tensor(obs, dtype=torch.float32, device=trainer.agent.device)
            
            manager_hidden = None
            goals_history = []
            encoded_states_history = []
            prev_r_int = 0.0
            
            episode_reward = 0
            step = 0
            max_steps = env.num_jobs * max(len(job.operations) for job in env.jobs.values()) * 2
            
            while not env.state.is_done() and step < max_steps:
                # Encode state
                z_t = trainer.agent.encode_state(obs)
                encoded_states_history.append(z_t)
                
                # Manager decision
                if step % trainer.dilation == 0:
                    goal, _, manager_hidden = trainer.agent.get_manager_goal(z_t, step, manager_hidden)
                    if goal is not None:
                        goals_history.append(goal)
                
                # Pool goals for worker
                pooled_goal = trainer.agent.pool_goals(goals_history, step)
                
                # Worker action (deterministic)
                action_mask = env.get_action_mask()
                if not action_mask.any():
                    break
                
                action = trainer.agent.get_deterministic_action(
                    obs, action_mask, pooled_goal, prev_r_int
                )
                
                # Environment step
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                episode_reward += reward
                next_obs = torch.tensor(next_obs, dtype=torch.float32, device=trainer.agent.device)
                obs = next_obs
                step += 1
                
                if done:
                    break
            
            gantt_save_path_trainer = os.path.join(training_process_dir, "hierarchical_gantt_trainer.png")
            print(f"✓ Trainer episode completed (Makespan: {info['makespan']:.2f}, TWT: {info['twt']:.2f})")
            
        except Exception as e:
            print(f"Trainer Gantt chart creation failed: {e}")
        
        # Also showcase using hierarchical policy showcase function
        print("Creating Gantt chart using hierarchical showcase function...")
        try:
            gantt_save_path_showcaser = os.path.join(training_process_dir, "hierarchical_gantt_showcase.png")
            showcaser_result = showcase_hierarchical_policy(model_dir=model_dir, env=env)
            
            # Create Gantt chart separately
            create_gantt_chart(showcaser_result, save_path=gantt_save_path_showcaser, title_suffix="Hierarchical RL")
            
            print("\nHierarchical Showcase Results:")
            print(f"  Makespan: {showcaser_result['makespan']:.2f}")
            print(f"  TWT: {showcaser_result['twt']:.2f}")
            print(f"  Total Reward: {showcaser_result['total_reward']:.2f}")
            print(f"  Steps Taken: {showcaser_result['steps_taken']}")
            print(f"  Manager Decisions: {showcaser_result['manager_decisions']}")
            print(f"  Valid Completion: {showcaser_result['is_valid_completion']}")
            
        except Exception as e:
            print(f"Hierarchical showcase function failed: {e}")
            print("Using trainer evaluation result instead.")
            import traceback
            traceback.print_exc()
        
        print(f"\nHierarchical RL experiment completed successfully!")
        print(f"Results saved in: {training_process_dir}")
        
        # Print summary comparison with flat RL structure
        print(f"\n" + "="*50)
        print("HIERARCHICAL RL EXPERIMENT SUMMARY")
        print("="*50)
        print(f"Manager-Worker Architecture:")
        print(f"  Dilation (c): {hrl_params['dilation']}")
        print(f"  Latent Dimension: {hrl_params['latent_dim']}")
        print(f"  Goal Dimension: {hrl_params['goal_dim']}")
        print(f"  Manager LR: {hrl_params['manager_lr']}")
        print(f"  Worker LR: {hrl_params['worker_lr']}")
        print(f"Final Performance:")
        print(f"  Avg Reward: {evaluation_result['avg_reward']:.2f}")
        print(f"  Avg Makespan: {evaluation_result['avg_makespan']:.2f}")
        print(f"  Avg TWT: {evaluation_result['avg_twt']:.2f}")
        print(f"Training Time: {results['training_time']:.2f}s")
        
    except Exception as e:
        print(f"Hierarchical training failed: {e}")
        import traceback
        traceback.print_exc()
        return

if __name__ == "__main__":
    main()

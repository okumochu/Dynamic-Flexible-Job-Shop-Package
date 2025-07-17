import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from RL.rl_env import RLEnv
from RL.hierarchical_rl_trainer import HierarchicalRLTrainer
from benchmarks.static_benchmark.data_handler import FlexibleJobShopDataHandler
import wandb
from utils.policy_utils import showcase_hierarchical_policy, create_gantt_chart
from config import config


def run_hierarchical_rl_experiment():
    """Run hierarchical RL experiment with centralized configuration"""
    
    print("="*60)
    print("🏗️  HIERARCHICAL REINFORCEMENT LEARNING EXPERIMENT")
    print("📋 Manager-Worker Architecture for Job Shop Scheduling")
    print("="*60)
        
    print("\n" + "="*50)
    print("Starting hierarchical RL experiment...")
    print("="*50)
    
    # Get configuration
    exp_config = config.get_hierarchical_rl_config()
    simulation_params = exp_config['simulation_params']
    hrl_params = exp_config['rl_params']
    result_dirs = exp_config['result_dirs']
    
    # Setup directories and wandb
    config.setup_directories(result_dirs)
    config.setup_wandb_env(result_dirs['training_process'], exp_config['wandb_project'])
    
    print("Creating data handler and hierarchical environment...")
    data_handler = FlexibleJobShopDataHandler(data_source=simulation_params, data_type="simulation")
    env = RLEnv(data_handler, alpha=hrl_params['alpha'], detailed_info=True)
    
    print(f"Hierarchical environment created: {env.num_jobs} jobs, {env.num_machines} machines")
    print(f"Observation dimension: {env.obs_len}")
    print(f"Action dimension: {env.action_dim}")
    print(f"Manager goal duration: {hrl_params['goal_duration']}")
    print(f"Latent dimension: {hrl_params['latent_dim']}")
    print(f"Goal dimension: {hrl_params['goal_dim']}")
    
    # Create hierarchical trainer with all parameters
    trainer = HierarchicalRLTrainer(
        env=env,
        epochs=hrl_params['epochs'],
        steps_per_epoch=hrl_params['steps_per_epoch'],
        goal_duration=hrl_params['goal_duration'],
        latent_dim=hrl_params['latent_dim'],
        goal_dim=hrl_params['goal_dim'],
        manager_lr=hrl_params['manager_lr'],
        worker_lr=hrl_params['worker_lr'],
        alpha=hrl_params['alpha'],
        gamma_manager=hrl_params['gamma_manager'],
        gamma_worker=hrl_params['gamma_worker'],
        gae_lambda=hrl_params['gae_lambda'],
        clip_ratio=hrl_params['clip_ratio'],
        entropy_coef=hrl_params['entropy_coef'],
        train_pi_iters=hrl_params['train_pi_iters'],
        train_v_iters=hrl_params['train_v_iters'],
        model_save_dir=result_dirs['model']
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
        print(f"Model saved in {result_dirs['model']}, wandb logs in {result_dirs['training_process']}.")
        
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
            
            goals_history = []
            
            episode_reward = 0
            step = 0
            max_steps = env.num_jobs * max(len(job.operations) for job in env.jobs.values()) * 2
            
            while not env.state.is_done() and step < max_steps:
                # Encode state
                z_t = trainer.agent.encode_state(obs)
                
                # Manager decision
                if step % trainer.goal_duration == 0:
                    goal = trainer.agent.get_manager_goal(z_t)
                    if goal is not None:
                        goals_history.append(goal)
                
                # Pool goals for worker
                pooled_goal = trainer.agent.pool_goals(goals_history, step, trainer.agent.goal_duration)
                
                # Worker action (deterministic)
                action_mask = env.get_action_mask()
                if not action_mask.any():
                    break
                
                action = trainer.agent.get_deterministic_action(
                    obs, action_mask, pooled_goal
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
            
            print(f"✓ Trainer episode completed (Makespan: {info['makespan']:.2f}, TWT: {info['twt']:.2f})")
            
        except Exception as e:
            print(f"Trainer Gantt chart creation failed: {e}")
        
        # Also showcase using hierarchical policy showcase function
        print("Creating Gantt chart using hierarchical showcase function...")
        try:
            gantt_save_path_showcaser = os.path.join(result_dirs['training_process'], "hierarchical_gantt_showcase.png")
            showcaser_result = showcase_hierarchical_policy(model_dir=result_dirs['model'], env=env)
            
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
        print(f"Results saved in: {result_dirs['training_process']}")
        
        # Print summary comparison with flat RL structure
        print(f"\n" + "="*50)
        print("HIERARCHICAL RL EXPERIMENT SUMMARY")
        print("="*50)
        print(f"Manager-Worker Architecture:")
        print(f"  Goal Duration (c): {hrl_params['goal_duration']}")
        print(f"  Latent Dimension: {hrl_params['latent_dim']}")
        print(f"  Goal Dimension: {hrl_params['goal_dim']}")
        print(f"  Manager LR: {hrl_params['manager_lr']}")
        print(f"  Worker LR: {hrl_params['worker_lr']}")
        print(f"Final Performance:")
        print(f"  Avg Reward: {evaluation_result['avg_reward']:.2f}")
        print(f"  Avg Makespan: {evaluation_result['avg_makespan']:.2f}")
        print(f"  Avg TWT: {evaluation_result['avg_twt']:.2f}")
        print(f"Training Time: {results['training_time']:.2f}s")
        
        return results, evaluation_result
        
    except Exception as e:
        print(f"Hierarchical training failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def main():
    """Main function to run hierarchical RL experiment"""
    run_hierarchical_rl_experiment()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Generalization Testing Utility for Flat RL Model
Loads a trained model and tests it on different problem instances to evaluate generalization.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
import matplotlib.pyplot as plt
from RL.PPO.ppo_worker import PPOWorker
from RL.flat_rl_env import FlatRLEnv
from benchmarks.static_benchmark.data_handler import FlexibleJobShopDataHandler
import argparse

def load_model_and_config(model_path: str):
    """Load the trained model and its configuration from the .pth file."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Get configuration from the saved model
    config = PPOWorker.get_saved_config(model_path)
    print(f"Loaded configuration from model: {config}")
    
    # Create PPO worker with the saved configuration
    ppo_worker = PPOWorker(
        input_dim=config['input_dim'],
        action_dim=config['action_dim'], 
        hidden_dim=config['hidden_dim'],
        pi_lr=config['pi_lr'],
        v_lr=config['v_lr'],
        gamma=config['gamma'],
        gae_lambda=config['gae_lambda'],
        clip_ratio=config['clip_ratio'],
        device=config['device']
    )
    
    # Load the trained weights
    ppo_worker.load(model_path)
    print(f"✓ Model loaded successfully from {model_path}")
    
    return ppo_worker, config

def test_model_on_seed(ppo_worker, config, seed: int, visualize: bool = True):
    """Test the model on a specific seed and return performance metrics."""
    print(f"\n{'='*50}")
    print(f"Testing model on seed: {seed}")
    print(f"{'='*50}")
    
    # Create environment with the same parameters as training but different seed
    simulation_params = {
        'num_jobs': 10, 
        'num_machines': 3,
        'operation_lb': 3,
        'operation_ub': 3,
        'processing_time_lb': 5,
        'processing_time_ub': 5,   
        'compatible_machines_lb': 3,
        'compatible_machines_ub': 3,
        'seed': seed,
    }
    
    data_handler = FlexibleJobShopDataHandler(data_source=simulation_params, data_type="simulation")
    env = FlatRLEnv(data_handler, alpha=0.5)  # Same alpha as training
    
    # Run one episode
    obs, info = env.reset()
    obs_tensor = torch.FloatTensor(obs)
    
    episode_actions = []
    episode_rewards = []
    done = False
    step_count = 0
    
    while not done:
        action_mask = env.get_action_mask()
        
        # Use deterministic action for evaluation
        action = ppo_worker.get_deterministic_action(obs_tensor, action_mask)
        episode_actions.append(action)
        
        obs, reward, done, truncated, info = env.step(action)
        obs_tensor = torch.FloatTensor(obs)
        episode_rewards.append(reward)
        step_count += 1
        
        if step_count > 1000:  # Safety break
            print("⚠️  Episode truncated after 1000 steps")
            break
    
    # Get final metrics
    final_metrics = env.get_current_objective()
    makespan = final_metrics['makespan']
    twt = final_metrics['twt']
    
    print(f"Episode completed in {step_count} steps")
    print(f"Final Makespan: {makespan}")
    print(f"Final Total Weighted Tardiness: {twt}")
    print(f"Total Episode Reward: {sum(episode_rewards):.2f}")
    
    # Validate solution
    try:
        if env.state.is_done():
            print("✓ Solution is complete")
        else:
            print("⚠️  Solution is not complete")
    except:
        print("⚠️  Could not validate solution completeness")
    
    # Visualize if requested (simplified for now - Gantt chart generation requires more complex data extraction)
    if visualize:
        print(f"📊 Visualization skipped for seed {seed} - would require additional implementation")
    
    return {
        'seed': seed,
        'makespan': makespan,
        'twt': twt,
        'total_reward': sum(episode_rewards),
        'steps': step_count
    }

def main():
    parser = argparse.ArgumentParser(description='Test generalization of trained Flat RL model')
    parser.add_argument('--model_path', type=str, default='result/flat_rl/model/final_model.pth',
                       help='Path to the trained model (.pth file)')
    parser.add_argument('--seeds', type=int, nargs='+', default=[123, 456, 789, 999, 1337],
                       help='Seeds to test on')
    parser.add_argument('--no_visualize', action='store_true',
                       help='Skip generating Gantt charts')
    
    args = parser.parse_args()
    
    print("🚀 Starting Flat RL Model Generalization Test")
    print(f"Model: {args.model_path}")
    print(f"Test seeds: {args.seeds}")
    
    try:
        # Load model and configuration from the .pth file
        ppo_worker, config = load_model_and_config(args.model_path)
        
        # Test on multiple seeds
        results = []
        for seed in args.seeds:
            try:
                result = test_model_on_seed(ppo_worker, config, seed, visualize=not args.no_visualize)
                results.append(result)
            except Exception as e:
                print(f"❌ Error testing seed {seed}: {e}")
                continue
        
        # Summary statistics
        if results:
            print(f"\n{'='*60}")
            print("GENERALIZATION TEST SUMMARY")
            print(f"{'='*60}")
            
            makespans = [r['makespan'] for r in results]
            twts = [r['twt'] for r in results]
            rewards = [r['total_reward'] for r in results]
            
            print(f"Tested {len(results)} seeds successfully")
            print(f"Makespan - Mean: {np.mean(makespans):.2f}, Std: {np.std(makespans):.2f}, Min: {min(makespans)}, Max: {max(makespans)}")
            print(f"TWT - Mean: {np.mean(twts):.2f}, Std: {np.std(twts):.2f}, Min: {min(twts):.2f}, Max: {max(twts):.2f}")
            print(f"Reward - Mean: {np.mean(rewards):.2f}, Std: {np.std(rewards):.2f}")
            
            print("\nDetailed Results:")
            for result in results:
                print(f"Seed {result['seed']:4d}: Makespan={result['makespan']:3d}, TWT={result['twt']:6.2f}, Reward={result['total_reward']:8.2f}, Steps={result['steps']:3d}")
        else:
            print("❌ No successful tests completed")
            
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        return 1
    
    print("\n✅ Generalization testing completed!")
    return 0

if __name__ == "__main__":
    exit(main()) 
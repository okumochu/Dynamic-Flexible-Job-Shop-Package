"""
Debug script to test the environment and identify issues
"""

from benchmarks.static_benchmark.data_handler import FlexibleJobShopDataHandler
from RL.flat_rl.flat_rl_env import FlatRLEnv
import torch
from gym import spaces
from typing import cast

def create_small_test_instance():
    """Create a small test instance: 5 jobs, 3 machines."""
    simulation_params = {
        'num_jobs': 5,
        'num_machines': 3,
        'operation_lb': 2,
        'operation_ub': 3,
        'processing_time_lb': 10,
        'processing_time_ub': 30,
        'compatible_machines_lb': 1,
        'compatible_machines_ub': 3,
        'seed': 42,
        'TF': 0.3,
        'RDD': 0.6
    }
    
    return FlexibleJobShopDataHandler(data_source=simulation_params, data_type="simulation")

def test_environment():
    """Test the environment step by step."""
    print("Creating test instance...")
    data_handler = create_small_test_instance()
    
    print("Creating environment...")
    env = FlatRLEnv(data_handler=data_handler, alpha=0.3, beta=0.7)
    
    action_space = cast(spaces.Discrete, env.action_space)
    print(f"Environment created: {env.num_jobs} jobs, {env.num_machines} machines")
    print(f"Action space: {action_space.n} actions")
    print(f"Observation space: {env.observation_space.shape}")
    
    # Reset environment
    obs = env.reset()
    print(f"Environment reset. Observation shape: {obs.shape}")
    
    step_count = 0
    max_steps = 100  # Prevent infinite loop
    
    while step_count < max_steps:
        # Get action mask
        action_mask = env.get_action_mask()
        valid_actions = action_mask.sum().item()
        
        print(f"\nStep {step_count}:")
        print(f"  Valid actions: {valid_actions}/{action_space.n}")
        print(f"  Current time: {env.current_time:.1f}")
        print(f"  Makespan: {env.current_makespan:.1f}")
        print(f"  TWT: {env.current_twt:.1f}")
        
        # Check if episode is done
        done = all(job_state['completed_ops'] == len(job.operations) 
                  for job, job_state in zip(env.jobs, env.job_states))
        
        if done:
            print("Episode completed!")
            break
        
        if valid_actions == 0:
            print("No valid actions available!")
            break
        
        # Take a random valid action
        valid_action_indices = torch.where(action_mask)[0]
        action = int(valid_action_indices[0].item())  # Take first valid action
        
        print(f"  Taking action: {action}")
        
        # Take step
        obs, reward, done, info = env.step(action)
        
        print(f"  Reward: {reward:.3f}")
        print(f"  Done: {done}")
        
        step_count += 1
        
        if done:
            print("Episode completed!")
            break
    
    print(f"\nFinal results:")
    print(f"Steps taken: {step_count}")
    print(f"Final makespan: {env.current_makespan:.1f}")
    print(f"Final TWT: {env.current_twt:.1f}")
    
    # Print job completion status
    for job_id, job_state in enumerate(env.job_states):
        print(f"Job {job_id}: {job_state['completed_ops']}/{len(env.jobs[job_id].operations)} operations completed")

if __name__ == "__main__":
    test_environment() 
import numpy as np
from RL.flat_rl_env import FlatRLEnv
from benchmarks.static_benchmark.data_handler import FlexibleJobShopDataHandler
import torch

def analyze_episode_end(env):
    """
    Analyze why the episode ended in the new FlatRLEnv:
    - If all jobs are finished, return 'all_jobs_finished'.
    - If there are no valid actions but jobs remain, return 'no_valid_actions'.
    """
    all_finished = all(js['completed_ops'] == len(env.jobs[jid].operations) for jid, js in enumerate(env.job_states))
    if all_finished:
        return "all_jobs_finished"
    # If not all finished, but no valid actions, it's a blockage or deadlock
    return "no_valid_actions"

def test_env(num_episodes=3, seed=123):
    data_params = {
        'num_jobs': 3,
        'num_machines': 3,
        'operation_lb': 2,
        'operation_ub': 3,
        'processing_time_lb': 1,
        'processing_time_ub': 5,
        'compatible_machines_lb': 1,
        'compatible_machines_ub': 3,
        'seed': seed
    }
    data_handler = FlexibleJobShopDataHandler(data_source=data_params, data_type="simulation")
    env = FlatRLEnv(data_handler)
    deadlock_count = 0
    finished_count = 0
    for ep in range(num_episodes):
        print(f"\n=== Episode {ep+1} ===")
        obs = env.reset()
        done = False
        step = 0
        while not done:
            action_mask = env.get_action_mask()
            valid_actions = torch.where(action_mask)[0].cpu().numpy()
            print(f"Step {step}: valid actions = {valid_actions}")
            if len(valid_actions) == 0:
                reason = analyze_episode_end(env)
                print(f"WARNING: No valid actions! Breaking episode. Reason: {reason}")
                if reason == "all_jobs_finished":
                    finished_count += 1
                else:
                    deadlock_count += 1
                break
            action = np.random.choice(valid_actions)
            next_obs, reward, done, info = env.step(action)
            print(f"  Action: {action}, Reward: {reward}, Done: {done}")
            print(f"  Makespan: {info['makespan']}, TWT: {info['twt']}")
            obs = next_obs
            step += 1
        else:
            # Episode ended with done=True
            print(f"Episode {ep+1} finished normally. Final Makespan: {info['makespan']}, Final TWT: {info['twt']}")
            finished_count += 1
    print(f"\nSummary: {finished_count} episodes finished normally, {deadlock_count} ended with no valid actions (blockage or deadlock).")

if __name__ == "__main__":
    test_env() 
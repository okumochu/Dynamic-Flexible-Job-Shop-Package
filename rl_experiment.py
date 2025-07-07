import os
from RL.flat_rl_env import FlatRLEnv
from RL.flat_rl_trainer import FlatRLTrainer
from benchmarks.static_benchmark.data_handler import FlexibleJobShopDataHandler
import wandb
from utils.policy_showcaser import PolicyShowcaser

# Set wandb output directory
training_process_dir = "result/flat_rl/training_process"
model_dir = "result/flat_rl/model"
os.environ["WANDB_DIR"] = training_process_dir

def main():
    # Example: generate a synthetic problem instance with correct params
    simulation_params = {
        'num_jobs': 12, 
        'num_machines': 4,
        'operation_lb': 2,
        'operation_ub': 2,
        'processing_time_lb': 3,
        'processing_time_ub': 8,   
        'compatible_machines_lb': 2,
        'compatible_machines_ub': 3,
        'seed': 42,
    }
    rl_params = {
        'alpha': 0.5,
        'gamma': 0.99,
        'gae_lambda': 0.97,
        'steps_per_epoch': 50,
        'epochs': 5000,
        'pi_lr': 1e-5,  # Reduced from 3e-5
        'v_lr': 1e-5,   # Reduced from 1e-4
        'target_kl': 0.5,  # Increased from 0.1
        'train_pi_iters': 50,
        'train_v_iters': 50,
    }
    data_handler = FlexibleJobShopDataHandler(data_source=simulation_params, data_type="simulation")
    env = FlatRLEnv(data_handler, alpha=rl_params['alpha'])
    trainer = FlatRLTrainer(env, model_save_dir=model_dir, gamma=rl_params['gamma'], gae_lambda=rl_params['gae_lambda'], steps_per_epoch=rl_params['steps_per_epoch'], epochs=rl_params['epochs'], pi_lr=rl_params['pi_lr'], v_lr=rl_params['v_lr'], target_kl=rl_params['target_kl'])
    print("Starting training for 100 episodes...")
    results = trainer.train()
    print("Training complete.")
    print(f"Final average reward: {sum(results['episode_rewards'][-10:]) / 10:.2f}")
    print(f"Final average makespan: {sum(results['episode_makespans'][-10:]) / 10:.2f}")
    print(f"Final average TWT: {sum(results['episode_twts'][-10:]) / 10:.2f}")
    print(f"Model saved in {model_dir}, wandb logs in {training_process_dir}.")

    # Showcase the trained policy
    showcaser = PolicyShowcaser(model_dir=model_dir, env=env)
    gantt_save_path = os.path.join(training_process_dir, "gantt_showcase.png")
    showcaser.showcase(render_gantt=True, gantt_save_path=gantt_save_path)

if __name__ == "__main__":
    main()

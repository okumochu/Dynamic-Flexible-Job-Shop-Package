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
    data_params = {
        'num_jobs': 5,
        'num_machines': 4,
        'operation_lb': 3,
        'operation_ub': 5,
        'processing_time_lb': 1,
        'processing_time_ub': 10,
        'compatible_machines_lb': 1,
        'compatible_machines_ub': 4,
        'seed': 42
    }
    data_handler = FlexibleJobShopDataHandler(data_source=data_params, data_type="simulation")
    env = FlatRLEnv(data_handler)
    trainer = FlatRLTrainer(env, model_save_dir=model_dir)
    print("Starting training for 100 episodes...")
    results = trainer.train(num_episodes=2000)
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

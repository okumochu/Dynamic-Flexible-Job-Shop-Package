import os
from RL.flat_rl_env import FlatRLEnv
from RL.flat_rl_trainer import FlatRLTrainer
from benchmarks.static_benchmark.data_handler import FlexibleJobShopDataHandler
from utils.policy_showcaser import PolicyShowcaser
from MILP.model import MILP
from utils.solution_utils import SolutionUtils
import time

# Set output directories
training_process_dir = "result/flat_rl/training_process"
model_dir = "result/flat_rl/model"
os.makedirs(training_process_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

# Set WANDB output directory
os.environ["WANDB_DIR"] = training_process_dir

def main():
    # 1. Generate a synthetic problem instance
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

    # 2. RL Experiment
    print("\n===== RL EXPERIMENT =====")
    env = FlatRLEnv(data_handler)
    trainer = FlatRLTrainer(env, model_save_dir=model_dir)
    print("Starting RL training for 2000 episodes...")
    rl_start = time.time()
    results = trainer.train(num_episodes=2000)
    rl_time = time.time() - rl_start
    print("RL Training complete.")
    print(f"Final average reward: {sum(results['episode_rewards'][-10:]) / 10:.2f}")
    print(f"Final average makespan: {sum(results['episode_makespans'][-10:]) / 10:.2f}")
    print(f"Final average TWT: {sum(results['episode_twts'][-10:]) / 10:.2f}")
    print(f"Model saved in {model_dir}, wandb logs in {training_process_dir}.")

    # RL Gantt chart
    showcaser = PolicyShowcaser(model_dir=model_dir, env=env)
    rl_gantt_path = os.path.join(training_process_dir, "rl_gantt.png")
    rl_showcase = showcaser.showcase(render_gantt=True, gantt_save_path=rl_gantt_path)
    print(f"RL Gantt chart saved to {rl_gantt_path}")

    # RL performance
    rl_makespan = rl_showcase['makespan']
    rl_twt = rl_showcase['twt']
    rl_objective = env.alpha * rl_makespan + (1 - env.alpha) * rl_twt if rl_makespan is not None and rl_twt is not None else None

    # 3. MILP Experiment
    print("\n===== MILP EXPERIMENT =====")
    milp_model = MILP(data_handler, twt_weight=env.beta)
    print("Building MILP model...")
    milp_model.build_model(time_limit=1200, MIPFocus=1, verbose=0)  # 20 min = 1200 sec
    print("Solving MILP model (20 min time limit)...")
    milp_start = time.time()
    solution = milp_model.solve()
    milp_time = time.time() - milp_start
    performance = solution["performance"]
    print(f"MILP Status: {performance['status']}")
    print(f"MILP Objective: {performance['objective']:.2f}")
    print(f"MILP Makespan: {performance['makespan']:.2f}")
    print(f"MILP TWT: {performance['total_weighted_tardiness']:.2f}")
    print(f"MILP Solve Time: {performance['solve_time']:.2f}s (wall: {milp_time:.2f}s)")

    # MILP Gantt chart
    milp_gantt_path = os.path.join(training_process_dir, "milp_gantt.png")
    if solution["schedule_result"]:
        machine_schedule = solution["schedule_result"]
        solution_utils = SolutionUtils(data_handler, machine_schedule)
        fig = solution_utils.draw_gantt(show_validation=True, show_due_dates=True)
        try:
            fig.write_image(milp_gantt_path)
            print(f"MILP Gantt chart saved to {milp_gantt_path}")
        except Exception as e:
            print(f"Could not save MILP Gantt chart as image: {e}")
    else:
        print("No MILP solution found, skipping Gantt chart.")

    # MILP performance
    milp_makespan = performance['makespan']
    milp_twt = performance['total_weighted_tardiness']
    milp_objective = performance['objective']

    # 4. Print summary table
    print("\n===== PERFORMANCE COMPARISON =====")
    print(f"{'Method':<10} | {'Objective':>10} | {'Makespan':>10} | {'TWT':>10} | {'Time (s)':>10}")
    print("-" * 60)
    print(f"{'RL':<10} | {rl_objective:>10.2f} | {rl_makespan:>10.2f} | {rl_twt:>10.2f} | {rl_time:>10.2f}")
    print(f"{'MILP':<10} | {milp_objective:>10.2f} | {milp_makespan:>10.2f} | {milp_twt:>10.2f} | {milp_time:>10.2f}")
    print("\nGantt charts saved in:")
    print(f"  RL:   {rl_gantt_path}")
    print(f"  MILP: {milp_gantt_path}")

if __name__ == "__main__":
    main()

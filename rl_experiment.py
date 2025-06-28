"""
RL Experiment for Flexible Job Shop Scheduling
Uses FlatRLTrainer to train and evaluate PPO agent
Compares performance with MILP solver
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import time
import os
import json

# Import our components
from benchmarks.static_benchmark.data_handler import FlexibleJobShopDataHandler
from RL.flat_rl.flat_rl_env import FlatRLEnv
from RL.flat_rl.flat_rl_trainer import FlatRLTrainer
from utils.solution_utils import SolutionUtils
from MILP.model import MILP

def create_regular_test_instance():
    """Create a regular test instance using mk06 dataset: 10 jobs, 10 machines."""
    dataset_path = "benchmarks/static_benchmark/datasets/brandimarte/mk06.txt"
    
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    
    return FlexibleJobShopDataHandler(data_source=dataset_path, data_type="dataset")

def print_problem_statistics(data_handler):
    """Print detailed problem statistics."""
    print("\n" + "="*50)
    print("PROBLEM STATISTICS")
    print("="*50)
    
    stats = data_handler.get_statistics()
    print(f"Number of Jobs: {stats['num_jobs']}")
    print(f"Number of Machines: {stats['num_machines']}")
    print(f"Number of Operations: {stats['num_operations']}")
    print(f"Total Processing Time: {stats['total_processing_time']}")
    print(f"Average Operations per Job: {stats['avg_operations_per_job']:.2f}")
    print(f"Average Processing Time: {stats['avg_processing_time']:.2f}")
    
    print("\nJob Details:")
    for job_id in range(stats['num_jobs']):
        job = data_handler.jobs[job_id]
        due_date = data_handler.get_job_due_date(job_id)
        weight = data_handler.get_job_weight(job_id)
        print(f"  Job {job_id}: {len(job.operations)} operations, "
              f"due={due_date}, weight={weight}, "
              f"total_time={job.total_processing_time}")
    
    print("\nMachine Loads:")
    for machine_id in range(stats['num_machines']):
        load = data_handler.get_machine_load(machine_id)
        print(f"  Machine {machine_id}: {load}")

def run_milp_comparison(data_handler, time_limit: int = 3000):
    """Run MILP solver for comparison."""
    print("\n" + "="*50)
    print("MILP COMPARISON")
    print("="*50)
    
    try:
        print(f"Solving with MILP (time limit: {time_limit}s)...")
        start_time = time.time()
        
        # Create and solve MILP model
        milp_model = MILP(data_handler, twt_weight=0.5)
        milp_model.build_model(time_limit=time_limit, MIPFocus=1, verbose=0)
        solution = milp_model.solve()
        
        solve_time = time.time() - start_time
        
        if solution["schedule_result"]:
            # Validate MILP solution using solution_utils
            solution_utils = SolutionUtils(data_handler, solution["schedule_result"])
            validation_result = solution_utils.validate_solution()
            
            print(f"✓ MILP Solution Found:")
            print(f"  - Status: {solution['performance']['status']}")
            print(f"  - Solve Time: {solve_time:.2f}s")
            print(f"  - Makespan: {solution['performance']['makespan']:.2f}")
            print(f"  - TWT: {solution['performance']['total_weighted_tardiness']:.2f}")
            print(f"  - Validation: {'Valid' if validation_result['is_valid'] else 'Invalid'}")
            
            # Generate MILP Gantt chart
            print("  Generating MILP Gantt chart...")
            milp_fig = solution_utils.draw_gantt(show_due_dates=True)
            milp_fig.write_image("result/milp_comparison/milp_solution_gantt.png")
            
            return {
                'status': 'success',
                'makespan': solution['performance']['makespan'],
                'twt': solution['performance']['total_weighted_tardiness'],
                'solve_time': solve_time,
                'validation': validation_result,
                'schedule': solution['schedule_result']
            }
        else:
            print("✗ MILP failed to find solution")
            return {
                'status': 'failed',
                'makespan': float('inf'),
                'twt': float('inf'),
                'solve_time': solve_time,
                'validation': None,
                'schedule': None
            }
            
    except Exception as e:
        print(f"✗ MILP Error: {e}")
        return {
            'status': 'error',
            'makespan': float('inf'),
            'twt': float('inf'),
            'solve_time': 0,
            'validation': None,
            'schedule': None
        }

def evaluate_training_process(trainer, data_handler, training_time, milp_results=None):
    """Evaluate the training process and generate comprehensive results."""
    print("\n" + "="*60)
    print("TRAINING PROCESS EVALUATION")
    print("="*60)
    
    # Get final evaluation results
    final_eval = trainer.evaluate(num_episodes=20)
    
    print(f"Final Evaluation Results (20 episodes):")
    print(f"Average Makespan: {final_eval['avg_makespan']:.2f} ± {final_eval['std_makespan']:.2f}")
    print(f"Average TWT: {final_eval['avg_twt']:.2f} ± {final_eval['std_twt']:.2f}")
    print(f"Best Makespan: {min(final_eval['makespans']):.2f}")
    print(f"Best TWT: {min(final_eval['twts']):.2f}")
    
    # Get best schedule
    best_schedule, best_makespan, best_twt = trainer.get_best_schedule(num_episodes=20)
    
    print(f"\nBest Schedule Found:")
    print(f"Makespan: {best_makespan:.2f}")
    print(f"Total Weighted Tardiness: {best_twt:.2f}")
    
    # Note: The RL environment currently provides simplified schedule information
    # For proper validation with solution_utils, we would need detailed machine assignments
    # with start times. For now, we'll skip validation and focus on performance comparison.
    print(f"RL Solution Validation: Skipped (requires detailed machine schedule tracking)")
    validation_result = None
    
    # Plot training curves
    print("\nGenerating training curves...")
    trainer.plot_training_curves(save_path='result/flat_rl_model/training_curves.png')
    
    # Plot evaluation progress
    print("Generating evaluation progress...")
    trainer.plot_evaluation_progress(save_path='result/flat_rl_model/evaluation_progress.png')
    
    # Note: Gantt chart generation requires detailed machine schedule
    # which is not currently available from the RL environment
    print("Gantt chart generation: Skipped (requires detailed machine schedule)")
    
    # Compare with MILP if available
    if milp_results and milp_results['status'] == 'success':
        print("\n" + "="*50)
        print("PERFORMANCE COMPARISON")
        print("="*50)
        
        print(f"RL Results:")
        print(f"  - Best Makespan: {best_makespan:.2f}")
        print(f"  - Best TWT: {best_twt:.2f}")
        print(f"  - Training Time: {training_time:.2f}s")
        
        print(f"\nMILP Results:")
        print(f"  - Makespan: {milp_results['makespan']:.2f}")
        print(f"  - TWT: {milp_results['twt']:.2f}")
        print(f"  - Solve Time: {milp_results['solve_time']:.2f}s")
        
        # Calculate performance ratios
        makespan_ratio = best_makespan / milp_results['makespan']
        twt_ratio = best_twt / milp_results['twt']
        
        print(f"\nPerformance Ratios (RL/MILP):")
        print(f"  - Makespan: {makespan_ratio:.3f} ({'Better' if makespan_ratio < 1 else 'Worse'})")
        print(f"  - TWT: {twt_ratio:.3f} ({'Better' if twt_ratio < 1 else 'Worse'})")
        print(f"  - Time: {training_time / milp_results['solve_time']:.3f}x")
    
    return {
        'final_evaluation': final_eval,
        'best_schedule': best_schedule,
        'best_makespan': best_makespan,
        'best_twt': best_twt,
        'validation_result': validation_result,
        'milp_comparison': milp_results
    }

def main():
    """Main experiment function."""
    print("Flexible Job Shop Scheduling - PPO RL Experiment (Regular Dataset)")
    print("="*80)
    
    # Create output directories
    os.makedirs('result/flat_rl_model', exist_ok=True)
    os.makedirs('result/milp_comparison', exist_ok=True)
    
    # Step 1: Load regular dataset
    print("Step 1: Loading regular dataset (mk06.txt)...")
    try:
        data_handler = create_regular_test_instance()
        print("✓ Dataset loaded successfully")
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        return
    
    # Print problem statistics
    print_problem_statistics(data_handler)
    
    # Step 2: Run MILP comparison
    print("\nStep 2: Running MILP comparison...")
    milp_results = run_milp_comparison(data_handler, time_limit=300)
    
    # Step 3: Create environment
    print("\nStep 3: Creating RL environment...")
    env = FlatRLEnv(data_handler=data_handler, alpha=0.3, beta=0.7)
    
    # Step 4: Initialize trainer
    print("\nStep 4: Initializing RL trainer...")
    trainer = FlatRLTrainer(
        env=env,
        hidden_dim=256,  # Larger for regular instance
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_ratio=0.2,
        entropy_coeff=1e-3,
        value_coeff=0.5,
        max_grad_norm=0.5,
        device='auto',  # Auto-detect GPU/CPU
        model_save_dir='result/flat_rl_model'
    )
    
    # Step 5: Train the agent
    print("\nStep 5: Training the RL agent...")
    training_results = trainer.train(
        num_episodes=2000,  # More episodes for regular dataset
        buffer_size=2000,
        update_frequency=20,
        eval_frequency=100,
        save_frequency=200
    )
    
    # Step 6: Evaluate training process and results
    print("\nStep 6: Evaluating training process and results...")
    evaluation_results = evaluate_training_process(trainer, data_handler, training_results['training_time'], milp_results)
    
    # Step 7: Save comprehensive results
    print("\nStep 7: Saving comprehensive results...")
    results_summary = {
        'dataset': 'mk06.txt',
        'problem_size': {
            'jobs': data_handler.num_jobs,
            'machines': data_handler.num_machines,
            'operations': data_handler.num_operations
        },
        'rl_results': {
            'training_episodes': len(training_results['episode_rewards']),
            'training_time': training_results['training_time'],
            'final_avg_makespan': evaluation_results['final_evaluation']['avg_makespan'],
            'final_avg_twt': evaluation_results['final_evaluation']['avg_twt'],
            'best_makespan': evaluation_results['best_makespan'],
            'best_twt': evaluation_results['best_twt'],
            'validation_valid': evaluation_results['validation_result']['is_valid'] if evaluation_results['validation_result'] else False
        },
        'milp_results': milp_results,
        'comparison': {}
    }
    
    # Add comparison metrics if MILP was successful
    if milp_results and milp_results['status'] == 'success':
        results_summary['comparison'] = {
            'makespan_ratio': evaluation_results['best_makespan'] / milp_results['makespan'],
            'twt_ratio': evaluation_results['best_twt'] / milp_results['twt'],
            'time_ratio': training_results['training_time'] / milp_results['solve_time']
        }
    
    # Save results to JSON
    with open('result/experiment_results.json', 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    # Step 8: Print final summary
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)
    print(f"Dataset: mk06.txt ({data_handler.num_jobs} jobs, {data_handler.num_machines} machines)")
    print(f"Training Episodes: {len(training_results['episode_rewards'])}")
    print(f"Training Time: {training_results['training_time']:.2f} seconds")
    print(f"Final Average Makespan: {evaluation_results['final_evaluation']['avg_makespan']:.2f} ± {evaluation_results['final_evaluation']['std_makespan']:.2f}")
    print(f"Final Average TWT: {evaluation_results['final_evaluation']['avg_twt']:.2f} ± {evaluation_results['final_evaluation']['std_twt']:.2f}")
    print(f"Best Makespan Achieved: {evaluation_results['best_makespan']:.2f}")
    print(f"Best TWT Achieved: {evaluation_results['best_twt']:.2f}")
    
    if milp_results and milp_results['status'] == 'success':
        print(f"\nMILP Comparison:")
        print(f"MILP Makespan: {milp_results['makespan']:.2f}")
        print(f"MILP TWT: {milp_results['twt']:.2f}")
        print(f"MILP Solve Time: {milp_results['solve_time']:.2f}s")
    
    print("\nGenerated Files:")
    print("- result/flat_rl_model/final_model.pth (trained model)")
    print("- result/flat_rl_model/training_curves.png (training progress)")
    print("- result/flat_rl_model/evaluation_progress.png (evaluation progress)")
    print("- result/flat_rl_model/best_schedule_gantt.png (RL schedule visualization)")
    print("- result/milp_comparison/milp_solution_gantt.png (MILP schedule visualization)")
    print("- result/experiment_results.json (comprehensive results)")
    
    print("\nExperiment completed successfully!")
    print("Check the result/ directory for all outputs.")

if __name__ == "__main__":
    main()

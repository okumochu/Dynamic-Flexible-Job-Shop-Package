#!/usr/bin/env python3
"""
Instance Viewer for Dynamic Flexible Job Shop Package

This script demonstrates:
1. Instance generation using config parameters
2. Random policy evaluation with Gantt chart visualization
3. Graph RL policy loading and evaluation
4. Comparison between different policies

Author: Generated for Dynamic Flexible Job Shop Package
"""

import os
import sys
import torch
import numpy as np
import random
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

# Add project root to path
project_root = os.path.dirname(os.path.abspath('.'))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import project modules
from config import config
from benchmarks.data_handler import FlexibleJobShopDataHandler
from RL.graph_rl_env import GraphRlEnv
from RL.graph_rl_trainer import GraphPPOTrainer
from utils.policy_utils import evaluate_graph_policy, create_baseline_schedule
from utils.solution_utils import SolutionUtils

print("Instance Viewer initialized successfully!")
print(f"Project root: {project_root}")
print(f"Config loaded: {config.simulation_params}")




def find_latest_model(model_dir="result"):
    """
    Find the most recent trained graph RL model.
    
    Args:
        model_dir: Base directory to search for models
        
    Returns:
        Path to the most recent model file, or None if not found
    """
    if not os.path.exists(model_dir):
        print(f"Model directory {model_dir} does not exist")
        return None
    
    # Look for exp_graph_rl directories
    exp_dirs = []
    for item in os.listdir(model_dir):
        item_path = os.path.join(model_dir, item)
        if os.path.isdir(item_path) and item.startswith("exp_graph_rl"):
            exp_dirs.append(item_path)
    
    if not exp_dirs:
        print("No exp_graph_rl directories found")
        return None
    
    # Find the most recent experiment directory
    latest_exp_dir = max(exp_dirs, key=os.path.getmtime)
    print(f"Found experiment directory: {latest_exp_dir}")
    
    # Look for model files in subdirectories
    model_files = []
    for root, dirs, files in os.walk(latest_exp_dir):
        for file in files:
            if file.endswith('.pth') and 'model' in file.lower():
                model_files.append(os.path.join(root, file))
    
    if not model_files:
        print("No model files found in experiment directory")
        return None
    
    # Return the most recent model file
    latest_model = max(model_files, key=os.path.getmtime)
    print(f"Found model file: {latest_model}")
    return latest_model

def main():
    """Main function to run the instance viewer."""
    
    print("Starting exp_instance_viewer.py...")
    print("This may take a few minutes to complete...")
    
    # 1. Instance Generation
    print("\n=== Instance Generation ===")
    simulation_params = config.simulation_params.copy()
    print("Generating instance with parameters:")
    for key, value in simulation_params.items():
        print(f"  {key}: {value}")

    # Create data handler for simulation
    data_handler = FlexibleJobShopDataHandler(
        data_source=simulation_params,
        data_type='simulation',
        TF=simulation_params['TF'],
        RDD=simulation_params['RDD'],
        seed=simulation_params['seed']
    )

    print(f"\nInstance generated successfully!")
    print(f"Number of jobs: {data_handler.num_jobs}")
    print(f"Number of machines: {data_handler.num_machines}")
    print(f"Number of operations: {data_handler.num_operations}")

    # Display job structure
    print(f"\nJob structure:")
    for job_id in range(min(3, data_handler.num_jobs)):  # Show first 3 jobs
        job = data_handler.jobs[job_id]
        print(f"Job {job_id}: {len(job.operations)} operations")
        for op_idx, operation in enumerate(job.operations):
            print(f"  Op {op_idx}: {len(operation.compatible_machines)} compatible machines")
            if op_idx < 2:  # Show first 2 operations per job
                print(f"    Processing times: {dict(list(operation.machine_processing_times.items())[:3])}...")

    # 2. Baseline Schedule Evaluation
    print("\n=== Baseline Schedule Evaluation ===")
    
    # Get device and alpha for later use
    device = "cuda" if torch.cuda.is_available() else "cpu"
    alpha = config.common_rl_params['alpha']
    
    # Create baseline schedule using FIFO method
    print("Creating baseline schedule using FIFO method...")
    baseline_result = create_baseline_schedule(data_handler, method="fifo")
    baseline_machine_schedule = baseline_result['machine_schedule']
    baseline_solution_data = baseline_result['solution_data']
    
    # Calculate metrics for baseline schedule
    def calculate_baseline_metrics(machine_schedule, data_handler):
        """Calculate metrics for baseline schedule."""
        # Calculate makespan
        makespan = 0.0
        for machine_id, operations in machine_schedule.items():
            if operations:
                last_operation = max(operations, key=lambda x: x[2])  # x[2] is end_time
                makespan = max(makespan, last_operation[2])
        
        # Calculate total weighted tardiness
        total_twt = 0.0
        due_dates = data_handler.get_jobs_due_date()
        weights = data_handler.get_jobs_weight()
        
        for job_id in range(data_handler.num_jobs):
            job_operations = data_handler.get_job_operations(job_id)
            last_op = job_operations[-1]
            
            # Find completion time of last operation
            completion_time = 0.0
            for machine_id, operations in machine_schedule.items():
                for op_id, start_time, end_time in operations:
                    if op_id == last_op.operation_id:
                        completion_time = end_time
                        break
            
            tardiness = max(0, completion_time - due_dates[job_id])
            total_twt += weights[job_id] * tardiness
        
        # Normalize TWT by total weight to ensure consistent scaling
        total_weight = data_handler.get_total_weight()
        if total_weight > 0:
            total_twt = total_twt / total_weight
        
        return {
            'makespan': makespan,
            'twt': total_twt,
            'objective': makespan,  # Single objective: makespan
            'machine_schedule': machine_schedule
        }
    
    baseline_results = calculate_baseline_metrics(baseline_machine_schedule, data_handler)
    
    print(f"\nBaseline Schedule Results:")
    print(f"  Makespan: {baseline_results['makespan']:.2f}")
    print(f"  Total Weighted Tardiness: {baseline_results['twt']:.2f}")
    print(f"  Objective: {baseline_results['objective']:.2f}")

    # Baseline schedule Gantt chart will be created later with detailed validation

    # 3. Graph RL Policy Evaluation
    print("\n=== Graph RL Policy Evaluation ===")
    
    # Set the specific model path as requested
    model_path = "result/exp_graph_rl/20250916_114416/model_20250916_1148.pth"
    
    if os.path.exists(model_path):
        print(f"Using specified model: {model_path}")
        
        # Create a new environment for graph RL evaluation (same instance)
        graph_env = GraphRlEnv(data_handler, alpha=alpha, device=device)
        
        # Create trainer instance for loading the model
        trainer = GraphPPOTrainer(
            problem_data=data_handler,
            hidden_dim=config.graph_rl_params['hidden_dim'],
            num_hgt_layers=config.graph_rl_params['num_hgt_layers'],
            num_heads=config.graph_rl_params['num_heads'],
            lr=config.graph_rl_params['lr'],
            gamma=config.graph_rl_params['gamma'],
            device=device,
            model_save_dir='result/temp'  # Prevent creating graph_rl directory
        )
        
        # Evaluate the graph RL policy
        graph_results = evaluate_graph_policy(model_path, graph_env, trainer)
        
        print(f"\nGraph RL Policy Results:")
        print(f"  Makespan: {graph_results['makespan']:.2f}")
        print(f"  Total Weighted Tardiness: {graph_results['twt']:.2f}")
        print(f"  Objective: {graph_results['objective']:.2f}")
        print(f"  Episode Length: {graph_results['episode_length']}")
        print(f"  Valid Completion: {graph_results['is_valid_completion']}")
        print(f"  Episode Reward: {graph_results['episode_reward']:.2f}")
        
        # Graph RL policy Gantt chart will be created later with detailed validation
        
    else:
        print(f"Model file not found: {model_path}")
        print("Trying to find latest model...")
        model_path = find_latest_model()
        
        if model_path is None:
            print("No trained model found. Skipping graph RL evaluation.")
            graph_results = None
        else:
            print(f"Using found model: {model_path}")
            
            # Create a new environment for graph RL evaluation (same instance)
            graph_env = GraphRlEnv(data_handler, alpha=alpha, device=device)
            
            # Create trainer instance for loading the model
            trainer = GraphPPOTrainer(
                problem_data=data_handler,
                hidden_dim=config.graph_rl_params['hidden_dim'],
                num_hgt_layers=config.graph_rl_params['num_hgt_layers'],
                num_heads=config.graph_rl_params['num_heads'],
                lr=config.graph_rl_params['lr'],
                gamma=config.graph_rl_params['gamma'],
                device=device,
                model_save_dir='result/temp'  # Prevent creating graph_rl directory
            )
            
            # Evaluate the graph RL policy
            graph_results = evaluate_graph_policy(model_path, graph_env, trainer)
            
            print(f"\nGraph RL Policy Results:")
            print(f"  Makespan: {graph_results['makespan']:.2f}")
            print(f"  Total Weighted Tardiness: {graph_results['twt']:.2f}")
            print(f"  Objective: {graph_results['objective']:.2f}")
            print(f"  Episode Length: {graph_results['episode_length']}")
            print(f"  Valid Completion: {graph_results['is_valid_completion']}")
            print(f"  Episode Reward: {graph_results['episode_reward']:.2f}")
            
            # Graph RL policy Gantt chart will be created later with detailed validation

    # 4. Solution Validation and Comparison
    print("\n=== Solution Validation and Comparison ===")
    
    # Data is already prepared by the policy evaluation functions
    print("Using pre-prepared solution data from policy evaluations...")
    
    if graph_results is not None:
        graph_solution_data = graph_results['solution_data']
    else:
        graph_solution_data = None

    print("Data preparation completed!")

    # Create SolutionUtils instances and validate solutions
    print("\n--- Baseline Schedule Validation ---")
    baseline_solution_utils = SolutionUtils(
        machine_schedule=baseline_results['machine_schedule'],
        num_machines=data_handler.num_machines,
        num_operations=data_handler.num_operations,
        job_assignments=baseline_solution_data['job_assignments'],
        job_due_dates=baseline_solution_data['job_due_dates'],
        machine_assignments=baseline_solution_data['machine_assignments']
    )

    baseline_validation = baseline_solution_utils.validate_solution()
    print(f"Baseline Schedule Validation:")
    print(f"  Valid: {baseline_validation['is_valid']}")
    print(f"  Makespan: {baseline_validation['makespan']:.2f}")
    print(f"  Scheduled Operations: {baseline_validation['scheduled_operations']}/{baseline_validation['total_operations']}")
    if not baseline_validation['is_valid']:
        print("  Violations:")
        for violation in baseline_validation['violations']:
            print(f"    - {violation}")

    # Graph RL policy validation (if available)
    if graph_results is not None:
        print("\n--- Graph RL Policy Validation ---")
        graph_solution_utils = SolutionUtils(
            machine_schedule=graph_results['machine_schedule'],
            num_machines=data_handler.num_machines,
            num_operations=data_handler.num_operations,
            job_assignments=graph_solution_data['job_assignments'],
            job_due_dates=graph_solution_data['job_due_dates'],
            machine_assignments=graph_solution_data['machine_assignments']
        )
        
        graph_validation = graph_solution_utils.validate_solution()
        print(f"Graph RL Policy Validation:")
        print(f"  Valid: {graph_validation['is_valid']}")
        print(f"  Makespan: {graph_validation['makespan']:.2f}")
        print(f"  Scheduled Operations: {graph_validation['scheduled_operations']}/{graph_validation['total_operations']}")
        if not graph_validation['is_valid']:
            print("  Violations:")
            for violation in graph_validation['violations']:
                print(f"    - {violation}")
    else:
        graph_solution_utils = None
        graph_validation = None

    # Create detailed Gantt charts using SolutionUtils
    print("\n=== Creating Detailed Gantt Charts ===")

    # Generate timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Baseline schedule Gantt chart
    print("\n--- Baseline Schedule Detailed Gantt Chart ---")
    baseline_gantt_detailed_path = f"result/exp_instance_viewer_baseline_schedule_detailed_gantt_{timestamp}.png"
    baseline_gantt_detailed = baseline_solution_utils.draw_gantt(
        show_validation=True,
        title=f"Baseline Schedule (Makespan: {baseline_validation['makespan']:.1f} min)"
    )
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(baseline_gantt_detailed_path), exist_ok=True)
    baseline_gantt_detailed.write_image(baseline_gantt_detailed_path, format="png", width=1200, height=600)
    print(f"Detailed Gantt chart saved to: {baseline_gantt_detailed_path}")

    # Graph RL policy Gantt chart (if available)
    if graph_solution_utils is not None:
        print("\n--- Graph RL Policy Detailed Gantt Chart ---")
        graph_gantt_detailed_path = f"result/exp_instance_viewer_graph_rl_policy_detailed_gantt_{timestamp}.png"
        graph_gantt_detailed = graph_solution_utils.draw_gantt(
            show_validation=True,
            title=f"Graph RL Policy Schedule (Makespan: {graph_validation['makespan']:.1f} min)"
        )
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(graph_gantt_detailed_path), exist_ok=True)
        graph_gantt_detailed.write_image(graph_gantt_detailed_path, format="png", width=1200, height=600)
        print(f"Detailed Gantt chart saved to: {graph_gantt_detailed_path}")

    # Performance comparison
    print("\n=== Performance Comparison ===")

    print(f"\nBaseline Schedule:")
    print(f"  Makespan: {baseline_validation['makespan']:.2f}")
    print(f"  TWT: {baseline_results['twt']:.2f}")
    print(f"  Objective: {baseline_results['objective']:.2f}")
    print(f"  Valid Completion: {baseline_validation['is_valid']}")

    if graph_validation is not None:
        print(f"\nGraph RL Policy:")
        print(f"  Makespan: {graph_validation['makespan']:.2f}")
        print(f"  TWT: {graph_results['twt']:.2f}")
        print(f"  Objective: {graph_results['objective']:.2f}")
        print(f"  Valid Completion: {graph_validation['is_valid']}")
        
        # Calculate improvement
        makespan_improvement = ((baseline_validation['makespan'] - graph_validation['makespan']) / baseline_validation['makespan']) * 100
        twt_improvement = ((baseline_results['twt'] - graph_results['twt']) / baseline_results['twt']) * 100 if baseline_results['twt'] > 0 else 0
        
        print(f"\nImprovement (Graph RL vs Baseline):")
        print(f"  Makespan: {makespan_improvement:.1f}%")
        print(f"  TWT: {twt_improvement:.1f}%")
    else:
        print("\nGraph RL Policy: Not available (no trained model found)")

    print(f"\nInstance Information:")
    print(f"  Jobs: {data_handler.num_jobs}")
    print(f"  Machines: {data_handler.num_machines}")
    print(f"  Operations: {data_handler.num_operations}")
    print(f"  Alpha (TWT weight): {alpha}")
    
    print("\n=== All Gantt charts saved as PNG files in result/ directory ===")
    print("Files created:")
    print(f"  - {baseline_gantt_detailed_path}")
    if graph_results is not None:
        print(f"  - {graph_gantt_detailed_path}")

if __name__ == "__main__":
    main()

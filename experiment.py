"""
Simple experiment to test MILP algorithm functionality.
Tests performance and generates Gantt chart visualization.
"""

from benchmark.data_handler import FlexibleJobShopDataHandler
from MILP.model import MILP
from utils.solution_utils import SolutionUtils
import os
import random

def run_experiment(dataset_path: str, time_limit: int = 120):
    """
    Run a complete experiment with MILP algorithm.
    
    Args:
        dataset_path: Path to the dataset file
        time_limit: Time limit for MILP solver in seconds
    """
    print("=" * 60)
    print("FLEXIBLE JOB SHOP - MILP EXPERIMENT")
    print("=" * 60)
    
    # Step 1: Load dataset
    print(f"\n1. Loading dataset: {dataset_path}")
    try:
        data_handler = FlexibleJobShopDataHandler(dataset_path)
        print(f"   ✓ Dataset loaded successfully")
        print(f"   - Jobs: {data_handler.num_jobs}")
        print(f"   - Machines: {data_handler.num_machines}")
        print(f"   - Operations: {data_handler.num_operations}")
    except Exception as e:
        print(f"   ❌ Error loading dataset: {e}")
        return
    
    # Step 2: Create and solve MILP model
    print(f"\n2. Solving with MILP (time limit: {time_limit}s)")
    try:
        milp_model = MILP(data_handler)
        
        # Build model
        print("   Building model...")
        milp_model.build_model(time_limit=time_limit, MIPFocus=1)
        
        # Solve model
        print("   Solving model...")
        solution = milp_model.solve()
        
        # Print results
        print("\n   Results:")
        perf = solution["performance"]
        sched = solution["schedule_result"]
        num_scheduled = sum(len(ops) for ops in sched.values())
        
        print(f"   Status: {perf['status']}")
        print(f"   Objective (Makespan): {perf['objective']:.2f if perf['objective'] is not None else 'N/A'}")
        print(f"   Solve Time: {perf['solve_time']:.2f} seconds")
        print(f"   Scheduled Operations: {num_scheduled}/{data_handler.num_operations}")
        
    except Exception as e:
        print(f"   ❌ Error solving MILP: {e}")
        return
    
    # Step 3: Validate and visualize solution
    if solution["schedule_result"]:
        print(f"\n3. Validating and visualizing solution")
        
        try:
            # Get machine schedule directly from solution
            machine_schedule = solution["schedule_result"]
            
            # Create solution utils
            solution_utils = SolutionUtils(data_handler, machine_schedule)
            
            # Validate solution
            validation_result = solution_utils.validate_solution()
            print(f"   ✓ Validation: {'Valid' if validation_result['is_valid'] else 'Invalid'}")
            
            if not validation_result['is_valid']:
                print("   ⚠ Validation violations:")
                for violation in validation_result['violations'][:3]:  # Show first 3 violations
                    print(f"     - {violation}")
                if len(validation_result['violations']) > 3:
                    print(f"     ... and {len(validation_result['violations']) - 3} more")
            
            # Generate Gantt chart
            print("   Generating Gantt chart...")
            fig = solution_utils.draw_gantt(figsize=(1200, 600))
            print("   ✓ Gantt chart generated (check browser window)")
            
        except Exception as e:
            print(f"   ❌ Error in validation/visualization: {e}")
    else:
        print(f"\n3. No solution found - skipping validation and visualization")
    
    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)

def main():
    """Main function to run experiments."""
    
    # List of available datasets
    datasets = [
        "benchmark/datasets/brandimarte/mk01.txt",
        "benchmark/datasets/brandimarte/mk02.txt", 
        "benchmark/datasets/brandimarte/mk03.txt",
        "benchmark/datasets/brandimarte/mk04.txt",
        "benchmark/datasets/brandimarte/mk05.txt",
        "benchmark/datasets/hurink/edata/la01.txt",
        "benchmark/datasets/hurink/edata/la02.txt",
        "benchmark/datasets/hurink/edata/la03.txt"
    ]
    
    # Filter to only existing datasets
    available_datasets = [d for d in datasets if os.path.exists(d)]
    
    if not available_datasets:
        print("❌ No datasets found!")
        return
    
    # Randomly select a dataset
    selected_dataset = random.choice(available_datasets)
    print(f"Selected dataset: {selected_dataset}")
    
    # Run experiment
    run_experiment(selected_dataset, time_limit=15*60)

if __name__ == "__main__":
    main()

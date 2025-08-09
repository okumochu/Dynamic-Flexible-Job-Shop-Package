"""
Simple experiment to test MILP algorithm functionality.
Tests performance and generates Gantt chart visualization.
"""

from benchmarks.static_benchmark.data_handler import FlexibleJobShopDataHandler
from MILP.model import MILP
from utils.solution_utils import SolutionUtils
import os
import random

def run_experiment(data_source, data_type: str, time_limit: int = 600, twt_weight: float = 0.5):
    """
    Run a complete experiment with MILP algorithm.
    
    Args:
        data_source: Path to dataset file or simulation parameters
        data_type: "dataset" or "simulation"
        time_limit: Time limit for MILP solver in seconds
        twt_weight: Weight for total weighted tardiness (0.0 = makespan only, 1.0 = TWT only)
    """
    print("=" * 60)
    print(f"FLEXIBLE JOB SHOP - MILP EXPERIMENT ({data_type.upper()})")
    print("=" * 60)
    
    # Step 1: Load data
    print(f"\n1. Loading {data_type}: {data_source}")
    data_handler = FlexibleJobShopDataHandler(data_source, data_type=data_type)
    print(f"   ✓ {data_type.capitalize()} loaded successfully")
    print(f"   - Jobs: {data_handler.num_jobs}")
    print(f"   - Machines: {data_handler.num_machines}")
    print(f"   - Operations: {data_handler.num_operations}")
    
    # Show due date and weight information
    due_dates = data_handler.get_jobs_due_date()
    weights = data_handler.get_jobs_weight()
    print(f"   - Average Due Date: {sum(due_dates) / len(due_dates):.1f}")
    print(f"   - Average Weight: {sum(weights) / len(weights):.1f}")
    
    # Step 2: Create and solve MILP model
    print(f"\n2. Solving with MILP (time limit: {time_limit}s, TWT weight: {twt_weight})")
    milp_model = MILP(data_handler, twt_weight=twt_weight)
    
    # Build model
    print("   Building model...")
    milp_model.build_model(time_limit=time_limit, MIPFocus=1, verbose=0)
    print("   ✓ Model built successfully")
    
    # Solve model
    print("   Solving model...")
    solution = milp_model.solve()
    
    # Print results
    print("\n   Results:")
    performance = solution["performance"]
    print(f"   - Status: {performance['status']}")
    print(f"   - Objective Value: {performance['objective']:.2f}")
    print(f"   - Solve Time: {performance['solve_time']:.2f}s")
    print(f"   - Makespan: {performance['makespan']:.2f}")
    print(f"   - Total Weighted Tardiness: {performance['total_weighted_tardiness']:.2f}")
    
    # Step 3: Validate and visualize solution
    if solution["schedule_result"]:
        print(f"\n3. Validating and visualizing solution")
        
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
        
        # Generate and save Gantt chart (headless, HTML)
        print("   Generating Gantt chart (headless)...")
        fig = solution_utils.draw_gantt(show_due_dates=True)
        output_dir = os.path.join("result", "milp_experiment")
        os.makedirs(output_dir, exist_ok=True)
        html_path = os.path.join(output_dir, "milp_gantt.html")
        fig.write_html(html_path, include_plotlyjs="cdn")
        print(f"   ✓ Gantt chart saved to: {html_path}")
    else:
        print(f"\n3. No solution found - skipping validation and visualization")
    
    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)

def run_simulation_experiment():
    """Run experiment with synthetic data using config.py parameters."""
    print("\n" + "=" * 80)
    print("SIMULATION EXPERIMENT")
    print("=" * 80)
    
    from config import config
    simulation_params = config.simulation_params
    
    run_experiment(simulation_params, "simulation", time_limit=1800, twt_weight=0.5)


def main():
    """Main function to run experiments."""
    
    print("FLEXIBLE JOB SHOP - MILP EXPERIMENTS")
    print("Testing complex simulation with TWT optimization")
    print("Parameters: 60 minutes time limit, TWT weight = 0.3, verbose = 0")
    
    # Run simulation experiment
    run_simulation_experiment()
    
    
    print("\n" + "=" * 80)
    print("ALL EXPERIMENTS COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()

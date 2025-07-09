"""
Comparison experiment between RL and MILP algorithms.
Tests both methods on the same synthetic data configuration and compares results.
"""

import os
import torch
import time
import matplotlib.pyplot as plt
from RL.flat_rl_env import FlatRLEnv
from RL.flat_rl_trainer import FlatRLTrainer
from MILP.model import MILP
from benchmarks.static_benchmark.data_handler import FlexibleJobShopDataHandler
from utils.policy_utils import showcase_flat_policy, create_gantt_chart
from utils.solution_utils import SolutionUtils
import wandb

class ComparisonExperiment:
    """Comparison experiment between RL and MILP algorithms."""
    
    def __init__(self, simulation_params, rl_params, milp_time_limit=3600, twt_weight=0.5):
        """
        Initialize comparison experiment.
        
        Args:
            simulation_params: Dictionary with synthetic data parameters
            rl_params: Dictionary with RL training parameters
            milp_time_limit: Time limit for MILP solver in seconds
            twt_weight: Weight for total weighted tardiness (0.0 = makespan only, 1.0 = TWT only)
        """
        self.simulation_params = simulation_params
        self.rl_params = rl_params
        self.milp_time_limit = milp_time_limit
        self.twt_weight = twt_weight
        
        # Create result directories
        self.result_dir = "result/comparison"
        self.rl_model_dir = os.path.join(self.result_dir, "rl_model")
        self.rl_training_dir = os.path.join(self.result_dir, "rl_training")
        self.comparison_dir = os.path.join(self.result_dir, "comparison_results")
        
        for dir_path in [self.rl_model_dir, self.rl_training_dir, self.comparison_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        # Set wandb directory
        os.environ["WANDB_DIR"] = self.rl_training_dir
        
        self.results = {}
    
    def create_data_handler(self):
        """Create data handler with synthetic data."""
        print("Creating synthetic data...")
        print(f"Configuration: {self.simulation_params}")
        
        data_handler = FlexibleJobShopDataHandler(
            data_source=self.simulation_params, 
            data_type="simulation"
        )
        
        print(f"✓ Data created: {data_handler.num_jobs} jobs, {data_handler.num_machines} machines")
        print(f"  Operations: {data_handler.num_operations}")
        
        # Show due date and weight information
        due_dates = data_handler.get_jobs_due_date()
        weights = data_handler.get_jobs_weight()
        print(f"  Average Due Date: {sum(due_dates) / len(due_dates):.1f}")
        print(f"  Average Weight: {sum(weights) / len(weights):.1f}")
        
        return data_handler
    
    def run_rl_experiment(self, data_handler):
        """Run RL experiment using pre-trained model."""
        print("\n" + "="*60)
        print("RUNNING RL EXPERIMENT (Using Pre-trained Model)")
        print("="*60)
        
        try:
            # Create environment
            env = FlatRLEnv(data_handler, alpha=self.rl_params['alpha'])
            print(f"Environment created: {env.num_jobs} jobs, {env.num_machines} machines")
            print(f"Observation dimension: {env.obs_len}, Action dimension: {env.action_dim}")
            
            # Use the pre-trained model directory
            pretrained_model_dir = "result/flat_rl/model"
            
            # Check if model exists
            model_path = os.path.join(pretrained_model_dir, "final_model.pth")
            if not os.path.exists(model_path):
                print(f"❌ Pre-trained model not found at {model_path}")
                print("Please train a model first or check the model path.")
                return False
            
            print(f"✓ Found pre-trained model at: {model_path}")
            
            # Use showcase function to evaluate the pre-trained model
            start_time = time.time()
            
            # Create Gantt chart path
            rl_gantt_path = os.path.join(self.comparison_dir, "rl_gantt.png")
            
            # Showcase the policy
            showcaser_result = showcase_flat_policy(model_dir=pretrained_model_dir, env=env)
            
            # Create Gantt chart separately
            create_gantt_chart(showcaser_result, save_path=rl_gantt_path, title_suffix="RL")
            evaluation_time = time.time() - start_time
            
            print(f"✓ RL evaluation completed in {evaluation_time:.2f}s")
            
            # Store results
            final_result = {
                'makespan': showcaser_result['makespan'],
                'twt': showcaser_result['twt'],
                'total_reward': showcaser_result['total_reward'],
                'objective': showcaser_result['makespan'] + self.twt_weight * showcaser_result['twt'],
                'solve_time': evaluation_time,  # Just evaluation time, not training time
                'method': 'RL (Pre-trained)',
                'gantt_path': rl_gantt_path,
                'steps_taken': showcaser_result['steps_taken'],
                'valid_completion': showcaser_result['is_valid_completion']
            }
            
            self.results['RL'] = final_result
            
            print(f"RL Results:")
            print(f"  Makespan: {final_result['makespan']:.2f}")
            print(f"  TWT: {final_result['twt']:.2f}")
            print(f"  Objective: {final_result['objective']:.2f}")
            print(f"  Total Reward: {final_result['total_reward']:.2f}")
            print(f"  Steps Taken: {final_result['steps_taken']}")
            print(f"  Valid Completion: {final_result['valid_completion']}")
            print(f"  Evaluation Time: {final_result['solve_time']:.2f}s")
            
            return True
            
        except Exception as e:
            print(f"❌ RL experiment failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def run_milp_experiment(self, data_handler):
        """Run MILP experiment."""
        print("\n" + "="*60)
        print("RUNNING MILP EXPERIMENT")
        print("="*60)
        
        try:
            # Create MILP model
            milp_model = MILP(data_handler, twt_weight=self.twt_weight)
            
            # Build and solve model
            print(f"Building MILP model (time limit: {self.milp_time_limit}s)...")
            milp_model.build_model(time_limit=self.milp_time_limit, MIPFocus=1, verbose=0)
            
            print("Solving MILP model...")
            start_time = time.time()
            solution = milp_model.solve()
            solve_time = time.time() - start_time
            
            # Extract results
            performance = solution["performance"]
            print(f"✓ MILP completed in {solve_time:.2f}s")
            print(f"  Status: {performance['status']}")
            print(f"  Objective: {performance['objective']:.2f}")
            print(f"  Makespan: {performance['makespan']:.2f}")
            print(f"  TWT: {performance['total_weighted_tardiness']:.2f}")
            
            # Create Gantt chart if solution exists
            milp_gantt_path = os.path.join(self.comparison_dir, "milp_gantt.png")
            
            if solution["schedule_result"]:
                try:
                    machine_schedule = solution["schedule_result"]
                    solution_utils = SolutionUtils(data_handler, machine_schedule)
                    
                    # Validate solution
                    validation_result = solution_utils.validate_solution()
                    print(f"  Validation: {'Valid' if validation_result['is_valid'] else 'Invalid'}")
                    
                    # Generate Gantt chart
                    fig = solution_utils.draw_gantt(show_due_dates=True)
                    fig.write_image(milp_gantt_path, width=1200, height=800)
                    print(f"  Gantt chart saved: {milp_gantt_path}")
                    
                except Exception as e:
                    print(f"  Warning: Could not create Gantt chart: {e}")
                    milp_gantt_path = None
            else:
                print("  No solution found - no Gantt chart created")
                milp_gantt_path = None
            
            # Store results
            self.results['MILP'] = {
                'makespan': performance['makespan'],
                'twt': performance['total_weighted_tardiness'],
                'objective': performance['objective'],
                'solve_time': performance['solve_time'],
                'status': performance['status'],
                'method': 'MILP',
                'gantt_path': milp_gantt_path
            }
            
            return True
            
        except Exception as e:
            print(f"❌ MILP experiment failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def compare_results(self):
        """Compare and display results from both methods."""
        print("\n" + "="*80)
        print("DETAILED COMPARISON RESULTS")
        print("="*80)
        
        if not self.results:
            print("No results to compare!")
            return
        
        # Print experiment configuration
        print("\nEXPERIMENT CONFIGURATION:")
        print("-" * 40)
        print(f"Problem Size: {self.simulation_params['num_jobs']} jobs × {self.simulation_params['num_machines']} machines")
        print(f"Operations per job: {self.simulation_params['operation_lb']}-{self.simulation_params['operation_ub']}")
        print(f"Processing time range: {self.simulation_params['processing_time_lb']}-{self.simulation_params['processing_time_ub']}")
        print(f"Compatible machines: {self.simulation_params['compatible_machines_lb']}-{self.simulation_params['compatible_machines_ub']}")
        print(f"TWT Weight: {self.twt_weight} (Objective = Makespan + {self.twt_weight} × TWT)")
        print(f"Random seed: {self.simulation_params['seed']}")
        
        # Create detailed comparison table
        print(f"\nPERFORMANCE COMPARISON:")
        print("=" * 100)
        header = f"{'Method':<8} {'Makespan':<12} {'TWT':<12} {'Objective':<12} {'Time (s)':<12} {'Status':<15} {'Efficiency':<10}"
        print(header)
        print("-" * 100)
        
        # Collect data for efficiency calculation
        best_objective = float('inf')
        best_time = float('inf')
        
        for method, result in self.results.items():
            if result['objective'] < best_objective:
                best_objective = result['objective']
            if result['solve_time'] < best_time:
                best_time = result['solve_time']
        
        # Print results with efficiency metrics
        for method, result in self.results.items():
            status = result.get('status', 'Completed')
            
            # Calculate efficiency (lower is better - objective quality vs time)
            if best_objective > 0 and best_time > 0:
                obj_ratio = result['objective'] / best_objective
                time_ratio = result['solve_time'] / best_time
                efficiency = obj_ratio / time_ratio if time_ratio > 0 else float('inf')
                efficiency_str = f"{efficiency:.2f}"
            else:
                efficiency_str = "N/A"
            
            print(f"{method:<8} {result['makespan']:<12.2f} {result['twt']:<12.2f} "
                  f"{result['objective']:<12.2f} {result['solve_time']:<12.2f} {status:<15} {efficiency_str:<10}")
        
        print("=" * 100)
        
        # Detailed winner analysis
        if len(self.results) == 2:
            rl_result = self.results.get('RL', {})
            milp_result = self.results.get('MILP', {})
            
            print(f"\nDETAILED COMPARISON ANALYSIS:")
            print("-" * 50)
            
            # Objective comparison
            rl_obj = rl_result.get('objective', float('inf'))
            milp_obj = milp_result.get('objective', float('inf'))
            
            if rl_obj < milp_obj:
                obj_improvement = ((milp_obj - rl_obj) / milp_obj) * 100
                print(f"🏆 OBJECTIVE WINNER: RL")
                print(f"   RL objective:   {rl_obj:.2f}")
                print(f"   MILP objective: {milp_obj:.2f}")
                print(f"   Improvement:    {obj_improvement:.1f}% better")
            elif milp_obj < rl_obj:
                obj_improvement = ((rl_obj - milp_obj) / rl_obj) * 100
                print(f"🏆 OBJECTIVE WINNER: MILP")
                print(f"   MILP objective: {milp_obj:.2f}")
                print(f"   RL objective:   {rl_obj:.2f}")
                print(f"   Improvement:    {obj_improvement:.1f}% better")
            else:
                print(f"🤝 OBJECTIVE TIE: Both achieve {rl_obj:.2f}")
            
            # Time comparison
            rl_time = rl_result.get('solve_time', float('inf'))
            milp_time = milp_result.get('solve_time', float('inf'))
            
            print(f"\n⏱️  TIME COMPARISON:")
            print(f"   Note: RL time is evaluation time only (using pre-trained model)")
            if rl_time < milp_time:
                time_improvement = ((milp_time - rl_time) / milp_time) * 100
                print(f"   RL evaluation time: {rl_time:.2f}s")
                print(f"   MILP solve time:    {milp_time:.2f}s")
                print(f"   RL evaluation is {time_improvement:.1f}% faster")
            elif milp_time < rl_time:
                time_improvement = ((rl_time - milp_time) / rl_time) * 100
                print(f"   MILP solve time:    {milp_time:.2f}s")
                print(f"   RL evaluation time: {rl_time:.2f}s")
                print(f"   MILP is {time_improvement:.1f}% faster")
            else:
                print(f"   Both methods took {rl_time:.2f}s")
            
            # Makespan comparison
            rl_makespan = rl_result.get('makespan', 0)
            milp_makespan = milp_result.get('makespan', 0)
            
            print(f"\n📊 MAKESPAN COMPARISON:")
            if rl_makespan < milp_makespan:
                makespan_improvement = ((milp_makespan - rl_makespan) / milp_makespan) * 100
                print(f"   RL makespan:   {rl_makespan:.2f}")
                print(f"   MILP makespan: {milp_makespan:.2f}")
                print(f"   RL is {makespan_improvement:.1f}% better")
            elif milp_makespan < rl_makespan:
                makespan_improvement = ((rl_makespan - milp_makespan) / rl_makespan) * 100
                print(f"   MILP makespan: {milp_makespan:.2f}")
                print(f"   RL makespan:   {rl_makespan:.2f}")
                print(f"   MILP is {makespan_improvement:.1f}% better")
            else:
                print(f"   Both achieve makespan: {rl_makespan:.2f}")
            
            # TWT comparison
            rl_twt = rl_result.get('twt', 0)
            milp_twt = milp_result.get('twt', 0)
            
            print(f"\n📈 TWT COMPARISON:")
            if rl_twt < milp_twt:
                twt_improvement = ((milp_twt - rl_twt) / milp_twt) * 100 if milp_twt > 0 else 100
                print(f"   RL TWT:   {rl_twt:.2f}")
                print(f"   MILP TWT: {milp_twt:.2f}")
                print(f"   RL is {twt_improvement:.1f}% better")
            elif milp_twt < rl_twt:
                twt_improvement = ((rl_twt - milp_twt) / rl_twt) * 100 if rl_twt > 0 else 100
                print(f"   MILP TWT: {milp_twt:.2f}")
                print(f"   RL TWT:   {rl_twt:.2f}")
                print(f"   MILP is {twt_improvement:.1f}% better")
            else:
                print(f"   Both achieve TWT: {rl_twt:.2f}")
            
            # Overall recommendation
            print(f"\n💡 RECOMMENDATION:")
            if rl_obj < milp_obj and rl_time < milp_time:
                print("   RL is superior in both objective value and computation time")
            elif milp_obj < rl_obj and milp_time < rl_time:
                print("   MILP is superior in both objective value and computation time")
            elif rl_obj < milp_obj:
                print("   RL achieves better objective but takes longer to compute")
            elif milp_obj < rl_obj:
                print("   MILP achieves better objective but takes longer to compute")
            else:
                if rl_time < milp_time:
                    print("   Both achieve same objective, but RL is faster")
                else:
                    print("   Both achieve same objective, but MILP is faster")
        
        # Save detailed comparison results
        comparison_file = os.path.join(self.comparison_dir, "detailed_comparison_results.txt")
        with open(comparison_file, 'w') as f:
            f.write("DETAILED COMPARISON EXPERIMENT RESULTS\n")
            f.write("="*60 + "\n\n")
            
            f.write("EXPERIMENT CONFIGURATION:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Problem Size: {self.simulation_params['num_jobs']} jobs × {self.simulation_params['num_machines']} machines\n")
            f.write(f"Operations per job: {self.simulation_params['operation_lb']}-{self.simulation_params['operation_ub']}\n")
            f.write(f"Processing time range: {self.simulation_params['processing_time_lb']}-{self.simulation_params['processing_time_ub']}\n")
            f.write(f"Compatible machines: {self.simulation_params['compatible_machines_lb']}-{self.simulation_params['compatible_machines_ub']}\n")
            f.write(f"TWT Weight: {self.twt_weight}\n")
            f.write(f"Random seed: {self.simulation_params['seed']}\n\n")
            
            f.write("ALGORITHM PARAMETERS:\n")
            f.write("-" * 25 + "\n")
            f.write(f"RL Parameters: {self.rl_params}\n")
            f.write(f"MILP Time Limit: {self.milp_time_limit}s\n\n")
            
            f.write("PERFORMANCE RESULTS:\n")
            f.write("-" * 20 + "\n")
            f.write(f"{'Method':<8} {'Makespan':<12} {'TWT':<12} {'Objective':<12} {'Time (s)':<12} {'Status':<15}\n")
            f.write("-" * 80 + "\n")
            
            for method, result in self.results.items():
                status = result.get('status', 'Completed')
                f.write(f"{method:<8} {result['makespan']:<12.2f} {result['twt']:<12.2f} "
                       f"{result['objective']:<12.2f} {result['solve_time']:<12.2f} {status:<15}\n")
        
        print(f"\nDetailed results saved to: {comparison_file}")
        print(f"Gantt charts saved in: {self.comparison_dir}")
        print(f"  - RL Gantt chart: rl_gantt.png")
        print(f"  - MILP Gantt chart: milp_gantt.png")
    
    def run_comparison(self):
        """Run complete comparison experiment."""
        print("FLEXIBLE JOB SHOP - RL vs MILP COMPARISON")
        print("="*60)
        
        # Create data handler
        data_handler = self.create_data_handler()
        
        # Run both experiments
        rl_success = self.run_rl_experiment(data_handler)
        milp_success = self.run_milp_experiment(data_handler)
        
        # Compare results
        if rl_success or milp_success:
            self.compare_results()
        else:
            print("Both experiments failed!")
        
        print(f"\nExperiment completed! Results in: {self.comparison_dir}")


def main():
    """Main function to run the comparison experiment."""
    
    # Use the same simulation parameters from rl_experiment.py
    simulation_params = {
        'num_jobs': 12,
        'num_machines': 4,
        'operation_lb': 4,
        'operation_ub': 4,
        'processing_time_lb': 5,
        'processing_time_ub': 5,   
        'compatible_machines_lb': 2,
        'compatible_machines_ub': 2,
        'seed': 42,
    }
    
    # Calculate total max steps for RL parameters
    total_max_steps = simulation_params['num_jobs'] * simulation_params['operation_ub'] * simulation_params['num_machines']
    
    # RL parameters (for reference - using pre-trained model)
    rl_params = {
        'alpha': 0.5,
        'gamma': 0.99,
        'gae_lambda': 0.9,
        'steps_per_epoch': total_max_steps,
        'epochs': 500,  
        'pi_lr': 1e-5,  # Reduced from 3e-5
        'v_lr': 1e-5,   # Reduced from 1e-4
        'target_kl': 0.5,  # Increased from 0.1
        'train_pi_iters': total_max_steps,
        'train_v_iters': total_max_steps,
    }
    
    # MILP parameters
    milp_time_limit = 1800  # 30 minutes
    twt_weight = 0.5  # Equal weight for makespan and TWT
    
    print("Starting RL vs MILP comparison experiment...")
    print(f"Problem size: {simulation_params['num_jobs']} jobs, {simulation_params['num_machines']} machines")
    print(f"RL: Using pre-trained model from result/flat_rl/model/")
    print(f"MILP: Time limit {milp_time_limit}s, TWT weight: {twt_weight}")
    
    # Create and run comparison experiment
    experiment = ComparisonExperiment(
        simulation_params=simulation_params,
        rl_params=rl_params,
        milp_time_limit=milp_time_limit,
        twt_weight=twt_weight
    )
    
    experiment.run_comparison()


if __name__ == "__main__":
    main() 
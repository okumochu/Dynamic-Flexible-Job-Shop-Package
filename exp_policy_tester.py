"""
Policy Tester for Flat RL vs Hierarchical RL Models on mk06 Dataset
Loads trained models and compares their performance with validation and visualization
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
import torch
from config import config

# Import necessary modules
from benchmarks.static_benchmark.data_handler import FlexibleJobShopDataHandler
from RL.rl_env import RLEnv
from utils.policy_utils import evaluate_flat_policy, evaluate_hierarchical_policy, visualize_policy_schedule
from utils.solution_utils import SolutionUtils

def create_test_environment(dataset_path: str) -> RLEnv:
    """Create test environment from mk06 dataset"""
    # Use TF and RDD from config for due date generation
    tf = config.simulation_params.get('TF', 0.4)
    rdd = config.simulation_params.get('RDD', 0.8)
    seed = config.simulation_params.get('seed', 42)
    
    data_handler = FlexibleJobShopDataHandler(
        data_source=dataset_path,
        data_type="dataset",
        TF=tf,
        RDD=rdd,
        seed=seed
    )
    
    # Create RL environment with dense rewards
    rl_env = RLEnv(
        data_handler, 
                    alpha=config.rl_params['alpha'], 
        use_reward_shaping=True  # Use dense rewards for real datasets
    )
    
    return rl_env

def load_model_paths() -> Tuple[str, str]:
    """Load the paths to the trained models"""
    # Use specific model paths provided by user
    flat_model_path = "/home/r13725046okumo/project/Dynamic-Flexible-Job-Shop-Package/result/exp_hrl_and_flarl/mk06/flat/flat_rl/model/model_20250804_1615.pth"
    hrl_model_path = "/home/r13725046okumo/project/Dynamic-Flexible-Job-Shop-Package/result/exp_hrl_and_flarl/mk06/hrl/hierarchical_rl/model/model_20250804_1657.pth"
    
    # Verify files exist
    if not os.path.exists(flat_model_path):
        raise FileNotFoundError(f"Flat RL model not found: {flat_model_path}")
    if not os.path.exists(hrl_model_path):
        raise FileNotFoundError(f"Hierarchical RL model not found: {hrl_model_path}")
    
    return flat_model_path, hrl_model_path

def test_flat_rl_model(model_path: str, test_env: RLEnv, num_episodes: int = 5) -> Dict[str, Any]:
    """Test flat RL model and return results"""
    print(f"\n🧪 Testing Flat RL Model: {os.path.basename(model_path)}")
    print("="*60)
    
    # Evaluate the model
    evaluation_result = evaluate_flat_policy(model_path, test_env, num_episodes)
    
    # Print results
    print(f"✅ Flat RL Evaluation Results:")
    print(f"   📊 Average Reward: {evaluation_result.get('avg_reward', evaluation_result.get('episode_reward', 0)):.2f}")
    print(f"   ⏱️  Average Makespan: {evaluation_result.get('avg_makespan', evaluation_result.get('makespan', 0)):.2f}")
    print(f"   📈 Average TWT: {evaluation_result.get('avg_twt', evaluation_result.get('twt', 0)):.2f}")
    print(f"   🎯 Average Objective: {evaluation_result.get('avg_objective', evaluation_result.get('objective', 0)):.2f}")
    print(f"   ✅ Valid Completion: {evaluation_result.get('is_valid_completion', True)}")
    
    return {
        'model_type': 'Flat RL',
        'model_path': model_path,
        'evaluation_result': evaluation_result,
        'success': True
    }

def test_hierarchical_rl_model(model_path: str, test_env: RLEnv, num_episodes: int = 5) -> Dict[str, Any]:
    """Test hierarchical RL model and return results"""
    print(f"\n🧪 Testing Hierarchical RL Model: {os.path.basename(model_path)}")
    print("="*60)
    
    # Evaluate the model
    evaluation_result = evaluate_hierarchical_policy(model_path, test_env, num_episodes)
    
    # Print results
    print(f"✅ Hierarchical RL Evaluation Results:")
    print(f"   📊 Average Reward: {evaluation_result.get('avg_reward', evaluation_result.get('episode_reward', 0)):.2f}")
    print(f"   ⏱️  Average Makespan: {evaluation_result.get('avg_makespan', evaluation_result.get('makespan', 0)):.2f}")
    print(f"   📈 Average TWT: {evaluation_result.get('avg_twt', evaluation_result.get('twt', 0)):.2f}")
    print(f"   🎯 Average Objective: {evaluation_result.get('avg_objective', evaluation_result.get('objective', 0)):.2f}")
    print(f"   ✅ Valid Completion: {evaluation_result.get('is_valid_completion', True)}")
    
    return {
        'model_type': 'Hierarchical RL',
        'model_path': model_path,
        'evaluation_result': evaluation_result,
        'success': True
    }

def validate_solution(evaluation_result: Dict, env: RLEnv) -> Dict[str, Any]:
    """Validate the solution using SolutionUtils"""
    print(f"\n🔍 Validating Solution...")
    
    machine_schedule = evaluation_result['machine_schedule']
    data_handler = env.data_handler
    
    # Create SolutionUtils instance
    solution_utils = SolutionUtils(data_handler, machine_schedule)
    
    # Validate the solution
    validation_result = solution_utils.validate_solution()
    
    print(f"   ✅ Solution Validation: {'VALID' if validation_result['is_valid'] else 'INVALID'}")
    
    if not validation_result['is_valid']:
        print("   ⚠️  Validation violations:")
        for violation in validation_result['violations']:
            print(f"      - {violation}")
    
    return validation_result

def generate_gantt_chart(evaluation_result: Dict, env: RLEnv, save_path: str, title: str) -> bool:
    """Generate Gantt chart for the evaluation result"""
    print(f"\n📊 Generating Gantt Chart: {title}")
    
    try:
        fig = visualize_policy_schedule(evaluation_result, env, save_path)
        
        if fig is not None:
            print(f"   ✅ Gantt chart saved to: {save_path}")
            return True
        else:
            print(f"   ❌ Failed to generate Gantt chart")
            return False
            
    except Exception as e:
        print(f"   ❌ Error generating Gantt chart: {e}")
        return False

def compare_results(flat_results: Dict, hrl_results: Dict) -> Dict[str, Any]:
    """Compare the results of both models"""
    print(f"\n📈 COMPARISON RESULTS")
    print("="*60)
    
    if not flat_results['success'] or not hrl_results['success']:
        print("❌ Cannot compare results - one or both models failed to load")
        return {}
    
    flat_eval = flat_results['evaluation_result']
    hrl_eval = hrl_results['evaluation_result']
    
    # Extract metrics
    flat_objective = flat_eval.get('avg_objective', flat_eval.get('objective', 0))
    hrl_objective = hrl_eval.get('avg_objective', hrl_eval.get('objective', 0))
    
    flat_makespan = flat_eval.get('avg_makespan', flat_eval.get('makespan', 0))
    hrl_makespan = hrl_eval.get('avg_makespan', hrl_eval.get('makespan', 0))
    
    flat_twt = flat_eval.get('avg_twt', flat_eval.get('twt', 0))
    hrl_twt = hrl_eval.get('avg_twt', hrl_eval.get('twt', 0))
    
    flat_reward = flat_eval.get('avg_reward', flat_eval.get('episode_reward', 0))
    hrl_reward = hrl_eval.get('avg_reward', hrl_eval.get('episode_reward', 0))
    
    # Determine winner for each metric
    comparison = {
        'objective': {
            'flat': flat_objective,
            'hrl': hrl_objective,
            'winner': 'Flat RL' if flat_objective < hrl_objective else 'Hierarchical RL' if hrl_objective < flat_objective else 'Tie',
            'improvement': abs(hrl_objective - flat_objective) / max(flat_objective, 1e-6) * 100 if flat_objective != hrl_objective else 0
        },
        'makespan': {
            'flat': flat_makespan,
            'hrl': hrl_makespan,
            'winner': 'Flat RL' if flat_makespan < hrl_makespan else 'Hierarchical RL' if hrl_makespan < flat_makespan else 'Tie',
            'improvement': abs(hrl_makespan - flat_makespan) / max(flat_makespan, 1e-6) * 100 if flat_makespan != hrl_makespan else 0
        },
        'twt': {
            'flat': flat_twt,
            'hrl': hrl_twt,
            'winner': 'Flat RL' if flat_twt < hrl_twt else 'Hierarchical RL' if hrl_twt < flat_twt else 'Tie',
            'improvement': abs(hrl_twt - flat_twt) / max(flat_twt, 1e-6) * 100 if flat_twt != hrl_twt else 0
        },
        'reward': {
            'flat': flat_reward,
            'hrl': hrl_reward,
            'winner': 'Hierarchical RL' if hrl_reward > flat_reward else 'Flat RL' if flat_reward > hrl_reward else 'Tie',
            'improvement': abs(hrl_reward - flat_reward) / max(abs(flat_reward), 1e-6) * 100 if flat_reward != hrl_reward else 0
        }
    }
    
    # Print comparison
    print(f"📊 METRICS COMPARISON:")
    print(f"   🎯 Objective: Flat RL {flat_objective:.2f} vs HRL {hrl_objective:.2f}")
    print(f"      Winner: {comparison['objective']['winner']} ({comparison['objective']['improvement']:.1f}% diff)")
    
    print(f"   ⏱️  Makespan: Flat RL {flat_makespan:.2f} vs HRL {hrl_makespan:.2f}")
    print(f"      Winner: {comparison['makespan']['winner']} ({comparison['makespan']['improvement']:.1f}% diff)")
    
    print(f"   📈 TWT: Flat RL {flat_twt:.2f} vs HRL {hrl_twt:.2f}")
    print(f"      Winner: {comparison['twt']['winner']} ({comparison['twt']['improvement']:.1f}% diff)")
    
    print(f"   🏆 Reward: Flat RL {flat_reward:.2f} vs HRL {hrl_reward:.2f}")
    print(f"      Winner: {comparison['reward']['winner']} ({comparison['reward']['improvement']:.1f}% diff)")
    
    # Overall winner based on objective (primary metric)
    overall_winner = comparison['objective']['winner']
    print(f"\n🏆 OVERALL WINNER: {overall_winner}")
    
    return comparison

def run_policy_test():
    """Main function to run the policy test"""
    print("🧪 POLICY TESTER: Flat RL vs Hierarchical RL on mk06 Dataset")
    print("="*70)
    
    # Set device
    device = config.rl_params.get('device', 'auto')
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🔧 Using device: {device}")
    
    # Load model paths
    flat_model_path, hrl_model_path = load_model_paths()
    print(f"📁 Flat RL Model: {flat_model_path}")
    print(f"📁 Hierarchical RL Model: {hrl_model_path}")
    
    # Create test environment
    dataset_path = 'benchmarks/static_benchmark/datasets/brandimarte/mk06.txt'
    print(f"\n🌍 Creating test environment from: {dataset_path}")
    
    test_env = create_test_environment(dataset_path)
    print(f"✅ Test environment created - Jobs: {test_env.num_jobs}, Machines: {test_env.num_machines}")
    
    # Test both models
    num_episodes = 5  # Test with multiple episodes for more reliable results
    
    flat_results = test_flat_rl_model(flat_model_path, test_env, num_episodes)
    hrl_results = test_hierarchical_rl_model(hrl_model_path, test_env, num_episodes)
    
    # Validate solutions
    if flat_results['success']:
        flat_validation = validate_solution(flat_results['evaluation_result'], test_env)
        flat_results['validation'] = flat_validation
    
    if hrl_results['success']:
        hrl_validation = validate_solution(hrl_results['evaluation_result'], test_env)
        hrl_results['validation'] = hrl_validation
    
    # Generate Gantt charts
    output_dir = "result/policy_test_mk06"
    os.makedirs(output_dir, exist_ok=True)
    
    if flat_results['success']:
        flat_gantt_path = os.path.join(output_dir, "flat_rl_gantt.png")
        flat_gantt_success = generate_gantt_chart(
            flat_results['evaluation_result'], 
            test_env, 
            flat_gantt_path, 
            "Flat RL Schedule"
        )
        flat_results['gantt_path'] = flat_gantt_path if flat_gantt_success else None
    
    if hrl_results['success']:
        hrl_gantt_path = os.path.join(output_dir, "hierarchical_rl_gantt.png")
        hrl_gantt_success = generate_gantt_chart(
            hrl_results['evaluation_result'], 
            test_env, 
            hrl_gantt_path, 
            "Hierarchical RL Schedule"
        )
        hrl_results['gantt_path'] = hrl_gantt_path if hrl_gantt_success else None
    
    # Compare results
    comparison = compare_results(flat_results, hrl_results)
    
    # Save results to CSV
    results_data = []
    
    if flat_results['success']:
        flat_eval = flat_results['evaluation_result']
        results_data.append({
            'method': 'Flat RL',
            'objective': flat_eval.get('avg_objective', flat_eval.get('objective', 0)),
            'makespan': flat_eval.get('avg_makespan', flat_eval.get('makespan', 0)),
            'twt': flat_eval.get('avg_twt', flat_eval.get('twt', 0)),
            'reward': flat_eval.get('avg_reward', flat_eval.get('episode_reward', 0)),
            'valid_completion': flat_eval.get('is_valid_completion', True),
            'validation_valid': flat_results.get('validation', {}).get('is_valid', False)
        })
    
    if hrl_results['success']:
        hrl_eval = hrl_results['evaluation_result']
        results_data.append({
            'method': 'Hierarchical RL',
            'objective': hrl_eval.get('avg_objective', hrl_eval.get('objective', 0)),
            'makespan': hrl_eval.get('avg_makespan', hrl_eval.get('makespan', 0)),
            'twt': hrl_eval.get('avg_twt', hrl_eval.get('twt', 0)),
            'reward': hrl_eval.get('avg_reward', hrl_eval.get('episode_reward', 0)),
            'valid_completion': hrl_eval.get('is_valid_completion', True),
            'validation_valid': hrl_results.get('validation', {}).get('is_valid', False)
        })
    
    if results_data:
        results_df = pd.DataFrame(results_data)
        csv_path = os.path.join(output_dir, "policy_test_results.csv")
        results_df.to_csv(csv_path, index=False)
        print(f"\n💾 Results saved to: {csv_path}")
    
    # Print summary
    print(f"\n📁 OUTPUT SUMMARY:")
    print("="*60)
    print(f"📄 Results CSV: {csv_path}")
    print(f"📊 Gantt Charts: {output_dir}")
    
    if flat_results.get('gantt_path'):
        print(f"   - Flat RL: {flat_results['gantt_path']}")
    if hrl_results.get('gantt_path'):
        print(f"   - Hierarchical RL: {hrl_results['gantt_path']}")
    
    print(f"\n✅ Policy testing completed successfully!")
    
    return {
        'flat_results': flat_results,
        'hrl_results': hrl_results,
        'comparison': comparison,
        'output_dir': output_dir
    }

if __name__ == "__main__":
    print("Remember to activate conda environment: conda activate dfjs")
    
    # Run the policy test
    results = run_policy_test()

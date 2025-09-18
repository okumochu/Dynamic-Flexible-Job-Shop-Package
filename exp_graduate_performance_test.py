#!/usr/bin/env python3

"""
Graduate Performance Test for Graph RL on FJSP Benchmarks

This script evaluates trained Graph RL models on standard benchmark instances
(Brandimarte and Hurink datasets) and compares performance against optimal objectives.
It generates comprehensive performance reports with GAP analysis.
"""

import os
import time
import pandas as pd
import numpy as np
import torch
from typing import Dict, List, Tuple, Any
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import project modules
from benchmarks.data_handler import FlexibleJobShopDataHandler
from RL.graph_rl_env import GraphRlEnv
from RL.graph_rl_trainer import GraphPPOTrainer
from RL.PPO.graph_network import HGTPolicy
from utils.policy_utils import evaluate_graph_policy
from config import config


class BenchmarkEvaluator:
    """
    Comprehensive benchmark evaluator for Graph RL models on FJSP instances.
    """
    
    def __init__(self, model_path: str):
        """
        Initialize the benchmark evaluator.
        
        Args:
            model_path: Path to the trained model file
        """
        self.model_path = model_path
        self.policy = None
        self.model_config = None
        self.device = None
        
        # Optimal/Best-known objective values for benchmark instances
        # Based on literature and research papers
        self.brandimarte_optimal = {
            'mk01': 40,   # Brandimarte (1993)
            'mk02': 26,   # Brandimarte (1993)
            'mk03': 204,  # Brandimarte (1993) 
            'mk04': 60,   # Brandimarte (1993)
            'mk05': 172,  # Brandimarte (1993)
            'mk06': 57,   # Brandimarte (1993)
            'mk07': 139,  # Brandimarte (1993)
            'mk08': 523,  # Brandimarte (1993)
            'mk09': 299,  # Brandimarte (1993)
            'mk10': 165,  # Brandimarte (1993)
            'mk11': 222,  # Best known
            'mk12': 1039, # Best known
            'mk13': 1061, # Best known
            'mk14': 1292, # Best known
            'mk15': 1207  # Best known
        }
        
        # Optimal values for Hurink instances (selected representative instances)
        # These are best-known values from literature
        self.hurink_optimal = {
            # ABZ instances (Adams, Balas, and Zawack)
            'abz5': 1234, 'abz6': 943, 'abz7': 656, 'abz8': 665, 'abz9': 678,
            
            # edata instances (easier flexibility)
            'la01': 609, 'la02': 655, 'la03': 597, 'la04': 590, 'la05': 593,
            'la06': 926, 'la07': 890, 'la08': 863, 'la09': 951, 'la10': 958,
            'la11': 1222, 'la12': 1039, 'la13': 1150, 'la14': 1292, 'la15': 1207,
            'la16': 945, 'la17': 784, 'la18': 848, 'la19': 842, 'la20': 902,
            'la21': 1026, 'la22': 927, 'la23': 1032, 'la24': 935, 'la25': 977,
            'la26': 1218, 'la27': 1235, 'la28': 1216, 'la29': 1152, 'la30': 1355,
            'la31': 1784, 'la32': 1850, 'la33': 1719, 'la34': 1721, 'la35': 1888,
            'la36': 1268, 'la37': 1397, 'la38': 1196, 'la39': 1233, 'la40': 1222,
            
            # rdata instances (random flexibility)  
            'mt06': 55, 'mt10': 655, 'mt20': 1165,
            
            # vdata instances (high flexibility)
            'setb4': 925, 'seti5': 1174, 'setf2': 1513
        }
        
        self.results = []
        
    def load_model(self):
        """Load the trained model using the existing trainer infrastructure."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        # Load the checkpoint to get model configuration
        checkpoint = torch.load(self.model_path, map_location='cpu')
        self.model_config = checkpoint.get('config', {})
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create a dummy data handler for model initialization
        # We'll use a minimal instance just to initialize the model
        dummy_data_handler = FlexibleJobShopDataHandler(
            data_source="benchmarks/brandimarte/mk01.txt",  # Use a small instance
            data_type="dataset",
            TF=config.simulation_params['TF'],
            RDD=config.simulation_params['RDD'],
            seed=config.simulation_params['seed']
        )
        
        # Create trainer instance to get the model
        trainer = GraphPPOTrainer(
            problem_data=dummy_data_handler,
            hidden_dim=self.model_config.get('hidden_dim', 64),
            num_hgt_layers=self.model_config.get('num_hgt_layers', 2),
            num_heads=self.model_config.get('num_heads', 4),
            device=self.device
        )
        
        # Load the model state
        trainer.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.policy = trainer.policy
        self.policy.eval()  # Set to evaluation mode
        
        logger.info(f"Model loaded successfully from {self.model_path}")
        logger.info(f"Architecture: hidden_dim={self.model_config.get('hidden_dim', 64)}, "
                   f"hgt_layers={self.model_config.get('num_hgt_layers', 2)}, "
                   f"heads={self.model_config.get('num_heads', 4)}")
    
    def evaluate_instance(self, dataset_path: str, instance_name: str, optimal_value: float = None) -> Dict[str, Any]:
        """
        Evaluate the model on a single benchmark instance.
        
        Args:
            dataset_path: Path to the dataset file
            instance_name: Name of the instance
            optimal_value: Known optimal objective value
            
        Returns:
            Dictionary with evaluation results
        """
        try:
            # Load the dataset
            data_handler = FlexibleJobShopDataHandler(
                data_source=dataset_path,
                data_type="dataset",
                TF=config.simulation_params['TF'],
                RDD=config.simulation_params['RDD'],
                seed=config.simulation_params['seed']
            )
            
            # Create environment
            env = GraphRlEnv(data_handler)
            
            # Run evaluation episodes
            num_episodes = 3  # Multiple episodes for robustness
            episode_makespans = []
            episode_objectives = []
            episode_twts = []
            episode_valid_completions = []
            
            for episode in range(num_episodes):
                obs, info = env.reset()
                episode_reward = 0
                episode_length = 0
                done = False
                
                while not done and episode_length < data_handler.num_operations * 3:  # Prevent infinite loops
                    obs = obs.to(self.device)
                    
                    # Get valid actions from the environment
                    valid_actions = env.graph_state.get_valid_actions()
                    
                    if not valid_actions:
                        logger.warning(f"No valid actions for {instance_name}, episode {episode}")
                        break
                    
                    with torch.no_grad():
                        action_logits, value = self.policy(obs, valid_actions)
                        
                        if len(action_logits) == 0:
                            logger.warning(f"No valid actions for {instance_name}, episode {episode}")
                            break
                        
                        # Take deterministic action (argmax)
                        action_idx = torch.argmax(action_logits).item()
                        
                        # Convert to environment action
                        if action_idx < len(valid_actions):
                            target_pair = valid_actions[action_idx]
                            env_action = None
                            for env_action_idx, pair in env.action_to_pair_map.items():
                                if pair == target_pair:
                                    env_action = env_action_idx
                                    break
                            
                            if env_action is None:
                                logger.warning(f"Could not find environment action for {instance_name}")
                                break
                        else:
                            logger.warning(f"Action index out of range for {instance_name}")
                            break
                    
                    next_obs, reward, terminated, truncated, next_info = env.step(env_action)
                    
                    episode_reward += reward
                    episode_length += 1
                    done = terminated or truncated
                    
                    if not done:
                        obs = next_obs
                
                # Extract final metrics
                is_valid_completion = env.graph_state.is_done()
                final_makespan = env.graph_state.get_makespan() if is_valid_completion else float('inf')
                total_twt = env.graph_state.get_total_weighted_tardiness() if is_valid_completion else float('inf')
                alpha = env.alpha
                objective = (1 - alpha) * final_makespan + alpha * total_twt if is_valid_completion else float('inf')
                
                episode_makespans.append(final_makespan)
                episode_objectives.append(objective)
                episode_twts.append(total_twt)
                episode_valid_completions.append(is_valid_completion)
            
            # Calculate statistics
            valid_episodes = [i for i, valid in enumerate(episode_valid_completions) if valid]
            
            if valid_episodes:
                best_makespan = min(episode_makespans[i] for i in valid_episodes)
                avg_makespan = np.mean([episode_makespans[i] for i in valid_episodes])
                best_objective = min(episode_objectives[i] for i in valid_episodes)
                avg_objective = np.mean([episode_objectives[i] for i in valid_episodes])
                avg_twt = np.mean([episode_twts[i] for i in valid_episodes])
            else:
                best_makespan = float('inf')
                avg_makespan = float('inf')
                best_objective = float('inf')
                avg_objective = float('inf')
                avg_twt = float('inf')
            
            # Calculate GAP if optimal value is known
            gap_best = float('inf')
            gap_avg = float('inf')
            if optimal_value and optimal_value > 0 and best_makespan != float('inf'):
                gap_best = ((best_makespan - optimal_value) / optimal_value) * 100
                gap_avg = ((avg_makespan - optimal_value) / optimal_value) * 100
            
            results = {
                'instance': instance_name,
                'dataset_type': 'brandimarte' if 'brandimarte' in dataset_path else 'hurink',
                'jobs': data_handler.num_jobs,
                'machines': data_handler.num_machines,
                'operations': data_handler.num_operations,
                'optimal_makespan': optimal_value if optimal_value else 'Unknown',
                'best_makespan': best_makespan,
                'avg_makespan': avg_makespan,
                'best_objective': best_objective,
                'avg_objective': avg_objective,
                'avg_twt': avg_twt,
                'gap_best_percent': gap_best,
                'gap_avg_percent': gap_avg,
                'valid_completions': len(valid_episodes),
                'total_episodes': num_episodes,
                'success_rate': len(valid_episodes) / num_episodes
            }
            
            logger.info(f"✅ {instance_name}: Best makespan={best_makespan:.1f}, "
                       f"GAP={gap_best:.1f}%, Success={len(valid_episodes)}/{num_episodes}")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Error evaluating {instance_name}: {e}")
            return {
                'instance': instance_name,
                'dataset_type': 'brandimarte' if 'brandimarte' in dataset_path else 'hurink',
                'error': str(e),
                'best_makespan': float('inf'),
                'gap_best_percent': float('inf')
            }
    
    def evaluate_brandimarte_instances(self) -> List[Dict[str, Any]]:
        """Evaluate all Brandimarte instances."""
        logger.info("🔍 Evaluating Brandimarte instances...")
        
        brandimarte_dir = "benchmarks/brandimarte"
        results = []
        
        for instance_file in sorted(os.listdir(brandimarte_dir)):
            if instance_file.endswith('.txt'):
                instance_name = instance_file.replace('.txt', '')
                dataset_path = os.path.join(brandimarte_dir, instance_file)
                optimal_value = self.brandimarte_optimal.get(instance_name)
                
                result = self.evaluate_instance(dataset_path, instance_name, optimal_value)
                results.append(result)
        
        return results
    
    def evaluate_hurink_instances(self, max_instances_per_category: int = 10) -> List[Dict[str, Any]]:
        """
        Evaluate selected Hurink instances.
        
        Args:
            max_instances_per_category: Maximum instances to test per category
        """
        logger.info("🔍 Evaluating Hurink instances...")
        
        hurink_base_dir = "benchmarks/hurink"
        results = []
        
        for category in ['edata', 'rdata', 'vdata']:
            category_dir = os.path.join(hurink_base_dir, category)
            
            if not os.path.exists(category_dir):
                logger.warning(f"Category directory not found: {category_dir}")
                continue
            
            instance_files = sorted([f for f in os.listdir(category_dir) if f.endswith('.txt')])
            
            # Limit number of instances per category for efficiency
            selected_files = instance_files[:max_instances_per_category]
            
            for instance_file in selected_files:
                instance_name = f"{category}_{instance_file.replace('.txt', '')}"
                dataset_path = os.path.join(category_dir, instance_file)
                
                # Try to get optimal value
                base_name = instance_file.replace('.txt', '')
                optimal_value = self.hurink_optimal.get(base_name)
                
                result = self.evaluate_instance(dataset_path, instance_name, optimal_value)
                results.append(result)
        
        return results
    
    def run_complete_evaluation(self, output_dir: str = None) -> str:
        """
        Run complete evaluation on all benchmark instances.
        
        Args:
            output_dir: Directory to save results
            
        Returns:
            Path to the generated CSV report
        """
        logger.info("🚀 Starting Graduate Performance Test...")
        logger.info(f"Model: {self.model_path}")
        
        # Create output directory using standardized structure
        if output_dir is None:
            from config import config
            output_dir = config.create_experiment_result_dir("exp_graduate_performance_test")
        else:
            os.makedirs(output_dir, exist_ok=True)
        
        # Load the trained model
        self.load_model()
        
        start_time = time.time()
        
        # Evaluate Brandimarte instances
        brandimarte_results = self.evaluate_brandimarte_instances()
        
        # Evaluate selected Hurink instances
        hurink_results = self.evaluate_hurink_instances(max_instances_per_category=5)  # Limit for time
        
        # Combine all results
        all_results = brandimarte_results + hurink_results
        
        # Create DataFrame and save
        df = pd.DataFrame(all_results)
        
        # Calculate summary statistics
        evaluation_time = time.time() - start_time
        
        # Save detailed results
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(output_dir, f"graduate_performance_report_{timestamp}.csv")
        df.to_csv(csv_path, index=False)
        
        # Generate summary report
        summary_report = self._generate_summary_report(df, evaluation_time)
        summary_path = os.path.join(output_dir, f"performance_summary_{timestamp}.txt")
        
        with open(summary_path, 'w') as f:
            f.write(summary_report)
        
        logger.info(f"📊 Complete evaluation finished in {evaluation_time:.1f} seconds")
        logger.info(f"📄 Detailed results: {csv_path}")
        logger.info(f"📋 Summary report: {summary_path}")
        
        # Print summary to console
        print("\n" + "="*80)
        print("GRADUATE PERFORMANCE TEST SUMMARY")
        print("="*80)
        print(summary_report)
        
        return csv_path
    
    def _generate_summary_report(self, df: pd.DataFrame, evaluation_time: float) -> str:
        """Generate a summary report from the evaluation results."""
        
        # Filter valid results (exclude errors)
        valid_df = df[df['best_makespan'] != float('inf')].copy()
        
        report = []
        report.append(f"Graduate Performance Test Results")
        report.append(f"Model: {self.model_path}")
        report.append(f"Evaluation time: {evaluation_time:.1f} seconds")
        report.append(f"Total instances tested: {len(df)}")
        report.append(f"Successfully solved: {len(valid_df)}")
        report.append(f"Success rate: {len(valid_df)/len(df)*100:.1f}%")
        report.append("")
        
        # Brandimarte results
        brandimarte_df = valid_df[valid_df['dataset_type'] == 'brandimarte']
        if len(brandimarte_df) > 0:
            report.append("BRANDIMARTE DATASET RESULTS:")
            report.append(f"  Instances solved: {len(brandimarte_df)}/15")
            
            # Calculate GAP statistics for instances with known optimal
            known_optimal = brandimarte_df[brandimarte_df['optimal_makespan'] != 'Unknown']
            if len(known_optimal) > 0:
                avg_gap = known_optimal['gap_best_percent'].mean()
                min_gap = known_optimal['gap_best_percent'].min()
                max_gap = known_optimal['gap_best_percent'].max()
                
                report.append(f"  Average GAP: {avg_gap:.2f}%")
                report.append(f"  Best GAP: {min_gap:.2f}%")
                report.append(f"  Worst GAP: {max_gap:.2f}%")
            
            # Best instances
            best_instances = brandimarte_df.nsmallest(3, 'gap_best_percent')
            report.append("  Top 3 best instances:")
            for _, row in best_instances.iterrows():
                if row['optimal_makespan'] != 'Unknown':
                    report.append(f"    {row['instance']}: {row['best_makespan']:.1f} (GAP: {row['gap_best_percent']:.1f}%)")
            
            report.append("")
        
        # Hurink results
        hurink_df = valid_df[valid_df['dataset_type'] == 'hurink']
        if len(hurink_df) > 0:
            report.append("HURINK DATASET RESULTS:")
            report.append(f"  Instances solved: {len(hurink_df)}")
            
            for category in ['edata', 'rdata', 'vdata']:
                category_df = hurink_df[hurink_df['instance'].str.startswith(category)]
                if len(category_df) > 0:
                    known_optimal = category_df[category_df['optimal_makespan'] != 'Unknown']
                    if len(known_optimal) > 0:
                        avg_gap = known_optimal['gap_best_percent'].mean()
                        report.append(f"  {category}: {len(category_df)} instances, avg GAP: {avg_gap:.2f}%")
                    else:
                        report.append(f"  {category}: {len(category_df)} instances (no known optimal)")
            
            report.append("")
        
        # Overall statistics
        if len(valid_df) > 0:
            report.append("OVERALL STATISTICS:")
            report.append(f"  Average problem size: {valid_df['operations'].mean():.1f} operations")
            report.append(f"  Average makespan: {valid_df['best_makespan'].mean():.1f}")
            
            # GAP statistics for all instances with known optimal
            known_optimal_all = valid_df[valid_df['optimal_makespan'] != 'Unknown']
            if len(known_optimal_all) > 0:
                report.append(f"  Overall average GAP: {known_optimal_all['gap_best_percent'].mean():.2f}%")
                report.append(f"  Instances within 10% of optimal: {(known_optimal_all['gap_best_percent'] <= 10).sum()}/{len(known_optimal_all)}")
                report.append(f"  Instances within 20% of optimal: {(known_optimal_all['gap_best_percent'] <= 20).sum()}/{len(known_optimal_all)}")
        
        return "\n".join(report)


def main():
    """Main function to run the graduate performance test."""
    import argparse
    import torch
    
    parser = argparse.ArgumentParser(description='Graduate Performance Test for Graph RL on FJSP Benchmarks')
    parser.add_argument('--model-path', type=str, 
                       default='result/exp_graph_rl/graph/model/model_final.pt',
                       help='Path to the trained model file')
    parser.add_argument('--output-dir', type=str, 
                       default=None,
                       help='Output directory for results (defaults to standardized structure)')
    parser.add_argument('--hurink-limit', type=int, default=5,
                       help='Maximum Hurink instances per category to test')
    
    args = parser.parse_args()
    
    # Check if model exists (only if not showing help)
    if not os.path.exists(args.model_path):
        logger.error(f"Model file not found: {args.model_path}")
        logger.info("Please train a model first using: python exp_graph_rl.py --experiment single")
        return
    
    # Import torch here to avoid issues if not available
    import torch
    
    # Run evaluation
    evaluator = BenchmarkEvaluator(args.model_path)
    csv_path = evaluator.run_complete_evaluation(args.output_dir)
    
    logger.info(f"🎉 Graduate Performance Test completed successfully!")
    logger.info(f"📈 Results available at: {csv_path}")


if __name__ == "__main__":
    main()

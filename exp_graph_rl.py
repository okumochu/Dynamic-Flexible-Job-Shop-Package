#!/usr/bin/env python3

"""
Graph-based Reinforcement Learning Experiment for FJSP

This script runs experiments using Heterogeneous Graph Transformer (HGT) networks
for solving the Flexible Job Shop Scheduling Problem. It follows the same patterns
as other experiment files in the project.

Gantt Chart Generation:
- Uses utils/policy_utils.py for baseline scheduling and policy evaluation
- Uses utils/solution_utils.py (SolutionUtils) for chart creation and validation
- Supports both baseline dispatching rules (FIFO, SPT) and trained policies
"""

import os
import time
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Optional
import wandb

# Import project modules
from benchmarks.data_handler import FlexibleJobShopDataHandler
from RL.graph_rl_env import GraphRlEnv
from RL.graph_rl_trainer import GraphPPOTrainer
from config import config


# Evaluation functions removed - only training code kept


def get_graph_rl_config(project_name: str = "exp_graph_rl", dataset_name: str = None, exp_name: str = "graph") -> Dict[str, Any]:
    """Get configuration for graph RL experiment."""
    return config.get_graph_rl_config(project_name, dataset_name)


def run_single_graph_rl_experiment():
    """Run a single graph RL experiment with simulation data."""
    
    print("="*60)
    print("🔗  GRAPH-BASED REINFORCEMENT LEARNING EXPERIMENT")
    print("📊 Heterogeneous Graph Transformer for Job Shop Scheduling")
    print("="*60)
    
    # Clear GPU cache before starting
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"GPU cache cleared. Available memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    
    # Get configuration
    exp_config = get_graph_rl_config()
    simulation_params = exp_config['simulation_params']
    rl_params = exp_config['rl_params']
    result_dir = exp_config['result_dir']
    
    # Setup wandb
    config.setup_wandb_env(result_dir, exp_config['wandb_project'])
    
    print("Creating data handler and graph environment...")
    data_handler = FlexibleJobShopDataHandler(
        data_source=simulation_params, 
        data_type="simulation",
        TF=simulation_params['TF'],
        RDD=simulation_params['RDD'],
        seed=simulation_params['seed']
    )
    env = GraphRlEnv(data_handler, alpha=rl_params['alpha'], device=rl_params['device'])
    
    print(f"Graph environment created: {env.problem_data.num_jobs} jobs, {env.problem_data.num_machines} machines")
    print(f"Total operations: {env.problem_data.num_operations}")
    print(f"Action space size: {env.action_space.n}")
    print(f"Graph features - Operations: 8, Machines: 7, Jobs: 7")
    
    # Create graph trainer using epoch/episode structure like flat_rl_trainer
    trainer = GraphPPOTrainer(
        problem_data=data_handler,
        epochs=rl_params['epochs'],
        episodes_per_epoch=rl_params['episodes_per_epoch'],
        train_per_episode=rl_params['train_per_episode'],
        hidden_dim=rl_params['hidden_dim'],
        num_hgt_layers=rl_params['num_hgt_layers'],
        num_heads=rl_params['num_heads'],
        lr=rl_params['lr'],
        gamma=rl_params['gamma'],
        gae_lambda=rl_params['gae_lambda'],
        clip_ratio=rl_params['clip_ratio'],
        device=rl_params['device'],
        project_name=exp_config['wandb_project'],
        run_name=rl_params['wandb_run_name'],
        model_save_dir=result_dir,
        seed=rl_params['seed']
    )
    
    print(f"Starting graph RL training for {rl_params['epochs']} epochs...")
    start_time = time.time()
    
    # Train the model
    training_results = trainer.train(seed=rl_params['seed'])
    
    training_time = training_results['training_time']
    print(f"Graph RL training complete. Training time: {training_time:.2f} seconds")
    
    # Clear GPU cache after training
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("GPU cache cleared after training")
    
    print(f"\nGraph RL experiment completed successfully!")
    print(f"Results saved in: {result_dir}")
    
    print(f"\n" + "="*50)
    print("GRAPH RL EXPERIMENT SUMMARY")
    print("="*50)
    print(f"Training Configuration:")
    print(f"  Epochs: {rl_params['epochs']}")
    print(f"  Episodes per Epoch: {rl_params['episodes_per_epoch']}")
    print(f"  Hidden Dimension: {rl_params['hidden_dim']}")
    print(f"  HGT Layers: {rl_params['num_hgt_layers']}")
    print(f"  Attention Heads: {rl_params['num_heads']}")
    print(f"  Learning Rate: {rl_params['lr']}")
    print(f"  Device: {rl_params['device']}")
    print(f"  Training Time: {training_time:.2f} seconds")
    
    return {
        'training_time': training_time,
        'model_path': os.path.join(result_dir, training_results['model_filename']),
        'training_results': training_results
    }


# Brandimarte experiment function removed - only training code kept


def main():
    """Main function to run graph RL training experiment."""
    print("Running single graph RL experiment...")
    results = run_single_graph_rl_experiment()
    print("\nGraph RL experiment completed successfully! 🎉")
    return results


if __name__ == "__main__":
    main()

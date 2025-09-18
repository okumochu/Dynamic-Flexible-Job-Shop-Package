#!/usr/bin/env python3

"""
Graph-based DDQN Experiment for FJSP

This script mirrors exp_graph_rl.py but uses a DDQN trainer that leverages the
HGT backbone for Q-learning over variable action spaces.
"""

import os
import time
from typing import Dict, Any

import wandb

from benchmarks.data_handler import FlexibleJobShopDataHandler
from RL.graph_DDQN_rl_trainer import GraphDDQNTrainer
from config import config


def get_graph_ddqn_config(project_name: str = "exp_DDQN_graph_rl", dataset_name: str = None) -> Dict[str, Any]:
    return config.get_graph_ddqn_config(project_name, dataset_name)


def run_single_graph_ddqn_experiment():
    print("="*60)
    print("🔗  GRAPH-BASED DDQN EXPERIMENT")
    print("📊 HGT Q-Network for Job Shop Scheduling")
    print("="*60)

    exp_config = get_graph_ddqn_config()
    simulation_params = exp_config['simulation_params']
    rl_params = exp_config['rl_params']
    result_dir = exp_config['result_dir']

    config.setup_wandb_env(result_dir, exp_config['wandb_project'])

    print("Creating data handler and environment...")
    data_handler = FlexibleJobShopDataHandler(
        data_source=simulation_params,
        data_type="simulation",
        TF=simulation_params['TF'],
        RDD=simulation_params['RDD'],
        seed=simulation_params['seed']
    )
    trainer = GraphDDQNTrainer(
        problem_data=data_handler,
        epochs=rl_params['epochs'],
        steps_per_epoch=rl_params['train_per_episode'] * data_handler.num_operations,
        hidden_dim=rl_params['hidden_dim'],
        num_hgt_layers=rl_params['num_hgt_layers'],
        num_heads=rl_params['num_heads'],
        lr=rl_params['lr'],
        gamma=rl_params['gamma'],
        batch_size=rl_params['batch_size'],
        buffer_size=rl_params['buffer_size'],
        target_update_freq=rl_params['target_update_freq'],
        epsilon_start=rl_params['epsilon_start'],
        epsilon_end=rl_params['epsilon_end'],
        epsilon_decay_steps=rl_params['epsilon_decay_steps'],
        device=rl_params['device'],
        project_name=exp_config['wandb_project'],
        run_name=rl_params['wandb_run_name'],
        model_save_dir=result_dir,
        seed=rl_params['seed']
    )

    print(f"Starting graph DDQN training for {rl_params['epochs']} epochs...")
    start_time = time.time()

    trainer.train()

    training_time = time.time() - start_time
    print(f"Graph DDQN training complete. Training time: {training_time:.2f} seconds")
    print(f"Results saved in: {result_dir}")

    return {
        'training_time': training_time,
        'result_dir': result_dir,
    }


def main():
    print("Running single graph DDQN experiment...")
    run_single_graph_ddqn_experiment()
    print("\nGraph DDQN experiment completed successfully! 🎉")


if __name__ == "__main__":
    main()



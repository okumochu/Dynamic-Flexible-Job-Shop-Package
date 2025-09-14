#!/usr/bin/env python3

"""
Graph-based PPO Trainer for Flexible Job Shop Scheduling Problem (FJSP)

This script implements the main training loop using PPO for graph-based RL
on the FJSP. It uses Heterogeneous Graph Transformer (HGT) networks and 
leverages the existing PPO infrastructure.
"""

import os
import time
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from torch_geometric.data import Batch
from typing import Dict, List, Tuple, Optional

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None
    print("Weights & Biases not available. Install with: pip install wandb")

# Import project modules
from benchmarks.static_benchmark.data_handler import FlexibleJobShopDataHandler
from RL.graph_rl_env import GraphRlEnv
from RL.PPO.graph_network import HGTPolicy, GraphPPOBuffer
from RL.PPO.networks import PPOUpdater
from config import config


class GraphPPOTrainer:
    """
    PPO Trainer for Graph-based FJSP using HGT networks.
    
    This trainer implements the standard PPO algorithm adapted for
    heterogeneous graph observations and variable action spaces.
    """
    
    def __init__(self,
                 problem_data: FlexibleJobShopDataHandler,
                 epochs: int = None,
                 episodes_per_epoch: int = None,
                 train_per_episode: int = None,
                 hidden_dim: int = None,
                 num_hgt_layers: int = None,
                 num_heads: int = None,
                 lr: float = None,
                 gamma: float = None,
                 gae_lambda: float = None,
                 clip_ratio: float = None,
                 value_coef: float = None,
                 max_grad_norm: float = None,
                 project_name: Optional[str] = None,
                 run_name: Optional[str] = None,
                 device: str = None,
                 model_save_dir: str = 'result/graph_rl/model',
                 seed: Optional[int] = None):
        """
        Initialize the Graph PPO Trainer.
        """
        
        # Use config defaults for any None parameters
        graph_config = config.get_graph_rl_config()
        rl_params = graph_config['rl_params']
        
        # Set defaults from config if not provided
        epochs = epochs if epochs is not None else rl_params['epochs']
        episodes_per_epoch = episodes_per_epoch or rl_params.get('episodes_per_epoch', 2)
        train_per_episode = train_per_episode if train_per_episode is not None else rl_params['train_per_episode']
        hidden_dim = hidden_dim if hidden_dim is not None else rl_params['hidden_dim']
        num_hgt_layers = num_hgt_layers if num_hgt_layers is not None else rl_params['num_hgt_layers']
        num_heads = num_heads if num_heads is not None else rl_params['num_heads']
        lr = lr if lr is not None else rl_params['lr']
        gamma = gamma if gamma is not None else rl_params['gamma']
        gae_lambda = gae_lambda if gae_lambda is not None else rl_params['gae_lambda']
        clip_ratio = clip_ratio if clip_ratio is not None else rl_params['clip_ratio']
        value_coef = value_coef if value_coef is not None else rl_params['value_coef']
        max_grad_norm = max_grad_norm if max_grad_norm is not None else rl_params['max_grad_norm']
        device = device if device is not None else rl_params['device']
        seed = seed if seed is not None else rl_params['seed']
        
        # Store training parameters
        self.problem_data = problem_data
        self.epochs = epochs
        self.episodes_per_epoch = episodes_per_epoch
        self.train_per_episode = train_per_episode
        self.model_save_dir = model_save_dir
        self.project_name = project_name
        self.run_name = run_name
        
        # Store hyperparameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        
        # Episode tracking like flat_rl_trainer
        self.episode_makespans = []
        self.episode_twts = []
        self.episode_objectives = []
        self.episode_rewards = []
        
        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Set random seeds
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        
        # Initialize environment with config parameters
        # Pass trainer device into environment/graph state to keep all tensors aligned
        self.env = GraphRlEnv(problem_data, alpha=rl_params['alpha'], device=str(self.device))
        print(f"Environment: {self.env.problem_data.num_jobs} jobs, "
              f"{self.env.problem_data.num_machines} machines, "
              f"{self.env.problem_data.num_operations} operations")
        print(f"Multi-objective weight (alpha): {rl_params['alpha']}")

        # Get feature dimensions from GraphState (not from observation)
        op_feature_dim, machine_feature_dim, job_feature_dim = self.env.graph_state.get_feature_dimensions()
        
        print(f"Feature dimensions - Operations: {op_feature_dim}, Machines: {machine_feature_dim}, Jobs: {job_feature_dim}")
        
        # Validate configuration parameters
        self._validate_config(hidden_dim, num_heads, rl_params)

        # Initialize policy network with DYNAMIC dimensions from config
        self.policy = HGTPolicy(
            op_feature_dim=op_feature_dim,
            machine_feature_dim=machine_feature_dim,
            job_feature_dim=job_feature_dim,
            hidden_dim=hidden_dim,
            num_hgt_layers=num_hgt_layers,
            num_heads=num_heads,
            dropout=rl_params['dropout']
        ).to(self.device)
        
        print(f"Policy network parameters: {sum(p.numel() for p in self.policy.parameters()):,}")
        
        # Initialize optimizer with better defaults
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, eps=1e-5)
        
        # Initialize buffer - better size estimation based on problem complexity
        estimated_steps_per_episode = min(self.env.problem_data.num_operations * 3, 1000)  # Cap for large problems
        buffer_size = max(2048, self.episodes_per_epoch * estimated_steps_per_episode)  # Minimum buffer size
        self.buffer = GraphPPOBuffer(buffer_size, self.device)
        
        # Learning rate scheduler for better convergence
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=max(1, self.epochs//4), gamma=0.8)
        
        # Create save directory
        os.makedirs(model_save_dir, exist_ok=True)
        
        print(f"Graph RL Trainer initialized")
        print(f"Environment: {self.env.problem_data.num_jobs} jobs, {self.env.problem_data.num_machines} machines")
    
    def _validate_config(self, hidden_dim: int, num_heads: int, rl_params: dict):
        """Validate configuration parameters for consistency."""
        
        # Validate hidden_dim is divisible by num_heads (required for multi-head attention)
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads}) "
                           f"for multi-head attention to work properly.")
        
        # Validate alpha parameter range
        alpha = rl_params['alpha']
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"Alpha parameter ({alpha}) must be between 0.0 and 1.0. "
                           f"0.0=pure makespan, 1.0=pure tardiness, 0.5=balanced")
        
        # Log configuration for verification
        print(f"✓ Configuration validated:")
        print(f"  Hidden dim: {hidden_dim} (divisible by {num_heads} heads)")
        print(f"  Multi-objective weight: {alpha} ({'makespan-only' if alpha == 0.0 else 'tardiness-only' if alpha == 1.0 else 'balanced'})")
        print(f"  Temporal encoding: DISABLED")
        print(f"  Dropout: {rl_params['dropout']}")
        print(f"  Learning rate: {rl_params['lr']}")
        print(f"  Simplified PPO: No entropy regularization or KL divergence (due to variable action spaces)")
    
    def collect_rollout(self, buffer: GraphPPOBuffer, epoch: int) -> Dict:
        """
        Collect rollout data for one epoch.
        
        Args:
            buffer: GraphPPOBuffer to store collected experiences
            epoch: Current epoch number
            
        Returns:
            Dictionary containing episode metrics for this epoch
        """
        self.policy.eval()
        
        # Collect data for episodes_per_epoch
        epoch_episodes_completed = 0
        epoch_total_steps = 0
        
        for episode in range(self.episodes_per_epoch):
            # Reset environment for new episode
            obs, info = self.env.reset()
            obs = obs.to(self.device)
            
            episode_reward = 0
            episode_steps = 0
            
            # Run episode until completion
            while not self.env.graph_state.is_done() and episode_steps < self.env.max_episode_steps:
                # Get valid actions from environment info (from last reset or step)
                env_valid_actions = info.get('valid_actions', [])
                
                # Get action from policy - always produces valid action logits
                with torch.no_grad():
                    action_logits, value = self.policy(obs, env_valid_actions)
                    
                    # Create distribution and sample action
                    dist = Categorical(logits=action_logits)
                    action_idx = dist.sample()
                    log_prob = dist.log_prob(action_idx)
                    
                    # Direct mapping from action index to environment action
                    target_pair = env_valid_actions[action_idx.item()]
                    # Use O(1) reverse map
                    env_action = self.env.pair_to_action_map.get(tuple(target_pair))
                
                # Step environment
                next_obs, reward, terminated, truncated, info = self.env.step(env_action)
                done = terminated or truncated
                
                # Store experience in buffer
                buffer.add(
                    obs=obs.cpu(),
                    action=action_idx.cpu().item(),
                    reward=reward,
                    value=value.cpu().item(),
                    log_prob=log_prob.cpu().item(),
                    valid_pairs=env_valid_actions.copy(),  # Store copy to avoid reference issues
                    done=done
                )
                
                # Update episode statistics
                episode_reward += reward
                episode_steps += 1
                
                # Store final info when episode ends
                if done:
                    final_info = info
                    break
                    
                obs = next_obs.to(self.device)
            
            # Track completion
            epoch_episodes_completed += 1
            epoch_total_steps += episode_steps
            
            # Extract final episode metrics from environment
            final_makespan = self.env.graph_state.get_makespan()
            
            # Extract TWT from final episode info if available
            final_twt = 0.0
            if 'final_info' in locals() and final_info is not None:
                final_twt = final_info.get('total_weighted_tardiness', 0.0)
            else:
                # Fallback: calculate TWT directly from environment
                final_twt = self.env._calculate_total_weighted_tardiness()
            
            # Calculate multi-objective using alpha parameter  
            alpha = self.env.alpha  # Get alpha from environment
            objective = (1 - alpha) * final_makespan + alpha * final_twt
            
            # Store episode metrics
            self.episode_makespans.append(final_makespan)
            self.episode_twts.append(final_twt)
            self.episode_objectives.append(objective)
            self.episode_rewards.append(episode_reward)
            
            # Store episode metrics for epoch-level aggregation (no individual logging to avoid step conflicts)
        
        # Get current buffer state
        current_buffer_size = self.buffer.size
        
        return {
            'episodes_completed': epoch_episodes_completed
        }
    
    def update_policy(self) -> Dict:
        """
        Update the policy using PPO.
        
        Returns:
            Dictionary with training statistics
        """
        self.policy.train()
        
        # Get batch from buffer
        batch = self.buffer.get_batch()
        
        if len(batch['observations']) == 0:
            return {'policy_loss': 0, 'value_loss': 0, 'optimizer_stepped': False}
        
        # Convert batch data to tensors for PPOUpdater
        rewards = torch.as_tensor(batch['rewards'], device=self.device, dtype=torch.float32)
        values = torch.as_tensor(batch['values'], device=self.device, dtype=torch.float32)
        dones = torch.as_tensor(batch['dones'], device=self.device, dtype=torch.bool)
        
        # Compute GAE advantages using tested PPOUpdater
        advantages, returns = PPOUpdater.compute_gae_advantages(
            rewards, values, dones, self.gamma, self.gae_lambda
        )
        
        # Normalize advantages
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Store old log probs for PPO clipping 
        old_log_probs = torch.as_tensor(batch['log_probs'], device=self.device, dtype=torch.float32)
        
        # Multiple epochs of PPO updates
        # Use episodes_per_epoch as num_ppo_epochs
        # Use full batch to respect episode boundaries and potential-based reward shaping
        total_policy_loss = 0
        total_value_loss = 0
        num_updates_performed = 0
        # Use full batch size to maintain episode coherence for potential-based rewards
        mini_batch_size = len(batch['observations'])
        
        for _ in range(self.episodes_per_epoch):
            # Shuffle observations while maintaining full batch for episode coherence
            indices = torch.randperm(len(batch['observations']))
            
            for start_idx in range(0, len(indices), mini_batch_size):
                end_idx = start_idx + mini_batch_size
                mini_batch_indices = indices[start_idx:end_idx]
                
                # Get batch data (full batch for episode coherence)
                mini_batch_obs = [batch['observations'][i] for i in mini_batch_indices]
                mini_batch_valid_pairs = [batch['valid_action_pairs'][i] for i in mini_batch_indices]
                mini_batch_actions = torch.as_tensor([batch['actions'][i] for i in mini_batch_indices], device=self.device)
                mini_batch_advantages = advantages[mini_batch_indices]
                mini_batch_returns = returns[mini_batch_indices]
                mini_batch_old_log_probs = old_log_probs[mini_batch_indices]
                
                # PERFORMANCE CRITICAL: Batched forward pass instead of sequential processing
                # Create a batch from individual HeteroData observations
                batch_graphs = Batch.from_data_list([obs.to(self.device) for obs in mini_batch_obs])
                
                # Batched forward pass through the policy network
                batch_action_logits, batch_values = self.policy.forward(batch_graphs, mini_batch_valid_pairs)
                
                # Process results for each graph in the batch
                policy_losses = []
                value_losses = []
                
                for i in range(len(mini_batch_obs)):
                    action_logits = batch_action_logits[i]  # Action logits for graph i
                    value = batch_values[i]  # Value estimate for graph i
                    action_idx = mini_batch_actions[i].item()
                    
                    # Create distribution and compute log prob for the stored action
                    if len(action_logits) > 0:
                        dist = Categorical(logits=action_logits)
                        new_log_prob = dist.log_prob(torch.as_tensor(action_idx, device=self.device))
                    else:
                        # Handle edge case where no valid actions exist
                        new_log_prob = torch.tensor(0.0, device=self.device)
                    
                    # PPO policy loss
                    policy_loss = PPOUpdater.ppo_policy_loss(
                        new_log_prob.unsqueeze(0), mini_batch_old_log_probs[i].unsqueeze(0), 
                        mini_batch_advantages[i].unsqueeze(0), self.clip_ratio
                    )
                    
                    # Value loss
                    value_loss = F.mse_loss(value, mini_batch_returns[i])
                    
                    # Store losses for this sample
                    policy_losses.append(policy_loss)
                    value_losses.append(value_loss)
                
                # Combine losses for this mini-batch
                policy_loss = torch.stack(policy_losses).mean()
                value_loss = torch.stack(value_losses).mean()
                
                # Total loss (simplified: only policy and value losses)
                loss = policy_loss + self.value_coef * value_loss
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                
                self.optimizer.step()
                
                # Accumulate statistics
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                num_updates_performed += 1
        
        return {
            'policy_loss': total_policy_loss / num_updates_performed,
            'value_loss': total_value_loss / num_updates_performed,
            'optimizer_stepped': True
        }
    
    
    def train(self, seed: Optional[int] = None):
        """
        Main training loop following the epoch/episode pattern from flat_rl_trainer.
        """
        # Configure wandb to save in proper directory  
        wandb_dir = os.path.join(os.path.dirname(self.model_save_dir), 'training_process')
        os.makedirs(wandb_dir, exist_ok=True)
        
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        if WANDB_AVAILABLE:
            wandb.init(
                name=self.run_name,
                project=self.project_name,
                dir=wandb_dir,
                config={
                    "epochs": self.epochs,
                    "episodes_per_epoch": self.episodes_per_epoch,
                    "train_per_episode": self.train_per_episode,
                    "lr": self.optimizer.param_groups[0]['lr'],
                    "gamma": self.gamma,
                    "gae_lambda": self.gae_lambda,
                    "clip_ratio": self.clip_ratio,
                    "hidden_dim": self.policy.hidden_dim,
                    "num_hgt_layers": self.policy.num_hgt_layers,
                }
            )
        else:
            print("Warning: W&B not available, training will proceed without logging")

        # Training loop
        start_time = time.time()
        
        from tqdm import tqdm
        pbar = tqdm(range(self.epochs), desc="Graph RL Training")

        total_optimization_steps = 0

        for epoch in pbar:
            # Collect rollout data
            collection_stats = self.collect_rollout(self.buffer, epoch)
            
            # Update agent multiple times on the same data (standard PPO practice)
            stats = {'policy_loss': 0, 'value_loss': 0, 'optimizer_stepped': False}
            
            # Train with collected data
            for train_step in range(self.train_per_episode):
                step_stats = self.update_policy()
                
                # Accumulate statistics
                for key in ['policy_loss', 'value_loss']:
                    stats[key] += step_stats.get(key, 0)
                
                if step_stats.get('optimizer_stepped', False):
                    stats['optimizer_stepped'] = True
                    total_optimization_steps += 1
                    
            
            # Average the statistics
            num_train_steps = self.train_per_episode
            for key in ['policy_loss', 'value_loss']:
                stats[key] /= max(1, num_train_steps)
            
            # Only update learning rate if we actually performed optimization
            if stats.get('optimizer_stepped', False):
                self.scheduler.step()

            # Log training metrics
            wandb_log = {
                "training/policy_loss": stats["policy_loss"],
                "training/value_loss": stats["value_loss"],
                "training/total_episodes": len(self.episode_makespans),
                "training/learning_rate": self.optimizer.param_groups[0]['lr']
            }
            
            # Add latest episode performance metrics if available
            if collection_stats['episodes_completed'] > 0 and len(self.episode_makespans) > 0:
                wandb_log.update({
                    "performance/episode_makespan": self.episode_makespans[-1],
                    "performance/episode_twt": self.episode_twts[-1],
                    "performance/episode_objective": self.episode_objectives[-1],
                    "performance/episode_reward": self.episode_rewards[-1],
                    "performance/alpha": self.env.alpha  # Log the multi-objective weight
                })
            
            # Log all metrics with epoch as step
            if WANDB_AVAILABLE:
                wandb.log(wandb_log, step=epoch)

            # Clear buffer unconditionally after each epoch's update phase
            self.buffer.clear()
            
            # Progress monitoring
            if (epoch + 1) % 10 == 0:
                if len(self.episode_makespans) > 0:
                    latest_makespan = self.episode_makespans[-1]
                    latest_twt = self.episode_twts[-1]
                    latest_objective = self.episode_objectives[-1]
                    latest_reward = self.episode_rewards[-1]
                    current_lr = self.optimizer.param_groups[0]['lr']
                    alpha = self.env.alpha
                    print(f"Epoch {epoch+1}/{self.epochs}: Makespan = {latest_makespan:.2f}, "
                          f"TWT = {latest_twt:.2f}, Objective = {latest_objective:.2f}, "
                          f"Reward = {latest_reward:.2f}, α = {alpha:.1f}, Policy Loss = {stats['policy_loss']:.4f}, "
                          f"Value Loss = {stats['value_loss']:.4f}, LR = {current_lr:.6f}")
        
        pbar.close()
        
        # Save model with timestamp
        timestamp = time.strftime('%Y%m%d_%H%M')
        model_filename = f"model_{timestamp}.pth"
        self.save_model(model_filename)
        
        if WANDB_AVAILABLE:
            wandb.finish()

        return {
            'training_time': time.time() - start_time,
            'model_filename': model_filename,
            'training_history': {
                'episode_makespans': self.episode_makespans,
                'episode_twts': self.episode_twts,
                'episode_objectives': self.episode_objectives,
                'episode_rewards': self.episode_rewards
            }
        }
    
    def save_model(self, filename: str):
        """Save model using standard pattern."""
        filepath = os.path.join(self.model_save_dir, filename)
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': {
                'hidden_dim': self.policy.hidden_dim,
                'num_hgt_layers': self.policy.num_hgt_layers,
                'gamma': self.gamma,
                'clip_ratio': self.clip_ratio,
                'value_coef': self.value_coef
            }
        }, filepath)
        print(f"Graph RL model saved to {filepath}")
    
    def load_model(self, filename: str):
        """Load model using standard pattern."""
        filepath = os.path.join(self.model_save_dir, filename)
        if os.path.exists(filepath):
            checkpoint = torch.load(filepath, map_location=self.device)
            self.policy.load_state_dict(checkpoint['policy_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print(f"Graph RL model loaded from {filepath}")
        else:
            print(f"Model file {filepath} not found")


def main():
    """Main training script."""
    parser = argparse.ArgumentParser(description='Train Graph PPO on FJSP')
    
    # Environment arguments
    parser.add_argument('--data-path', type=str, required=True,
                       help='Path to FJSP data file')
    parser.add_argument('--data-type', type=str, default='dataset',
                       choices=['dataset', 'simulation'],
                       help='Type of data source')
    
    # Training arguments
    parser.add_argument('--total-timesteps', type=int, default=1000000,
                       help='Total timesteps to train')
    parser.add_argument('--buffer-size', type=int, default=2048,
                       help='Buffer size for PPO')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size for training')
    parser.add_argument('--learning-rate', type=float, default=3e-4,
                       help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99,
                       help='Discount factor')
    parser.add_argument('--gae-lambda', type=float, default=0.95,
                       help='GAE lambda')
    parser.add_argument('--clip-ratio', type=float, default=0.2,
                       help='PPO clip ratio')
    
    # Network arguments
    parser.add_argument('--hidden-dim', type=int, default=128,
                       help='Hidden dimension for HGT network')
    parser.add_argument('--num-hgt-layers', type=int, default=3,
                       help='Number of HGT layers')
    parser.add_argument('--num-heads', type=int, default=4,
                       help='Number of attention heads')
    
    # Logging and saving
    parser.add_argument('--log-interval', type=int, default=10,
                       help='Logging interval')
    parser.add_argument('--save-interval', type=int, default=100,
                       help='Model saving interval')
    parser.add_argument('--save-path', type=str, default='models/graph_ppo',
                       help='Path to save models')
    parser.add_argument('--use-tensorboard', action='store_true',
                       help='Use TensorBoard logging')
    parser.add_argument('--use-wandb', action='store_true',
                       help='Use Weights & Biases logging')
    
    # Other arguments
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='Device to use')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Load problem data
    print(f"Loading FJSP data from {args.data_path}")
    problem_data = FlexibleJobShopDataHandler(
        data_source=args.data_path,
        data_type=args.data_type,
        seed=args.seed
    )
    
    print(f"Problem: {problem_data.num_jobs} jobs, "
          f"{problem_data.num_machines} machines, "
          f"{problem_data.num_operations} operations")
    
    # Initialize trainer with config-driven parameters
    trainer = GraphPPOTrainer(
        problem_data=problem_data,
        hidden_dim=args.hidden_dim,
        num_hgt_layers=args.num_hgt_layers,
        num_heads=args.num_heads,
        lr=args.learning_rate,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_ratio=args.clip_ratio,
        device=args.device,
        seed=args.seed
    )
    
    # Start training (using the actual train method signature)
    trainer.train(seed=args.seed)


if __name__ == '__main__':
    main()

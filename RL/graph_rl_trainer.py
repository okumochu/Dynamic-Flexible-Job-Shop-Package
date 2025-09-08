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
                 pi_lr: float = None,
                 v_lr: float = None,
                 gamma: float = None,
                 gae_lambda: float = None,
                 clip_ratio: float = None,
                 entropy_coef: float = None,
                 value_coef: float = None,
                 max_grad_norm: float = None,
                 target_kl: float = None,
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
        episodes_per_epoch = episodes_per_epoch if episodes_per_epoch is not None else rl_params['episodes_per_epoch']
        train_per_episode = train_per_episode if train_per_episode is not None else rl_params['train_per_episode']
        hidden_dim = hidden_dim if hidden_dim is not None else rl_params['hidden_dim']
        num_hgt_layers = num_hgt_layers if num_hgt_layers is not None else rl_params['num_hgt_layers']
        num_heads = num_heads if num_heads is not None else rl_params['num_heads']
        pi_lr = pi_lr if pi_lr is not None else rl_params['pi_lr']
        v_lr = v_lr if v_lr is not None else rl_params['v_lr']
        gamma = gamma if gamma is not None else rl_params['gamma']
        gae_lambda = gae_lambda if gae_lambda is not None else rl_params['gae_lambda']
        clip_ratio = clip_ratio if clip_ratio is not None else rl_params['clip_ratio']
        entropy_coef = entropy_coef if entropy_coef is not None else rl_params['entropy_coef']
        value_coef = value_coef if value_coef is not None else rl_params['value_coef']
        max_grad_norm = max_grad_norm if max_grad_norm is not None else rl_params['max_grad_norm']
        target_kl = target_kl if target_kl is not None else rl_params['target_kl']
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
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl
        
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
        
        # Initialize environment FIRST
        self.env = GraphRlEnv(problem_data)
        print(f"Environment: {self.env.problem_data.num_jobs} jobs, "
              f"{self.env.problem_data.num_machines} machines, "
              f"{self.env.problem_data.num_operations} operations")

        # Get an initial observation to determine feature dimensions
        initial_obs, _ = self.env.reset()
        op_feature_dim = initial_obs['op'].x.shape[1]
        machine_feature_dim = initial_obs['machine'].x.shape[1]
        
        print(f"Feature dimensions - Operations: {op_feature_dim}, Machines: {machine_feature_dim}")

        # Initialize policy network with the CORRECT dimensions
        self.policy = HGTPolicy(
            op_feature_dim=op_feature_dim,
            machine_feature_dim=machine_feature_dim,
            hidden_dim=hidden_dim,
            num_hgt_layers=num_hgt_layers,
            num_heads=num_heads,
            dropout=rl_params['dropout']
        ).to(self.device)
        
        print(f"Policy network parameters: {sum(p.numel() for p in self.policy.parameters()):,}")
        
        # Initialize optimizer with better defaults
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=pi_lr, eps=1e-5)
        
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
            consecutive_invalid_actions = 0
            max_invalid_actions = 5  # Prevent getting stuck
            
            # Run episode until completion
            while not self.env.graph_state.is_done() and episode_steps < self.env.max_episode_steps:
                # Get valid actions from environment info (from last reset or step)
                env_valid_actions = info.get('valid_actions', [])
                
                # Get action from policy with optimized action selection
                with torch.no_grad():
                    action_logits, value, _, policy_valid_pairs = self.policy(obs, env_valid_actions)
                    
                    if len(action_logits) == 0:
                        print("Warning: No valid actions available!")
                        consecutive_invalid_actions += 1
                        if consecutive_invalid_actions >= max_invalid_actions:
                            break
                        continue
                    
                    # Create distribution and sample action
                    dist = Categorical(logits=action_logits)
                    action_idx = dist.sample()
                    log_prob = dist.log_prob(action_idx)
                    
                    # OPTIMIZED: Direct mapping from action index to environment action
                    # action_idx corresponds directly to env_valid_actions[action_idx]
                    if action_idx.item() < len(env_valid_actions):
                        target_pair = env_valid_actions[action_idx.item()]
                        
                        # Find corresponding environment action (this lookup is now much simpler)
                        env_action = None
                        for env_action_idx, pair in self.env.action_to_pair_map.items():
                            if pair == target_pair:
                                env_action = env_action_idx
                                break
                        
                        if env_action is None:
                            print(f"Warning: Could not find environment action for pair {target_pair}")
                            break
                    else:
                        print(f"Warning: Action index {action_idx.item()} out of range")
                        break
                
                # Step environment
                next_obs, reward, terminated, truncated, info = self.env.step(env_action)
                done = terminated or truncated
                
                # Reset invalid action counter on successful action
                consecutive_invalid_actions = 0
                
                # Store experience in buffer
                buffer.add(
                    obs=obs.cpu(),
                    action=action_idx.cpu().item(),
                    reward=reward,
                    value=value.cpu().item(),
                    log_prob=log_prob.cpu().item(),
                    valid_pairs=env_valid_actions,  # Use environment's valid actions (optimization)
                    done=done
                )
                
                # Update episode statistics
                episode_reward += reward
                episode_steps += 1
                
                if done:
                    break
                    
                obs = next_obs.to(self.device)
            
            # Track completion
            epoch_episodes_completed += 1
            epoch_total_steps += episode_steps
            
            # Track episode metrics - focus on makespan only
            final_makespan = self.env.graph_state.get_makespan()
            
            # For single objective optimization, objective = makespan
            objective = final_makespan
            
            # Store episode metrics
            self.episode_makespans.append(final_makespan)
            self.episode_twts.append(0.0)  # Keep for compatibility but not used
            self.episode_objectives.append(objective)
            self.episode_rewards.append(episode_reward)
            
            # Store episode metrics for epoch-level aggregation (no individual logging to avoid step conflicts)
        
        # Get current buffer state for debugging
        current_buffer_size = self.buffer.size
        
        return {
            'episodes_completed': epoch_episodes_completed,
            'total_steps_collected': epoch_total_steps,
            'buffer_size_after_collection': current_buffer_size,
            'mean_episode_reward': np.mean(self.episode_rewards[-self.episodes_per_epoch:]) if len(self.episode_rewards) >= self.episodes_per_epoch else 0,
            'mean_makespan': np.mean(self.episode_makespans[-self.episodes_per_epoch:]) if len(self.episode_makespans) >= self.episodes_per_epoch else 0,
            'mean_objective': np.mean(self.episode_objectives[-self.episodes_per_epoch:]) if len(self.episode_objectives) >= self.episodes_per_epoch else 0
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
            return {'policy_loss': 0, 'value_loss': 0, 'entropy': 0, 'kl_div': 0, 'optimizer_stepped': False}
        
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
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        total_kl_div = 0
        
        num_updates = 4  # Number of PPO epochs
        batch_size = 32   # Mini-batch size
        
        for _ in range(num_updates):
            # Create mini-batches
            indices = torch.randperm(len(batch['observations']))
            
            for start_idx in range(0, len(indices), batch_size):
                end_idx = start_idx + batch_size
                mini_batch_indices = indices[start_idx:end_idx]
                
                if len(mini_batch_indices) == 0:
                    continue
                
                # Get mini-batch data
                mini_batch_obs = [batch['observations'][i] for i in mini_batch_indices]
                mini_batch_valid_pairs = [batch['valid_action_pairs'][i] for i in mini_batch_indices]
                mini_batch_actions = torch.as_tensor([batch['actions'][i] for i in mini_batch_indices], device=self.device)
                mini_batch_advantages = advantages[mini_batch_indices]
                mini_batch_returns = returns[mini_batch_indices]
                mini_batch_old_log_probs = old_log_probs[mini_batch_indices]
                
                # Forward pass for each observation in mini-batch
                policy_losses = []
                value_losses = []
                entropies = []
                kl_divs = []
                
                for i, obs in enumerate(mini_batch_obs):
                    obs = obs.to(self.device)
                    stored_valid_pairs = mini_batch_valid_pairs[i]
                    action_logits, value, _, valid_pairs = self.policy(obs, stored_valid_pairs)
                    
                    if len(action_logits) == 0:
                        continue
                    
                    # Find the action index in current valid pairs
                    action_idx = mini_batch_actions[i].item()
                    stored_valid_pairs = mini_batch_valid_pairs[i]
                    
                    if action_idx >= len(stored_valid_pairs):
                        continue
                        
                    target_pair = stored_valid_pairs[action_idx]
                    
                    # Find corresponding index in current valid pairs
                    current_action_idx = None
                    for j, pair in enumerate(valid_pairs):
                        if pair == target_pair:
                            current_action_idx = j
                            break
                    
                    # If the action is no longer valid, compute logits for the stored valid pairs instead
                    if current_action_idx is None:
                        # Get logits for the original valid pairs using the specialized method
                        stored_action_logits = self.policy.get_action_logits_for_pairs(obs, stored_valid_pairs)
                        if len(stored_action_logits) > action_idx:
                            # Use the stored action logits and index
                            action_logits_to_use = stored_action_logits
                            current_action_idx = action_idx
                        else:
                            continue
                    else:
                        action_logits_to_use = action_logits
                    
                    # Create distribution and compute log prob
                    dist = Categorical(logits=action_logits_to_use)
                    new_log_prob = dist.log_prob(torch.as_tensor(current_action_idx, device=self.device))
                    
                    # Compute entropy using tested PPOUpdater
                    entropy = PPOUpdater.compute_entropy(action_logits_to_use.unsqueeze(0))
                    
                    # PPO policy loss using tested PPOUpdater
                    policy_loss = PPOUpdater.ppo_policy_loss(
                        new_log_prob.unsqueeze(0), mini_batch_old_log_probs[i].unsqueeze(0), 
                        mini_batch_advantages[i].unsqueeze(0), self.clip_ratio
                    )
                    
                    # Value loss
                    value_loss = F.mse_loss(value.squeeze(), mini_batch_returns[i])
                    
                    # KL divergence (approximate)
                    kl_div = mini_batch_old_log_probs[i] - new_log_prob
                    
                    policy_losses.append(policy_loss)
                    value_losses.append(value_loss)
                    entropies.append(entropy)
                    kl_divs.append(kl_div)
                
                if not policy_losses:
                    continue
                
                # Combine losses
                policy_loss = torch.stack(policy_losses).mean()
                value_loss = torch.stack(value_losses).mean()
                entropy_loss = torch.stack(entropies).mean()
                kl_div = torch.stack(kl_divs).mean()
                
                # Total loss
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy_loss
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                
                # Clip gradients and monitor gradient norms
                total_norm_before = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                
                # Check for gradient issues
                if torch.isnan(total_norm_before) or torch.isinf(total_norm_before):
                    print(f"Warning: Invalid gradient norm detected: {total_norm_before}")
                    continue  # Skip this update
                
                self.optimizer.step()
                
                # Accumulate statistics
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy_loss.item()
                total_kl_div += kl_div.item()
                
                # Early stopping if KL divergence is too high
                if kl_div.item() > self.target_kl:
                    break
        
        num_mini_batches = max(1, (len(batch['observations']) // batch_size) * num_updates)
        
        # Debug info (removed verbose logging)
        if len(batch['observations']) > 0 and num_mini_batches > 0:
            pass  # Debug info available in wandb metrics
        
        return {
            'policy_loss': total_policy_loss / max(1, num_mini_batches) if total_policy_loss > 0 else 0.0,
            'value_loss': total_value_loss / max(1, num_mini_batches) if total_value_loss > 0 else 0.0,
            'entropy': total_entropy / max(1, num_mini_batches) if total_entropy > 0 else 0.0,
            'kl_div': total_kl_div / max(1, num_mini_batches) if total_kl_div > 0 else 0.0,
            'optimizer_stepped': True,
            'buffer_size': len(batch['observations']),
            'mini_batches_processed': num_mini_batches
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
                    "pi_lr": self.optimizer.param_groups[0]['lr'],
                    "gamma": self.gamma,
                    "gae_lambda": self.gae_lambda,
                    "clip_ratio": self.clip_ratio,
                    "entropy_coef": self.entropy_coef,
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
            stats = {'policy_loss': 0, 'value_loss': 0, 'entropy': 0, 'kl_div': 0, 'optimizer_stepped': False}
            
            # Only train if we have enough data
            if self.buffer.size >= max(32, self.episodes_per_epoch // 2):
                for train_step in range(self.train_per_episode):
                    step_stats = self.update_policy()
                    
                    # Accumulate statistics
                    for key in ['policy_loss', 'value_loss', 'entropy', 'kl_div']:
                        stats[key] += step_stats.get(key, 0)
                    
                    if step_stats.get('optimizer_stepped', False):
                        stats['optimizer_stepped'] = True
                        total_optimization_steps += 1
                        
                    # Early stopping if KL divergence is too high
                    if step_stats.get('kl_div', 0) > self.target_kl:
                        break
                
                # Average the statistics
                num_train_steps = self.train_per_episode
                for key in ['policy_loss', 'value_loss', 'entropy', 'kl_div']:
                    stats[key] /= max(1, num_train_steps)
                
                # Only update learning rate if we actually performed optimization
                if stats.get('optimizer_stepped', False):
                    self.scheduler.step()

            # Log training metrics
            wandb_log = {
                "training/policy_loss": stats["policy_loss"],
                "training/value_loss": stats["value_loss"],
                "training/entropy": stats["entropy"],
                "training/kl_div": stats["kl_div"],
                "training/epoch": epoch,
                "training/total_episodes": len(self.episode_makespans),
                "training/learning_rate": self.optimizer.param_groups[0]['lr'],
                "training/buffer_size": stats.get("buffer_size", 0),
                "training/mini_batches_processed": stats.get("mini_batches_processed", 0),
                "training/steps_collected": collection_stats.get("total_steps_collected", 0),
                "training/buffer_after_collection": collection_stats.get("buffer_size_after_collection", 0)
            }
            
            # Add collection metrics if available
            if collection_stats['episodes_completed'] > 0:
                recent_episodes = self.episodes_per_epoch
                if len(self.episode_makespans) >= recent_episodes:
                    recent_makespans = self.episode_makespans[-recent_episodes:]
                    recent_objectives = self.episode_objectives[-recent_episodes:]
                    recent_rewards = self.episode_rewards[-recent_episodes:]
                    
                wandb_log.update({
                        "performance/mean_makespan": collection_stats['mean_makespan'],
                        "performance/mean_objective": collection_stats['mean_objective'],
                        "performance/mean_reward": collection_stats['mean_episode_reward'],
                        "performance/best_makespan": min(recent_makespans),
                        "performance/worst_makespan": max(recent_makespans),
                        "performance/makespan_std": np.std(recent_makespans),
                        "performance/best_objective": min(recent_objectives),
                        "performance/latest_makespan": recent_makespans[-1],
                        "performance/latest_reward": recent_rewards[-1],
                    })
            
            # Log all metrics with epoch as step
            if WANDB_AVAILABLE:
                wandb.log(wandb_log, step=epoch)

            # Clear buffer only when it's getting full to improve sample efficiency
            if self.buffer.size >= self.buffer.buffer_size * 0.8:
                self.buffer.clear()
            
            # Progress monitoring
            if (epoch + 1) % 10 == 0:
                recent_makespans = self.episode_makespans[-20:] if len(self.episode_makespans) >= 20 else self.episode_makespans
                if recent_makespans:
                    avg_recent_makespan = np.mean(recent_makespans)
                    current_lr = self.optimizer.param_groups[0]['lr']
                    print(f"Epoch {epoch+1}/{self.epochs}: Recent avg makespan = {avg_recent_makespan:.2f}, LR = {current_lr:.6f}")
        
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
                'entropy_coef': self.entropy_coef,
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
    parser.add_argument('--entropy-coef', type=float, default=0.01,
                       help='Entropy coefficient')
    
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
    
    # Initialize trainer
    trainer = GraphPPOTrainer(
        problem_data=problem_data,
        hidden_dim=args.hidden_dim,
        num_hgt_layers=args.num_hgt_layers,
        num_heads=args.num_heads,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_ratio=args.clip_ratio,
        entropy_coef=args.entropy_coef,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        device=args.device,
        seed=args.seed
    )
    
    # Start training
    trainer.train(
        total_timesteps=args.total_timesteps,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        save_path=args.save_path,
        use_tensorboard=args.use_tensorboard,
        use_wandb=args.use_wandb
    )


if __name__ == '__main__':
    main()

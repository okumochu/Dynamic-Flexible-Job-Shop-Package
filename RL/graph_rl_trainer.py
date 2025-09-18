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
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from torch_geometric.data import Batch
from typing import Dict, List, Tuple, Optional
from config import config
import wandb

# Import project modules
from benchmarks.data_handler import FlexibleJobShopDataHandler
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
        
        # Episode tracking for current epoch only
        self.episode_makespans = []
        self.episode_twts = []
        self.episode_objectives = []
        self.episode_rewards = []
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
        self.optimizer = torch.optim.AdamW(self.policy.parameters(), lr=lr)
        
        # Initialize buffer - use upper bound: max_operations * episodes_per_epoch
        # Buffer will be cleared after each update (PPO is on-policy)
        max_steps_per_episode = self.env.problem_data.num_operations
        # Initialize buffer for storing experiences
        buffer_size = max_steps_per_episode * self.episodes_per_epoch
        self.buffer = GraphPPOBuffer(buffer_size, self.device)
        
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
    
    def collect_rollout(self, buffer: GraphPPOBuffer) -> Dict:
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
            self.env.reset()
            
            episode_reward = 0
            episode_steps = 0
            
            # Run episode until completion
            while not self.env.graph_state.is_done():
                # Get current observation and valid actions from environment state
                current_obs = self.env.graph_state.get_observation().to(self.device)
                env_valid_actions = self.env.graph_state.get_valid_actions()
                
                # Get action from policy - always produces valid action logits
                with torch.no_grad():
                    action_logits, value = self.policy(current_obs, env_valid_actions)
                    
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
                
                # Store experience in buffer (keep on device for efficiency)
                buffer.add(
                    obs=current_obs,  # Use current observation, not stale obs
                    action=action_idx.item(),
                    reward=reward,
                    value=value.item(),
                    log_prob=log_prob.item(),
                    valid_pairs=env_valid_actions.copy(),  # Store copy to avoid reference issues
                    done=terminated
                )
                
                # Update episode statistics
                episode_reward += reward
                episode_steps += 1                    
            
            # Track completion
            epoch_episodes_completed += 1
            epoch_total_steps += episode_steps
            
            # Extract final episode metrics from environment
            final_makespan = self.env.graph_state.get_makespan()
            final_twt = self.env.graph_state.get_total_weighted_tardiness()
            
            # Calculate multi-objective using alpha parameter  
            alpha = self.env.alpha  # Get alpha from environment
            objective = (1 - alpha) * final_makespan + alpha * final_twt
            
            # Store episode metrics
            self.episode_makespans.append(final_makespan)
            self.episode_twts.append(final_twt)
            self.episode_objectives.append(objective)
            self.episode_rewards.append(episode_reward)
        
        # Return collection statistics with epoch mean performance metrics
        return {
            'episodes_completed': epoch_episodes_completed,
            'total_steps': epoch_total_steps,
            'makespan_mean': float(np.mean(self.episode_makespans)),
            'twt_mean': float(np.mean(self.episode_twts)),
            'objective_mean': float(np.mean(self.episode_objectives)),
            'reward_mean': float(np.mean(self.episode_rewards))
        }
    
    def update_policy(self) -> Dict:
        """
        Update the policy using PPO.
        
        Returns:
            Dictionary with training statistics
        """
        self.policy.train()
        
        # Get batch from buffer (already on device)
        batch = self.buffer.get_batch()
        
        # Extract tensors (already on correct device)
        rewards = batch['rewards']
        values = batch['values']
        dones = batch['dones']
        
        # Compute GAE advantages using tested PPOUpdater
        advantages, returns = PPOUpdater.compute_gae_advantages(
            rewards, values, dones, self.gamma, self.gae_lambda
        )
        
        # Store old log probs for PPO clipping (already on device)
        old_log_probs = batch['log_probs']
        
        # Process full batch without mini-batching for small problems
        total_policy_loss = 0
        total_value_loss = 0
        num_updates_performed = 0
        
        # Multiple PPO updates on the same data (train_per_episode times)
        for _ in range(self.train_per_episode):
            # Process batch in order (no shuffling for size-variant action spaces)
            
            # Get batch data in original order
            batch_obs = batch['observations']
            batch_valid_pairs = batch['valid_action_pairs']
            batch_actions = torch.as_tensor(batch['actions'], device=self.device)
            batch_advantages = advantages
            batch_returns = returns
            batch_old_log_probs = old_log_probs
            
            # Create batch from individual HeteroData observations
            batch_graphs = Batch.from_data_list([obs.to(self.device) for obs in batch_obs])
            
            # Forward pass through the policy network
            batch_action_logits, batch_values = self.policy.forward(batch_graphs, batch_valid_pairs, use_batch=True)
            
            # Process results for each graph in the batch
            policy_losses = []
            value_losses = []
            
            for i in range(len(batch_obs)):
                action_logits = batch_action_logits[i]  # Action logits for graph i
                value = batch_values[i]  # Value estimate for graph i
                action_idx = batch_actions[i].item()
                
                # Create distribution and compute log prob for the stored action
                dist = Categorical(logits=action_logits)
                new_log_prob = dist.log_prob(torch.as_tensor(action_idx, device=self.device))

                # PPO policy loss
                policy_loss = PPOUpdater.ppo_policy_loss(
                    new_log_prob.unsqueeze(0), batch_old_log_probs[i].unsqueeze(0), 
                    batch_advantages[i].unsqueeze(0), self.clip_ratio
                )

                # Scale loss by number of valid actions to normalize learning across variable action spaces
                K_t = max(2, len(batch_valid_pairs[i]))  # avoid log(1) or log(0)
                scaling_weight = 1.0 / float(np.log(K_t))
                policy_loss = policy_loss * scaling_weight
                
                # Value loss
                value_loss = F.mse_loss(value, batch_returns[i])
                
                # Store losses for this sample
                policy_losses.append(policy_loss)
                value_losses.append(value_loss)
            
            # Combine losses for the batch
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


        # Training loop
        start_time = time.time()
        
        from tqdm import tqdm
        pbar = tqdm(range(self.epochs), desc="Graph RL Training")


        for epoch in pbar:
            # Collect rollout data
            collection_stats = self.collect_rollout(self.buffer)
            
            # Update agent multiple times on the same data (standard PPO practice)
            # train_per_episode is now handled inside update_policy()
            stats = self.update_policy()
            
            # Learning rate is updated inside update_policy()

            # Log all metrics together
            wandb_log = {
                "policy_loss": stats["policy_loss"],
                "value_loss": stats["value_loss"],
                "total_epochs": epoch + 1,
                "learning_rate": self.optimizer.param_groups[0]['lr'],
                "performance/makespan_mean": collection_stats["makespan_mean"],
                "performance/twt_mean": collection_stats["twt_mean"],
                "performance/objective_mean": collection_stats["objective_mean"],
                "performance/reward_mean": collection_stats["reward_mean"],
                "performance/alpha": self.env.alpha
            }
            
            # Log all metrics with epoch as step
            wandb.log(wandb_log, step=epoch)

            # Clear buffer and episode lists after each epoch (PPO is on-policy)
            self.buffer.clear()
            self.episode_makespans.clear()
            self.episode_twts.clear()
            self.episode_objectives.clear()
            self.episode_rewards.clear()
        
        pbar.close()
        
        # Save model with timestamp
        model_filename = config.create_model_filename()
        self.save_model(model_filename)
        
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
            print(f"Graph RL model loaded from {filepath}")
        else:
            print(f"Model file {filepath} not found")



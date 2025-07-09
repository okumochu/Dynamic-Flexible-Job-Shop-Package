"""
Hierarchical RL Agent Components
Contains Manager and Worker classes for feudal-style hierarchical RL
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Optional, Tuple, Dict, List
from .networks import PerceptualEncoder, DilatedLSTMManager, HierarchicalWorker
from .buffer import PPOBuffer


class HierarchicalAgent:
    """
    Hierarchical RL Agent with Manager-Worker architecture.
    Contains all update logic, similar to FlatAgent.
    """
    
    def __init__(self,
                 input_dim: int,
                 action_dim: int,
                 latent_dim: int = 256,
                 goal_dim: int = 32,
                 hidden_dim: int = 512,
                 dilation: int = 10,
                 manager_lr: float = 3e-4,
                 worker_lr: float = 3e-4,
                 gamma_manager: float = 0.995,
                 gamma_worker: float = 0.95,
                 gae_lambda: float = 0.95,
                 clip_ratio: float = 0.2,
                 entropy_coef: float = 0.01,
                 epsilon_greedy: float = 0.1,
                 device: str = 'auto'):
        
        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Hierarchical Agent initialized on device: {self.device}")
        
        # Store configuration
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.goal_dim = goal_dim
        self.hidden_dim = hidden_dim
        self.dilation = dilation
        self.manager_lr = manager_lr
        self.worker_lr = worker_lr
        self.gamma_manager = gamma_manager
        self.gamma_worker = gamma_worker
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.entropy_coef = entropy_coef
        self.epsilon_greedy = epsilon_greedy
        
        # Create models
        self.encoder = PerceptualEncoder(input_dim, latent_dim).to(self.device)
        self.manager = DilatedLSTMManager(latent_dim, hidden_dim, dilation).to(self.device)
        self.worker = HierarchicalWorker(latent_dim, action_dim, goal_dim, hidden_dim).to(self.device)
        
        # Optimizers
        self.manager_optimizer = optim.Adam(
            list(self.encoder.parameters()) + list(self.manager.parameters()),
            lr=manager_lr
        )
        self.worker_optimizer = optim.Adam(
            list(self.encoder.parameters()) + list(self.worker.parameters()),
            lr=worker_lr
        )
        
        # Storage for hierarchical data during training
        self.manager_data = {
            'states': [],
            'goals': [],
            'values': [],
            'rewards': [],
            'dones': []
        }
        
        # Training statistics
        self.training_stats = {
            'manager_loss': [],
            'manager_policy_loss': [],
            'manager_value_loss': [],
            'worker_policy_loss': [],
            'worker_value_loss': [],
            'worker_total_loss': [],
            'avg_goal_norm': [],
            'avg_cosine_alignment': []
        }
    
    def take_action(self, obs: torch.Tensor, action_mask: torch.Tensor, 
                   pooled_goals: torch.Tensor, prev_r_int: float) -> Tuple[int, float, float]:
        """Take action using hierarchical policy"""
        with torch.no_grad():
            obs = obs.unsqueeze(0).to(self.device)
            action_mask = action_mask.to(self.device)
            pooled_goals = pooled_goals.unsqueeze(0).to(self.device)
            prev_r_int_tensor = torch.tensor([prev_r_int], device=self.device).unsqueeze(0)
            
            # Encode state
            z_t = self.encoder(obs)
            
            # Worker action
            action, log_prob, value = self.worker.get_action_and_value(
                z_t, pooled_goals, prev_r_int_tensor, action_mask.unsqueeze(0)
            )
            
            return int(action.item()), float(log_prob.item()), float(value.item())
    
    def get_deterministic_action(self, obs: torch.Tensor, action_mask: torch.Tensor,
                                pooled_goals: torch.Tensor, prev_r_int: float) -> int:
        """Get deterministic action for evaluation"""
        with torch.no_grad():
            obs = obs.unsqueeze(0).to(self.device)
            action_mask = action_mask.to(self.device)
            pooled_goals = pooled_goals.unsqueeze(0).to(self.device)
            prev_r_int_tensor = torch.tensor([prev_r_int], device=self.device).unsqueeze(0)
            
            # Encode state
            z_t = self.encoder(obs)
            
            # Worker deterministic action
            action = self.worker.get_deterministic_action(
                z_t, pooled_goals, prev_r_int_tensor, action_mask.unsqueeze(0)
            )
            
            return int(action.item()) if hasattr(action, 'item') else int(action)
    
    def encode_state(self, obs: torch.Tensor) -> torch.Tensor:
        """Encode observation to latent space"""
        with torch.no_grad():
            obs = obs.unsqueeze(0).to(self.device)
            return self.encoder(obs).squeeze(0)
    
    def get_manager_goal(self, z_t: torch.Tensor, step: int, hidden=None):
        """Get manager goal at current step"""
        return self.manager.get_goal_at_step(z_t, step, hidden)
    
    def compute_intrinsic_reward(self, encoded_states: List[torch.Tensor], 
                                goals: List[torch.Tensor], step: int) -> float:
        """Compute intrinsic reward: r_int = (1/c) Σ_{i=1}^c cos(s_t – s_{t−i}, g_{t−i})"""
        if step < self.dilation or step >= len(encoded_states):
            return 0.0
        
        r_int = 0.0
        current_state = encoded_states[step]
        
        for i in range(1, min(self.dilation + 1, step + 1)):
            if step - i >= 0:
                past_state = encoded_states[step - i]
                goal_idx = max(0, (step - i) // self.dilation)
                if goal_idx < len(goals):
                    state_diff = current_state - past_state
                    goal_vector = goals[goal_idx]
                    cosine_sim = F.cosine_similarity(state_diff.unsqueeze(0), goal_vector.unsqueeze(0))
                    r_int += cosine_sim.item()
        
        return r_int / self.dilation
    
    def pool_goals(self, goals: List[torch.Tensor], step: int) -> torch.Tensor:
        """Pool goals: Σ_{i=t−c+1}^t g_i"""
        if len(goals) == 0:
            return torch.zeros(self.latent_dim, device=self.device)
        
        start_idx = max(0, (step - self.dilation + 1) // self.dilation)
        end_idx = min(len(goals), step // self.dilation + 1)
        
        if start_idx >= end_idx:
            return goals[-1] if len(goals) > 0 else torch.zeros(self.latent_dim, device=self.device)
        
        # Stack the goal tensors and then sum
        selected_goals = goals[start_idx:end_idx]
        if len(selected_goals) == 1:
            pooled = selected_goals[0]
        else:
            pooled = torch.sum(torch.stack(selected_goals), dim=0)
        return pooled
    
    def add_manager_experience(self, state: torch.Tensor, goal: torch.Tensor, 
                              value: float, reward: float, done: bool):
        """Add manager experience"""
        self.manager_data['states'].append(state)
        self.manager_data['goals'].append(goal)
        self.manager_data['values'].append(value)
        self.manager_data['rewards'].append(reward)
        self.manager_data['dones'].append(done)
    
    def compute_manager_reward(self, encoded_states: List[torch.Tensor], 
                              goals: List[torch.Tensor], goal_idx: int, current_step: int, 
                              episode_ended: bool = False, final_reward: float = 0.0) -> float:
        """
        Compute manager reward: R_t only when episode ends
        Args:
            encoded_states: List of encoded states
            goals: List of goals
            goal_idx: Index of the goal
            current_step: Current step
            episode_ended: Whether the episode has ended
            final_reward: The true external reward when episode ends
        """
        if goal_idx >= len(goals) or current_step < self.dilation:
            return 0.0
        
        # Manager only receives reward when episode ends
        if episode_ended:
            return final_reward
        else:
            return 0.0
    
    def compute_gae(self, rewards, values, dones, gamma, gae_lambda):
        """Compute Generalized Advantage Estimation"""
        advantages = torch.zeros_like(rewards)
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0 if dones[t] else values[t]
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + gamma * next_value * (1 - dones[t].float()) - values[t]
            gae = delta + gamma * gae_lambda * (1 - dones[t].float()) * gae
            advantages[t] = gae
        
        returns = advantages + values
        return advantages, returns
    
    def update_manager(self) -> Dict:
        """Update manager using transition-policy gradient"""
        if len(self.manager_data['states']) == 0:
            return {'loss': 0.0, 'policy_loss': 0.0, 'value_loss': 0.0, 
                   'avg_goal_norm': 0.0, 'avg_cosine_alignment': 0.0}
        
        # Convert to tensors
        states = torch.stack(self.manager_data['states'])
        goals = torch.stack(self.manager_data['goals'])
        values = torch.tensor(self.manager_data['values'], device=self.device)
        rewards = torch.tensor(self.manager_data['rewards'], device=self.device)
        dones = torch.tensor(self.manager_data['dones'], device=self.device)
        
        # Compute advantages for manager
        advantages, returns = self.compute_gae(
            rewards, values, dones, self.gamma_manager, self.gae_lambda
        )
        
        # Manager update
        self.manager_optimizer.zero_grad()
        
        # Forward pass through manager
        states_seq = states.unsqueeze(0)  # Add batch dim
        pred_goals, pred_values, _ = self.manager(states_seq)
        pred_goals = pred_goals.squeeze(0)  # Remove batch dim
        pred_values = pred_values.squeeze(0)
        
        # Policy gradient loss
        policy_loss = -torch.mean(advantages.detach() * torch.sum(pred_goals * goals, dim=1))
        
        # Value loss
        value_loss = F.mse_loss(pred_values, returns)
        
        # Total loss
        total_loss = policy_loss + 0.5 * value_loss
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.manager.parameters(), 0.5)
        self.manager_optimizer.step()
        
        # Statistics
        with torch.no_grad():
            avg_goal_norm = torch.mean(torch.norm(pred_goals, dim=1)).item()
            avg_cosine_alignment = torch.mean(
                F.cosine_similarity(pred_goals, goals)
            ).item()
        
        stats = {
            'loss': total_loss.item(),
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'avg_goal_norm': avg_goal_norm,
            'avg_cosine_alignment': avg_cosine_alignment
        }
        
        # Update training stats
        for key, value in stats.items():
            if f'manager_{key}' in self.training_stats:
                self.training_stats[f'manager_{key}'].append(value)
        
        return stats
    
    def update_worker(self, buffer: PPOBuffer, train_pi_iters: int = 10, train_v_iters: int = 10) -> Dict:
        """Update worker using PPO with standard buffer"""
        worker_data = buffer.get_batch()
        
        if worker_data['observations'].shape[0] == 0:
            return {'policy_loss': 0.0, 'value_loss': 0.0, 'total_loss': 0.0, 'entropy': 0.0}
        
        # Compute advantages for worker
        advantages, returns = self.compute_gae(
            worker_data['rewards'], worker_data['values'],
            worker_data['dones'], self.gamma_worker, self.gae_lambda
        )
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # For simplified version, use zero goals and intrinsic rewards
        # This would need to be enhanced to properly handle hierarchical structure
        
        # PPO updates
        for _ in range(train_pi_iters):
            self.worker_optimizer.zero_grad()
            
            # Encode states
            encoded_states = self.encoder(worker_data['observations'])
            
            # For this simplified version, use zero goals and intrinsic rewards
            batch_size = encoded_states.shape[0]
            zero_goals = torch.zeros(batch_size, self.latent_dim, device=self.device)
            zero_r_int = torch.zeros(batch_size, 1, device=self.device)
            
            # Forward pass through worker
            action_probs, values = self.worker(
                encoded_states, zero_goals, zero_r_int, worker_data['action_masks']
            )
            
            # New log probabilities
            dist = torch.distributions.Categorical(action_probs)
            new_log_probs = dist.log_prob(worker_data['actions'])
            entropy = dist.entropy().mean()
            
            # PPO ratio
            ratio = torch.exp(new_log_probs - worker_data['log_probs'])
            
            # PPO loss
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = F.mse_loss(values.squeeze(-1), returns)
            
            # Total loss
            total_loss = policy_loss + 0.5 * value_loss - self.entropy_coef * entropy
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.worker.parameters(), 0.5)
            self.worker_optimizer.step()
        
        stats = {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'total_loss': total_loss.item(),
            'entropy': entropy.item()
        }
        
        # Update training stats
        for key, value in stats.items():
            if f'worker_{key}' in self.training_stats:
                self.training_stats[f'worker_{key}'].append(value)
        
        return stats
    
    def clear_manager_data(self):
        """Clear manager data storage"""
        self.manager_data = {
            'states': [],
            'goals': [],
            'values': [],
            'rewards': [],
            'dones': []
        }
    
    def save(self, path: str):
        """Save all models and training stats"""
        torch.save({
            'encoder_state_dict': self.encoder.state_dict(),
            'manager_state_dict': self.manager.state_dict(),
            'worker_state_dict': self.worker.state_dict(),
            'manager_optimizer_state_dict': self.manager_optimizer.state_dict(),
            'worker_optimizer_state_dict': self.worker_optimizer.state_dict(),
            'training_stats': self.training_stats,
            # Save configuration parameters
            'input_dim': self.input_dim,
            'action_dim': self.action_dim,
            'latent_dim': self.latent_dim,
            'goal_dim': self.goal_dim,
            'hidden_dim': self.hidden_dim,
            'dilation': self.dilation,
            'manager_lr': self.manager_lr,
            'worker_lr': self.worker_lr,
            'gamma_manager': self.gamma_manager,
            'gamma_worker': self.gamma_worker,
            'gae_lambda': self.gae_lambda,
            'clip_ratio': self.clip_ratio,
            'entropy_coef': self.entropy_coef,
            'epsilon_greedy': self.epsilon_greedy,
            'device': str(self.device)
        }, path)
    
    def load(self, path: str):
        """Load all models and training stats"""
        checkpoint = torch.load(path, map_location=self.device)
        try:
            self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
            self.manager.load_state_dict(checkpoint['manager_state_dict'])
            self.worker.load_state_dict(checkpoint['worker_state_dict'])
        except RuntimeError as e:
            print("\n[ERROR] Model architecture mismatch while loading state_dict!")
            print("This usually means your current network architecture does not match the one used for training.")
            print("Details:", e)
            raise
        
        self.manager_optimizer.load_state_dict(checkpoint['manager_optimizer_state_dict'])
        self.worker_optimizer.load_state_dict(checkpoint['worker_optimizer_state_dict'])
        self.training_stats = checkpoint['training_stats']
    
    @staticmethod
    def get_saved_config(path: str):
        """Get the configuration parameters from a saved model."""
        checkpoint = torch.load(path, map_location='cpu')
        if 'input_dim' in checkpoint:
            return {
                'input_dim': checkpoint['input_dim'],
                'action_dim': checkpoint['action_dim'],
                'latent_dim': checkpoint['latent_dim'],
                'goal_dim': checkpoint['goal_dim'],
                'hidden_dim': checkpoint['hidden_dim'],
                'dilation': checkpoint['dilation'],
                'manager_lr': checkpoint['manager_lr'],
                'worker_lr': checkpoint['worker_lr'],
                'gamma_manager': checkpoint['gamma_manager'],
                'gamma_worker': checkpoint['gamma_worker'],
                'gae_lambda': checkpoint['gae_lambda'],
                'clip_ratio': checkpoint['clip_ratio'],
                'entropy_coef': checkpoint['entropy_coef'],
                'epsilon_greedy': checkpoint['epsilon_greedy'],
                'device': checkpoint['device']
            }
        else:
            raise ValueError("This checkpoint does not contain configuration parameters.")

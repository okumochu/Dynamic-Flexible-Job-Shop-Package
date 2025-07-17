"""
Hierarchical RL Agent Components
Contains Manager and Worker classes for feudal-style hierarchical RL
"""

import torch
import torch.nn.functional as F
import torch.optim as optim
from typing import Optional, Tuple, Dict, List
from RL.PPO.networks import PerceptualEncoder, ManagerPolicy, WorkerPolicy, ValueNetwork, PPOUpdater
from RL.PPO.buffer import PPOBuffer, HierarchicalPPOBuffer


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
                 goal_duration: int = 10,  # c parameter - goal duration
                 manager_lr: float = 3e-4,
                 worker_lr: float = 3e-4,
                 gamma_manager: float = 0.995,
                 gamma_worker: float = 0.95,
                 gae_lambda: float = 0.95,
                 clip_ratio: float = 0.2,
                 entropy_coef: float = 0.01,
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
        self.goal_duration = goal_duration  # Store goal duration (c)
        self.manager_lr = manager_lr
        self.worker_lr = worker_lr
        self.gamma_manager = gamma_manager
        self.gamma_worker = gamma_worker
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.entropy_coef = entropy_coef
        
        # Create models
        self.manager_encoder = PerceptualEncoder(input_dim, latent_dim).to(self.device)
        self.manager_policy = ManagerPolicy(latent_dim).to(self.device)
        self.manager_value = ValueNetwork(latent_dim).to(self.device)

        self.worker_policy = WorkerPolicy(latent_dim, action_dim, goal_dim).to(self.device)
        self.worker_value = ValueNetwork(latent_dim).to(self.device)
        
        # FIX: Use separate optimizers to avoid conflicts with shared encoder
        # Create shared encoder optimizer that will be used by both manager and worker
        self.encoder_optimizer = optim.Adam(self.manager_encoder.parameters(), lr=min(manager_lr, worker_lr))
        
        # Manager optimizers (without encoder parameters)
        self.manager_policy_optimizer = optim.Adam(self.manager_policy.parameters(), lr=manager_lr)
        self.manager_value_optimizer = optim.Adam(self.manager_value.parameters(), lr=manager_lr)
        
        # Worker optimizers (without encoder parameters)
        self.worker_policy_optimizer = optim.Adam(self.worker_policy.parameters(), lr=worker_lr)
        self.worker_value_optimizer = optim.Adam(self.worker_value.parameters(), lr=worker_lr)
        
        # Training statistics
        self.training_stats = {
            'manager_policy_loss': [],
            'manager_value_loss': [],
            'worker_policy_loss': [],
            'worker_value_loss': [],
            'worker_entropy': [],
            'avg_goal_norm': [],
            'avg_cosine_alignment': []
        }
    
    def get_manager_goal(self, z_t: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Get manager goal for current encoded state"""
        with torch.no_grad():
            z_input = z_t.unsqueeze(0).to(self.device)  # Add batch dimension
            goals = self.manager_policy(z_input)
            return goals.squeeze(0)
    
    def get_worker_action_and_value(self, z_t, pooled_goals, action_mask=None):
        """Sample action and get value estimate from worker"""
        action_logits = self.worker_policy(z_t, pooled_goals)
        
        # Apply action mask if provided
        if action_mask is not None:
            masked_logits = action_logits.clone()
            masked_logits[~action_mask] = float('-inf')
        else:
            masked_logits = action_logits
        
        dist = torch.distributions.Categorical(logits=masked_logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        # Get value estimate
        value = self.worker_value(z_t)
        
        return action, log_prob, value.squeeze(-1)
    
    def take_action(self, obs: torch.Tensor, action_mask: torch.Tensor, 
                   pooled_goals: torch.Tensor) -> Tuple[int, float, float]:
        """Take action using hierarchical policy"""
        with torch.no_grad():
            obs = obs.unsqueeze(0).to(self.device)
            action_mask = action_mask.to(self.device)
            pooled_goals = pooled_goals.unsqueeze(0).to(self.device)
            
            # Encode state
            z_t = self.manager_encoder(obs)
            
            # Worker action
            action, log_prob, value = self.get_worker_action_and_value(
                z_t, pooled_goals, action_mask.unsqueeze(0)
            )
            
            return int(action.item()), float(log_prob.item()), float(value.item())
    
    def get_deterministic_action(self, obs: torch.Tensor, action_mask: torch.Tensor,
                                pooled_goals: torch.Tensor) -> int:
        """Get deterministic action for evaluation"""
        with torch.no_grad():
            obs = obs.unsqueeze(0).to(self.device)
            action_mask = action_mask.to(self.device)
            pooled_goals = pooled_goals.unsqueeze(0).to(self.device)
            
            # Encode state
            z_t = self.manager_encoder(obs)
            
            # Worker deterministic action
            action_logits = self.worker_policy(z_t, pooled_goals)
            
            if action_mask is not None:
                # Set invalid actions to -inf before argmax
                masked_logits = action_logits.clone()
                masked_logits[~action_mask.unsqueeze(0)] = float('-inf')
                action = torch.argmax(masked_logits, dim=-1)
            else:
                action = torch.argmax(action_logits, dim=-1)

            return int(action.item()) if hasattr(action, 'item') else int(action)
    
    def encode_state(self, obs: torch.Tensor) -> torch.Tensor:
        """Encode observation to latent space"""
        with torch.no_grad():
            obs = obs.unsqueeze(0).to(self.device)
            return self.manager_encoder(obs).squeeze(0)
    
    def compute_intrinsic_reward(self, encoded_states: List[torch.Tensor], 
                                goals: List[torch.Tensor], step: int, c: int) -> float:
        """
        Compute intrinsic reward based on goal achievement.
        For hierarchical RL: r_int = avg(cos(s_t - s_{t-c}, g_{t-c}))
        where g_{t-c} is the goal that was active during the period [t-c, t]
        """
        # FIX: Better bounds checking
        if step < c or len(encoded_states) <= step or len(goals) == 0:
            return 0.0
        
        # FIX: Ensure we don't access negative indices
        past_step_idx = max(0, step - c)
        if past_step_idx >= len(encoded_states):
            return 0.0
        
        current_state = encoded_states[step]
        past_state = encoded_states[past_step_idx]
        
        # Find the goal that was active during this period
        goal_idx = past_step_idx // c
        if goal_idx >= len(goals):
            return 0.0
        
        goal_vector = goals[goal_idx]
        state_diff = current_state - past_state
        
        # Avoid zero norm states
        if torch.norm(state_diff) < 1e-8 or torch.norm(goal_vector) < 1e-8:
            return 0.0
        
        # Compute cosine similarity between state difference and goal
        cosine_sim = F.cosine_similarity(state_diff.unsqueeze(0), goal_vector.unsqueeze(0))
        
        return cosine_sim.item()
    
    def pool_goals(self, goals: List[torch.Tensor], step: int, c: int) -> torch.Tensor:
        """
        Pool goals using linear interpolation for smoother transitions.
        After the first c steps, the pooled goal is a mix of the last two goals.
        """
        # FIX: Better handling of empty goals
        if not goals:
            return torch.zeros(self.latent_dim, device=self.device)

        if len(goals) == 1:
            # Only one goal available, use it
            return goals[-1]
        
        if step < c:
            # Before first goal transition, use current goal
            return goals[-1]
        
        # FIX: Ensure we have at least 2 goals for interpolation
        if len(goals) < 2:
            return goals[-1]
        
        # After the first c steps, interpolate between the last two goals
        last_goal = goals[-2]
        current_goal = goals[-1]
        
        # Interpolation weight alpha
        alpha = (step % c) / c
        
        # Pooled goal: g_t = (1 - alpha) * g_{t-1} + alpha * g_t
        pooled_goal = (1 - alpha) * last_goal + alpha * current_goal
        
        return pooled_goal
    
    def compute_manager_reward(self, encoded_states_over_period: List[torch.Tensor], 
                                           goal: torch.Tensor) -> float:
        """
        Compute manager reward based on cosine similarity between state change and goal.
        """
        if len(encoded_states_over_period) < 2:
            return 0.0
        
        # Get state transition and goal
        s_t = encoded_states_over_period[0]
        s_t_plus_c = encoded_states_over_period[-1]
        g_t = goal
        
        # State transition vector
        state_diff = s_t_plus_c - s_t
        
        # FIX: Check for zero norm vectors
        if torch.norm(state_diff) < 1e-8 or torch.norm(g_t) < 1e-8:
            return 0.0
        
        # Cosine similarity is used as the reward signal
        with torch.no_grad():
            cosine_sim = F.cosine_similarity(state_diff.unsqueeze(0), g_t.unsqueeze(0))
            return cosine_sim.item()
    
    def update_manager(self, buffer: HierarchicalPPOBuffer) -> Dict:
        """Update manager using transition-policy gradient with proper FuN objective."""
        if buffer.manager_size == 0:
            return {'policy_loss': 0.0, 'value_loss': 0.0, 
                   'avg_goal_norm': 0.0, 'avg_cosine_alignment': 0.0}
        
        # Get batch from buffer
        batch = buffer.get_manager_batch()
        states = batch['states']
        goals = batch['goals']
        values = batch['values']
        rewards = batch['rewards']
        dones = batch['dones']
        
        # Compute advantages for manager using PPOUpdater
        advantages, returns = PPOUpdater.compute_gae_advantages(
            rewards, values, dones, self.gamma_manager, self.gae_lambda
        )
        
        # Forward pass through manager
        pred_goals = self.manager_policy(states)
        pred_values = self.manager_value(states).squeeze(-1)
        
        # FuN-style policy gradient loss: E[A^M_t · ∇_θ cos(st+c − st, gt)]
        cosine_alignment = F.cosine_similarity(pred_goals, goals, dim=1)
        policy_loss = -torch.mean(advantages.detach() * cosine_alignment)
        
        # Value loss
        value_loss = F.mse_loss(pred_values, returns)
        
        # --- Update manager policy ---
        self.manager_policy_optimizer.zero_grad()
        policy_loss.backward()
        self.manager_policy_optimizer.step()
        
        # --- Update manager value ---
        self.manager_value_optimizer.zero_grad()
        value_loss.backward()
        self.manager_value_optimizer.step()
        
        # Statistics
        with torch.no_grad():
            avg_goal_norm = torch.mean(torch.norm(pred_goals, dim=1)).item()
            avg_cosine_alignment = torch.mean(cosine_alignment).item()
        
        stats = {
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
    
    def update_worker(self, buffer: PPOBuffer, pooled_goals: torch.Tensor,
                     train_pi_iters: int = 10, train_v_iters: int = 10) -> Dict:
        """
        Update worker using PPO with mixed rewards (extrinsic + intrinsic).
        """
        worker_data = buffer.get_batch()
        
        if worker_data['observations'].shape[0] == 0:
            return {'policy_loss': 0.0, 'value_loss': 0.0, 'entropy': 0.0}
        
        # Use the actual rewards from buffer (already contain extrinsic + intrinsic mix)
        worker_rewards = worker_data['rewards']
        batch_size = worker_data['observations'].shape[0]
        
        # Standard PPO update for worker using PPOUpdater
        accumulated_policy_loss = 0.0
        accumulated_value_loss = 0.0
        accumulated_entropy = 0.0
        
        num_updates = train_pi_iters

        for _ in range(num_updates):
            # Fresh forward pass for each update iteration
            batch_encoded_states = self.manager_encoder(worker_data['observations'])
            
            # Use pooled goals for worker 
            batch_pooled_goals = pooled_goals[:batch_size]
            
            # Get worker predictions
            action_logits = self.worker_policy(batch_encoded_states, batch_pooled_goals)
            
            # Worker value estimation
            values = self.worker_value(batch_encoded_states).squeeze(-1)
            
            # Apply action mask
            masked_logits = action_logits.clone()
            masked_logits[~worker_data['action_masks']] = float('-inf')
            
            # Create distribution and get log probabilities
            dist = torch.distributions.Categorical(logits=masked_logits)
            new_log_probs = dist.log_prob(worker_data['actions'])
            entropy = dist.entropy().mean()

            # Compute advantages using GAE from PPOUpdater
            advantages, returns = PPOUpdater.compute_gae_advantages(
                worker_rewards, worker_data['values'], worker_data['dones'], 
                self.gamma_worker, self.gae_lambda
            )
            
            # PPO policy loss using PPOUpdater
            policy_loss = PPOUpdater.ppo_policy_loss(
                new_log_probs, worker_data['log_probs'], advantages.detach(), self.clip_ratio
            )
            
            # Total policy loss
            total_policy_loss = policy_loss - self.entropy_coef * entropy
            
            # Value loss
            value_loss = F.mse_loss(values, returns.detach())
            
            # Combined loss for all worker components including encoder
            total_loss = total_policy_loss + value_loss
            
            # --- Update all worker components together (including encoder) ---
            self.worker_policy_optimizer.zero_grad()
            self.worker_value_optimizer.zero_grad()
            self.encoder_optimizer.zero_grad()
            
            total_loss.backward()
            
            self.worker_policy_optimizer.step()
            self.worker_value_optimizer.step()
            self.encoder_optimizer.step()
            
            accumulated_policy_loss += policy_loss.item()
            accumulated_value_loss += value_loss.item()
            accumulated_entropy += entropy.item()

        stats = {
            'policy_loss': accumulated_policy_loss / num_updates,
            'value_loss': accumulated_value_loss / num_updates,
            'entropy': accumulated_entropy / num_updates
        }
        
        # Update training stats
        for key, value in stats.items():
            if f'worker_{key}' in self.training_stats:
                self.training_stats[f'worker_{key}'].append(value)
        
        return stats
    
    def clear_manager_data(self, buffer: HierarchicalPPOBuffer):
        """Clear manager data storage in the provided buffer"""
        buffer.clear()
    
    def save(self, path: str):
        """Save all models and training stats"""
        torch.save({
            'manager_encoder_state_dict': self.manager_encoder.state_dict(),
            'manager_policy_state_dict': self.manager_policy.state_dict(),
            'manager_value_state_dict': self.manager_value.state_dict(),
            'worker_policy_state_dict': self.worker_policy.state_dict(),
            'worker_value_state_dict': self.worker_value.state_dict(),
            'encoder_optimizer_state_dict': self.encoder_optimizer.state_dict(),
            'manager_policy_optimizer_state_dict': self.manager_policy_optimizer.state_dict(),
            'manager_value_optimizer_state_dict': self.manager_value_optimizer.state_dict(),
            'worker_policy_optimizer_state_dict': self.worker_policy_optimizer.state_dict(),
            'worker_value_optimizer_state_dict': self.worker_value_optimizer.state_dict(),
            'training_stats': self.training_stats,
            # Save configuration parameters
            'input_dim': self.input_dim,
            'action_dim': self.action_dim,
            'latent_dim': self.latent_dim,
            'goal_dim': self.goal_dim,
            'goal_duration': self.goal_duration,
            'manager_lr': self.manager_lr,
            'worker_lr': self.worker_lr,
            'gamma_manager': self.gamma_manager,
            'gamma_worker': self.gamma_worker,
            'gae_lambda': self.gae_lambda,
            'clip_ratio': self.clip_ratio,
            'entropy_coef': self.entropy_coef,
            'device': str(self.device)
        }, path)
    
    def load(self, path: str):
        """Load all models and training stats"""
        checkpoint = torch.load(path, map_location=self.device)
        try:
            self.manager_encoder.load_state_dict(checkpoint['manager_encoder_state_dict'])
            self.manager_policy.load_state_dict(checkpoint['manager_policy_state_dict'])
            self.manager_value.load_state_dict(checkpoint['manager_value_state_dict'])
            self.worker_policy.load_state_dict(checkpoint['worker_policy_state_dict'])
            self.worker_value.load_state_dict(checkpoint['worker_value_state_dict'])
        except RuntimeError as e:
            print("\n[ERROR] Model architecture mismatch while loading state_dict!")
            print("This usually means your current network architecture does not match the one used for training.")
            print("Details:", e)
            raise
        
        # Load optimizers (with backward compatibility)
        if 'encoder_optimizer_state_dict' in checkpoint:
            self.encoder_optimizer.load_state_dict(checkpoint['encoder_optimizer_state_dict'])
        self.manager_policy_optimizer.load_state_dict(checkpoint['manager_policy_optimizer_state_dict'])
        self.manager_value_optimizer.load_state_dict(checkpoint['manager_value_optimizer_state_dict'])
        self.worker_policy_optimizer.load_state_dict(checkpoint['worker_policy_optimizer_state_dict'])
        self.worker_value_optimizer.load_state_dict(checkpoint['worker_value_optimizer_state_dict'])
        self.training_stats = checkpoint['training_stats']
        
        # Load goal_duration if available (for backward compatibility)
        if 'goal_duration' in checkpoint:
            self.goal_duration = checkpoint['goal_duration']
    
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
                'goal_duration': checkpoint.get('goal_duration', 10),  # Default to 10 if not found
                'manager_lr': checkpoint['manager_lr'],
                'worker_lr': checkpoint['worker_lr'],
                'gamma_manager': checkpoint['gamma_manager'],
                'gamma_worker': checkpoint['gamma_worker'],
                'gae_lambda': checkpoint['gae_lambda'],
                'clip_ratio': checkpoint['clip_ratio'],
                'entropy_coef': checkpoint['entropy_coef'],
                'device': checkpoint['device']
            }
        else:
            raise ValueError("This checkpoint does not contain configuration parameters.")

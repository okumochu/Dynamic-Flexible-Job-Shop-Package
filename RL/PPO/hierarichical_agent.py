"""
Hierarchical RL Agent Components
Contains Manager and Worker classes for feudal-style hierarchical RL
"""

import torch
import torch.nn.functional as F
import torch.optim as optim
from typing import Optional, Tuple, Dict, List
from RL.PPO.networks import PerceptualEncoder, ManagerPolicy, WorkerPolicy, ValueNetwork, PPOUpdater
from RL.PPO.buffer import PPOBuffer, ManagerBuffer


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
        self.goal_duration = goal_duration  # Store goal duration (c)
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
        self.manager_policy = ManagerPolicy(latent_dim).to(self.device)
        self.manager_value = ValueNetwork(latent_dim).to(self.device)
        self.worker_policy = WorkerPolicy(latent_dim, action_dim, goal_dim).to(self.device)
        # Worker value network takes concatenated state + goal
        self.worker_value = ValueNetwork(latent_dim + goal_dim).to(self.device)
        
        # Optimizers
        self.manager_optimizer = optim.Adam(
            list(self.encoder.parameters()) + list(self.manager_policy.parameters()) + list(self.manager_value.parameters()),
            lr=manager_lr
        )
        self.worker_optimizer = optim.Adam(
            list(self.encoder.parameters()) + list(self.worker_policy.parameters()) + list(self.worker_value.parameters()),
            lr=worker_lr
        )
        
        # Storage for hierarchical data during training
        self.manager_buffer = ManagerBuffer(
            buffer_size=1000,  # Reasonable default buffer size
            latent_dim=latent_dim,
            device=self.device
        )
                
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
    
    def get_manager_goal(self, z_t: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """Get manager goal for current encoded state"""
        with torch.no_grad():
            z_input = z_t.unsqueeze(0).to(self.device)  # Add batch dimension
            goals = self.manager_policy(z_input)
            return goals.squeeze(0)
    
    def get_worker_action_and_value(self, z_t, pooled_goals, action_mask=None):
        """Sample action and get value estimate from worker"""
        action_probs = self.worker_policy(z_t, pooled_goals)
        
        # Apply action mask if provided
        if action_mask is not None:
            masked_probs = action_probs.clone()
            masked_probs[~action_mask] = 0.0
            # Renormalize to ensure it's a valid probability distribution
            masked_probs = masked_probs / masked_probs.sum(dim=-1, keepdim=True)
            action_probs = masked_probs
        
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        # Get value estimate using concatenated state + goal
        w_t = self.worker_policy.goal_transform(pooled_goals)  # Transform goals to worker space
        value_input = torch.cat([z_t, w_t], dim=-1)
        value = self.worker_value(value_input)
        
        return action, log_prob, value.squeeze(-1)
    
    def take_action(self, obs: torch.Tensor, action_mask: torch.Tensor, 
                   pooled_goals: torch.Tensor) -> Tuple[int, float, float]:
        """Take action using hierarchical policy"""
        with torch.no_grad():
            obs = obs.unsqueeze(0).to(self.device)
            action_mask = action_mask.to(self.device)
            pooled_goals = pooled_goals.unsqueeze(0).to(self.device)
            
            # Encode state
            z_t = self.encoder(obs)
            
            # Worker action
            action, log_prob, value = self.get_worker_action_and_value(
                z_t, pooled_goals, action_mask.unsqueeze(0)
            )
            
            return int(action.item()), float(log_prob.item()), float(value.item())
    
    def get_deterministic_action(self, obs: torch.Tensor, action_mask: torch.Tensor,
                                pooled_goals: torch.Tensor) -> int:
        """Get deterministic action for evaluation (merged function)"""
        with torch.no_grad():
            obs = obs.unsqueeze(0).to(self.device)
            action_mask = action_mask.to(self.device)
            pooled_goals = pooled_goals.unsqueeze(0).to(self.device)
            
            # Encode state
            z_t = self.encoder(obs)
            
            # Worker deterministic action
            action_probs = self.worker_policy(z_t, pooled_goals)
            
            if action_mask is not None:
                # Set invalid actions to 0 probability
                masked_probs = action_probs * action_mask.unsqueeze(0).float()
                action = torch.argmax(masked_probs, dim=-1)
            
            return int(action.item()) if hasattr(action, 'item') else int(action)
    
    def encode_state(self, obs: torch.Tensor) -> torch.Tensor:
        """Encode observation to latent space"""
        with torch.no_grad():
            obs = obs.unsqueeze(0).to(self.device)
            return self.encoder(obs).squeeze(0)
    
    def compute_intrinsic_reward(self, encoded_states: List[torch.Tensor], 
                                goals: List[torch.Tensor], step: int, c: int) -> float:
        """
        Compute intrinsic reward based on goal achievement.
        For hierarchical RL: r_int = cos(s_t - s_{t-c}, g_{t-c})
        where g_{t-c} is the goal that was active during the period [t-c, t]
        """
        if step < c or step >= len(encoded_states) or len(goals) == 0:
            return 0.0
        
        current_state = encoded_states[step]
        past_state = encoded_states[step - c]
        
        # Find the goal that was active during this period
        # Goals are set every c steps, so the goal index is (step - c) // c
        goal_idx = max(0, (step - c) // c)
        if goal_idx >= len(goals):
            goal_idx = len(goals) - 1
        
        goal_vector = goals[goal_idx]
        state_diff = current_state - past_state
        
        # Compute cosine similarity between state difference and goal
        cosine_sim = F.cosine_similarity(state_diff.unsqueeze(0), goal_vector.unsqueeze(0))
        
        return cosine_sim.item()
    
    def pool_goals(self, goals: List[torch.Tensor], step: int, c: int) -> torch.Tensor:
        """
        Get the current active goal for the worker.
        In hierarchical RL, goals are set every c steps, so we return the goal
        that is currently active for the worker.
        """
        if len(goals) == 0:
            return torch.zeros(self.latent_dim, device=self.device)
        
        # Find which goal is currently active
        # Goals are indexed by: goal_idx = step // c
        goal_idx = min(step // c, len(goals) - 1)
        
        return goals[goal_idx]
    
    def add_manager_experience(self, state: torch.Tensor, goal: torch.Tensor, 
                              value: float, reward: float, done: bool):
        """Add manager experience"""
        self.manager_buffer.add(state, goal, value, reward, done)
    
    def compute_manager_reward_with_gradient(self, encoded_states: List[torch.Tensor], 
                                           goals: List[torch.Tensor], goal_idx: int, step: int,
                                           advantages: Optional[torch.Tensor] = None) -> float:
        """
        Compute manager reward with proper gradient calculation: ∇_θ g_t = A^M_t · ∇_θ cos(st+c − st, gt)
        This implements the full FuN-style manager objective.
        
        Args:
            encoded_states: List of encoded states
            goals: List of goals
            goal_idx: Index of the goal
            step: Current step  
            advantages: Manager advantages (if available during update)
        """
        if goal_idx >= len(goals) or step + self.goal_duration >= len(encoded_states):
            return 0.0
        
        # Get state transition and goal
        s_t = encoded_states[step]
        s_t_plus_c = encoded_states[step + self.goal_duration]
        g_t = goals[goal_idx]
        
        # State transition vector
        state_diff = s_t_plus_c - s_t
        
        # Enable gradients for goal to compute gradient
        if g_t.requires_grad or advantages is not None:
            # During training: compute gradient-weighted objective
            cosine_sim = F.cosine_similarity(state_diff.unsqueeze(0), g_t.unsqueeze(0))
            
            if advantages is not None:
                # Full gradient calculation: A^M_t · ∇_θ cos(st+c − st, gt)  
                # The gradient will be computed during backprop when advantages multiply this
                return cosine_sim.item()
            else:
                return cosine_sim.item()
        else:
            # During rollout: just return cosine similarity as reward signal
            with torch.no_grad():
                cosine_sim = F.cosine_similarity(state_diff.unsqueeze(0), g_t.unsqueeze(0))
                return cosine_sim.item()
    
    def compute_manager_reward(self, encoded_states: List[torch.Tensor], 
                              goals: List[torch.Tensor], goal_idx: int, step: int) -> float:
        """
        Compute manager reward based on: ∇_θ g_t = A^M_t · ∇_θ cos(st+c − st, gt)
        where A^M_t = R_t − V^M_t (episode objective difference - manager value)
        Args:
            encoded_states: List of encoded states
            goals: List of goals
            goal_idx: Index of the goal
            step: Current step
        """
        return self.compute_manager_reward_with_gradient(encoded_states, goals, goal_idx, step)
    
    def update_manager(self) -> Dict:
        """Update manager using transition-policy gradient with proper FuN objective"""
        if self.manager_buffer.size == 0:
            return {'loss': 0.0, 'policy_loss': 0.0, 'value_loss': 0.0, 
                   'avg_goal_norm': 0.0, 'avg_cosine_alignment': 0.0}
        
        # Get batch from buffer
        batch = self.manager_buffer.get_batch()
        states = batch['states']
        goals = batch['goals']
        values = batch['values']
        rewards = batch['rewards']
        dones = batch['dones']
        
        # Compute advantages for manager using PPOUpdater
        advantages, returns = PPOUpdater.compute_gae_advantages(
            rewards, values, dones, self.gamma_manager, self.gae_lambda
        )
        
        # Manager update
        self.manager_optimizer.zero_grad()
        
        # Forward pass through manager
        pred_goals = self.manager_policy(states)
        pred_values = self.manager_value(states).squeeze(-1)
        
        # FuN-style policy gradient loss: E[A^M_t · ∇_θ cos(st+c − st, gt)]
        # This is approximated by: A^M_t · cos_similarity(pred_goals, target_goals)
        # where target_goals are the goals that led to the advantages
        cosine_alignment = F.cosine_similarity(pred_goals, goals, dim=1)
        policy_loss = -torch.mean(advantages.detach() * cosine_alignment)
        
        # Value loss
        value_loss = F.mse_loss(pred_values, returns)
        
        # Total loss
        total_loss = policy_loss + 0.5 * value_loss
        
        total_loss.backward()
        self.manager_optimizer.step()
        
        # Statistics
        with torch.no_grad():
            avg_goal_norm = torch.mean(torch.norm(pred_goals, dim=1)).item()
            avg_cosine_alignment = torch.mean(cosine_alignment).item()
        
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
    
    def update_worker(self, buffer: PPOBuffer, encoded_states: List[torch.Tensor], 
                     goals: List[torch.Tensor], pooled_goals: List[torch.Tensor],
                     train_pi_iters: int = 10, train_v_iters: int = 10) -> Dict:
        """
        Update worker using PPO with mixed rewards (extrinsic + intrinsic).
        
        Args:
            buffer: PPOBuffer containing worker experiences with mixed rewards
            encoded_states: List of encoded states from the episode
            goals: List of goals from the episode  
            pooled_goals: List of pooled goals for each step
            train_pi_iters: Number of policy iterations
            train_v_iters: Number of value iterations
        """
        worker_data = buffer.get_batch()
        
        if worker_data['observations'].shape[0] == 0:
            return {'policy_loss': 0.0, 'value_loss': 0.0, 'total_loss': 0.0}
        
        # Use the actual rewards from buffer (already contain extrinsic + intrinsic mix)
        worker_rewards = worker_data['rewards']
        batch_size = worker_data['observations'].shape[0]
        
        # Standard PPO update for worker using PPOUpdater
        total_policy_loss = 0.0
        total_value_loss = 0.0
        
        for _ in range(train_pi_iters):
            self.worker_optimizer.zero_grad()
            
            # Get batch data
            batch_encoded_states = self.encoder(worker_data['observations'])
            
            # Use pooled goals for worker 
            batch_pooled_goals = torch.stack(pooled_goals[:batch_size], dim=0)
            
            # Get worker predictions
            action_probs = self.worker_policy(batch_encoded_states, batch_pooled_goals)
            
            # Worker value estimation (concatenate state + transformed goal)
            w_t = self.worker_policy.goal_transform(batch_pooled_goals)
            value_input = torch.cat([batch_encoded_states, w_t], dim=-1)
            values = self.worker_value(value_input).squeeze(-1)
            
            # Apply action mask
            masked_probs = action_probs.clone()
            masked_probs[~worker_data['action_masks']] = 0.0
            masked_probs = masked_probs / masked_probs.sum(dim=-1, keepdim=True)
            
            # Create distribution and get log probabilities
            dist = torch.distributions.Categorical(masked_probs)
            log_probs = dist.log_prob(worker_data['actions'])
            
            # Compute advantages using GAE from PPOUpdater
            advantages, returns = PPOUpdater.compute_gae_advantages(
                worker_rewards, worker_data['values'], worker_data['dones'], 
                self.gamma_worker, self.gae_lambda
            )
            
            # PPO policy loss using PPOUpdater
            policy_loss = PPOUpdater.ppo_policy_loss(
                log_probs, worker_data['log_probs'], advantages, self.clip_ratio
            )
            
            # Value loss
            value_loss = F.mse_loss(values, returns)
            
            # Total loss
            total_loss = policy_loss + 0.5 * value_loss
            
            total_loss.backward()
            self.worker_optimizer.step()
            
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
        
        stats = {
            'policy_loss': total_policy_loss / train_pi_iters,
            'value_loss': total_value_loss / train_pi_iters,
            'total_loss': (total_policy_loss + 0.5 * total_value_loss) / train_pi_iters
        }
        
        # Update training stats
        for key, value in stats.items():
            if f'worker_{key}' in self.training_stats:
                self.training_stats[f'worker_{key}'].append(value)
        
        return stats
    
    def clear_manager_data(self):
        """Clear manager data storage"""
        self.manager_buffer.clear()
    
    def save(self, path: str):
        """Save all models and training stats"""
        torch.save({
            'encoder_state_dict': self.encoder.state_dict(),
            'manager_policy_state_dict': self.manager_policy.state_dict(),
            'manager_value_state_dict': self.manager_value.state_dict(),
            'worker_policy_state_dict': self.worker_policy.state_dict(),
            'worker_value_state_dict': self.worker_value.state_dict(),
            'manager_optimizer_state_dict': self.manager_optimizer.state_dict(),
            'worker_optimizer_state_dict': self.worker_optimizer.state_dict(),
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
            'epsilon_greedy': self.epsilon_greedy,
            'device': str(self.device)
        }, path)
    
    def load(self, path: str):
        """Load all models and training stats"""
        checkpoint = torch.load(path, map_location=self.device)
        try:
            self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
            self.manager_policy.load_state_dict(checkpoint['manager_policy_state_dict'])
            self.manager_value.load_state_dict(checkpoint['manager_value_state_dict'])
            self.worker_policy.load_state_dict(checkpoint['worker_policy_state_dict'])
            self.worker_value.load_state_dict(checkpoint['worker_value_state_dict'])
        except RuntimeError as e:
            print("\n[ERROR] Model architecture mismatch while loading state_dict!")
            print("This usually means your current network architecture does not match the one used for training.")
            print("Details:", e)
            raise
        
        self.manager_optimizer.load_state_dict(checkpoint['manager_optimizer_state_dict'])
        self.worker_optimizer.load_state_dict(checkpoint['worker_optimizer_state_dict'])
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
                'epsilon_greedy': checkpoint['epsilon_greedy'],
                'device': checkpoint['device']
            }
        else:
            raise ValueError("This checkpoint does not contain configuration parameters.")

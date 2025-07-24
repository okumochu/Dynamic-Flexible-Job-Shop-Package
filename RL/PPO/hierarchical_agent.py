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
                 latent_dim: int ,
                 goal_dim: int ,
                 goal_duration: int ,  # c parameter - goal duration
                 manager_lr: float ,
                 worker_lr: float ,
                 gamma_manager: float ,
                 gamma_worker: float ,
                 gae_lambda: float ,
                 clip_ratio: float ,
                 entropy_coef: float ,
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
        
        # FIX: Use separate optimizers - encoder updated ONLY by worker
        # Encoder optimizer (used only by worker)
        self.encoder_optimizer = optim.Adam(self.manager_encoder.parameters(), lr=worker_lr)
        
        # Manager optimizers (without encoder parameters)
        self.manager_policy_optimizer = optim.Adam(self.manager_policy.parameters(), lr=manager_lr)
        self.manager_value_optimizer = optim.Adam(self.manager_value.parameters(), lr=manager_lr)
        
        # Worker optimizers (without encoder parameters)
        self.worker_policy_optimizer = optim.Adam(self.worker_policy.parameters(), lr=worker_lr)
        self.worker_value_optimizer = optim.Adam(self.worker_value.parameters(), lr=worker_lr)
    
    def get_manager_goal(self, z_t: torch.Tensor) -> torch.Tensor:
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
    
    def get_deterministic_action(self, obs: torch.Tensor, action_mask: Optional[torch.Tensor],
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
    
    def compute_intrinsic_reward(self, encoded_state_history: List[torch.Tensor], 
                                pooled_goal_history: List[torch.Tensor], step: int, c: int) -> float:
        """
        Compute FuN-style intrinsic reward: average cosine similarity between 
        encoded state differences and pooled goals over the last c steps.
        
        Based on FeUdal Networks: r_int = (1/c) * Σ cos(s_t - s_{t-i}, g_pooled)
        where the sum is over i from 1 to c.
        
        Args:
            encoded_states: List of encoded states
            goals: List of manager goals
            step: Current timestep
            c: Number of steps to look back (goal_duration)
            
        Returns:
            Intrinsic reward as float
        """
        
        # Calculate cosine similarity for each of the last at most c steps
        cosine_sims = []
        last_state = None
        for state, pooled_goal in zip(encoded_state_history, pooled_goal_history):
            
            # skip the first step of the episode
            if last_state is None:
                last_state = state
                continue
                
            state_diff = state - last_state
            cosine_sim = F.cosine_similarity(state_diff.unsqueeze(0), pooled_goal.unsqueeze(0)).item()
            cosine_sims.append(cosine_sim)

        # Return average cosine similarity over the last c steps
        return sum(cosine_sims) / len(cosine_sims) if cosine_sims else 0.0
    
    def pool_goals(self, last_goal: Optional[torch.Tensor], current_goal: torch.Tensor, step: int, c: int) -> torch.Tensor:
        """
        Pool goals using linear interpolation for smoother transitions in latent space.
        Each goal is active for exactly c steps.
        
        Args:
            last_goal: Last goal tensor in latent space (latent_dim)
            current_goal: Current goal tensor in latent space (latent_dim)
            step: Current timestep
            c: Goal duration
            
        Returns:
            pooled_goal: Interpolated goal in latent space (latent_dim)
                        Will be transformed via bias-free linear layer in WorkerPolicy
        """
        
        # If it is first c step, then use the current goal without pooling
        if last_goal is None:
            return current_goal
        else:          
            # Position within the current goal period (0 to c-1)
            step_in_c_steps = step % c
            
            # Interpolation weight: starts at 0 (use prev_goal) and goes to 1 (use current_goal)
            alpha = step_in_c_steps / c
            
            # Smoothly transition from previous goal to current goal
            pooled_goal = (1 - alpha) * last_goal + alpha * current_goal
            
            return pooled_goal

    
    def compute_manager_reward(self, s_t: Optional[torch.Tensor], s_t_plus_c: torch.Tensor, g_t: Optional[torch.Tensor]) -> float:
        """
        Compute manager reward based on cosine similarity between state change and goal.
        """
        # If s_t is None, means it is the first step, so reward is 0
        if s_t is None:
            return 0.0
        else:
            with torch.no_grad():
                # State transition vector
                state_diff = s_t_plus_c - s_t
                
                # Cosine similarity is used as the reward signal
                cosine_sim = F.cosine_similarity(state_diff.unsqueeze(0), g_t.unsqueeze(0))
                return cosine_sim.item()
    
    def update_manager(self, buffer: HierarchicalPPOBuffer) -> Dict:
        """Update manager using transition-policy gradient with proper FuN objective."""
        batch = buffer.get_manager_batch()
        states = batch['states']
        goals = batch['goals']
        values = batch['values']
        rewards = batch['rewards']
        dones = batch['dones']
        
        # Compute GAE advantages and returns
        advantages, returns = PPOUpdater.compute_gae_advantages(
            rewards, values, dones, self.gamma_manager, self.gae_lambda
        )
        pred_goals = self.manager_policy(states)
        pred_values = self.manager_value(states).squeeze(-1)
        value_loss = F.mse_loss(pred_values, returns)
        
        # Compute cosine similarity between predicted goals and actual goals
        cosine_alignment = F.cosine_similarity(pred_goals, goals, dim=1)
        policy_loss = -torch.mean(advantages.detach() * cosine_alignment)
       
        # Update manager policy and value (encoder updated only with value loss)
        self.manager_policy_optimizer.zero_grad()
        self.manager_value_optimizer.zero_grad()
        self.encoder_optimizer.zero_grad()
        
        # Backward only value loss for encoder
        value_loss.backward(retain_graph=True)
        
        # Backward policy loss for policy only
        policy_loss.backward()
        
        self.manager_policy_optimizer.step()
        self.manager_value_optimizer.step()
        self.encoder_optimizer.step()
        
        with torch.no_grad():
            avg_goal_norm = torch.mean(torch.norm(pred_goals, dim=1)).item()
            avg_cosine_alignment = torch.mean(cosine_alignment).item()
            avg_manager_reward = torch.mean(rewards).item() if len(rewards) > 0 else 0.0
        
        stats = {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'avg_goal_norm': avg_goal_norm,
            'avg_cosine_alignment': avg_cosine_alignment,
            'avg_manager_reward': avg_manager_reward
        }

        return stats
    
    def update_worker(self, buffer: PPOBuffer, manager_buffer: HierarchicalPPOBuffer,
                     train_pi_iters: int = 10, train_v_iters: int = 10) -> Dict:
        worker_data = buffer.get_batch()
        worker_rewards = worker_data['rewards']
        batch_size = worker_data['observations'].shape[0]
        accumulated_policy_loss = 0.0
        accumulated_value_loss = 0.0
        accumulated_entropy = 0.0
        pooled_goals = manager_buffer.get_worker_pooled_goals()
        
        # Policy update
        for _ in range(train_pi_iters):
            batch_encoded_states = self.manager_encoder(worker_data['observations'])
            if pooled_goals.shape[0] < batch_size:
                last_goal = pooled_goals[-1:] if pooled_goals.shape[0] > 0 else torch.zeros(1, self.latent_dim, device=self.device)
                missing_goals = batch_size - pooled_goals.shape[0]
                padding_goals = last_goal.repeat(missing_goals, 1)
                batch_pooled_goals = torch.cat([pooled_goals, padding_goals], dim=0)
            else:
                batch_pooled_goals = pooled_goals[:batch_size]
            action_logits = self.worker_policy(batch_encoded_states, batch_pooled_goals)
            values = self.worker_value(batch_encoded_states).squeeze(-1)
            masked_logits = action_logits.clone()
            masked_logits[~worker_data['action_masks']] = float('-inf')
            dist = torch.distributions.Categorical(logits=masked_logits)
            new_log_probs = dist.log_prob(worker_data['actions'])
            entropy = dist.entropy().mean()
            advantages, returns = PPOUpdater.compute_gae_advantages(
                worker_rewards, worker_data['values'], worker_data['dones'], 
                self.gamma_worker, self.gae_lambda
            )
            policy_loss = PPOUpdater.ppo_policy_loss(
                new_log_probs, worker_data['log_probs'], advantages.detach(), self.clip_ratio
            )
            total_policy_loss = policy_loss - self.entropy_coef * entropy
            self.worker_policy_optimizer.zero_grad()
            self.encoder_optimizer.zero_grad()
            total_policy_loss.backward(retain_graph=True)
            self.worker_policy_optimizer.step()
            self.encoder_optimizer.step()
            accumulated_policy_loss += policy_loss.item()
            accumulated_entropy += entropy.item()
        # Value update
        for _ in range(train_v_iters):
            batch_encoded_states = self.manager_encoder(worker_data['observations'])
            if pooled_goals.shape[0] < batch_size:
                last_goal = pooled_goals[-1:] if pooled_goals.shape[0] > 0 else torch.zeros(1, self.latent_dim, device=self.device)
                missing_goals = batch_size - pooled_goals.shape[0]
                padding_goals = last_goal.repeat(missing_goals, 1)
                batch_pooled_goals = torch.cat([pooled_goals, padding_goals], dim=0)
            else:
                batch_pooled_goals = pooled_goals[:batch_size]
            values = self.worker_value(self.manager_encoder(worker_data['observations'])).squeeze(-1)
            advantages, returns = PPOUpdater.compute_gae_advantages(
                worker_rewards, worker_data['values'], worker_data['dones'], 
                self.gamma_worker, self.gae_lambda
            )
            value_loss = F.mse_loss(values, returns.detach())
            self.worker_value_optimizer.zero_grad()
            self.encoder_optimizer.zero_grad()
            value_loss.backward()
            self.worker_value_optimizer.step()
            self.encoder_optimizer.step()
            accumulated_value_loss += value_loss.item()
        stats = {
            'policy_loss': accumulated_policy_loss / train_pi_iters,
            'value_loss': accumulated_value_loss / train_v_iters,
            'entropy': accumulated_entropy / train_pi_iters
        }
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

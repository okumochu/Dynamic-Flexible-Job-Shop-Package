"""
Hierarchical RL Agent Components
Contains Manager and Worker classes for feudal-style hierarchical RL
"""

import torch
import torch.nn.functional as F
import torch.optim as optim
from typing import Optional, Tuple, Dict, List
from RL.PPO.networks import PerceptualEncoder, ManagerEncoder, ManagerPolicy, WorkerPolicy, ValueNetwork, PPOUpdater, HybridWorkerPolicy
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
                 device: str = 'auto',
                 target_kl: float = 0.01,
                 seed: Optional[int] = None,
                 max_grad_norm: float = 0.5):
        
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
        self.target_kl = target_kl
        self.max_grad_norm = max_grad_norm
        self.seed = seed
        
        # Create models
        self.state_encoder = PerceptualEncoder(input_dim, latent_dim).to(self.device)
        self.manager_encoder = ManagerEncoder(latent_dim).to(self.device)  # Manager-specific encoder
        self.manager_policy = ManagerPolicy(latent_dim).to(self.device)
        self.manager_value = ValueNetwork(latent_dim).to(self.device)

        self.worker_policy = WorkerPolicy(latent_dim, action_dim, goal_dim).to(self.device)
        self.worker_value = ValueNetwork(latent_dim).to(self.device)
        
        # IMPORTANT: Updated optimizer strategy
        # 1. PerceptualEncoder: Updated by worker (policy & value losses) and manager (value loss)
        # 2. ManagerEncoder: Updated by manager value loss only (no policy loss)
        self.perceptual_encoder_optimizer = optim.Adam(self.state_encoder.parameters(), lr=worker_lr)
        self.manager_encoder_optimizer = optim.Adam(self.manager_encoder.parameters(), lr=manager_lr)
        
        # Manager optimizers
        self.manager_policy_optimizer = optim.Adam(self.manager_policy.parameters(), lr=manager_lr)
        self.manager_value_optimizer = optim.Adam(self.manager_value.parameters(), lr=manager_lr)
        
        # Worker optimizers  
        self.worker_policy_optimizer = optim.Adam(self.worker_policy.parameters(), lr=worker_lr)
        self.worker_value_optimizer = optim.Adam(self.worker_value.parameters(), lr=worker_lr)
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
    
    def get_manager_goal(self, z_t: torch.Tensor) -> torch.Tensor:
        """Get manager goal for current encoded state"""
        with torch.no_grad():
            z_input = z_t.unsqueeze(0).to(self.device)  # Add batch dimension
            # Pass through manager encoder first
            manager_z = self.manager_encoder(z_input)
            goals = self.manager_policy(manager_z)
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
            z_t = self.state_encoder(obs)
            
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
            z_t = self.state_encoder(obs)
            
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
            return self.state_encoder(obs).squeeze(0)
    
    def compute_intrinsic_reward(self, encoded_state_history: List[torch.Tensor], 
                                pooled_goal_history: List[torch.Tensor]) -> float:
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
            with torch.no_grad():
                manager_state = self.manager_encoder(state)
            
            # skip the first step of the episode
            if last_state is None:
                last_state = manager_state
                continue
                
            state_diff = manager_state - last_state
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
                # Apply manager encoder to both states for consistency with manager policy
                manager_s_t = self.manager_encoder(s_t.unsqueeze(0)).squeeze(0)
                manager_s_t_plus_c = self.manager_encoder(s_t_plus_c.unsqueeze(0)).squeeze(0)
                
                # State transition vector in manager-encoded space
                state_diff = manager_s_t_plus_c - manager_s_t
                
                # Cosine similarity is used as the reward signal
                cosine_sim = F.cosine_similarity(state_diff.unsqueeze(0), g_t.unsqueeze(0))
                return cosine_sim.item()
    
    def update_manager(self, buffer: HierarchicalPPOBuffer) -> Dict:
        """Update manager using transition-policy gradient with proper FuN objective."""
        # Get manager batch data from buffer
        manager_data = buffer.get_manager_batch()
        values = manager_data['values']
        rewards = manager_data['rewards']
        dones = manager_data['dones']
        observations = manager_data['observations']
        
        # Compute GAE advantages and returns
        advantages, returns = PPOUpdater.compute_gae_advantages(
            rewards, values, dones, self.gamma_manager, self.gae_lambda
        )
        
        # IMPORTANT: Re-encode observations and pass through manager encoder
        # PerceptualEncoder output goes to ManagerEncoder for manager-specific processing
        perceptual_encoded = self.state_encoder(observations)
        manager_encoded_with_grads = self.manager_encoder(perceptual_encoded)
        
        # Recompute manager values with current network to get gradients (including manager encoder)
        current_values = self.manager_value(manager_encoded_with_grads).squeeze(-1)
        value_loss = F.mse_loss(current_values, returns)
        
        # Recompute manager goals with current policy to get gradients
        current_goals = self.manager_policy(manager_encoded_with_grads)
        
        # For transition policy loss, use detached re-encoded states
        # These represent actual state transitions (same as stored encoded_states but from current encoder)
        # Only manager policy should get gradients from this loss, not encoders
        encoded_states_detached = perceptual_encoded.detach()  # Detached = same functionality as buffer's encoded_states
        policy_loss = PPOUpdater.transition_policy_loss(encoded_states_detached, current_goals, advantages)

        # Update manager networks, manager encoder, AND perceptual encoder (from value loss)
        self.manager_policy_optimizer.zero_grad()
        self.manager_value_optimizer.zero_grad()
        self.manager_encoder_optimizer.zero_grad()
        self.perceptual_encoder_optimizer.zero_grad()  # Add PerceptualEncoder update
        
        # Backward pass: Both encoders get gradients from value loss, manager policy from transition loss
        value_loss.backward(retain_graph=True)
        # Clip value/encoder grads
        torch.nn.utils.clip_grad_norm_(self.manager_value.parameters(), getattr(self, 'max_grad_norm', 0.5))
        torch.nn.utils.clip_grad_norm_(self.manager_encoder.parameters(), getattr(self, 'max_grad_norm', 0.5))
        torch.nn.utils.clip_grad_norm_(self.state_encoder.parameters(), getattr(self, 'max_grad_norm', 0.5))

        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.manager_policy.parameters(), getattr(self, 'max_grad_norm', 0.5))
        
        self.manager_policy_optimizer.step()
        self.manager_value_optimizer.step()
        self.manager_encoder_optimizer.step()
        self.perceptual_encoder_optimizer.step()  # Update PerceptualEncoder too
        
        stats = {
            'manager_value_loss': value_loss.item(),
            'manager_policy_loss': policy_loss.item()
        }

        return stats
    
    def update_worker(self, worker_buffer: PPOBuffer, manager_buffer: HierarchicalPPOBuffer,
                     train_pi_iters: int, train_v_iters: int) -> Dict:
        
        # Get worker data
        worker_data = worker_buffer.get_batch()
        worker_rewards = worker_data['rewards']
        worker_observations = worker_data['observations']  # Raw observations for re-encoding
        
        # Get step information from manager buffer
        step_information = manager_buffer.get_step_information()
        pooled_goals = step_information['pooled_goals']
        
        # Compute GAE advantages once (outside the loops)
        advantages, returns = PPOUpdater.compute_gae_advantages(
            worker_rewards, worker_data['values'], worker_data['dones'], 
            self.gamma_worker, self.gae_lambda
        )
        
        # Initialize accumulators
        accumulated_policy_loss = 0.0
        accumulated_value_loss = 0.0
        accumulated_entropy = 0.0

        # Policy update loop
        for _ in range(train_pi_iters):
            # Re-encode observations for each policy update to get fresh gradients
            encoded_states = self.state_encoder(worker_observations)
            
            # Forward pass through worker policy
            action_logits = self.worker_policy(encoded_states, pooled_goals)
            masked_logits = action_logits.clone()
            masked_logits[~worker_data['action_masks']] = float('-inf')
            
            # Create distribution and compute new log probabilities
            dist = torch.distributions.Categorical(logits=masked_logits)
            new_log_probs = dist.log_prob(worker_data['actions'])
            
            # Compute entropy for regularization
            entropy = PPOUpdater.compute_entropy(action_logits, worker_data['action_masks'])
            
            # Compute PPO policy loss
            policy_loss = PPOUpdater.ppo_policy_loss(
                new_log_probs, worker_data['log_probs'], advantages.detach(), self.clip_ratio
            )
            
            # Total policy loss with entropy regularization
            total_policy_loss = policy_loss - self.entropy_coef * entropy
            
            # Update worker policy and perceptual encoder
            self.worker_policy_optimizer.zero_grad()
            self.perceptual_encoder_optimizer.zero_grad()
            total_policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.worker_policy.parameters(), getattr(self, 'max_grad_norm', 0.5))
            torch.nn.utils.clip_grad_norm_(self.state_encoder.parameters(), getattr(self, 'max_grad_norm', 0.5))
            self.worker_policy_optimizer.step()
            self.perceptual_encoder_optimizer.step()
            
            # KL early stop (approximate) for discrete policy
            with torch.no_grad():
                approx_kl = (worker_data['log_probs'] - new_log_probs).mean().abs().item()
            if approx_kl > self.target_kl:
                break

            accumulated_policy_loss += policy_loss.item()
            accumulated_entropy += entropy.item()

        # Value function update loop
        for _ in range(train_v_iters):
            # Re-encode observations for each value update to get fresh gradients
            encoded_states = self.state_encoder(worker_observations)
            
            # Forward pass through worker value function
            values = self.worker_value(encoded_states).squeeze(-1)
            value_loss = F.mse_loss(values, returns.detach())
            
            # Update worker value function and perceptual encoder
            self.worker_value_optimizer.zero_grad()
            self.perceptual_encoder_optimizer.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.worker_value.parameters(), getattr(self, 'max_grad_norm', 0.5))
            torch.nn.utils.clip_grad_norm_(self.state_encoder.parameters(), getattr(self, 'max_grad_norm', 0.5))
            self.worker_value_optimizer.step()
            self.perceptual_encoder_optimizer.step()
            
            accumulated_value_loss += value_loss.item()
            
        # Return training statistics
        stats = {
            'policy_loss': accumulated_policy_loss / train_pi_iters,
            'value_loss': accumulated_value_loss / train_v_iters,
            'entropy': accumulated_entropy / train_pi_iters
        }
        return stats
    
    def save(self, path: str):
        """Save all models and training stats"""
        torch.save({
            'perceptual_encoder_state_dict': self.state_encoder.state_dict(),
            'manager_encoder_state_dict': self.manager_encoder.state_dict(),
            'manager_policy_state_dict': self.manager_policy.state_dict(),
            'manager_value_state_dict': self.manager_value.state_dict(),
            'worker_policy_state_dict': self.worker_policy.state_dict(),
            'worker_value_state_dict': self.worker_value.state_dict(),
            'perceptual_encoder_optimizer_state_dict': self.perceptual_encoder_optimizer.state_dict(),
            'manager_encoder_optimizer_state_dict': self.manager_encoder_optimizer.state_dict(),
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
            # Load perceptual encoder (backward compatibility)
            if 'perceptual_encoder_state_dict' in checkpoint:
                self.state_encoder.load_state_dict(checkpoint['perceptual_encoder_state_dict'])
            elif 'manager_encoder_state_dict' in checkpoint:  # Old format
                self.state_encoder.load_state_dict(checkpoint['manager_encoder_state_dict'])
            
            # Load manager encoder
            if 'manager_encoder_state_dict' in checkpoint:
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
        if 'perceptual_encoder_optimizer_state_dict' in checkpoint:
            self.perceptual_encoder_optimizer.load_state_dict(checkpoint['perceptual_encoder_optimizer_state_dict'])
        elif 'encoder_optimizer_state_dict' in checkpoint:  # Old format
            self.perceptual_encoder_optimizer.load_state_dict(checkpoint['encoder_optimizer_state_dict'])
            
        if 'manager_encoder_optimizer_state_dict' in checkpoint:
            self.manager_encoder_optimizer.load_state_dict(checkpoint['manager_encoder_optimizer_state_dict'])
            
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


class HybridHierarchicalAgent:
    """
    Hierarchical RL Agent with Manager-Worker architecture for hybrid action spaces.
    Contains all update logic for both manager and worker, using hybrid policies.
    """
    
    def __init__(self,
                 input_dim: int,
                 discrete_action_dim: int,
                 continuous_action_dim: int,
                 latent_dim: int,
                 goal_dim: int,
                 goal_duration: int,  # c parameter - goal duration
                 manager_lr: float,
                 worker_lr: float,
                 gamma_manager: float,
                 gamma_worker: float,
                 gae_lambda: float,
                 clip_ratio: float,
                 entropy_coef: float,
                 device: str = 'auto',
                 max_grad_norm: float = 0.5,
                 target_kl: float = 0.01,
                 seed: int | None = None):
        
        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Hybrid Hierarchical Agent initialized on device: {self.device}")
        
        # Store configuration
        self.input_dim = input_dim
        self.discrete_action_dim = discrete_action_dim
        self.continuous_action_dim = continuous_action_dim
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
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl
        
        # Create models
        self.state_encoder = PerceptualEncoder(input_dim, latent_dim).to(self.device)
        self.manager_encoder = ManagerEncoder(latent_dim).to(self.device)  # Manager-specific encoder
        self.manager_policy = ManagerPolicy(latent_dim).to(self.device)
        self.manager_value = ValueNetwork(latent_dim).to(self.device)

        self.worker_policy = HybridWorkerPolicy(latent_dim, discrete_action_dim, continuous_action_dim, goal_dim).to(self.device)
        self.worker_value = ValueNetwork(latent_dim).to(self.device)
        
        # IMPORTANT: Updated optimizer strategy
        # 1. PerceptualEncoder: Updated by worker (policy & value losses) and manager (value loss)
        # 2. ManagerEncoder: Updated by manager value loss only (no policy loss)
        self.perceptual_encoder_optimizer = optim.Adam(self.state_encoder.parameters(), lr=worker_lr)
        self.manager_encoder_optimizer = optim.Adam(self.manager_encoder.parameters(), lr=manager_lr)
        
        # Manager optimizers
        self.manager_policy_optimizer = optim.Adam(self.manager_policy.parameters(), lr=manager_lr)
        self.manager_value_optimizer = optim.Adam(self.manager_value.parameters(), lr=manager_lr)
        
        # Worker optimizers  
        self.worker_policy_optimizer = optim.Adam(self.worker_policy.parameters(), lr=worker_lr)
        self.worker_value_optimizer = optim.Adam(self.worker_value.parameters(), lr=worker_lr)
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
    
    def get_manager_goal(self, z_t: torch.Tensor) -> torch.Tensor:
        """Get manager goal for current encoded state"""
        with torch.no_grad():
            z_input = z_t.unsqueeze(0).to(self.device)  # Add batch dimension
            # Pass through manager encoder first
            manager_z = self.manager_encoder(z_input)
            goals = self.manager_policy(manager_z)
            return goals.squeeze(0)
    
    def get_worker_action_and_value(self, z_t, pooled_goals, action_mask=None):
        """Sample action and get value estimate from worker, for hybrid action space."""
        discrete_logits, continuous_mean, continuous_log_std = self.worker_policy(z_t, pooled_goals)
        
        # Discrete action sampling
        if action_mask is not None:
            masked_discrete_logits = discrete_logits.clone()
            masked_discrete_logits[~action_mask] = float('-inf')
        else:
            masked_discrete_logits = discrete_logits
        
        discrete_dist = torch.distributions.Categorical(logits=masked_discrete_logits)
        discrete_action = discrete_dist.sample()
        discrete_log_prob = discrete_dist.log_prob(discrete_action)
        
        # Continuous action sampling using softplus for unbounded positive constraint
        continuous_std = torch.exp(continuous_log_std)
        # Use reparameterization with softplus
        eps = torch.randn_like(continuous_mean)
        pre_softplus = continuous_mean + continuous_std * eps
        continuous_action = F.softplus(pre_softplus)
        # Compute log_prob for softplus-transformed normal
        normal_log_prob = torch.distributions.Normal(continuous_mean, continuous_std).log_prob(pre_softplus)
        softplus_derivative = torch.sigmoid(pre_softplus)
        continuous_log_prob = (normal_log_prob - torch.log(softplus_derivative + 1e-8)).sum(axis=-1)
        
        # Get value estimate
        value = self.worker_value(z_t)
        
        return discrete_action, continuous_action, discrete_log_prob, continuous_log_prob, value.squeeze(-1)
    
    def take_action(self, obs: torch.Tensor, action_mask: torch.Tensor, 
                   pooled_goals: torch.Tensor) -> Tuple[int, float, float, float, float]:
        """Take action using hierarchical policy, returning both discrete and continuous actions and log_probs."""
        with torch.no_grad():
            obs = obs.unsqueeze(0).to(self.device)
            action_mask = action_mask.to(self.device)
            pooled_goals = pooled_goals.unsqueeze(0).to(self.device)
            
            # Encode state
            z_t = self.state_encoder(obs)
            
            # Worker action
            discrete_action, continuous_action, discrete_log_prob, continuous_log_prob, value = \
                self.get_worker_action_and_value(z_t, pooled_goals, action_mask.unsqueeze(0))
            
            return int(discrete_action.item()), float(continuous_action.item()), \
                   float(discrete_log_prob.item()), float(continuous_log_prob.item()), float(value.item())
    
    def get_deterministic_action(self, obs: torch.Tensor, action_mask: Optional[torch.Tensor],
                                pooled_goals: torch.Tensor) -> Tuple[int, float]:
        """Get deterministic action for evaluation for hybrid action space."""
        with torch.no_grad():
            obs = obs.unsqueeze(0).to(self.device)
            action_mask = action_mask.to(self.device)
            pooled_goals = pooled_goals.unsqueeze(0).to(self.device)
            
            # Encode state
            z_t = self.state_encoder(obs)
            
            # Worker deterministic action
            discrete_logits, continuous_mean, _ = self.worker_policy(z_t, pooled_goals)
            
            if action_mask is not None:
                # Set invalid discrete actions to -inf before argmax
                masked_discrete_logits = discrete_logits.clone()
                masked_discrete_logits[~action_mask.unsqueeze(0)] = float('-inf')
                discrete_action = torch.argmax(masked_discrete_logits, dim=-1)
            else:
                discrete_action = torch.argmax(discrete_logits, dim=-1)
            
            # For deterministic, use softplus to ensure positive values
            continuous_action = F.softplus(continuous_mean)

            return int(discrete_action.item()), float(continuous_action.item())
    
    def encode_state(self, obs: torch.Tensor) -> torch.Tensor:
        """Encode observation to latent space"""
        with torch.no_grad():
            obs = obs.unsqueeze(0).to(self.device)
            return self.state_encoder(obs).squeeze(0)
    
    def compute_intrinsic_reward(self, encoded_state_history: List[torch.Tensor], 
                                pooled_goal_history: List[torch.Tensor]) -> float:
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
            with torch.no_grad():
                manager_state = self.manager_encoder(state)
            
            # skip the first step of the episode
            if last_state is None:
                last_state = manager_state
                continue
                
            state_diff = manager_state - last_state
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
                # Apply manager encoder to both states for consistency with manager policy
                manager_s_t = self.manager_encoder(s_t.unsqueeze(0)).squeeze(0)
                manager_s_t_plus_c = self.manager_encoder(s_t_plus_c.unsqueeze(0)).squeeze(0)
                
                # State transition vector in manager-encoded space
                state_diff = manager_s_t_plus_c - manager_s_t
                
                # Cosine similarity is used as the reward signal
                cosine_sim = F.cosine_similarity(state_diff.unsqueeze(0), g_t.unsqueeze(0))
                return cosine_sim.item()
    
    def update_manager(self, buffer: HierarchicalPPOBuffer) -> Dict:
        """Update manager using transition-policy gradient with proper FuN objective."""
        # Get manager batch data from buffer
        manager_data = buffer.get_manager_batch()
        values = manager_data['values']
        rewards = manager_data['rewards']
        dones = manager_data['dones']
        observations = manager_data['observations']
        
        # Compute GAE advantages and returns
        advantages, returns = PPOUpdater.compute_gae_advantages(
            rewards, values, dones, self.gamma_manager, self.gae_lambda
        )
        
        # IMPORTANT: Re-encode observations and pass through manager encoder
        # PerceptualEncoder output goes to ManagerEncoder for manager-specific processing
        perceptual_encoded = self.state_encoder(observations)
        manager_encoded_with_grads = self.manager_encoder(perceptual_encoded)
        
        # Recompute manager values with current network to get gradients (including manager encoder)
        current_values = self.manager_value(manager_encoded_with_grads).squeeze(-1)
        value_loss = F.mse_loss(current_values, returns)
        
        # Recompute manager goals with current policy to get gradients
        current_goals = self.manager_policy(manager_encoded_with_grads)
        
        # For transition policy loss, use detached re-encoded states
        # These represent actual state transitions (same as stored encoded_states but from current encoder)
        # Only manager policy should get gradients from this loss, not encoders
        encoded_states_detached = perceptual_encoded.detach()  # Detached = same functionality as buffer's encoded_states
        policy_loss = PPOUpdater.transition_policy_loss(encoded_states_detached, current_goals, advantages)

        # Update manager networks, manager encoder, AND perceptual encoder (from value loss)
        self.manager_policy_optimizer.zero_grad()
        self.manager_value_optimizer.zero_grad()
        self.manager_encoder_optimizer.zero_grad()
        self.perceptual_encoder_optimizer.zero_grad()  # Add PerceptualEncoder update
        
        # Backward pass: Both encoders get gradients from value loss, manager policy from transition loss
        value_loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(self.manager_value.parameters(), self.max_grad_norm)
        torch.nn.utils.clip_grad_norm_(self.manager_encoder.parameters(), self.max_grad_norm)
        # PerceptualEncoder is updated via manager_value_loss.backward(retain_graph=True)

        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.manager_policy.parameters(), self.max_grad_norm)
        
        self.manager_policy_optimizer.step()
        self.manager_value_optimizer.step()
        self.manager_encoder_optimizer.step()
        self.perceptual_encoder_optimizer.step()  # Update PerceptualEncoder too
        
        stats = {
            'manager_value_loss': value_loss.item(),
            'manager_policy_loss': policy_loss.item()
        }

        return stats
    
    def update_worker(self, worker_buffer: PPOBuffer, manager_buffer: HierarchicalPPOBuffer,
                     train_pi_iters: int, train_v_iters: int) -> Dict:
        
        # Get worker data
        worker_data = worker_buffer.get_batch()
        worker_rewards = worker_data['rewards']
        worker_observations = worker_data['observations']  # Raw observations for re-encoding
        
        # Get step information from manager buffer
        step_information = manager_buffer.get_step_information()
        pooled_goals = step_information['pooled_goals']
        
        # Compute GAE advantages once (outside the loops)
        
        advantages, returns = PPOUpdater.compute_gae_advantages(
            worker_rewards, worker_data['values'], worker_data['dones'], 
            self.gamma_worker, self.gae_lambda
        )
        
        # Initialize accumulators
        accumulated_policy_loss = 0.0
        accumulated_value_loss = 0.0
        accumulated_entropy = 0.0

        # Policy update loop
        for _ in range(train_pi_iters):
            # Re-encode observations for each policy update to get fresh gradients
            encoded_states = self.state_encoder(worker_observations)
            
            # Forward pass through worker policy
            discrete_logits, continuous_mean, continuous_log_std = self.worker_policy(encoded_states, pooled_goals)
            
            masked_discrete_logits = discrete_logits.clone()
            masked_discrete_logits[~worker_data['action_masks']] = float('-inf')
            
            # Discrete policy loss
            discrete_dist = torch.distributions.Categorical(logits=masked_discrete_logits)
            new_discrete_log_probs = discrete_dist.log_prob(worker_data['actions'])
            discrete_policy_loss = PPOUpdater.ppo_policy_loss(
                new_discrete_log_probs, worker_data['log_probs'], advantages.detach(), self.clip_ratio
            )

            # Continuous policy loss with softplus transformation
            continuous_std = torch.exp(continuous_log_std)
            # Calculate new log_probs for stored continuous actions using inverse softplus (stable)
            continuous_actions = worker_data['continuous_actions']
            pre_softplus = torch.where(
                continuous_actions > 20,
                continuous_actions,
                torch.log(torch.expm1(continuous_actions) + 1e-8)
            )
            
            # New log_probs using current policy parameters
            normal_log_prob = torch.distributions.Normal(continuous_mean, continuous_std).log_prob(pre_softplus)
            # Apply softplus transformation log_prob correction
            softplus_derivative = torch.sigmoid(pre_softplus)
            new_continuous_log_probs = (normal_log_prob - torch.log(softplus_derivative + 1e-8)).sum(axis=-1)
            continuous_ratio = torch.exp(new_continuous_log_probs - worker_data['continuous_log_probs'])
            continuous_surr1 = continuous_ratio * advantages
            continuous_surr2 = torch.clamp(continuous_ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
            continuous_policy_loss = -torch.min(continuous_surr1, continuous_surr2).mean()

            # Entropy for regularization (discrete only)
            discrete_entropy = PPOUpdater.compute_entropy(discrete_logits, worker_data['action_masks'])
            total_entropy_val = discrete_entropy
            
            # Total policy loss with entropy regularization
            total_policy_loss = discrete_policy_loss + continuous_policy_loss - self.entropy_coef * total_entropy_val
            
            # Update worker policy and perceptual encoder
            self.worker_policy_optimizer.zero_grad()
            self.perceptual_encoder_optimizer.zero_grad()
            total_policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.worker_policy.parameters(), self.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(self.state_encoder.parameters(), self.max_grad_norm)
            self.worker_policy_optimizer.step()
            self.perceptual_encoder_optimizer.step()

            # KL early stop (approximate) for discrete policy
            with torch.no_grad():
                new_logp = new_discrete_log_probs
                approx_kl = (worker_data['log_probs'] - new_logp).mean().abs().item()
            if approx_kl > self.target_kl:
                break
            
            accumulated_policy_loss += (discrete_policy_loss.item() + continuous_policy_loss.item())
            accumulated_entropy += total_entropy_val.item()

        # Value function update loop
        for _ in range(train_v_iters):
            # Re-encode observations for each value update to get fresh gradients
            encoded_states = self.state_encoder(worker_observations)
            
            # Forward pass through worker value function
            values = self.worker_value(encoded_states).squeeze(-1)
            value_loss = F.mse_loss(values, returns.detach())
            
            # Update worker value function and perceptual encoder
            self.worker_value_optimizer.zero_grad()
            self.perceptual_encoder_optimizer.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.worker_value.parameters(), self.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(self.state_encoder.parameters(), self.max_grad_norm)
            self.worker_value_optimizer.step()
            self.perceptual_encoder_optimizer.step()
            
            accumulated_value_loss += value_loss.item()
            
        # Return training statistics
        stats = {
            'policy_loss': accumulated_policy_loss / train_pi_iters,
            'value_loss': accumulated_value_loss / train_v_iters,
            'entropy': accumulated_entropy / train_pi_iters
        }
        return stats
    
    def save(self, path: str):
        """Save all models and training stats"""
        torch.save({
            'perceptual_encoder_state_dict': self.state_encoder.state_dict(),
            'manager_encoder_state_dict': self.manager_encoder.state_dict(),
            'manager_policy_state_dict': self.manager_policy.state_dict(),
            'manager_value_state_dict': self.manager_value.state_dict(),
            'worker_policy_state_dict': self.worker_policy.state_dict(),
            'worker_value_state_dict': self.worker_value.state_dict(),
            'perceptual_encoder_optimizer_state_dict': self.perceptual_encoder_optimizer.state_dict(),
            'manager_encoder_optimizer_state_dict': self.manager_encoder_optimizer.state_dict(),
            'manager_policy_optimizer_state_dict': self.manager_policy_optimizer.state_dict(),
            'manager_value_optimizer_state_dict': self.manager_value_optimizer.state_dict(),
            'worker_policy_optimizer_state_dict': self.worker_policy_optimizer.state_dict(),
            'worker_value_optimizer_state_dict': self.worker_value_optimizer.state_dict(),
            
            # Save configuration parameters
            'input_dim': self.input_dim,
            'discrete_action_dim': self.discrete_action_dim,
            'continuous_action_dim': self.continuous_action_dim,
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
            'device': str(self.device),
            'max_grad_norm': self.max_grad_norm
        }, path)
    
    def load(self, path: str):
        """Load all models and training stats"""
        checkpoint = torch.load(path, map_location=self.device)
        try:
            # Load perceptual encoder (backward compatibility)
            if 'perceptual_encoder_state_dict' in checkpoint:
                self.state_encoder.load_state_dict(checkpoint['perceptual_encoder_state_dict'])
            elif 'manager_encoder_state_dict' in checkpoint:  # Old format
                self.state_encoder.load_state_dict(checkpoint['manager_encoder_state_dict'])
            
            # Load manager encoder
            if 'manager_encoder_state_dict' in checkpoint:
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
        if 'perceptual_encoder_optimizer_state_dict' in checkpoint:
            self.perceptual_encoder_optimizer.load_state_dict(checkpoint['perceptual_encoder_optimizer_state_dict'])
        elif 'encoder_optimizer_state_dict' in checkpoint:  # Old format
            self.perceptual_encoder_optimizer.load_state_dict(checkpoint['encoder_optimizer_state_dict'])
            
        if 'manager_encoder_optimizer_state_dict' in checkpoint:
            self.manager_encoder_optimizer.load_state_dict(checkpoint['manager_encoder_optimizer_state_dict'])
            
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
                'discrete_action_dim': checkpoint['discrete_action_dim'],
                'continuous_action_dim': checkpoint['continuous_action_dim'],
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
                'device': checkpoint['device'],
                'max_grad_norm': checkpoint.get('max_grad_norm', 0.5) # Default for backward compatibility
            }
        else:
            raise ValueError("This checkpoint does not contain configuration parameters.")

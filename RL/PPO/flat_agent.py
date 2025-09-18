import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from RL.PPO.networks import PolicyNetwork, ValueNetwork, PPOUpdater, HybridPolicyNetwork
from RL.PPO.buffer import PPOBuffer
from typing import Tuple, Optional

class FlatAgent:
    """
    PPO Worker Agent for Flexible Job Shop Scheduling with separate policy and value networks.
    """
    def __init__(self, 
                 input_dim,
                 action_dim,
                 pi_lr,
                 v_lr,
                 gamma,
                 gae_lambda,
                 clip_ratio,
                 entropy_coef,
                 device='auto',
                 target_kl: float = 0.01,
                 max_grad_norm: float = 0.5,
                 seed: Optional[int] = None):
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        print(f"PPO Agent initialized on device: {self.device}")
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.entropy_coef = entropy_coef
        self.target_kl = target_kl
        self.max_grad_norm = max_grad_norm
        self.seed = seed
        self.policy = PolicyNetwork(input_dim, action_dim).to(self.device)
        self.value = ValueNetwork(input_dim).to(self.device)
        self.pi_lr = pi_lr
        self.v_lr = v_lr
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=pi_lr)
        self.value_optimizer = optim.Adam(self.value.parameters(), lr=v_lr)
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        
        # Store configuration parameters for saving
        self.input_dim = input_dim
        
        self.training_stats = {
            'policy_loss': [],
            'value_loss': [],
            'entropy': []
        }
    
    def take_action(self, obs: torch.Tensor, action_mask: torch.Tensor, deterministic: bool = False) -> Tuple[int, float, float]:
        with torch.no_grad():
            obs = obs.unsqueeze(0).to(self.device)
            action_mask = action_mask.to(self.device)
            action_logits = self.policy(obs)
            action_logits = action_logits.squeeze(0)
            masked_logits = action_logits.clone()
            
            # Mask out invalid actions (~ action_mask reverse, e.g. action_logits = [0.3, 0.5, 0.2] -> masked_logits = [0.3, -inf, 0.2])
            masked_logits[~action_mask] = float('-inf')

            # Sample action
            dist = Categorical(logits=masked_logits)
            if deterministic:
                action = torch.argmax(masked_logits)
            else:
                action = dist.sample()

            # return log_prob, value
            log_prob = dist.log_prob(action)
            value = self.value(obs)
            return int(action.item()), float(log_prob.item()), float(value.item())
    
    def update(self, buffer: PPOBuffer, train_pi_iters: int, train_v_iters: int):
        """
        Update PPO agent using buffer.
        Args:
            buffer: PPOBuffer with experience
            train_pi_iters: Number of policy update iterations
            train_v_iters: Number of value update iterations
        Returns:
            stats: dict with policy_loss, value_loss, and entropy
        """
        worker_data = buffer.get_batch()
        
        # Compute GAE advantages using PPOUpdater
        advantages, returns = PPOUpdater.compute_gae_advantages(
            worker_data['rewards'], worker_data['values'], worker_data['dones'], 
            self.gamma, self.gae_lambda
        )
        
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0

        """
        Use same data to train policy and value function with setting ammount of steps.
        """
        # Policy update
        for i in range(train_pi_iters):

            """
            In first iteration, a new policy is generated based on the old one.
            But, in the second iteration, the policy is updated.
            """
            action_logits = self.policy(worker_data['observations'])
            masked_logits = action_logits.clone()
            masked_logits[~worker_data['action_masks']] = float('-inf')
            dist = Categorical(logits=masked_logits)
            new_log_probs = dist.log_prob(worker_data['actions'])
            
            # Compute entropy using PPOUpdater
            entropy = PPOUpdater.compute_entropy(action_logits, worker_data['action_masks'])

            # PPO policy loss using PPOUpdater
            policy_loss = PPOUpdater.ppo_policy_loss(
                new_log_probs, worker_data['log_probs'], advantages, self.clip_ratio
            )
            # Approximate KL for early stopping
            with torch.no_grad():
                approx_kl = (worker_data['log_probs'] - new_log_probs).mean().abs().item()
            if approx_kl > self.target_kl:
                # Early stop policy iterations
                break
            
            # Total policy loss with entropy penalty
            total_policy_loss_tensor = policy_loss - self.entropy_coef * entropy
            
            # update policy
            self.policy_optimizer.zero_grad()
            total_policy_loss_tensor.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy_optimizer.step()
            
            total_policy_loss += total_policy_loss_tensor.item()
            total_entropy += entropy.item()

        # Value function update (off-policy, no need for early stopping)
        for _ in range(train_v_iters):
            values = self.value(worker_data['observations']).squeeze(-1)
            value_loss = F.mse_loss(values, returns)
            self.value_optimizer.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.value.parameters(), self.max_grad_norm)
            self.value_optimizer.step()
            total_value_loss += value_loss.item()
        
        # Logging
        stats = {
            'policy_loss': total_policy_loss / train_pi_iters,
            'value_loss': total_value_loss / train_v_iters,
            'entropy': total_entropy / train_pi_iters
        }
        for k in stats:
            if k in self.training_stats:
                self.training_stats[k].append(stats[k])
        return stats
    
    
    def save(self, path: str):
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'value_state_dict': self.value.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'value_optimizer_state_dict': self.value_optimizer.state_dict(),
            'training_stats': self.training_stats,
            # Save configuration parameters
            'input_dim': self.input_dim,
            'action_dim': self.action_dim,
            'pi_lr': self.pi_lr,
            'v_lr': self.v_lr,
            'gamma': self.gamma,
            'gae_lambda': self.gae_lambda,
            'clip_ratio': self.clip_ratio,
            'entropy_coef': self.entropy_coef,
            'device': str(self.device),
            'target_kl': self.target_kl,
            'max_grad_norm': self.max_grad_norm
        }, path)

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        try:
            self.policy.load_state_dict(checkpoint['policy_state_dict'])
            self.value.load_state_dict(checkpoint['value_state_dict'])
        except RuntimeError as e:
            print("\n[ERROR] Model architecture mismatch while loading state_dict!")
            print("This usually means your current network architecture (e.g., hidden_dim, obs_dim, action_dim) does not match the one used for training.")
            print("Details:", e)
            print("\nTo fix: use the same hidden_dim and architecture as when you trained the model, or retrain from scratch if you changed the environment or network.")
            raise
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        self.value_optimizer.load_state_dict(checkpoint['value_optimizer_state_dict'])
        self.training_stats = checkpoint['training_stats']
    

    @staticmethod
    def get_saved_config(path: str):
        """Get the configuration parameters from a saved model."""
        checkpoint = torch.load(path, map_location='cpu')
        if 'input_dim' in checkpoint:
            return {
                'input_dim': checkpoint['input_dim'],
                'action_dim': checkpoint['action_dim'],
                'pi_lr': checkpoint['pi_lr'],
                'v_lr': checkpoint['v_lr'],
                'gamma': checkpoint['gamma'],
                'gae_lambda': checkpoint['gae_lambda'],
                'clip_ratio': checkpoint['clip_ratio'],
                'entropy_coef': checkpoint.get('entropy_coef', 0.01),  # Default for backward compatibility
                'device': checkpoint['device']
            }
        else:
            raise ValueError("This checkpoint does not contain configuration parameters. It was likely saved with an older version.") 

    def get_deterministic_action(self, obs: torch.Tensor, action_mask: torch.Tensor) -> int:
        """Selects the action with the highest probability (argmax) among valid actions."""
        with torch.no_grad():
            obs = obs.unsqueeze(0).to(self.device)
            action_mask = action_mask.to(self.device)
            action_logits = self.policy(obs)
            action_logits = action_logits.squeeze(0)
            masked_logits = action_logits.clone()
            masked_logits[~action_mask] = float('-inf')
            action = torch.argmax(masked_logits).item()
            return int(action)


class HybridFlatAgent:
    """
    PPO Agent for Flexible Job Shop Scheduling with hybrid action space (discrete + continuous).
    Uses separate policy and value networks.
    """
    def __init__(self, 
                 input_dim: int,
                 discrete_action_dim: int,
                 continuous_action_dim: int,
                 pi_lr: float,
                 v_lr: float,
                 gamma: float,
                 gae_lambda: float,
                 clip_ratio: float,
                 entropy_coef: float,
                 device: str = 'auto',
                 max_grad_norm: float = 0.5,
                 target_kl: float = 0.01,
                 seed: Optional[int] = None):
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        print(f"Hybrid PPO Agent initialized on device: {self.device}")
        
        self.discrete_action_dim = discrete_action_dim
        self.continuous_action_dim = continuous_action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl
        
        self.policy = HybridPolicyNetwork(input_dim, discrete_action_dim, continuous_action_dim).to(self.device)
        self.value = ValueNetwork(input_dim).to(self.device)
        
        self.pi_lr = pi_lr
        self.v_lr = v_lr
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=pi_lr)
        self.value_optimizer = optim.Adam(self.value.parameters(), lr=v_lr)
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        
        # Store configuration parameters for saving
        self.input_dim = input_dim
        
        self.training_stats = {
            'policy_loss': [],
            'value_loss': [],
            'entropy': [],
            'continuous_policy_loss': [],
            'discrete_policy_loss': []
        }
    
    def take_action(self, obs: torch.Tensor, action_mask: torch.Tensor, 
                   deterministic: bool = False) -> Tuple[int, float, float, float, float]:
        with torch.no_grad():
            obs = obs.unsqueeze(0).to(self.device)
            action_mask = action_mask.to(self.device)
            
            discrete_logits, continuous_mean, continuous_log_std = self.policy(obs)
            discrete_logits = discrete_logits.squeeze(0)
            continuous_mean = continuous_mean.squeeze(0)
            continuous_log_std = continuous_log_std.squeeze(0)
            
            masked_discrete_logits = discrete_logits.clone()
            masked_discrete_logits[~action_mask] = float('-inf')

            # Discrete action sampling
            discrete_dist = Categorical(logits=masked_discrete_logits)
            if deterministic:
                discrete_action = torch.argmax(masked_discrete_logits)
            else:
                discrete_action = discrete_dist.sample()
            discrete_log_prob = discrete_dist.log_prob(discrete_action)
            
            # Continuous action sampling using softplus for unbounded positive constraint
            continuous_std = torch.exp(continuous_log_std)
            if deterministic:
                continuous_action = F.softplus(continuous_mean)
                continuous_log_prob = torch.zeros_like(continuous_action)
            else:
                # For sampling, use reparameterization with softplus to ensure positive values
                eps = torch.randn_like(continuous_mean)
                pre_softplus = continuous_mean + continuous_std * eps
                continuous_action = F.softplus(pre_softplus)
                # Compute log_prob for softplus-transformed normal using change of variables
                # log_prob = normal_log_prob - log(softplus'(x)) = normal_log_prob - log(sigmoid(x))
                normal_log_prob = torch.distributions.Normal(continuous_mean, continuous_std).log_prob(pre_softplus)
                softplus_derivative = torch.sigmoid(pre_softplus)
                continuous_log_prob = (normal_log_prob - torch.log(softplus_derivative + 1e-8)).sum(axis=-1)
            
            # Value estimate
            value = self.value(obs)
            
            return int(discrete_action.item()), float(continuous_action.item()), \
                   float(discrete_log_prob.item()), float(continuous_log_prob.item()), float(value.item())

    def update(self, buffer: PPOBuffer, train_pi_iters: int, train_v_iters: int): # type: ignore
        """
        Update Hybrid PPO agent using buffer.
        Args:
            buffer: PPOBuffer with experience
            train_pi_iters: Number of policy update iterations
            train_v_iters: Number of value update iterations
        Returns:
            stats: dict with policy_loss, value_loss, and entropy
        """
        data = buffer.get_batch()
        
        # Compute GAE advantages using PPOUpdater
        advantages, returns = PPOUpdater.compute_gae_advantages(
            data['rewards'], data['values'], data['dones'], 
            self.gamma, self.gae_lambda
        )
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        total_discrete_policy_loss = 0.0
        total_continuous_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0

        # Policy update
        for i in range(train_pi_iters):

            discrete_logits, continuous_mean, continuous_log_std = self.policy(data['observations'])
            
            # Debug prints
            # print(f"Observations min: {data['observations'].min()}, max: {data['observations'].max()}")
            # print(f"Discrete logits (before mask) min: {discrete_logits.min()}, max: {discrete_logits.max()}")
            
            masked_discrete_logits = discrete_logits.clone()
            masked_discrete_logits[~data['action_masks']] = float('-inf')

            # print(f"Masked discrete logits min: {masked_discrete_logits.min()}, max: {masked_discrete_logits.max()}")
            # print(f"Action masks sum: {data['action_masks'].sum()}")

            discrete_dist = Categorical(logits=masked_discrete_logits)
            new_discrete_log_probs = discrete_dist.log_prob(data['actions'])
            discrete_policy_loss = PPOUpdater.ppo_policy_loss(
                new_discrete_log_probs, data['log_probs'], advantages, self.clip_ratio
            )
            
            # Continuous policy update with softplus transformation
            continuous_std = torch.exp(continuous_log_std)
            # Calculate new log_probs for stored continuous actions using inverse softplus
            continuous_actions = data['continuous_actions']
            # Numerically stable inverse softplus using expm1
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
            # PPO for continuous action: clip the ratio of new_log_probs to old_log_probs
            continuous_ratio = torch.exp(new_continuous_log_probs - data['continuous_log_probs'])
            continuous_surr1 = continuous_ratio * advantages
            continuous_surr2 = torch.clamp(continuous_ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
            continuous_policy_loss = -torch.min(continuous_surr1, continuous_surr2).mean()

            # Entropy for regularization (discrete only; no extra penalty for continuous)
            discrete_entropy = PPOUpdater.compute_entropy(discrete_logits, data['action_masks'])
            total_entropy_val = discrete_entropy
            
            # Total policy loss
            total_policy_loss_tensor = discrete_policy_loss + continuous_policy_loss - self.entropy_coef * total_entropy_val
            
            self.policy_optimizer.zero_grad()
            total_policy_loss_tensor.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy_optimizer.step()

            # KL early stop (approximate) using stored old log_probs
            with torch.no_grad():
                approx_kl = (data['log_probs'] - new_discrete_log_probs).mean().abs().item()
            if approx_kl > self.target_kl:
                break
            
            total_discrete_policy_loss += discrete_policy_loss.item()
            total_continuous_policy_loss += continuous_policy_loss.item()
            total_entropy += total_entropy_val.item()

        # Value function update
        for _ in range(train_v_iters):
            values = self.value(data['observations']).squeeze(-1)
            value_loss = F.mse_loss(values, returns)
            self.value_optimizer.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.value.parameters(), self.max_grad_norm)
            self.value_optimizer.step()
            total_value_loss += value_loss.item()
        
        # Logging
        stats = {
            'policy_loss': (total_discrete_policy_loss + total_continuous_policy_loss) / train_pi_iters,
            'discrete_policy_loss': total_discrete_policy_loss / train_pi_iters,
            'continuous_policy_loss': total_continuous_policy_loss / train_pi_iters,
            'value_loss': total_value_loss / train_v_iters,
            'entropy': total_entropy / train_pi_iters
        }
        for k in stats:
            if k in self.training_stats:
                self.training_stats[k].append(stats[k])
        return stats
    
    
    def save(self, path: str):
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'value_state_dict': self.value.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'value_optimizer_state_dict': self.value_optimizer.state_dict(),
            'training_stats': self.training_stats,
            # Save configuration parameters
            'input_dim': self.input_dim,
            'discrete_action_dim': self.discrete_action_dim,
            'continuous_action_dim': self.continuous_action_dim,
            'pi_lr': self.pi_lr,
            'v_lr': self.v_lr,
            'gamma': self.gamma,
            'gae_lambda': self.gae_lambda,
            'clip_ratio': self.clip_ratio,
            'entropy_coef': self.entropy_coef,
            'device': str(self.device),
            'max_grad_norm': self.max_grad_norm,
            'target_kl': self.target_kl
        }, path)

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        try:
            self.policy.load_state_dict(checkpoint['policy_state_dict'])
            self.value.load_state_dict(checkpoint['value_state_dict'])
        except RuntimeError as e:
            print("\n[ERROR] Model architecture mismatch while loading state_dict!")
            print("This usually means your current network architecture (e.g., hidden_dim, obs_dim, action_dim) does not match the one used for training.")
            print("Details:", e)
            print("\nTo fix: use the same hidden_dim and architecture as when you trained the model, or retrain from scratch if you changed the environment or network.")
            raise
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        self.value_optimizer.load_state_dict(checkpoint['value_optimizer_state_dict'])
        self.training_stats = checkpoint['training_stats']
    

    @staticmethod
    def get_saved_config(path: str):
        """Get the configuration parameters from a saved model."""
        checkpoint = torch.load(path, map_location='cpu')
        if 'input_dim' in checkpoint:
            return {
                'input_dim': checkpoint['input_dim'],
                'discrete_action_dim': checkpoint['discrete_action_dim'],
                'continuous_action_dim': checkpoint['continuous_action_dim'],
                'pi_lr': checkpoint['pi_lr'],
                'v_lr': checkpoint['v_lr'],
                'gamma': checkpoint['gamma'],
                'gae_lambda': checkpoint['gae_lambda'],
                'clip_ratio': checkpoint['clip_ratio'],
                'entropy_coef': checkpoint.get('entropy_coef', 0.01),  # Default for backward compatibility
                'device': checkpoint['device'],
                'max_grad_norm': checkpoint.get('max_grad_norm', 0.5) # Default for backward compatibility
            }
        else:
            raise ValueError("This checkpoint does not contain configuration parameters. It was likely saved with an older version.") 

    def get_deterministic_action(self, obs: torch.Tensor, action_mask: torch.Tensor) -> Tuple[int, float]:
        """Selects the action with the highest probability (argmax) among valid actions.
        Returns: discrete_action, continuous_action
        """
        with torch.no_grad():
            obs = obs.unsqueeze(0).to(self.device)
            action_mask = action_mask.to(self.device)
            
            discrete_logits, continuous_mean, _ = self.policy(obs)
            discrete_logits = discrete_logits.squeeze(0)
            continuous_mean = continuous_mean.squeeze(0)
            
            masked_discrete_logits = discrete_logits.clone()
            masked_discrete_logits[~action_mask] = float('-inf')
            
            discrete_action = torch.argmax(masked_discrete_logits).item()
            # For deterministic, use softplus to ensure positive values
            continuous_action = F.softplus(continuous_mean).item()
            
            return int(discrete_action), float(continuous_action)
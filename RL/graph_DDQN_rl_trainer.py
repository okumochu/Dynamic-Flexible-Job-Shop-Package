#!/usr/bin/env python3

import os
import time
import random
from typing import List, Tuple, Optional, Dict

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from torch_geometric.data import Batch

from config import config
from benchmarks.data_handler import FlexibleJobShopDataHandler
from RL.DDQN.graph_network import HGTQNetwork
from RL.graph_rl_env import GraphRlEnv


class ReplayBuffer:
    def __init__(self, capacity: int, device: torch.device):
        self.capacity = capacity
        self.device = device
        self.ptr = 0
        self.size = 0
        self.observations: List = []
        self.actions = torch.zeros(capacity, dtype=torch.long, device=device)
        self.rewards = torch.zeros(capacity, dtype=torch.float32, device=device)
        self.dones = torch.zeros(capacity, dtype=torch.bool, device=device)
        self.next_observations: List = []
        self.valid_action_pairs: List[List[Tuple[int, int]]] = []
        self.next_valid_action_pairs: List[List[Tuple[int, int]]] = []

    def add(self, obs, action: int, reward: float, done: bool, next_obs, valid_pairs, next_valid_pairs):
        if len(self.observations) < self.capacity:
            self.observations.append(obs.clone())
            self.next_observations.append(next_obs.clone())
            self.valid_action_pairs.append(valid_pairs.copy())
            self.next_valid_action_pairs.append(next_valid_pairs.copy())
        else:
            self.observations[self.ptr] = obs.clone()
            self.next_observations[self.ptr] = next_obs.clone()
            self.valid_action_pairs[self.ptr] = valid_pairs.copy()
            self.next_valid_action_pairs[self.ptr] = next_valid_pairs.copy()

        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = bool(done)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample_indices(self, batch_size: int) -> torch.Tensor:
        idxs = np.random.randint(0, self.size, size=batch_size)
        return torch.as_tensor(idxs, device=self.device)


class GraphDDQNTrainer:
    def __init__(self,
                 problem_data: FlexibleJobShopDataHandler,
                 epochs: int = None,
                 steps_per_epoch: int = None,
                 hidden_dim: int = None,
                 num_hgt_layers: int = None,
                 num_heads: int = None,
                 lr: float = None,
                 gamma: float = None,
                 batch_size: int = None,
                 buffer_size: int = None,
                 target_update_freq: int = None,
                 epsilon_start: float = None,
                 epsilon_end: float = None,
                 epsilon_decay_steps: int = None,
                 project_name: Optional[str] = None,
                 run_name: Optional[str] = None,
                 device: str = None,
                 model_save_dir: str = 'result/graph_rl_ddqn/model',
                 seed: Optional[int] = None):

        graph_config = config.get_graph_rl_config()
        rl_params = graph_config['rl_params']

        epochs = epochs if epochs is not None else rl_params.get('epochs', 10)
        steps_per_epoch = steps_per_epoch if steps_per_epoch is not None else rl_params.get('train_per_episode', 4) * problem_data.num_operations
        hidden_dim = hidden_dim if hidden_dim is not None else rl_params['hidden_dim']
        num_hgt_layers = num_hgt_layers if num_hgt_layers is not None else rl_params['num_hgt_layers']
        num_heads = num_heads if num_heads is not None else rl_params['num_heads']
        lr = lr if lr is not None else rl_params['lr']
        gamma = gamma if gamma is not None else rl_params['gamma']
        batch_size = batch_size if batch_size is not None else 64
        buffer_size = buffer_size if buffer_size is not None else problem_data.num_operations * 50
        target_update_freq = target_update_freq if target_update_freq is not None else 200
        epsilon_start = epsilon_start if epsilon_start is not None else 1.0
        epsilon_end = epsilon_end if epsilon_end is not None else 0.05
        epsilon_decay_steps = epsilon_decay_steps if epsilon_decay_steps is not None else 10_000
        device = device if device is not None else rl_params['device']
        seed = seed if seed is not None else rl_params['seed']

        self.problem_data = problem_data
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.project_name = project_name
        self.run_name = run_name
        self.device = torch.device(device)

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        # Env
        self.env = GraphRlEnv(problem_data, alpha=rl_params['alpha'], device=str(self.device))

        # Feature dims
        op_feature_dim, machine_feature_dim, job_feature_dim = self.env.graph_state.get_feature_dimensions()

        # Networks
        self.q_net = HGTQNetwork(op_feature_dim, machine_feature_dim, job_feature_dim, hidden_dim, num_hgt_layers, num_heads, rl_params['dropout']).to(self.device)
        self.target_q_net = HGTQNetwork(op_feature_dim, machine_feature_dim, job_feature_dim, hidden_dim, num_hgt_layers, num_heads, rl_params['dropout']).to(self.device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.target_q_net.eval()

        self.optimizer = torch.optim.AdamW(self.q_net.parameters(), lr=lr)

        # Replay buffer
        self.replay = ReplayBuffer(buffer_size, self.device)

        # DDQN params
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.total_steps = 0

        # Metrics accumulators per epoch
        self.episode_makespans: list = []
        self.episode_twts: list = []
        self.episode_objectives: list = []
        self.episode_rewards: list = []

        os.makedirs(model_save_dir, exist_ok=True)
        self.model_save_dir = model_save_dir

    def epsilon(self) -> float:
        frac = min(1.0, self.total_steps / max(1, self.epsilon_decay_steps))
        return float(self.epsilon_start + (self.epsilon_end - self.epsilon_start) * frac)

    @torch.no_grad()
    def select_action(self, obs: Batch) -> Tuple[int, Tuple[int, int]]:
        current_obs = obs.to(self.device)
        valid_actions = self.env.graph_state.get_valid_actions()
        if len(valid_actions) == 0:
            return 0, (0, 0)

        if random.random() < self.epsilon():
            action_idx = random.randrange(len(valid_actions))
            return action_idx, valid_actions[action_idx]

        q_vals_list = self.q_net(current_obs, valid_actions)
        q_vals = q_vals_list[0]
        action_idx = int(torch.argmax(q_vals).item())
        return action_idx, valid_actions[action_idx]

    def train(self):
        start_time = time.time()
        # match GraphPPOTrainer wandb init style
        wandb_dir = os.path.join(os.path.dirname(self.model_save_dir), 'training_process')
        os.makedirs(wandb_dir, exist_ok=True)
        wandb.init(
            name=self.run_name,
            project=self.project_name,
            dir=wandb_dir,
            config={
                "epochs": self.epochs,
                "steps_per_epoch": self.steps_per_epoch,
                "lr": self.optimizer.param_groups[0]['lr'],
                "gamma": self.gamma,
                "hidden_dim": self.q_net.hidden_dim,
                "num_hgt_layers": self.q_net.num_hgt_layers,
                "buffer_size": self.replay.capacity,
                "batch_size": self.batch_size,
                "target_update_freq": self.target_update_freq,
            }
        )

        # Progress bar for epochs (mirror PPO verbosity)
        from tqdm import tqdm
        pbar = tqdm(range(self.epochs), desc="Graph DDQN Training")

        for epoch in pbar:
            obs, _ = self.env.reset()
            done = False

            steps_this_epoch = 0
            episode_reward = 0.0
            episode_makespans = []
            episode_twts = []

            q_losses = []
            while steps_this_epoch < self.steps_per_epoch:
                # Capture current state's valid actions before stepping
                current_valid = self.env.graph_state.get_valid_actions()

                action_idx, pair = self.select_action(obs)
                env_action = self.env.pair_to_action_map.get(tuple(pair), None)
                if env_action is None:
                    # fallback random valid action
                    env_action = self.env.pair_to_action_map.get(tuple(random.choice(current_valid)))

                next_obs, reward, terminated, truncated, info = self.env.step(env_action)
                next_valid = info['valid_actions']

                # Store transition: use previous state's valid actions
                self.replay.add(
                    obs.to(self.device),
                    action_idx,
                    float(reward),
                    bool(terminated),
                    next_obs.to(self.device),
                    current_valid,
                    next_valid
                )

                obs = next_obs
                done = terminated
                steps_this_epoch += 1
                self.total_steps += 1
                episode_reward += float(reward)

                # collect per-step performance
                episode_makespans.append(float(info.get('makespan', 0.0)))
                episode_twts.append(float(info.get('total_weighted_tardiness', 0.0)))

                if self.replay.size >= max(10, self.batch_size):
                    q_loss = self.update()
                    q_losses.append(float(q_loss))

                if self.total_steps % self.target_update_freq == 0:
                    self.target_q_net.load_state_dict(self.q_net.state_dict())

                if done:
                    obs, _ = self.env.reset()
                    done = False
                    # finalize episode stats
                    makespan_final = episode_makespans[-1] if episode_makespans else 0.0
                    twt_final = episode_twts[-1] if episode_twts else 0.0
                    objective_final = (1 - self.env.alpha) * makespan_final + self.env.alpha * twt_final
                    self.episode_makespans.append(makespan_final)
                    self.episode_twts.append(twt_final)
                    self.episode_objectives.append(objective_final)
                    self.episode_rewards.append(episode_reward)
                    # reset
                    episode_reward = 0.0
                    episode_makespans.clear()
                    episode_twts.clear()

            # epoch-end logging (mirror GraphPPOTrainer keys; use q_loss instead of policy/value)
            if len(self.episode_makespans) > 0:
                wandb_log = {
                    'q_loss': float(np.mean(q_losses)) if q_losses else 0.0,
                    'total_epochs': epoch + 1,
                    'learning_rate': float(self.optimizer.param_groups[0]['lr']),
                    'performance/makespan_mean': float(np.mean(self.episode_makespans)),
                    'performance/twt_mean': float(np.mean(self.episode_twts)),
                    'performance/objective_mean': float(np.mean(self.episode_objectives)),
                    'performance/reward_mean': float(np.mean(self.episode_rewards)),
                    'performance/alpha': self.env.alpha
                }
                wandb.log(wandb_log, step=epoch)

                # Update progress bar/status line
                pbar.set_postfix({
                    'q_loss': f"{wandb_log['q_loss']:.4f}",
                    'mkspn': f"{wandb_log['performance/makespan_mean']:.1f}",
                    'obj': f"{wandb_log['performance/objective_mean']:.1f}",
                    'rew': f"{wandb_log['performance/reward_mean']:.1f}",
                })

            # clear accumulators for next epoch
            self.episode_makespans.clear()
            self.episode_twts.clear()
            self.episode_objectives.clear()
            self.episode_rewards.clear()

        pbar.close()

        # Save model with timestamp and finish wandb (mirror GraphPPOTrainer)
        model_filename = config.create_model_filename()
        self.save_model(model_filename)
        wandb.finish()

        return {
            "training_time": time.time() - start_time,
            "model_filename": model_filename
        }

    def update(self):
        idxs = self.replay.sample_indices(self.batch_size)

        # Gather batch observations
        obs_list = [self.replay.observations[i] for i in idxs.tolist()]
        next_obs_list = [self.replay.next_observations[i] for i in idxs.tolist()]
        valid_list = [self.replay.valid_action_pairs[i] for i in idxs.tolist()]
        next_valid_list = [self.replay.next_valid_action_pairs[i] for i in idxs.tolist()]

        actions = self.replay.actions[idxs]
        rewards = self.replay.rewards[idxs]
        dones = self.replay.dones[idxs]

        # Forward current Q(s,a)
        batch_graphs = Batch.from_data_list([g.to(self.device) for g in obs_list])
        q_values_lists = self.q_net(batch_graphs, valid_list, use_batch=True)

        # For indexing varying-length lists, compute offsets
        action_offsets = []
        offset = 0
        for pairs in valid_list:
            action_offsets.append(offset)
            offset += len(pairs)

        flat_q_values = torch.cat(q_values_lists, dim=0) if len(q_values_lists) > 0 else torch.empty(0, device=self.device)

        # Map per-sample action index into flattened indices
        flat_indices = []
        for i, a in enumerate(actions.tolist()):
            base = action_offsets[i]
            flat_indices.append(base + a)
        flat_indices = torch.as_tensor(flat_indices, dtype=torch.long, device=self.device)

        chosen_q = flat_q_values[flat_indices]

        # Target: r + gamma * Q_target(s', argmax_a' Q_online(s', a'))
        with torch.no_grad():
            next_batch_graphs = Batch.from_data_list([g.to(self.device) for g in next_obs_list])
            next_q_online_lists = self.q_net(next_batch_graphs, next_valid_list, use_batch=True)
            next_q_target_lists = self.target_q_net(next_batch_graphs, next_valid_list, use_batch=True)

            next_max_q = []
            for q_online, q_target in zip(next_q_online_lists, next_q_target_lists):
                if q_online.numel() == 0:
                    next_max_q.append(torch.tensor(0.0, device=self.device))
                    continue
                best_idx = torch.argmax(q_online)
                next_max_q.append(q_target[best_idx])
            next_max_q = torch.stack(next_max_q)

            targets = rewards + (1.0 - dones.float()) * self.gamma * next_max_q

        loss = F.mse_loss(chosen_q, targets)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=1.0)
        self.optimizer.step()
        return loss.detach().item()

    def save_model(self, filename: str):
        filepath = os.path.join(self.model_save_dir, filename)
        torch.save({
            'q_net': self.q_net.state_dict(),
            'target_q_net': self.target_q_net.state_dict(),
        }, filepath)

    def load_model(self, filename: str):
        filepath = os.path.join(self.model_save_dir, filename)
        if os.path.exists(filepath):
            checkpoint = torch.load(filepath, map_location=self.device)
            self.q_net.load_state_dict(checkpoint['q_net'])
            self.target_q_net.load_state_dict(checkpoint['target_q_net'])



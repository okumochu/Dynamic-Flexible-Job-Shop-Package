import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.data import HeteroData, Batch
from torch_geometric.nn import HGTConv, Linear, MLP, GlobalAttention
from torch_geometric.nn.norm import LayerNorm as PygLayerNorm
from torch_geometric.utils import softmax, scatter
from typing import Dict, List, Tuple, Optional


class _MultiTypeGlobalAttentionPooling(nn.Module):
    """
    Wrapper around three GlobalAttention layers (one per node type) returning
    concatenated pooled embeddings of size 3*hidden_dim.
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        gate_nn_factory = lambda: nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        self.op_attn = GlobalAttention(gate_nn_factory())
        self.machine_attn = GlobalAttention(gate_nn_factory())
        self.job_attn = GlobalAttention(gate_nn_factory())

    def forward(
        self,
        op_x: torch.Tensor, op_batch: torch.Tensor,
        machine_x: torch.Tensor, machine_batch: torch.Tensor,
        job_x: torch.Tensor, job_batch: torch.Tensor,
        batch_size: int
    ) -> torch.Tensor:
        op_pool = self.op_attn(op_x, op_batch)  # [B, H]
        machine_pool = self.machine_attn(machine_x, machine_batch)  # [B, H]
        job_pool = self.job_attn(job_x, job_batch)  # [B, H]
        return torch.cat([op_pool, machine_pool, job_pool], dim=1)  # [B, 3H]


class HGTPolicy(nn.Module):
    """
    Heterogeneous Graph Transformer (HGT) Policy Network for FJSP.
    
    This network processes a heterogeneous graph with operation and machine nodes,
    uses HGT layers for message passing, and outputs action logits for valid
    (operation, machine) pairs along with value estimates.
    """
    
    def __init__(self,
                 op_feature_dim: int,  # Dynamic: determined from graph state structure
                 machine_feature_dim: int,  # Dynamic: determined from graph state structure
                 job_feature_dim: int,  # Dynamic: determined from graph state structure
                 hidden_dim: int,  # From config.graph_rl_params['hidden_dim']
                 num_hgt_layers: int,  # From config.graph_rl_params['num_hgt_layers']
                 num_heads: int,  # From config.graph_rl_params['num_heads']
                 dropout: float):
        """
        Initialize the HGT Policy Network with hierarchical job-operation-machine structure.
        
        Args:
            op_feature_dim: Dimensionality of operation node features (auto-detected: 8)
            machine_feature_dim: Dimensionality of machine node features (auto-detected: 7) 
            job_feature_dim: Dimensionality of job node features (auto-detected: 7)
            hidden_dim: Hidden dimension for embeddings and HGT layers (from config)
            num_hgt_layers: Number of HGT conv layers (from config)
            num_heads: Number of attention heads in HGT (from config)
            dropout: Dropout rate (from config)
            temporal_dim: Dimension for temporal embeddings in RTE (from config)
            max_temporal_freq: Maximum frequency for sinusoidal temporal encoding (from config)
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_hgt_layers = num_hgt_layers
        self.dropout = dropout
        self.use_temporal_encoding = False
        
        # Input embedding layers for different node types
        self.op_embedding = Linear(op_feature_dim, hidden_dim, bias=True)
        self.machine_embedding = Linear(machine_feature_dim, hidden_dim, bias=True)
        self.job_embedding = Linear(job_feature_dim, hidden_dim, bias=True)
        
        # HGT layers for message passing (hierarchical structure)
        self.hgt_layers = nn.ModuleList()
        # Per-layer LayerNorm modules for residual stabilization
        self.per_layer_norms = nn.ModuleList()
        # Dropout applied after residual addition
        self.residual_dropout = nn.Dropout(dropout)
        metadata = (
            ['op', 'machine', 'job'],
            [
                # Job-operation hierarchy
                ('job', 'contains', 'op'),
                ('op', 'belongs_to', 'job'),
                
                # Operation precedence and machine scheduling
                ('op', 'precedes', 'op'),
                ('op', 'machine_precedes', 'op'),
                
                # Operation-machine compatibility and assignment
                ('op', 'on_machine', 'machine'),
                ('op', 'assigned_to', 'machine'),
                ('machine', 'can_process', 'op'),
                ('machine', 'processes', 'op')
            ]
        )
        
        for _ in range(num_hgt_layers):
            self.hgt_layers.append(
                HGTConv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    metadata=metadata,
                    heads=num_heads
                )
            )
            # Create one LayerNorm per node type per layer
            self.per_layer_norms.append(
                nn.ModuleDict({
                    'op': PygLayerNorm(hidden_dim),
                    'machine': PygLayerNorm(hidden_dim),
                    'job': PygLayerNorm(hidden_dim)
                })
            )
        
        # Policy head: input [op_emb, machine_emb, global_embedding] -> scalar
        # Input dim = hidden_dim (op) + hidden_dim (machine) + 3*hidden_dim (global) = 5*hidden_dim
        self.policy_head = MLP(
            channel_list=[5 * hidden_dim, hidden_dim, hidden_dim // 2, 1],
            dropout=dropout,
            act='tanh',
            norm=None
        )
        
        # Value head: map global embedding directly to scalar value
        self.value_head = MLP(
            channel_list=[3 * hidden_dim, hidden_dim, 1],
            dropout=dropout,
            act='tanh',
            norm=None
        )
        
        # Separate attention pooling for actor and critic using GlobalAttention
        self.actor_pool = _MultiTypeGlobalAttentionPooling(hidden_dim)
        self.critic_pool = _MultiTypeGlobalAttentionPooling(hidden_dim)

        # Layer normalization - use PyG LayerNorm
        self.op_norm = PygLayerNorm(hidden_dim)
        self.machine_norm = PygLayerNorm(hidden_dim)
        self.job_norm = PygLayerNorm(hidden_dim)
    
    
    def forward(self, obs, valid_actions=None, use_batch=False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass that always works with batches internally.
        
        Args:
            obs: Either a single HeteroData graph or a Batch containing multiple graphs
            valid_actions: For single graph: List of valid (operation, machine) pairs
                          For batch: List of valid action pairs for each graph in the batch
            use_batch: If True, treat obs as already in batch format and use directly
            
        Returns:
            For single graph: (action_logits, value)
            For batch: (batch_action_logits, batch_values)
        """
        # Use batch format directly if use_batch is True, otherwise convert single graph to batch
        if use_batch:
            # Input is already in batch format - use directly
            batch = obs
            batch_valid_action_pairs = valid_actions
            is_single_graph = False
        else:
            # Single graph case - convert to batch
            batch = Batch.from_data_list([obs])
            batch_valid_action_pairs = [valid_actions] if valid_actions else [[]]
            is_single_graph = True
        
        # Process the batch
        batch_action_logits, batch_values = self._process_batch(batch, batch_valid_action_pairs)
        
        # Return appropriate format based on input type
        if is_single_graph:
            # Extract single graph results
            action_logits = batch_action_logits[0] if len(batch_action_logits) > 0 else torch.tensor([], device=obs['op'].x.device)
            value = batch_values[0] if len(batch_values) > 0 else torch.tensor(0.0, device=obs['op'].x.device)
            
            return action_logits, value
        else:
            # Return batch results
            return batch_action_logits, batch_values
    
    def _process_batch(self, batch: Batch, batch_valid_action_pairs: List[List[Tuple[int, int]]]) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Internal method for batched forward pass.
        
        Args:
            batch: torch_geometric.data.Batch containing multiple HeteroData graphs
            batch_valid_action_pairs: List of valid action pairs for each graph in the batch
            
        Returns:
            batch_action_logits: List of action logits tensors for each graph
            batch_values: Tensor of value estimates for each graph [batch_size]
        """
        # Extract node features and edge indices from batched graph (EXTENDED: includes job nodes)
        op_x = batch['op'].x  # [total_ops_across_batch, op_feature_dim]
        machine_x = batch['machine'].x  # [total_machines_across_batch, machine_feature_dim]
        job_x = batch['job'].x  # [total_jobs_across_batch, job_feature_dim] NEW
        
        # Embed node features to hidden dimension
        op_embeddings = self.op_embedding(op_x)  # [total_ops_across_batch, hidden_dim]
        machine_embeddings = self.machine_embedding(machine_x)  # [total_machines_across_batch, hidden_dim]
        job_embeddings = self.job_embedding(job_x)  # [total_jobs_across_batch, hidden_dim] NEW
        
        # Apply layer normalization
        op_embeddings = self.op_norm(op_embeddings)
        machine_embeddings = self.machine_norm(machine_embeddings)
        job_embeddings = self.job_norm(job_embeddings)  # NEW
        
        # Create node embedding dictionary for HGT (EXTENDED: includes job embeddings)
        x_dict = {
            'op': op_embeddings,
            'machine': machine_embeddings,
            'job': job_embeddings  # NEW: hierarchical job information
        }
        
        # Extract edge indices
        """
        edge_index_dict = {
            ('job', 'contains', 'op'): tensor([
                [0, 0, 1, 1, 2],  # source nodes (jobs)
                [0, 1, 2, 3, 4]   # target nodes (operations)
            ])
            # J0→O0, J0→O1, J1→O2, J1→O3, J2→O4
        """
        edge_index_dict = {}
        # Process all edge types from the batch (all edge types are guaranteed to exist)
        for edge_type in batch.edge_types:
            edge_index_dict[edge_type] = batch[edge_type].edge_index
        
        # Apply HGT layers with residual connections and per-layer LayerNorm
        for layer_index, hgt_layer in enumerate(self.hgt_layers):
            residual_dict = {k: v for k, v in x_dict.items()}
            out_dict = hgt_layer(x_dict, edge_index_dict)
            # Residual add + dropout + norm per node type
            x_dict = {
                'op': self.per_layer_norms[layer_index]['op'](self.residual_dropout(out_dict['op'] + residual_dict['op'])),
                'machine': self.per_layer_norms[layer_index]['machine'](self.residual_dropout(out_dict['machine'] + residual_dict['machine'])),
                'job': self.per_layer_norms[layer_index]['job'](self.residual_dropout(out_dict['job'] + residual_dict['job']))
            }
        
        # Get final embeddings
        final_op_embeddings = x_dict['op']  # [total_ops_across_batch, hidden_dim]
        final_machine_embeddings = x_dict['machine']  # [total_machines_across_batch, hidden_dim]
        final_job_embeddings = x_dict['job']  # [total_jobs_across_batch, hidden_dim]
        
        # Split embeddings back to individual graphs using batch indices
        # Get batch information (using index to split the mixed embeddings across graphs)
        # e.g. op_batch = [0, 0, 0, 1, 1], it means the first 3 ops belong to the first graph, and the last 2 ops belong to the second graph
        op_batch = batch['op'].batch  # [total_ops_across_batch]
        machine_batch = batch['machine'].batch  # [total_machines_across_batch]
        job_batch = batch['job'].batch  # [total_jobs_across_batch]
        batch_size = batch.num_graphs
        
        # Attention-based global pooling (separate modules for actor and critic)
        actor_global_embedding_batched = self.actor_pool(
            final_op_embeddings, op_batch,
            final_machine_embeddings, machine_batch,
            final_job_embeddings, job_batch,
            batch_size
        )  # [batch_size, 3*hidden_dim]

        critic_global_embedding_batched = self.critic_pool(
            final_op_embeddings, op_batch,
            final_machine_embeddings, machine_batch,
            final_job_embeddings, job_batch,
            batch_size
        )  # [batch_size, 3*hidden_dim]
        
        # Compute action logits for each graph in a vectorized manner
        device = final_op_embeddings.device
        
        # Step 1: Get action counts and node offsets
        num_actions_per_graph = torch.tensor(
            [len(pairs) for pairs in batch_valid_action_pairs],
            device=device,
            dtype=torch.long
        )
        op_offsets = batch['op'].ptr[:-1]
        machine_offsets = batch['machine'].ptr[:-1]
                
        # Step 2: Create global action indices
        global_op_indices = torch.cat([
            (torch.tensor([pair[0] for pair in pairs], device=device, dtype=torch.long) + op_offsets[i])
            for i, pairs in enumerate(batch_valid_action_pairs)
        ])
        global_machine_indices = torch.cat([
            (torch.tensor([pair[1] for pair in pairs], device=device, dtype=torch.long) + machine_offsets[i])
            for i, pairs in enumerate(batch_valid_action_pairs)
        ])
        
        # Step 3: Gather all embeddings
        valid_op_embeds = final_op_embeddings[global_op_indices]
        valid_machine_embeds = final_machine_embeddings[global_machine_indices]
        
        # Step 4: Broadcast global context per action
        broadcast_global = torch.repeat_interleave(
            actor_global_embedding_batched,
            num_actions_per_graph,
            dim=0
        )
        
        # Step 5: Compute all logits at once
        combined_embeddings = torch.cat(
            [valid_op_embeds, valid_machine_embeds, broadcast_global], dim=1
        )
        all_logits = self.policy_head(combined_embeddings).squeeze(-1)
        
        # Step 6: Split logits back into list per graph
        batch_action_logits = list(torch.split(all_logits, num_actions_per_graph.tolist()))
        
        # Critic: values from critic-specific global embeddings
        batch_values = self.value_head(critic_global_embedding_batched).squeeze(-1)  # [batch_size]
        
        return batch_action_logits, batch_values    


class GraphPPOBuffer:
    """
    Buffer specifically designed for graph-based PPO with variable action spaces.
    """
    
    def __init__(self, buffer_size: int, device: torch.device):
        self.buffer_size = buffer_size
        self.device = device
        self.ptr = 0
        self.size = 0
        
        # Store graph observations (as list since they can vary in size)
        self.observations = []
        self.actions = torch.zeros(buffer_size, dtype=torch.long, device=device)
        self.rewards = torch.zeros(buffer_size, dtype=torch.float32, device=device)
        self.values = torch.zeros(buffer_size, dtype=torch.float32, device=device)
        self.log_probs = torch.zeros(buffer_size, dtype=torch.float32, device=device)
        self.dones = torch.zeros(buffer_size, dtype=torch.bool, device=device)
        
        # Store valid action pairs (as lists) - no action masks needed for graph version
        self.valid_action_pairs = []
    
    def add(self, obs: HeteroData, action: int, reward: float, value: float,
            log_prob: float, valid_pairs: List[Tuple[int, int]], done: bool):
        """Add a transition to the buffer."""
        
        # Handle buffer overflow by overwriting oldest entries
        if len(self.observations) < self.buffer_size:
            self.observations.append(obs.clone())
            self.valid_action_pairs.append(valid_pairs.copy())
        else:
            self.observations[self.ptr] = obs.clone()
            self.valid_action_pairs[self.ptr] = valid_pairs.copy()
        
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value
        self.log_probs[self.ptr] = log_prob
        self.dones[self.ptr] = bool(done)
        
        self.ptr = (self.ptr + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)
    
    def get_batch(self) -> Dict[str, any]:
        """Get a batch of all stored transitions."""
        return {
            'observations': self.observations[:self.size],
            'actions': self.actions[:self.size],
            'rewards': self.rewards[:self.size],
            'values': self.values[:self.size],
            'log_probs': self.log_probs[:self.size],
            'dones': self.dones[:self.size],
            'valid_action_pairs': self.valid_action_pairs[:self.size]
        }
    
    def clear(self):
        """Clear the buffer."""
        self.ptr = 0
        self.size = 0
        self.observations.clear()
        self.valid_action_pairs.clear()
        self.actions.zero_()
        self.rewards.zero_()
        self.values.zero_()
        self.log_probs.zero_()
        self.dones.zero_()

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.data import HeteroData
from torch_geometric.nn import HGTConv, Linear
from typing import Dict, List, Tuple, Optional


class TemporalHGTConv(nn.Module):
    """
    Enhanced HGT layer with Relative Temporal Encoding (RTE) for machine_precedes edges.
    
    This layer implements the sinusoidal temporal encoding from the HGT paper (Section 3.5)
    to capture temporal dynamics in scheduling problems.
    """
    
    def __init__(self, in_channels: int, out_channels: int, metadata: Tuple[List[str], List[Tuple[str, str, str]]], 
                 heads: int = 1, temporal_dim: int = 16, max_temporal_freq: float = 1000.0):
        super().__init__()
        
        self.temporal_dim = temporal_dim
        self.out_channels = out_channels
        self.max_temporal_freq = max_temporal_freq
        
        # Standard HGT layer
        self.hgt_conv = HGTConv(in_channels, out_channels, metadata, heads)
        
        # Sinusoidal temporal encoding parameters
        # Generate frequency basis for sinusoidal encoding (Equation 6 from HGT paper)
        freq_positions = torch.arange(0, temporal_dim // 2, dtype=torch.float32)
        frequencies = 1.0 / (max_temporal_freq ** (2 * freq_positions / temporal_dim))
        self.register_buffer('frequencies', frequencies)
        
        # T-Linear: trainable linear projection for temporal encoding (from HGT paper)
        self.temporal_projection = nn.Linear(temporal_dim, out_channels)
        
        # Layer normalization for stability
        self.temporal_norm = nn.LayerNorm(out_channels)
        
    def encode_temporal_information(self, delta_t: torch.Tensor) -> torch.Tensor:
        """
        Apply sinusoidal temporal encoding to time delays (Equation 6 from HGT paper).
        
        Args:
            delta_t: Time delays [num_edges, 1] 
            
        Returns:
            Sinusoidal temporal embeddings [num_edges, temporal_dim]
        """
        # delta_t shape: [num_edges, 1]
        # frequencies shape: [temporal_dim // 2]
        
        # Compute arguments for sine and cosine functions
        # Shape: [num_edges, temporal_dim // 2]
        args = delta_t * self.frequencies.unsqueeze(0)
        
        # Apply sine and cosine functions alternately
        sin_embeddings = torch.sin(args)  # [num_edges, temporal_dim // 2]
        cos_embeddings = torch.cos(args)  # [num_edges, temporal_dim // 2]
        
        # Interleave sine and cosine to create full temporal embedding
        # Shape: [num_edges, temporal_dim]
        temporal_embeddings = torch.zeros(delta_t.shape[0], self.temporal_dim, 
                                        device=delta_t.device, dtype=delta_t.dtype)
        temporal_embeddings[:, 0::2] = sin_embeddings  # Even indices = sine
        temporal_embeddings[:, 1::2] = cos_embeddings  # Odd indices = cosine
        
        return temporal_embeddings
    
    def forward(self, x_dict: Dict[str, torch.Tensor], edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
                edge_attr_dict: Optional[Dict[Tuple[str, str, str], torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass with Relative Temporal Encoding (RTE) for machine_precedes edges.
        
        Implements the HGT paper's temporal encoding: sinusoidal basis functions + T-Linear projection.
        
        Args:
            x_dict: Node embeddings for each node type
            edge_index_dict: Edge indices for each edge type
            edge_attr_dict: Edge attributes (temporal information) for each edge type
            
        Returns:
            Updated node embeddings with temporal information
        """
        # Apply temporal encoding BEFORE standard HGT (as per HGT paper)
        x_dict_with_temporal = x_dict.copy()
        
        # Apply Relative Temporal Encoding if machine_precedes edges exist with attributes
        if (edge_attr_dict is not None and 
            ('op', 'machine_precedes', 'op') in edge_attr_dict and
            ('op', 'machine_precedes', 'op') in edge_index_dict):
            
            machine_precedes_edges = edge_index_dict[('op', 'machine_precedes', 'op')]
            temporal_attrs = edge_attr_dict[('op', 'machine_precedes', 'op')]
            
            if machine_precedes_edges.shape[1] > 0 and temporal_attrs.shape[0] > 0:
                # Step 1: Apply sinusoidal temporal encoding to ΔT (processing times)
                temporal_embeddings = self.encode_temporal_information(temporal_attrs)  # [num_edges, temporal_dim]
                
                # Step 2: Apply T-Linear projection (trainable transformation)
                projected_temporal = self.temporal_projection(temporal_embeddings)  # [num_edges, out_channels]
                projected_temporal = self.temporal_norm(projected_temporal)
                
                # Step 3: Add temporal encoding to source node features (as per HGT paper)
                # H^(l)[s] + RTE(ΔT) where s is the source node
                src_ops = machine_precedes_edges[0]  # Source operations
                
                # Accumulate temporal updates for source nodes (operations can appear multiple times)
                temporal_updates = torch.zeros_like(x_dict_with_temporal['op'])
                temporal_updates.index_add_(0, src_ops, projected_temporal)
                
                # Add temporal encoding to source node features
                x_dict_with_temporal['op'] = x_dict_with_temporal['op'] + temporal_updates
        
        # Apply standard HGT message passing on temporally-augmented features  
        x_dict_updated = self.hgt_conv(x_dict_with_temporal, edge_index_dict)
        
        return x_dict_updated


class HGTPolicy(nn.Module):
    """
    Heterogeneous Graph Transformer (HGT) Policy Network for FJSP.
    
    This network processes a heterogeneous graph with operation and machine nodes,
    uses HGT layers for message passing, and outputs action logits for valid
    (operation, machine) pairs along with value estimates.
    """
    
    def __init__(self,
                 op_feature_dim: int = 8,
                 machine_feature_dim: int = 2,
                 hidden_dim: int = 128,
                 num_hgt_layers: int = 3,
                 num_heads: int = 4,
                 dropout: float = 0.1):
        """
        Initialize the HGT Policy Network.
        
        Args:
            op_feature_dim: Dimensionality of operation node features
            machine_feature_dim: Dimensionality of machine node features  
            hidden_dim: Hidden dimension for embeddings and HGT layers
            num_hgt_layers: Number of HGT conv layers
            num_heads: Number of attention heads in HGT
            dropout: Dropout rate
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_hgt_layers = num_hgt_layers
        self.dropout = dropout
        
        # Input embedding layers for different node types
        self.op_embedding = Linear(op_feature_dim, hidden_dim, bias=True, weight_initializer='glorot')
        self.machine_embedding = Linear(machine_feature_dim, hidden_dim, bias=True, weight_initializer='glorot')
        
        # Temporal HGT layers for message passing with RTE
        self.hgt_layers = nn.ModuleList()
        metadata = (['op', 'machine'], [('op', 'precedes', 'op'),
                                       ('op', 'machine_precedes', 'op'),  # CRITICAL: Disjunctive arcs with temporal info
                                       ('op', 'on_machine', 'machine'),
                                       ('op', 'assigned_to', 'machine'),
                                       ('machine', 'can_process', 'op'),
                                       ('machine', 'processes', 'op')])
        
        for _ in range(num_hgt_layers):
            self.hgt_layers.append(
                TemporalHGTConv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    metadata=metadata,
                    heads=num_heads,
                    temporal_dim=16  # Dimension for temporal embeddings
                )
            )
        
        # Action head: processes concatenated (op, machine) embeddings
        self.action_head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)  # Single logit for (op, machine) pair
        )
        
        # Value head: processes global graph representation
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)  # Single value estimate
        )
        
        # Layer normalization
        self.op_norm = nn.LayerNorm(hidden_dim)
        self.machine_norm = nn.LayerNorm(hidden_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights with improved initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Use He initialization for ReLU networks
                nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        
        # Initialize final layers of action and value heads with smaller weights
        # This helps with initial policy stability
        for module in [self.action_head[-1], self.value_head[-1]]:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=0.01)
                nn.init.zeros_(module.bias)
    
    def forward(self, hetero_data: HeteroData, valid_action_pairs: Optional[List[Tuple[int, int]]] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[Tuple[int, int]]]:
        """
        Forward pass through the HGT network with optimized action selection.
        
        Args:
            hetero_data: HeteroData object containing the graph
            valid_action_pairs: Optional list of valid (op_idx, machine_idx) pairs from environment.
                               If provided, logits will be computed only for these pairs (performance optimization).
            
        Returns:
            action_logits: Logits for valid (operation, machine) pairs [num_valid_actions]
            value: Global value estimate [1]
            action_mask: Boolean mask for valid actions [num_total_possible_actions]
            valid_action_pairs: List of (op_idx, machine_idx) pairs corresponding to logits
        """
        # Extract node features and edge indices
        op_x = hetero_data['op'].x  # [num_ops, op_feature_dim]
        machine_x = hetero_data['machine'].x  # [num_machines, machine_feature_dim]
        
        # Embed node features to hidden dimension
        op_embeddings = self.op_embedding(op_x)  # [num_ops, hidden_dim]
        machine_embeddings = self.machine_embedding(machine_x)  # [num_machines, hidden_dim]
        
        # Apply layer normalization
        op_embeddings = self.op_norm(op_embeddings)
        machine_embeddings = self.machine_norm(machine_embeddings)
        
        # Create node embedding dictionary for HGT
        x_dict = {
            'op': op_embeddings,
            'machine': machine_embeddings
        }
        
        # Extract edge indices (handle missing edge types gracefully)
        edge_index_dict = {}
        
        # Job-level precedence (conjunctive arcs)
        if ('op', 'precedes', 'op') in hetero_data.edge_types:
            edge_index_dict[('op', 'precedes', 'op')] = hetero_data['op', 'precedes', 'op'].edge_index
            
        # Machine-level precedence (disjunctive arcs) - CRITICAL for temporal relationships
        if ('op', 'machine_precedes', 'op') in hetero_data.edge_types:
            edge_index_dict[('op', 'machine_precedes', 'op')] = hetero_data['op', 'machine_precedes', 'op'].edge_index
            
        # Compatibility edges
        if ('op', 'on_machine', 'machine') in hetero_data.edge_types:
            edge_index_dict[('op', 'on_machine', 'machine')] = hetero_data['op', 'on_machine', 'machine'].edge_index
        if ('machine', 'can_process', 'op') in hetero_data.edge_types:
            edge_index_dict[('machine', 'can_process', 'op')] = hetero_data['machine', 'can_process', 'op'].edge_index
            
        # Assignment edges (created dynamically during scheduling)
        if ('op', 'assigned_to', 'machine') in hetero_data.edge_types:
            edge_index_dict[('op', 'assigned_to', 'machine')] = hetero_data['op', 'assigned_to', 'machine'].edge_index
        if ('machine', 'processes', 'op') in hetero_data.edge_types:
            edge_index_dict[('machine', 'processes', 'op')] = hetero_data['machine', 'processes', 'op'].edge_index
        
        # Extract edge attributes for temporal encoding
        edge_attr_dict = {}
        if ('op', 'machine_precedes', 'op') in hetero_data.edge_types:
            edge_data = hetero_data['op', 'machine_precedes', 'op']
            if hasattr(edge_data, 'edge_attr') and edge_data.edge_attr is not None:
                edge_attr_dict[('op', 'machine_precedes', 'op')] = edge_data.edge_attr
        
        # Apply Temporal HGT layers with edge attributes
        for hgt_layer in self.hgt_layers:
            x_dict = hgt_layer(x_dict, edge_index_dict, edge_attr_dict)
            # Apply dropout and residual connections
            x_dict['op'] = F.dropout(x_dict['op'], p=self.dropout, training=self.training)
            x_dict['machine'] = F.dropout(x_dict['machine'], p=self.dropout, training=self.training)
        
        # Get final embeddings
        final_op_embeddings = x_dict['op']  # [num_ops, hidden_dim]
        final_machine_embeddings = x_dict['machine']  # [num_machines, hidden_dim]
        
        # Use provided valid action pairs or compute from graph (optimized path vs fallback)
        if valid_action_pairs is not None:
            # OPTIMIZED PATH: Use environment-provided valid actions (eliminates graph analysis)
            valid_pairs_to_use = valid_action_pairs
        else:
            # FALLBACK PATH: Compute from graph (for compatibility)
            valid_pairs_to_use = self._get_valid_action_pairs(hetero_data)
        
        # Compute action logits efficiently for valid pairs only
        action_logits = []
        for op_idx, machine_idx in valid_pairs_to_use:
            # Concatenate operation and machine embeddings
            combined_embedding = torch.cat([
                final_op_embeddings[op_idx],
                final_machine_embeddings[machine_idx]
            ], dim=0)  # [2 * hidden_dim]
            
            # Compute logit for this (op, machine) pair
            logit = self.action_head(combined_embedding)  # [1]
            action_logits.append(logit)
        
        if action_logits:
            action_logits = torch.stack(action_logits).squeeze(-1)  # [num_valid_actions]
        else:
            # No valid actions - return empty tensor
            action_logits = torch.empty(0, device=op_x.device)
        
        # Compute global value estimate using mean pooling of all node embeddings
        all_embeddings = torch.cat([
            final_op_embeddings.mean(dim=0, keepdim=True),
            final_machine_embeddings.mean(dim=0, keepdim=True)
        ], dim=0)  # [2, hidden_dim]
        
        global_embedding = all_embeddings.mean(dim=0)  # [hidden_dim]
        value = self.value_head(global_embedding)  # [1]
        
        # Create full action mask for all possible (op, machine) pairs
        num_ops = op_x.shape[0]
        num_machines = machine_x.shape[0]
        action_mask = self._create_full_action_mask(hetero_data, valid_pairs_to_use, num_ops, num_machines)
        
        return action_logits, value, action_mask, valid_pairs_to_use
    
    def _get_valid_action_pairs(self, hetero_data: HeteroData) -> List[Tuple[int, int]]:
        """
        Extract valid (operation, machine) action pairs from the graph.
        
        Args:
            hetero_data: HeteroData object containing the graph
            
        Returns:
            List of valid (op_idx, machine_idx) pairs
        """
        valid_pairs = []
        
        # Get operation status (column 5 in operation features)
        op_status = hetero_data['op'].x[:, 5]  # [num_ops]
        ready_ops = torch.where(op_status == 0)[0]  # Operations with status == 0 (unscheduled)
        
        # For each ready operation, find its compatible machines
        if ('op', 'on_machine', 'machine') in hetero_data.edge_types:
            edge_index = hetero_data['op', 'on_machine', 'machine'].edge_index
            
            for op_idx in ready_ops:
                # Find machines connected to this operation
                machine_indices = edge_index[1][edge_index[0] == op_idx]
                for machine_idx in machine_indices:
                    valid_pairs.append((int(op_idx.item()), int(machine_idx.item())))
        
        return valid_pairs
    
    def _create_full_action_mask(self, hetero_data: HeteroData, valid_pairs: List[Tuple[int, int]], 
                                num_ops: int, num_machines: int) -> torch.Tensor:
        """
        Create a full action mask for all possible (op, machine) combinations.
        
        This is used for environments that expect a fixed-size action space.
        """
        # Create mapping from (op, machine) pair to linear index
        total_actions = 0
        pair_to_idx = {}
        
        # Build mapping based on actual compatibility (from edges)
        if ('op', 'on_machine', 'machine') in hetero_data.edge_types:
            edge_index = hetero_data['op', 'on_machine', 'machine'].edge_index
            for i in range(edge_index.shape[1]):
                op_idx = int(edge_index[0, i].item())
                machine_idx = int(edge_index[1, i].item())
                pair_to_idx[(op_idx, machine_idx)] = total_actions
                total_actions += 1
        
        # Create boolean mask
        if total_actions > 0:
            action_mask = torch.zeros(total_actions, dtype=torch.bool, device=hetero_data['op'].x.device)
            for op_idx, machine_idx in valid_pairs:
                if (op_idx, machine_idx) in pair_to_idx:
                    action_mask[pair_to_idx[(op_idx, machine_idx)]] = True
        else:
            action_mask = torch.empty(0, dtype=torch.bool, device=hetero_data['op'].x.device)
        
        return action_mask
    
    def get_action_logits_for_pairs(self, hetero_data: HeteroData, 
                                   action_pairs: List[Tuple[int, int]]) -> torch.Tensor:
        """
        Get action logits for specific (operation, machine) pairs.
        
        This is useful for environments with custom action mappings.
        
        Args:
            hetero_data: HeteroData object containing the graph
            action_pairs: List of (op_idx, machine_idx) pairs to evaluate
            
        Returns:
            Logits for the specified pairs [len(action_pairs)]
        """
        # Run forward pass to get embeddings
        _, _, _, _ = self.forward(hetero_data)
        
        # Extract embeddings (recompute to avoid storing intermediate results)
        op_x = hetero_data['op'].x
        machine_x = hetero_data['machine'].x
        
        op_embeddings = self.op_norm(self.op_embedding(op_x))
        machine_embeddings = self.machine_norm(self.machine_embedding(machine_x))
        
        x_dict = {'op': op_embeddings, 'machine': machine_embeddings}
        
        # Extract edge indices (consistent with forward method)
        edge_index_dict = {}
        
        # Job-level precedence (conjunctive arcs)
        if ('op', 'precedes', 'op') in hetero_data.edge_types:
            edge_index_dict[('op', 'precedes', 'op')] = hetero_data['op', 'precedes', 'op'].edge_index
            
        # Machine-level precedence (disjunctive arcs) - CRITICAL for temporal relationships
        if ('op', 'machine_precedes', 'op') in hetero_data.edge_types:
            edge_index_dict[('op', 'machine_precedes', 'op')] = hetero_data['op', 'machine_precedes', 'op'].edge_index
            
        # Compatibility edges
        if ('op', 'on_machine', 'machine') in hetero_data.edge_types:
            edge_index_dict[('op', 'on_machine', 'machine')] = hetero_data['op', 'on_machine', 'machine'].edge_index
        if ('machine', 'can_process', 'op') in hetero_data.edge_types:
            edge_index_dict[('machine', 'can_process', 'op')] = hetero_data['machine', 'can_process', 'op'].edge_index
            
        # Assignment edges (created dynamically during scheduling)
        if ('op', 'assigned_to', 'machine') in hetero_data.edge_types:
            edge_index_dict[('op', 'assigned_to', 'machine')] = hetero_data['op', 'assigned_to', 'machine'].edge_index
        if ('machine', 'processes', 'op') in hetero_data.edge_types:
            edge_index_dict[('machine', 'processes', 'op')] = hetero_data['machine', 'processes', 'op'].edge_index
        
        # Extract edge attributes for temporal encoding (consistent with forward method)
        edge_attr_dict = {}
        if ('op', 'machine_precedes', 'op') in hetero_data.edge_types:
            edge_data = hetero_data['op', 'machine_precedes', 'op']
            if hasattr(edge_data, 'edge_attr') and edge_data.edge_attr is not None:
                edge_attr_dict[('op', 'machine_precedes', 'op')] = edge_data.edge_attr
        
        # Apply Temporal HGT layers with edge attributes
        for hgt_layer in self.hgt_layers:
            x_dict = hgt_layer(x_dict, edge_index_dict, edge_attr_dict)
            x_dict['op'] = F.dropout(x_dict['op'], p=self.dropout, training=self.training)
            x_dict['machine'] = F.dropout(x_dict['machine'], p=self.dropout, training=self.training)
        
        final_op_embeddings = x_dict['op']
        final_machine_embeddings = x_dict['machine']
        
        # Compute logits for specified pairs
        logits = []
        for op_idx, machine_idx in action_pairs:
            combined_embedding = torch.cat([
                final_op_embeddings[op_idx],
                final_machine_embeddings[machine_idx]
            ], dim=0)
            logit = self.action_head(combined_embedding)
            logits.append(logit)
        
        if logits:
            return torch.stack(logits).squeeze(-1)
        else:
            return torch.empty(0, device=op_x.device)


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

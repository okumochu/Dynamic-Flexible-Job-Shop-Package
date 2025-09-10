import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.data import HeteroData, Batch
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
                 op_feature_dim: int,  # Dynamic: determined from graph state structure
                 machine_feature_dim: int,  # Dynamic: determined from graph state structure
                 job_feature_dim: int,  # Dynamic: determined from graph state structure
                 hidden_dim: int,  # From config.graph_rl_params['hidden_dim']
                 num_hgt_layers: int,  # From config.graph_rl_params['num_hgt_layers']
                 num_heads: int,  # From config.graph_rl_params['num_heads']
                 dropout: float,  # From config.graph_rl_params['dropout']
                 temporal_dim: int = 16,  # From config.graph_rl_params['temporal_dim']
                 max_temporal_freq: float = 1000.0):  # From config.graph_rl_params['max_temporal_freq']
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
        
        # Input embedding layers for different node types (EXTENDED: includes job nodes)
        self.op_embedding = Linear(op_feature_dim, hidden_dim, bias=True, weight_initializer='glorot')
        self.machine_embedding = Linear(machine_feature_dim, hidden_dim, bias=True, weight_initializer='glorot')
        self.job_embedding = Linear(job_feature_dim, hidden_dim, bias=True, weight_initializer='glorot')  # NEW
        
        # Temporal HGT layers for message passing with RTE (EXTENDED: hierarchical structure)
        self.hgt_layers = nn.ModuleList()
        metadata = (
            ['op', 'machine', 'job'],  # NEW: Added job nodes for hierarchical structure
            [
                # Job-operation hierarchy (NEW: hierarchical edges)
                ('job', 'contains', 'op'),
                ('op', 'belongs_to', 'job'),
                
                # Operation precedence and machine scheduling
                ('op', 'precedes', 'op'),
                ('op', 'machine_precedes', 'op'),  # CRITICAL: Disjunctive arcs with temporal info
                
                # Operation-machine compatibility and assignment
                ('op', 'on_machine', 'machine'),
                ('op', 'assigned_to', 'machine'),
                ('machine', 'can_process', 'op'),
                ('machine', 'processes', 'op')
            ]
        )
        
        for _ in range(num_hgt_layers):
            self.hgt_layers.append(
                TemporalHGTConv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    metadata=metadata,
                    heads=num_heads,
                    temporal_dim=temporal_dim,  # From config parameter
                    max_temporal_freq=max_temporal_freq  # From config parameter
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
        
        # Layer normalization (EXTENDED: includes job nodes)
        self.op_norm = nn.LayerNorm(hidden_dim)
        self.machine_norm = nn.LayerNorm(hidden_dim)
        self.job_norm = nn.LayerNorm(hidden_dim)  # NEW
        
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
    
    def forward(self, hetero_data: HeteroData, valid_action_pairs: Optional[List[Tuple[int, int]]] = None) -> Tuple[torch.Tensor, torch.Tensor, None, List[Tuple[int, int]]]:
        """
        Forward pass through the HGT network with optimized action selection.
        
        Args:
            hetero_data: HeteroData object containing the graph
            valid_action_pairs: Optional list of valid (op_idx, machine_idx) pairs from environment.
                               If provided, logits will be computed only for these pairs (performance optimization).
            
        Returns:
            action_logits: Logits for valid (operation, machine) pairs [num_valid_actions]
            value: Global value estimate [1]
            action_mask: None (not used, saves memory)
            valid_action_pairs: List of (op_idx, machine_idx) pairs corresponding to logits
        """
        # Extract node features and edge indices (EXTENDED: includes job nodes)
        op_x = hetero_data['op'].x  # [num_ops, op_feature_dim]
        machine_x = hetero_data['machine'].x  # [num_machines, machine_feature_dim]
        job_x = hetero_data['job'].x  # [num_jobs, job_feature_dim] NEW
        
        # Embed node features to hidden dimension
        op_embeddings = self.op_embedding(op_x)  # [num_ops, hidden_dim]
        machine_embeddings = self.machine_embedding(machine_x)  # [num_machines, hidden_dim]
        job_embeddings = self.job_embedding(job_x)  # [num_jobs, hidden_dim] NEW
        
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
        
        # Apply Temporal HGT layers with edge attributes (EXTENDED: includes job embeddings)
        for hgt_layer in self.hgt_layers:
            x_dict = hgt_layer(x_dict, edge_index_dict, edge_attr_dict)
            # Apply dropout to all node types
            x_dict['op'] = F.dropout(x_dict['op'], p=self.dropout, training=self.training)
            x_dict['machine'] = F.dropout(x_dict['machine'], p=self.dropout, training=self.training)
            x_dict['job'] = F.dropout(x_dict['job'], p=self.dropout, training=self.training)  # NEW
        
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
        
        # Compute global value estimate using mean pooling of all node embeddings (EXTENDED: includes jobs)
        final_job_embeddings = x_dict['job']  # [num_jobs, hidden_dim] NEW
        
        all_embeddings = torch.cat([
            final_op_embeddings.mean(dim=0, keepdim=True),
            final_machine_embeddings.mean(dim=0, keepdim=True),
            final_job_embeddings.mean(dim=0, keepdim=True)  # NEW: job-level global information
        ], dim=0)  # [3, hidden_dim]
        
        global_embedding = all_embeddings.mean(dim=0)  # [hidden_dim]
        value = self.value_head(global_embedding)  # [1]
        
        return action_logits, value, None, valid_pairs_to_use
    
    def forward_batch(self, batch: Batch, batch_valid_action_pairs: List[List[Tuple[int, int]]]) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        PERFORMANCE CRITICAL: Batched forward pass for multiple graphs.
        
        This method processes multiple HeteroData graphs in parallel, providing
        significant GPU utilization improvements over sequential processing.
        
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
        
        # Extract edge indices (handle missing edge types gracefully)
        edge_index_dict = {}
        edge_attr_dict = {}
        
        # Process all edge types from the batch
        for edge_type in batch.edge_types:
            if hasattr(batch[edge_type], 'edge_index'):
                edge_index_dict[edge_type] = batch[edge_type].edge_index
                
                # Handle edge attributes for temporal encoding
                if edge_type == ('op', 'machine_precedes', 'op') and hasattr(batch[edge_type], 'edge_attr'):
                    edge_attr_dict[edge_type] = batch[edge_type].edge_attr
        
        # Apply Temporal HGT layers with edge attributes (EXTENDED: includes job embeddings)
        for hgt_layer in self.hgt_layers:
            x_dict = hgt_layer(x_dict, edge_index_dict, edge_attr_dict)
            # Apply dropout to all node types
            x_dict['op'] = F.dropout(x_dict['op'], p=self.dropout, training=self.training)
            x_dict['machine'] = F.dropout(x_dict['machine'], p=self.dropout, training=self.training)
            x_dict['job'] = F.dropout(x_dict['job'], p=self.dropout, training=self.training)  # NEW
        
        # Get final embeddings (EXTENDED: includes job embeddings)
        final_op_embeddings = x_dict['op']  # [total_ops_across_batch, hidden_dim]
        final_machine_embeddings = x_dict['machine']  # [total_machines_across_batch, hidden_dim]
        final_job_embeddings = x_dict['job']  # [total_jobs_across_batch, hidden_dim] NEW
        
        # Split embeddings back to individual graphs using batch indices
        op_batch_slices = []
        machine_batch_slices = []
        job_batch_slices = []  # NEW
        
        # Get batch information
        op_batch = batch['op'].batch  # [total_ops_across_batch] - which graph each op belongs to
        machine_batch = batch['machine'].batch  # [total_machines_across_batch] - which graph each machine belongs to
        job_batch = batch['job'].batch  # [total_jobs_across_batch] - which graph each job belongs to NEW
        batch_size = batch.num_graphs
        
        # Split embeddings by graph
        for graph_idx in range(batch_size):
            op_mask = (op_batch == graph_idx)
            machine_mask = (machine_batch == graph_idx)
            job_mask = (job_batch == graph_idx)  # NEW
            
            op_batch_slices.append(final_op_embeddings[op_mask])
            machine_batch_slices.append(final_machine_embeddings[machine_mask])
            job_batch_slices.append(final_job_embeddings[job_mask])  # NEW
        
        # Compute action logits for each graph
        batch_action_logits = []
        batch_values = []
        
        for graph_idx in range(batch_size):
            graph_op_embeddings = op_batch_slices[graph_idx]  # [num_ops_in_graph, hidden_dim]
            graph_machine_embeddings = machine_batch_slices[graph_idx]  # [num_machines_in_graph, hidden_dim]
            graph_job_embeddings = job_batch_slices[graph_idx]  # [num_jobs_in_graph, hidden_dim] NEW
            valid_pairs = batch_valid_action_pairs[graph_idx]
            
            # Compute action logits for this graph (vectorized where possible)
            if valid_pairs:
                # Use vectorized approach when we have valid pairs
                op_indices = torch.tensor([pair[0] for pair in valid_pairs], dtype=torch.long, device=graph_op_embeddings.device)
                machine_indices = torch.tensor([pair[1] for pair in valid_pairs], dtype=torch.long, device=graph_machine_embeddings.device)
                
                # Extract embeddings for valid pairs
                valid_op_embeds = graph_op_embeddings[op_indices]  # [num_valid_actions, hidden_dim]
                valid_machine_embeds = graph_machine_embeddings[machine_indices]  # [num_valid_actions, hidden_dim]
                
                # Concatenate and compute logits in batch
                combined_embeddings = torch.cat([valid_op_embeds, valid_machine_embeds], dim=1)  # [num_valid_actions, 2*hidden_dim]
                action_logits = self.action_head(combined_embeddings).squeeze(-1)  # [num_valid_actions]
            else:
                action_logits = torch.empty(0, device=op_x.device)
            
            batch_action_logits.append(action_logits)
            
            # Compute value estimate for this graph (EXTENDED: includes job embeddings)
            op_pool = graph_op_embeddings.mean(dim=0)  # [hidden_dim]
            machine_pool = graph_machine_embeddings.mean(dim=0)  # [hidden_dim]
            job_pool = graph_job_embeddings.mean(dim=0)  # [hidden_dim] NEW
            
            all_embeddings = torch.cat([op_pool.unsqueeze(0), machine_pool.unsqueeze(0), job_pool.unsqueeze(0)], dim=0)  # [3, hidden_dim]
            global_embedding = all_embeddings.mean(dim=0)  # [hidden_dim]
            value = self.value_head(global_embedding)  # [1]
            batch_values.append(value)
        
        # Stack values into a single tensor
        batch_values = torch.stack(batch_values).squeeze(-1)  # [batch_size]
        
        return batch_action_logits, batch_values
    
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

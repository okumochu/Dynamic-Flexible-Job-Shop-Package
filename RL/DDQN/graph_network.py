import torch
import torch.nn as nn
from typing import List, Tuple
from torch_geometric.data import HeteroData, Batch
from torch_geometric.nn import HGTConv, Linear, MLP
from torch_geometric.nn.norm import LayerNorm as PygLayerNorm
from torch_geometric.utils import scatter


class HGTQNetwork(nn.Module):
    """
    HGT-based Q-network for DDQN.

    Reuses the PPO backbone (embeddings + HGT) but replaces heads with a single
    Q-head that scores (op, machine) actions given global context.
    """

    def __init__(self,
                 op_feature_dim: int,
                 machine_feature_dim: int,
                 job_feature_dim: int,
                 hidden_dim: int,
                 num_hgt_layers: int,
                 num_heads: int,
                 dropout: float):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_hgt_layers = num_hgt_layers

        # Input embeddings
        self.op_embedding = Linear(op_feature_dim, hidden_dim, bias=True)
        self.machine_embedding = Linear(machine_feature_dim, hidden_dim, bias=True)
        self.job_embedding = Linear(job_feature_dim, hidden_dim, bias=True)

        # Norms
        self.op_norm = PygLayerNorm(hidden_dim)
        self.machine_norm = PygLayerNorm(hidden_dim)
        self.job_norm = PygLayerNorm(hidden_dim)

        # HGT layers and per-layer norms
        metadata = (
            ['op', 'machine', 'job'],
            [
                ('job', 'contains', 'op'),
                ('op', 'belongs_to', 'job'),
                ('op', 'precedes', 'op'),
                ('op', 'machine_precedes', 'op'),
                ('op', 'on_machine', 'machine'),
                ('op', 'assigned_to', 'machine'),
                ('machine', 'can_process', 'op'),
                ('machine', 'processes', 'op')
            ]
        )

        self.hgt_layers = nn.ModuleList()
        self.per_layer_norms = nn.ModuleList()
        self.residual_dropout = nn.Dropout(dropout)

        for _ in range(num_hgt_layers):
            self.hgt_layers.append(
                HGTConv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    metadata=metadata,
                    heads=num_heads
                )
            )
            self.per_layer_norms.append(
                nn.ModuleDict({
                    'op': PygLayerNorm(hidden_dim),
                    'machine': PygLayerNorm(hidden_dim),
                    'job': PygLayerNorm(hidden_dim)
                })
            )

        # Q head: [op, machine, global(3H)] -> scalar with deeper architecture
        self.q_head = MLP(
            channel_list=[5 * hidden_dim, hidden_dim, hidden_dim, hidden_dim // 2, 1],
            dropout=dropout,
            act='tanh',
            norm=None
        )

    def forward(self, obs, valid_actions=None, use_batch=False) -> List[torch.Tensor]:
        """
        Returns list of Q-value tensors, one per graph, aligned with provided valid_actions lists.
        """
        if use_batch:
            batch = obs
            batch_valid_action_pairs = valid_actions
        else:
            batch = Batch.from_data_list([obs])
            batch_valid_action_pairs = [valid_actions] if valid_actions else [[]]

        return self._process_batch(batch, batch_valid_action_pairs)

    def _process_batch(self, batch: Batch, batch_valid_action_pairs: List[List[Tuple[int, int]]]) -> List[torch.Tensor]:
        op_x = batch['op'].x
        machine_x = batch['machine'].x
        job_x = batch['job'].x

        op_embeddings = self.op_embedding(op_x)
        machine_embeddings = self.machine_embedding(machine_x)
        job_embeddings = self.job_embedding(job_x)

        op_embeddings = self.op_norm(op_embeddings)
        machine_embeddings = self.machine_norm(machine_embeddings)
        job_embeddings = self.job_norm(job_embeddings)

        x_dict = {
            'op': op_embeddings,
            'machine': machine_embeddings,
            'job': job_embeddings
        }

        edge_index_dict = {edge_type: batch[edge_type].edge_index for edge_type in batch.edge_types}

        for layer_index, hgt_layer in enumerate(self.hgt_layers):
            residual_dict = {k: v for k, v in x_dict.items()}
            out_dict = hgt_layer(x_dict, edge_index_dict)
            x_dict = {
                'op': self.per_layer_norms[layer_index]['op'](self.residual_dropout(out_dict['op'] + residual_dict['op'])),
                'machine': self.per_layer_norms[layer_index]['machine'](self.residual_dropout(out_dict['machine'] + residual_dict['machine'])),
                'job': self.per_layer_norms[layer_index]['job'](self.residual_dropout(out_dict['job'] + residual_dict['job']))
            }

        final_op_embeddings = x_dict['op']
        final_machine_embeddings = x_dict['machine']
        final_job_embeddings = x_dict['job']

        op_batch = batch['op'].batch
        machine_batch = batch['machine'].batch
        job_batch = batch['job'].batch

        # Mean pooling per node type to form global context [B, 3H]
        op_mean = scatter(final_op_embeddings, op_batch, dim=0, reduce='mean')
        machine_mean = scatter(final_machine_embeddings, machine_batch, dim=0, reduce='mean')
        job_mean = scatter(final_job_embeddings, job_batch, dim=0, reduce='mean')
        global_embedding_batched = torch.cat([op_mean, machine_mean, job_mean], dim=1)

        device = final_op_embeddings.device

        num_actions_per_graph = torch.tensor(
            [len(pairs) for pairs in batch_valid_action_pairs],
            device=device,
            dtype=torch.long
        )

        op_offsets = batch['op'].ptr[:-1]
        machine_offsets = batch['machine'].ptr[:-1]

        global_op_indices = torch.cat([
            (torch.tensor([pair[0] for pair in pairs], device=device, dtype=torch.long) + op_offsets[i])
            for i, pairs in enumerate(batch_valid_action_pairs)
        ]) if len(batch_valid_action_pairs) > 0 else torch.empty(0, dtype=torch.long, device=device)

        global_machine_indices = torch.cat([
            (torch.tensor([pair[1] for pair in pairs], device=device, dtype=torch.long) + machine_offsets[i])
            for i, pairs in enumerate(batch_valid_action_pairs)
        ]) if len(batch_valid_action_pairs) > 0 else torch.empty(0, dtype=torch.long, device=device)

        valid_op_embeds = final_op_embeddings[global_op_indices]
        valid_machine_embeds = final_machine_embeddings[global_machine_indices]

        broadcast_global = torch.repeat_interleave(
            global_embedding_batched,
            num_actions_per_graph,
            dim=0
        )

        combined_embeddings = torch.cat(
            [valid_op_embeds, valid_machine_embeds, broadcast_global], dim=1
        )

        all_q_values = self.q_head(combined_embeddings).squeeze(-1)

        batch_q_values = list(torch.split(all_q_values, num_actions_per_graph.tolist()))
        return batch_q_values



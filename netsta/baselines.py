"""
Baseline GNN-shaped models for benchmarking against NetSTA.

All baseline models expose the same interface as `NetSTAModel`:
  - forward(x, edge_index, edge_attr, batch) -> Dict[str, Tensor]
  - compute_loss(preds, targets) -> (total, per_task)
  - heads: nn.ModuleDict (so train_epoch/validate can introspect active tasks)
  - get_param_groups() -> list of dicts for the optimizer

Tasks share NetSTA's SlackHead / CriticalPathHead so loss math is identical.
Sklearn baselines (Random Forest / Linear) are not nn.Modules and live in the
benchmark script directly.
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    BatchNorm,
    GCNConv,
    SAGEConv,
    global_max_pool,
    global_mean_pool,
)

from .model import CriticalPathHead, SlackHead


def _build_heads(
    in_dim: int,
    hidden: int,
    dropout: float,
    pos_weight_cap: float,
    slack_mean: float = 0.0,
    slack_std: float = 1.0,
):
    return nn.ModuleDict(
        {
            "slack": SlackHead(
                in_dim, hidden, dropout,
                slack_mean=slack_mean, slack_std=slack_std,
            ),
            "critical_path": CriticalPathHead(in_dim, hidden, dropout, pos_weight_cap),
        }
    )


def _weighted_total(per_task: Dict[str, torch.Tensor], weights: Dict[str, float]):
    out = None
    for name, val in per_task.items():
        w = weights.get(name, 1.0)
        out = w * val if out is None else out + w * val
    return out


class _BaselineMixin:
    """Common compute_loss / get_param_groups for the GNN-shaped baselines."""

    task_weights: Dict[str, float]
    heads: nn.ModuleDict
    base_lr: float

    def compute_loss(self, preds, targets):
        per_task = {
            name: self.heads[name].loss(preds[name], targets[name])
            for name in self.heads
            if name in targets
        }
        if not per_task:
            raise ValueError("compute_loss called with no matching targets")
        return _weighted_total(per_task, self.task_weights), per_task

    def get_param_groups(self):
        return [{"params": list(self.parameters()), "lr": self.base_lr}]


class MLPBaseline(_BaselineMixin, nn.Module):
    """Per-node 3-layer MLP with graph-mean/max context. No message passing.

    Each node sees its own features concatenated with mean+max pooled
    graph-level features, then a small MLP predicts each task. This is the
    "does the graph structure matter?" baseline -- the model can use bulk
    graph statistics but no neighbor signal.
    """

    def __init__(self, node_feature_dim, hidden=128, dropout=0.1, lr=1e-3,
                 task_weights=None, pos_weight_cap=10.0,
                 slack_mean: float = 0.0, slack_std: float = 1.0):
        super().__init__()
        in_dim = node_feature_dim * 3  # [self, mean, max]
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ELU(),
        )
        self.heads = _build_heads(
            hidden, hidden, dropout, pos_weight_cap,
            slack_mean=slack_mean, slack_std=slack_std,
        )
        self.task_weights = task_weights or {"slack": 0.5, "critical_path": 0.5}
        self.base_lr = lr

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        mean_g = global_mean_pool(x, batch)
        max_g = global_max_pool(x, batch)
        per_node_mean = mean_g[batch]
        per_node_max = max_g[batch]
        h = self.encoder(torch.cat([x, per_node_mean, per_node_max], dim=-1))
        graph_emb = torch.cat([global_mean_pool(h, batch), global_max_pool(h, batch)], -1)
        out = {name: head(h) for name, head in self.heads.items()}
        out["_node_emb"] = h
        out["_graph_emb"] = graph_emb
        return out


class _MessagePassingBaseline(_BaselineMixin, nn.Module):
    """Shared chassis for the GCN and GraphSAGE baselines."""

    def __init__(
        self,
        conv_factory,
        node_feature_dim: int,
        hidden: int = 256,
        num_layers: int = 4,
        dropout: float = 0.1,
        lr: float = 1e-3,
        task_weights=None,
        pos_weight_cap: float = 10.0,
        slack_mean: float = 0.0,
        slack_std: float = 1.0,
    ):
        super().__init__()
        self.dropout = dropout
        self.input_proj = nn.Linear(node_feature_dim, hidden)
        self.convs = nn.ModuleList([conv_factory(hidden, hidden) for _ in range(num_layers)])
        self.norms = nn.ModuleList([BatchNorm(hidden) for _ in range(num_layers)])
        self.heads = _build_heads(
            hidden, hidden, dropout, pos_weight_cap,
            slack_mean=slack_mean, slack_std=slack_std,
        )
        self.task_weights = task_weights or {"slack": 0.5, "critical_path": 0.5}
        self.base_lr = lr

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        h = self.input_proj(x)
        for conv, norm in zip(self.convs, self.norms):
            h_in = h
            h = conv(h, edge_index)
            h = norm(h)
            h = F.elu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = h + h_in  # residual
        if batch is None:
            batch = torch.zeros(h.size(0), dtype=torch.long, device=h.device)
        graph_emb = torch.cat([global_mean_pool(h, batch), global_max_pool(h, batch)], -1)
        out = {name: head(h) for name, head in self.heads.items()}
        out["_node_emb"] = h
        out["_graph_emb"] = graph_emb
        return out


class GCNBaseline(_MessagePassingBaseline):
    """4-layer GCN baseline -- the 'does attention matter?' control."""

    def __init__(self, node_feature_dim, **kw):
        super().__init__(GCNConv, node_feature_dim, **kw)


class GraphSAGEBaseline(_MessagePassingBaseline):
    """4-layer GraphSAGE (mean aggregation) -- 'is GAT better than vanilla MP?'."""

    def __init__(self, node_feature_dim, **kw):
        super().__init__(lambda i, o: SAGEConv(i, o, aggr="mean"), node_feature_dim, **kw)

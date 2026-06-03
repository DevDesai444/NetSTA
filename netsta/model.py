"""
NetSTA GNN backbone and multi-task heads for EDA timing prediction.

Components:
  - NetSTABackbone: shared GATv2 encoder producing per-node and graph-level
    embeddings, with edge-feature-aware attention and residual connections.
  - TaskHead: base class for downstream prediction heads.
  - SlackHead / CriticalPathHead / CongestionHead: concrete heads.
  - NetSTAModel: wrapper that activates configured heads and combines their
    losses with configurable weights.
"""

from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    BatchNorm,
    GATv2Conv,
    GCNConv,
    MessagePassing,
    global_max_pool,
    global_mean_pool,
)
from torch_geometric.utils import softmax as pyg_softmax

from .config import NetSTAConfig


# ---------------------------------------------------------------------------
# Symmetry-aware attention
# ---------------------------------------------------------------------------


class SymmetryAwareAttention(MessagePassing):
    """GATv2-style attention with a learnable bias on matched-device edges.

    Analog circuits encode 'this edge connects matched devices' as a binary
    indicator in the last edge-feature column. This layer reads that
    indicator and adds a learnable bias `b_sym` to the attention logits for
    matched edges *before* the softmax, so message passing concentrates more
    weight on matched pairs (current mirrors, diff pairs, etc.).

    The standard GATv2 path is preserved when the matching column is 0 (the
    bias contributes nothing), so the layer is a drop-in replacement for
    digital circuits too.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int,
        edge_dim: int,
        dropout: float = 0.1,
        matching_idx: int = -1,
        negative_slope: float = 0.2,
    ):
        super().__init__(aggr="add", node_dim=0)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.dropout = dropout
        self.matching_idx = matching_idx
        self.negative_slope = negative_slope

        # GATv2 parametrization: separate projections for source and target.
        self.lin_l = torch.nn.Linear(in_channels, heads * out_channels, bias=False)
        self.lin_r = torch.nn.Linear(in_channels, heads * out_channels, bias=False)
        self.lin_e = torch.nn.Linear(edge_dim, heads * out_channels, bias=False)
        self.att = torch.nn.Parameter(torch.empty(1, heads, out_channels))
        # Learnable bias added to attention logits for matched-pair edges.
        self.symmetry_bias = torch.nn.Parameter(torch.zeros(heads))
        torch.nn.init.xavier_uniform_(self.att)

    def forward(self, x, edge_index, edge_attr=None):
        h_l = self.lin_l(x).view(-1, self.heads, self.out_channels)
        h_r = self.lin_r(x).view(-1, self.heads, self.out_channels)
        if edge_attr is None or edge_attr.numel() == 0:
            # Build a zero-padded edge_attr so message() can still run.
            num_edges = edge_index.size(1)
            edge_attr = torch.zeros(
                num_edges, self.lin_e.in_features,
                device=x.device, dtype=x.dtype,
            )
        out = self.propagate(edge_index, x=(h_l, h_r), edge_attr=edge_attr)
        return out.reshape(-1, self.heads * self.out_channels)

    def message(self, x_i, x_j, edge_attr, index, ptr, size_i):
        # x_i = h_r at target, x_j = h_l at source (PyG passes x as tuple).
        e = self.lin_e(edge_attr).view(-1, self.heads, self.out_channels)
        # GATv2 attention logit.
        x = F.leaky_relu(x_i + x_j + e, self.negative_slope)
        alpha = (x * self.att).sum(dim=-1)  # [E, heads]
        # Symmetry bias: pull the matching indicator out and add.
        match = edge_attr[:, self.matching_idx:self.matching_idx + 1]  # [E, 1]
        alpha = alpha + self.symmetry_bias.unsqueeze(0) * match
        alpha = pyg_softmax(alpha, index, ptr=ptr, num_nodes=size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_j * alpha.unsqueeze(-1)


# ---------------------------------------------------------------------------
# Backbone
# ---------------------------------------------------------------------------


class NetSTABackbone(nn.Module):
    """Shared GATv2 encoder.

    Produces per-node embeddings and an optional graph-level embedding built
    by concatenating mean and max pooling over node embeddings.
    """

    def __init__(self, config: NetSTAConfig):
        super().__init__()
        self.config = config
        self.num_layers = config.num_layers
        self.dropout = config.dropout

        hidden = config.hidden_dim
        heads = config.num_heads
        # GAT with concat=True multiplies output by `heads`. To keep parameter
        # counts roughly comparable when attention is ablated, GCN uses the
        # same effective channel width (hidden*heads).
        layer_out = hidden * heads
        self.use_residual = config.use_residual
        self.use_attention = config.use_attention

        self.input_proj = nn.Linear(config.node_feature_dim, layer_out if not config.use_attention else hidden)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.use_symmetry_attention = config.use_symmetry_attention
        for i in range(config.num_layers):
            if config.use_attention:
                in_ch = hidden if i == 0 else layer_out
                if config.use_symmetry_attention:
                    self.convs.append(
                        SymmetryAwareAttention(
                            in_ch, hidden, heads=heads,
                            edge_dim=config.edge_feature_dim,
                            dropout=config.dropout,
                            # matching_constraint is the LAST edge feature.
                            matching_idx=config.edge_feature_dim - 1,
                        )
                    )
                else:
                    self.convs.append(
                        GATv2Conv(
                            in_ch,
                            hidden,
                            heads=heads,
                            edge_dim=config.edge_feature_dim,
                            dropout=config.dropout,
                            concat=True,
                        )
                    )
            else:
                # Uniform message passing; GCNConv doesn't consume edge_attr.
                self.convs.append(GCNConv(layer_out, layer_out))
            self.norms.append(BatchNorm(layer_out))

        self.node_emb_dim = layer_out
        self.graph_emb_dim = 2 * layer_out

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        h = self.input_proj(x)

        for i in range(self.num_layers):
            h_in = h
            if self.use_attention:
                h = self.convs[i](h, edge_index, edge_attr=edge_attr)
            else:
                h = self.convs[i](h, edge_index)
            h = self.norms[i](h)
            h = F.elu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            # Residual only when input and output dims match. With attention
            # they only match from layer 1 onward; without attention the
            # input_proj already maps to layer_out so every layer can residual.
            if self.use_residual and (i > 0 or not self.use_attention):
                h = h + h_in

        if batch is None:
            batch = torch.zeros(h.size(0), dtype=torch.long, device=h.device)

        graph_emb = torch.cat(
            [global_mean_pool(h, batch), global_max_pool(h, batch)], dim=-1
        )

        return {"node_emb": h, "graph_emb": graph_emb}


# ---------------------------------------------------------------------------
# Directional STA-aware backbone
# ---------------------------------------------------------------------------


class _TimingMessageLayer(MessagePassing):
    """One step of timing-style relaxation along a directed edge.

    Forward semantics (when called with the original edge_index):
      AT_new[child] = MAX over input drivers of
                      (AT[driver] + delay_proj(edge_attr[driver -> child]))

    The aggregator is element-wise max — directly analogous to the STA forward
    pass where arrival time at a node is the max over its input drivers of
    (driver_AT + propagation delay). Each embedding dimension can capture a
    different facet of timing (e.g. critical path through different cell
    types), so per-dimension max is a sensible generalization of scalar STA.

    For backward (RT) propagation, instantiate a second copy and call with
    edge_index flipped — and use `aggr_kind="min"`.
    """

    def __init__(self, hidden: int, edge_dim: int, aggr_kind: str = "max"):
        if aggr_kind not in ("max", "min"):
            raise ValueError(f"aggr_kind must be 'max' or 'min', got {aggr_kind}")
        # PyG's scatter_max ignores nodes with no incoming edges (returns the
        # default fill) — we re-combine with `x` outside this layer so source
        # nodes keep their initial embedding.
        super().__init__(aggr=aggr_kind, node_dim=0)
        self.aggr_kind = aggr_kind
        self.delay_proj = nn.Linear(edge_dim, hidden)
        # A small mixer applied after aggregation so the layer can shape the
        # propagated signal (otherwise a pure max-of-(parent + delay) might be
        # too restrictive a function class).
        self.mix = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ELU(),
            nn.Linear(hidden, hidden),
        )

    def forward(self, x, edge_index, edge_attr):
        agg = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        return self.mix(agg)

    def message(self, x_j, edge_attr):
        # x_j: source-node embedding (the "driver" under forward propagation,
        # or the "sink" under backward propagation since we flip edge_index).
        # edge_attr: matched per-edge features (wire delay, distance, etc.).
        delay = self.delay_proj(edge_attr)
        if self.aggr_kind == "min":
            # Backward semantics: RT propagates by SUBTRACTING delay.
            return x_j - delay
        return x_j + delay


class TimingPropagationBackbone(nn.Module):
    """STA-aware backbone with separate forward (AT) and backward (RT) passes.

    Forward pass iterates a shared `_TimingMessageLayer(aggr='max')` over the
    DAG `num_layers` times, mimicking K relaxation sweeps of arrival-time
    propagation. Backward pass iterates a separately-parameterized layer over
    the flipped DAG with min-aggregation, mimicking required-time propagation.

    Per-node output = concat([AT_embedding, RT_embedding]); the slack head's
    job is then approximately (RT - AT), which is exactly the STA formula.

    With K = `num_layers` iterations, messages can reach gates K hops deep —
    matched empirically to the dataset's typical max-depth (1-14 for 40-gate
    circuits, hence default num_layers=8 in train.py once we wire that up).
    """

    def __init__(self, config: NetSTAConfig):
        super().__init__()
        self.config = config
        self.num_iterations = config.num_layers
        hidden = config.hidden_dim
        self.dropout = config.dropout

        # Inputs are mapped into a shared "timing embedding" space; the same
        # initial embedding seeds both the AT and RT passes — they then evolve
        # in opposite directions along the DAG.
        self.input_proj = nn.Linear(config.node_feature_dim, hidden)
        self.input_norm = BatchNorm(hidden)

        self.forward_layer = _TimingMessageLayer(
            hidden, config.edge_feature_dim, aggr_kind="max",
        )
        self.backward_layer = _TimingMessageLayer(
            hidden, config.edge_feature_dim, aggr_kind="min",
        )

        # Pooled node embedding = [AT, RT]. The graph embedding doubles in
        # width because we concatenate mean + max pool over that.
        self.node_emb_dim = 2 * hidden
        self.graph_emb_dim = 2 * self.node_emb_dim

    def _iterate(
        self,
        h: torch.Tensor,
        layer: _TimingMessageLayer,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Run `num_iterations` relaxation sweeps with monotonic update.

        For forward (max-agg) propagation we keep `max(h, propagated)` so the
        AT embedding is non-decreasing — the STA invariant. Symmetric for
        backward (min-agg).
        """
        ea = edge_attr
        if ea is None or ea.numel() == 0 or edge_index.numel() == 0:
            # Disconnected graph: nothing to propagate, return input as-is.
            return h
        combine = torch.maximum if layer.aggr_kind == "max" else torch.minimum
        for _ in range(self.num_iterations):
            propagated = layer(h, edge_index, ea)
            h = combine(h, propagated)
            h = F.dropout(h, p=self.dropout, training=self.training)
        return h

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        h = self.input_proj(x)
        h = self.input_norm(h)
        h = F.elu(h)

        # Forward / AT pass over the DAG as given.
        h_at = self._iterate(h, self.forward_layer, edge_index, edge_attr)

        # Backward / RT pass over the flipped DAG.
        rev_edge_index = edge_index.flip(0) if edge_index.numel() else edge_index
        h_rt = self._iterate(h, self.backward_layer, rev_edge_index, edge_attr)

        node_emb = torch.cat([h_at, h_rt], dim=-1)
        if batch is None:
            batch = torch.zeros(node_emb.size(0), dtype=torch.long, device=node_emb.device)
        graph_emb = torch.cat(
            [global_mean_pool(node_emb, batch), global_max_pool(node_emb, batch)], dim=-1,
        )
        return {"node_emb": node_emb, "graph_emb": graph_emb}


# ---------------------------------------------------------------------------
# Task heads
# ---------------------------------------------------------------------------


class TaskHead(nn.Module):
    """Base class for downstream prediction tasks.

    Subclasses set `name` and implement `forward` (returning the prediction
    tensor) and `loss` (returning a scalar loss given prediction + target).
    """

    name: str = ""

    def forward(
        self,
        node_emb: torch.Tensor,
        graph_emb: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        raise NotImplementedError

    def loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


def _mlp(in_dim: int, hidden: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden, 1),
    )


class SlackHead(TaskHead):
    """Per-node absolute-slack regression (ns).

    The head returns predictions directly in nanoseconds. To keep training
    well-conditioned when ns values are tiny (slack std is typically 0.05 -
    0.15 ns), the loss standardizes both prediction and target by the
    train-set slack std before computing MSE — equivalent to optimizing on
    z-scores while keeping the forward API in physical units. The mean and
    std are persisted on the head via register_buffer so checkpoints
    round-trip them automatically and downstream callers (predict.py,
    streamlit app) get ns out of the box.
    """

    name = "slack"

    def __init__(
        self,
        in_dim: int,
        hidden: int,
        dropout: float,
        slack_mean: float = 0.0,
        slack_std: float = 1.0,
    ):
        super().__init__()
        self.mlp = _mlp(in_dim, hidden, dropout)
        self.register_buffer("slack_mean", torch.tensor(float(slack_mean)))
        self.register_buffer("slack_std", torch.tensor(max(float(slack_std), 1e-3)))

    def forward(self, node_emb, graph_emb=None, batch=None):
        # Predict in normalized z-space, then map back to ns for the caller.
        z = self.mlp(node_emb).squeeze(-1)
        return z * self.slack_std + self.slack_mean

    def loss(self, pred, target):
        # Cancel the additive mean in the standardization for cleaner gradients.
        return F.mse_loss(pred / self.slack_std, target / self.slack_std)


class CriticalPathHead(TaskHead):
    """Per-node critical-path classification (BCE with pos-weighting)."""

    name = "critical_path"

    def __init__(self, in_dim: int, hidden: int, dropout: float, pos_weight_cap: float):
        super().__init__()
        self.mlp = _mlp(in_dim, hidden, dropout)
        self.pos_weight_cap = pos_weight_cap

    def forward(self, node_emb, graph_emb=None, batch=None):
        return self.mlp(node_emb).squeeze(-1)

    def loss(self, pred, target):
        num_pos = target.sum()
        num_total = target.numel()
        # Avoid divide-by-zero when a batch has no positives.
        pos_weight = (num_total - num_pos) / num_pos.clamp(min=1.0)
        pos_weight = pos_weight.clamp(max=self.pos_weight_cap).to(pred.device)
        return F.binary_cross_entropy_with_logits(
            pred, target, pos_weight=pos_weight
        )


class CongestionHead(TaskHead):
    """Per-node routing-congestion regression (MSE)."""

    name = "congestion"

    def __init__(self, in_dim: int, hidden: int, dropout: float):
        super().__init__()
        self.mlp = _mlp(in_dim, hidden, dropout)

    def forward(self, node_emb, graph_emb=None, batch=None):
        return self.mlp(node_emb).squeeze(-1)

    def loss(self, pred, target):
        return F.mse_loss(pred, target)


class DRCHead(TaskHead):
    """Per-node DRC-hotspot classification with focal loss.

    Focal loss (Lin et al. 2017) down-weights well-classified examples so the
    rare hotspot class dominates the gradient even at strong imbalance.
    """

    name = "drc"

    def __init__(
        self, in_dim: int, hidden: int, dropout: float,
        focal_alpha: float = 0.25, focal_gamma: float = 2.0,
    ):
        super().__init__()
        self.mlp = _mlp(in_dim, hidden, dropout)
        self.alpha = focal_alpha
        self.gamma = focal_gamma

    def forward(self, node_emb, graph_emb=None, batch=None):
        return self.mlp(node_emb).squeeze(-1)

    def loss(self, pred, target):
        # Standard focal-loss formulation on top of BCE-with-logits.
        ce = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
        p = torch.sigmoid(pred)
        p_t = p * target + (1.0 - p) * (1.0 - target)
        focal_weight = (1.0 - p_t).pow(self.gamma)
        alpha_t = self.alpha * target + (1.0 - self.alpha) * (1.0 - target)
        return (alpha_t * focal_weight * ce).mean()


class AnalogPerformanceHead(TaskHead):
    """Per-node analog small-signal regression head.

    Predicts a 2-vector (gbw_score, parasitic_impact) per node and applies
    MSE against the corresponding 2-column ground-truth label tensor stored
    on PyG Data as `y_analog_performance`.
    """

    name = "analog_performance"

    def __init__(self, in_dim: int, hidden: int, dropout: float):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 2),
        )

    def forward(self, node_emb, graph_emb=None, batch=None):
        return self.mlp(node_emb)  # [N, 2]

    def loss(self, pred, target):
        # target is [N, 2]; if it arrived flat (legacy), reshape.
        if target.dim() == 1 and pred.dim() == 2 and pred.size(-1) == 2:
            target = target.view(-1, 2)
        return F.mse_loss(pred, target)


TASK_HEAD_REGISTRY = {
    SlackHead.name: SlackHead,
    CriticalPathHead.name: CriticalPathHead,
    CongestionHead.name: CongestionHead,
    DRCHead.name: DRCHead,
    AnalogPerformanceHead.name: AnalogPerformanceHead,
}


def _build_head(name: str, config: NetSTAConfig, in_dim: int) -> TaskHead:
    if name not in TASK_HEAD_REGISTRY:
        raise ValueError(
            f"Unknown task '{name}'. Available: {sorted(TASK_HEAD_REGISTRY)}"
        )
    cls = TASK_HEAD_REGISTRY[name]
    if cls is CriticalPathHead:
        return cls(in_dim, config.hidden_dim, config.dropout, config.critical_pos_weight_cap)
    if cls is SlackHead:
        return cls(
            in_dim, config.hidden_dim, config.dropout,
            slack_mean=config.slack_mean, slack_std=config.slack_std,
        )
    return cls(in_dim, config.hidden_dim, config.dropout)


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------


_BACKBONE_REGISTRY = {
    "gatv2": NetSTABackbone,
    "timing": TimingPropagationBackbone,
}


class NetSTAModel(nn.Module):
    """Backbone + active task heads with weighted multi-task loss.

    `config.backbone_kind` selects the encoder:
      - "gatv2"  -> the original GATv2 stack (used by ablation comparisons)
      - "timing" -> directional STA-aware propagation (default, for the
                    timing-sensitive tasks slack and critical-path)
    """

    def __init__(self, config: NetSTAConfig):
        super().__init__()
        config.validate()
        self.config = config

        backbone_cls = _BACKBONE_REGISTRY.get(config.backbone_kind)
        if backbone_cls is None:
            raise ValueError(
                f"Unknown backbone_kind '{config.backbone_kind}'. "
                f"Available: {sorted(_BACKBONE_REGISTRY)}"
            )
        self.backbone = backbone_cls(config)
        self.heads = nn.ModuleDict(
            {
                name: _build_head(name, config, self.backbone.node_emb_dim)
                for name in config.active_tasks
            }
        )
        self.task_weights = dict(config.task_weights)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        emb = self.backbone(x, edge_index, edge_attr=edge_attr, batch=batch)
        preds: Dict[str, torch.Tensor] = {}
        for name, head in self.heads.items():
            preds[name] = head(emb["node_emb"], emb["graph_emb"], batch)
        preds["_node_emb"] = emb["node_emb"]
        preds["_graph_emb"] = emb["graph_emb"]
        return preds

    def compute_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Return (total_loss, per_task_losses).

        `targets` only needs to contain entries for the active heads.
        """
        per_task: Dict[str, torch.Tensor] = {}
        total = None
        for name, head in self.heads.items():
            if name not in targets:
                continue
            ltask = head.loss(predictions[name], targets[name])
            per_task[name] = ltask
            weight = self.task_weights[name]
            total = ltask * weight if total is None else total + weight * ltask
        if total is None:
            raise ValueError("compute_loss called with no matching targets")
        return total, per_task

    def get_param_groups(
        self, base_lr: Optional[float] = None, layer_decay: Optional[float] = None
    ) -> List[dict]:
        """Build optimizer parameter groups with optional layer-wise LR decay.

        Layer-wise LR is a pretraining-fine-tuning trick: deeper (closer-to-
        head) layers get the base LR, earlier layers get LR scaled by
        `layer_decay ** distance_from_head`. Only the GATv2 backbone has a
        stack of independently-parameterized layers — the timing backbone
        uses shared weights across iterations, so it gets a single group at
        the base LR. Setting layer_decay = 1.0 also collapses to that.
        """
        lr = self.config.learning_rate if base_lr is None else base_lr
        decay = self.config.layer_decay if layer_decay is None else layer_decay

        if self.config.backbone_kind != "gatv2" or decay == 1.0:
            return [{"params": list(self.parameters()), "lr": lr}]

        n = self.config.num_layers
        groups: List[dict] = []
        groups.append(
            {"params": list(self.backbone.input_proj.parameters()), "lr": lr * (decay ** n)}
        )
        for i in range(n):
            scale = decay ** (n - 1 - i)
            params = list(self.backbone.convs[i].parameters()) + list(
                self.backbone.norms[i].parameters()
            )
            groups.append({"params": params, "lr": lr * scale})
        groups.append({"params": list(self.heads.parameters()), "lr": lr})
        return groups

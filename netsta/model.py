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
      AT_new[child] = aggr over input drivers of
                      (AT[driver] + delay_proj(edge_attr[driver -> child]))

    `aggr_kind="max"` mirrors the STA forward pass (arrival_time at a node is
    the max over its input drivers of driver_AT + propagation delay).
    `aggr_kind="min"` mirrors the backward pass (required_time at a node is
    the min over output sinks of sink_RT - propagation delay). Each embedding
    dimension can capture a different facet of timing, so per-dimension
    max/min is the natural generalization of scalar STA.

    There is no post-aggregation MLP — message + aggregation IS the entire
    layer. Keeping the recurrence purely additive (message = x_j +/- delay)
    means the per-iteration update is structurally analogous to the STA
    relaxation step, and stacking K iterations composes meaningfully (each
    iteration extends propagation by one DAG hop without an MLP twist on top
    that would let the model wander away from the STA semantics).

    During training we aggregate via temperature-controlled LogSumExp instead
    of hard max — gradient flows through every input driver in proportion to
    its contribution to the soft maximum, rather than only through the
    single arg-max input. The temperature controls the softness:
    `soft_temperature -> infinity` reduces to hard max; finite values let
    weaker contenders also receive gradient. At eval time we switch to hard
    max for STA-correct semantics.
    """

    def __init__(
        self,
        hidden: int,
        edge_dim: int,
        aggr_kind: str = "max",
        soft_temperature: float = 8.0,
    ):
        if aggr_kind not in ("max", "min"):
            raise ValueError(f"aggr_kind must be 'max' or 'min', got {aggr_kind}")
        # Register a hard aggregator with PyG; we override aggregate() so the
        # registered kind only matters as a label.
        super().__init__(aggr=aggr_kind, node_dim=0)
        self.aggr_kind = aggr_kind
        self.soft_temperature = float(soft_temperature)
        self.delay_proj = nn.Linear(edge_dim, hidden)
        # Initialize the delay projection at a small scale so the first few
        # iterations do not overwhelm the input projection's initial AT/RT
        # estimate. Large initial deltas can otherwise saturate the
        # accumulation in 1-2 iterations and starve later updates of gradient.
        nn.init.normal_(self.delay_proj.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.delay_proj.bias)

    def forward(self, x, edge_index, edge_attr):
        if edge_index.numel() == 0:
            # Disconnected graph: aggregation has no inputs anywhere, so the
            # caller's combine step will simply keep the existing embedding.
            return x
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_attr):
        # x_j is the source-node embedding (driver under forward propagation,
        # sink under backward propagation since we flip edge_index in the
        # backbone). Delay is signed so the same layer handles both modes.
        delay = self.delay_proj(edge_attr)
        if self.aggr_kind == "min":
            return x_j - delay
        return x_j + delay

    def aggregate(self, inputs, index, ptr=None, dim_size=None):
        """Override PyG's aggregator: hard max/min at eval, soft at train."""
        from torch_geometric.utils import scatter

        if not self.training:
            reduce = "max" if self.aggr_kind == "max" else "min"
            return scatter(inputs, index, dim=0, dim_size=dim_size, reduce=reduce)

        # LogSumExp with temperature beta:
        #   soft_max(z; beta) = max(z) + (1/beta) * log(sum_i exp(beta * (z_i - max)))
        # Soft min reuses the soft max on negated inputs:
        #   soft_min(z; beta) = -soft_max(-z; beta)
        beta = self.soft_temperature
        signed = inputs if self.aggr_kind == "max" else -inputs
        # Per-destination max for numerical stability.
        max_per_dst = scatter(signed, index, dim=0, dim_size=dim_size, reduce="max")
        shifted = signed - max_per_dst[index]
        exp_vals = (shifted * beta).exp()
        sum_exp = scatter(exp_vals, index, dim=0, dim_size=dim_size, reduce="sum")
        soft = max_per_dst + (1.0 / beta) * sum_exp.clamp_min(1e-12).log()
        return soft if self.aggr_kind == "max" else -soft


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
        # Projection from a per-graph summary of the AT pass (mean + max pool)
        # into the per-node seed for the backward / RT pass. Initialized at
        # small scale so early training stays close to the "no seed" baseline
        # and the model learns to use the clock-period signal incrementally
        # rather than all at once.
        self.rt_seed_proj = nn.Linear(2 * hidden, hidden)
        nn.init.normal_(self.rt_seed_proj.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.rt_seed_proj.bias)

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
        if batch is None:
            batch = torch.zeros(h.size(0), dtype=torch.long, device=h.device)

        # Forward / AT pass over the DAG. Arrival-time accumulates outward
        # from the PIs.
        h_at = self._iterate(h, self.forward_layer, edge_index, edge_attr)

        # Boundary condition for the backward pass: required_time at the POs
        # equals the clock period, which in our STA scales with max_AT and
        # therefore VARIES per graph. A purely local backward sweep cannot
        # discover this from local features alone, so we inject a per-graph
        # AT summary (mean + max pool of h_at) into every node's RT seed.
        # This gives the backward sweep the clock-period context it needs
        # while keeping each node's local embedding intact.
        graph_at = torch.cat(
            [global_mean_pool(h_at, batch), global_max_pool(h_at, batch)], dim=-1,
        )                                                       # [B, 2*hidden]
        seed_offset = self.rt_seed_proj(graph_at)               # [B, hidden]
        h_rt_seed = h + seed_offset[batch]

        rev_edge_index = edge_index.flip(0) if edge_index.numel() else edge_index
        h_rt = self._iterate(h_rt_seed, self.backward_layer, rev_edge_index, edge_attr)

        node_emb = torch.cat([h_at, h_rt], dim=-1)
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


def _ns_head(in_dim: int, dropout: float) -> nn.Sequential:
    """Single hidden-layer MLP -> 1 scalar, used by all ns regression heads.

    Kept intentionally small: the goal is for the backbone to do the
    representational work and for the head to be approximately a linear
    readout. A tiny non-linearity helps fit the residual but does not
    introduce capacity to absorb backbone mistakes.
    """
    return nn.Sequential(
        nn.Linear(in_dim, in_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(in_dim, 1),
    )


class _NsRegressionHead(TaskHead):
    """Base class for the slack / arrival_time / required_time heads.

    Each subclass sets `name` and the buffer names. The shared loss is
    z-space MSE: divide pred and target by the train-set std before MSE so
    optimization stays well-conditioned even when raw ns values are O(0.1).
    Mean/std are register_buffer slots so checkpoints round-trip the scale
    without the caller needing the stats file at inference time.

    `slice_offset` and `slice_dim` let the head read a specific contiguous
    slice of the backbone's per-node embedding. The TimingPropagationBackbone
    concatenates [AT_emb, RT_emb] of width 2 * hidden; the AT and RT heads
    use the corresponding half so they get architectural alignment with the
    quantity they predict. Slack head uses the full vector.
    """

    def __init__(
        self,
        in_dim: int,
        dropout: float,
        target_mean: float,
        target_std: float,
        slice_offset: int = 0,
        slice_dim: int = 0,
    ):
        super().__init__()
        self.slice_offset = int(slice_offset)
        self.slice_dim = int(slice_dim) if slice_dim > 0 else int(in_dim)
        self.head = _ns_head(self.slice_dim, dropout)
        self.register_buffer("target_mean", torch.tensor(float(target_mean)))
        self.register_buffer(
            "target_std", torch.tensor(max(float(target_std), 1e-3))
        )

    def _slice(self, node_emb: torch.Tensor) -> torch.Tensor:
        if self.slice_dim == node_emb.size(-1) and self.slice_offset == 0:
            return node_emb
        return node_emb[:, self.slice_offset : self.slice_offset + self.slice_dim]

    def forward(self, node_emb, graph_emb=None, batch=None):
        z = self.head(self._slice(node_emb)).squeeze(-1)
        return z * self.target_std + self.target_mean

    def loss(self, pred, target):
        # Cancel the additive mean so the gradient is independent of it.
        return F.mse_loss(pred / self.target_std, target / self.target_std)


class SlackHead(_NsRegressionHead):
    """Per-node slack regression in nanoseconds.

    Two construction modes:

    * Direct (default): a one-hidden-layer MLP regresses slack directly from
      the full per-node embedding. Used with any backbone.

    * Compositional: set `compositional=True` to derive slack as
      RT_pred - AT_pred, with the RT and AT readouts each taking a clean
      half of the directional backbone's [AT_emb, RT_emb] concatenation.
      The head then has no learnable parameters of its own beyond what the
      auxiliary AT/RT heads already learn — the STA identity is baked in
      structurally and the slack prediction is mechanically consistent with
      the AT and RT predictions. Recommended whenever the auxiliary AT/RT
      heads are active.

    Loss is Huber (smooth-L1) in z-space rather than MSE. Slack has a
    long-left-tail distribution (nodes with timing violations sit at large
    negative slack); MSE over-weights those, pulling the regression away
    from the bulk of well-behaved nodes. Huber stays quadratic for small
    residuals (so it matches MSE in the normal regime) but is linear past
    the delta threshold, capping the influence of any single outlier on
    the gradient.
    """

    name = "slack"

    def __init__(
        self,
        in_dim: int,
        hidden: int,
        dropout: float,
        slack_mean: float = 0.0,
        slack_std: float = 1.0,
        compositional: bool = False,
        huber_delta: float = 0.5,
    ):
        super().__init__(
            in_dim=in_dim, dropout=dropout,
            target_mean=slack_mean, target_std=slack_std,
        )
        # Backwards-compat buffer aliases so older checkpoints expecting
        # `slack_mean` / `slack_std` keys still load and so external tools
        # (predict.py, app.py) can inspect the scale without depending on the
        # base-class internals.
        self.register_buffer("slack_mean", self.target_mean)
        self.register_buffer("slack_std", self.target_std)
        self.compositional = bool(compositional)
        # Delta is interpreted in z-space units after dividing both pred and
        # target by target_std. delta=0.5 means "treat residuals up to 0.5 std
        # as quadratic, beyond that as linear" — fits ~38% of a unit-normal
        # distribution and clips the outlier tail aggressively.
        self.huber_delta = float(huber_delta)

    def loss(self, pred, target):
        return F.smooth_l1_loss(
            pred / self.target_std,
            target / self.target_std,
            beta=self.huber_delta,
        )

    def attach_components(
        self,
        at_head: "ArrivalTimeHead | None",
        rt_head: "RequiredTimeHead | None",
    ) -> None:
        """Wire the AT and RT heads in for compositional inference.

        Called by NetSTAModel after head construction so the slack head can
        request RT - AT from heads that already exist in the ModuleDict
        rather than carrying duplicate readouts.
        """
        self._at_head = at_head
        self._rt_head = rt_head

    def forward(self, node_emb, graph_emb=None, batch=None):
        if self.compositional and getattr(self, "_at_head", None) is not None \
                and getattr(self, "_rt_head", None) is not None:
            at_pred = self._at_head(node_emb, graph_emb, batch)
            rt_pred = self._rt_head(node_emb, graph_emb, batch)
            return rt_pred - at_pred
        z = self.head(node_emb).squeeze(-1)
        return z * self.target_std + self.target_mean


class ArrivalTimeHead(_NsRegressionHead):
    """Per-node arrival-time regression in ns.

    With the directional timing backbone, the AT head reads only the AT half
    of the per-node embedding ([0 : hidden]) so its readout is structurally
    aligned with the forward-pass embedding. With other backbones the head
    falls back to the full vector — `slice_dim = 0` selects the entire
    embedding.
    """

    name = "arrival_time"

    def __init__(
        self,
        in_dim: int,
        dropout: float,
        arrival_time_mean: float = 0.0,
        arrival_time_std: float = 1.0,
        slice_offset: int = 0,
        slice_dim: int = 0,
    ):
        super().__init__(
            in_dim=in_dim, dropout=dropout,
            target_mean=arrival_time_mean, target_std=arrival_time_std,
            slice_offset=slice_offset, slice_dim=slice_dim,
        )


class RequiredTimeHead(_NsRegressionHead):
    """Per-node required-time regression in ns.

    With the directional timing backbone, the RT head reads only the RT half
    of the per-node embedding ([hidden : 2*hidden]) so its readout is
    structurally aligned with the backward-pass embedding. With other
    backbones the head falls back to the full vector.
    """

    name = "required_time"

    def __init__(
        self,
        in_dim: int,
        dropout: float,
        required_time_mean: float = 0.0,
        required_time_std: float = 1.0,
        slice_offset: int = 0,
        slice_dim: int = 0,
    ):
        super().__init__(
            in_dim=in_dim, dropout=dropout,
            target_mean=required_time_mean, target_std=required_time_std,
            slice_offset=slice_offset, slice_dim=slice_dim,
        )


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
    ArrivalTimeHead.name: ArrivalTimeHead,
    RequiredTimeHead.name: RequiredTimeHead,
    CriticalPathHead.name: CriticalPathHead,
    CongestionHead.name: CongestionHead,
    DRCHead.name: DRCHead,
    AnalogPerformanceHead.name: AnalogPerformanceHead,
}


def _directional_slices(config: NetSTAConfig, in_dim: int):
    """Return (at_offset, at_dim, rt_offset, rt_dim) for the AT/RT readouts.

    The TimingPropagationBackbone concatenates [AT_emb, RT_emb] of width
    2 * hidden_dim. Slicing per-head aligns the readout with the half of the
    backbone that produced the corresponding quantity. Any other backbone has
    no such structure, so the heads fall back to the full embedding.
    """
    if config.backbone_kind == "timing" and in_dim == 2 * config.hidden_dim:
        h = config.hidden_dim
        return (0, h, h, h)
    return (0, in_dim, 0, in_dim)


def _build_head(name: str, config: NetSTAConfig, in_dim: int) -> TaskHead:
    if name not in TASK_HEAD_REGISTRY:
        raise ValueError(
            f"Unknown task '{name}'. Available: {sorted(TASK_HEAD_REGISTRY)}"
        )
    cls = TASK_HEAD_REGISTRY[name]
    if cls is CriticalPathHead:
        return cls(in_dim, config.hidden_dim, config.dropout, config.critical_pos_weight_cap)
    if cls is SlackHead:
        # Compositional whenever AT + RT supervision is also active — that's
        # the case in which RT_pred - AT_pred is well-defined and trained.
        compositional = (
            "arrival_time" in config.active_tasks
            and "required_time" in config.active_tasks
        )
        return cls(
            in_dim, config.hidden_dim, config.dropout,
            slack_mean=config.slack_mean, slack_std=config.slack_std,
            compositional=compositional,
        )
    if cls is ArrivalTimeHead:
        at_off, at_dim, _, _ = _directional_slices(config, in_dim)
        return cls(
            in_dim, config.dropout,
            arrival_time_mean=config.arrival_time_mean,
            arrival_time_std=config.arrival_time_std,
            slice_offset=at_off, slice_dim=at_dim,
        )
    if cls is RequiredTimeHead:
        _, _, rt_off, rt_dim = _directional_slices(config, in_dim)
        return cls(
            in_dim, config.dropout,
            required_time_mean=config.required_time_mean,
            required_time_std=config.required_time_std,
            slice_offset=rt_off, slice_dim=rt_dim,
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

        # The compositional SlackHead defers to the AT and RT heads at forward
        # time; wire them in now that all heads are constructed. ModuleDict
        # uses __contains__/__getitem__ semantics rather than dict.get.
        slack = self.heads["slack"] if "slack" in self.heads else None
        at = self.heads["arrival_time"] if "arrival_time" in self.heads else None
        rt = self.heads["required_time"] if "required_time" in self.heads else None
        if (
            isinstance(slack, SlackHead) and slack.compositional
            and at is not None and rt is not None
        ):
            slack.attach_components(at, rt)

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

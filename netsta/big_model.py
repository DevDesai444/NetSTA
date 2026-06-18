"""
Larger NetSTA backbone: GraphGPS-style transformer fused with the directional
STA prior.

Two parallel branches per forward pass:

  STA prior branch
    The existing TimingPropagationBackbone (forward max-agg AT pass + backward
    min-agg RT pass). Carries the physics inductive bias — message passing
    structured as STA relaxation. Output: [N, 2*hidden_sta].

  GraphGPS transformer branch
    Laplacian positional encoding (top-k smallest non-trivial eigenvectors of
    the graph Laplacian) + L stacked blocks of:
        local MPNN (GINE-style with edge features)
        global multi-head self-attention over all nodes in the graph
    Output: [N, hidden_gps].

Fusion: concat([sta_emb, gps_emb]) -> MLP -> node_emb of width = node_emb_dim.
Same 6-head structure on top, but heads see a much richer representation.

Parameter count target: ~3-5M (vs ~77K in the small model).
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    BatchNorm,
    GINEConv,
    global_max_pool,
    global_mean_pool,
)

from .config import NetSTAConfig
from .model import TimingPropagationBackbone


# ---------------------------------------------------------------------------
# Laplacian positional encoding
# ---------------------------------------------------------------------------


def laplacian_pe(edge_index: torch.Tensor, num_nodes: int, k: int) -> torch.Tensor:
    """Compute per-node positional encoding from the graph Laplacian.

    Builds the symmetric normalized Laplacian L = I - D^-1/2 A D^-1/2 (on the
    undirected version of the graph), takes its k smallest non-trivial
    eigenvectors as per-node positional features. Random sign flips per
    eigenvector are applied so the model isn't tied to a particular sign
    convention.

    Returns [num_nodes, k]. For tiny graphs (num_nodes < k+1), the remaining
    columns are zero-padded.
    """
    device = edge_index.device
    if num_nodes < 2 or edge_index.numel() == 0:
        return torch.zeros(num_nodes, k, device=device)

    # Build the dense adjacency. The dataset is small per-graph (cones cap at
    # ~6k nodes), so the dense path is fine and avoids sparse-eigsh fragility.
    edges = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    a = torch.zeros(num_nodes, num_nodes, device=device)
    a[edges[0], edges[1]] = 1.0
    deg = a.sum(dim=1)
    deg = deg.clamp(min=1.0)
    d_inv_sqrt = deg.pow(-0.5)
    laplacian = torch.eye(num_nodes, device=device) - (
        d_inv_sqrt.unsqueeze(1) * a * d_inv_sqrt.unsqueeze(0)
    )
    # Numerical safety: enforce symmetry.
    laplacian = (laplacian + laplacian.transpose(0, 1)) / 2.0
    try:
        eigvals, eigvecs = torch.linalg.eigh(laplacian)
    except RuntimeError:
        return torch.zeros(num_nodes, k, device=device)
    # Drop the trivial constant eigenvector (eigval 0); take the next k.
    pe = eigvecs[:, 1 : 1 + k]
    if pe.size(1) < k:
        pad = torch.zeros(num_nodes, k - pe.size(1), device=device)
        pe = torch.cat([pe, pad], dim=1)
    # Random sign flip per eigenvector (training augmentation).
    signs = torch.randint(0, 2, (pe.size(1),), device=device).float() * 2 - 1
    return pe * signs.unsqueeze(0)


def _batched_lap_pe(
    edge_index: torch.Tensor, batch: torch.Tensor, num_nodes: int, k: int
) -> torch.Tensor:
    """Compute Laplacian PE per-graph in a batched PyG forward."""
    pe = torch.zeros(num_nodes, k, device=edge_index.device)
    if edge_index.numel() == 0:
        return pe
    num_graphs = int(batch.max().item()) + 1
    for g in range(num_graphs):
        node_mask = batch == g
        node_idx = node_mask.nonzero(as_tuple=True)[0]
        if node_idx.numel() == 0:
            continue
        # Restrict to edges where both endpoints belong to this graph.
        e_mask = node_mask[edge_index[0]] & node_mask[edge_index[1]]
        sub_edges = edge_index[:, e_mask]
        if sub_edges.numel() == 0:
            continue
        # Re-index nodes locally [0..n_g-1].
        global_to_local = torch.full(
            (num_nodes,), -1, dtype=torch.long, device=edge_index.device,
        )
        global_to_local[node_idx] = torch.arange(node_idx.numel(), device=edge_index.device)
        local_edges = global_to_local[sub_edges]
        pe[node_idx] = laplacian_pe(local_edges, node_idx.numel(), k)
    return pe


# ---------------------------------------------------------------------------
# GraphGPS block
# ---------------------------------------------------------------------------


class GraphGPSBlock(nn.Module):
    """One block of: local MPNN (GINE) + global self-attention + FFN.

    Both branches operate on the SAME node embedding and are summed
    (pre-LayerNorm residual). Attention is computed per-graph by chunking on
    `batch` so different graphs in a batch don't attend to each other.
    """

    def __init__(self, dim: int, heads: int, dropout: float = 0.1, edge_dim: int = 5):
        super().__init__()
        self.norm_local = nn.LayerNorm(dim)
        self.norm_global = nn.LayerNorm(dim)
        self.norm_ffn = nn.LayerNorm(dim)

        # Local: GINEConv with an MLP message function.
        mlp = nn.Sequential(
            nn.Linear(dim, dim * 2), nn.GELU(),
            nn.Linear(dim * 2, dim),
        )
        self.local = GINEConv(mlp, train_eps=True, edge_dim=edge_dim)

        # Global self-attention.
        self.attn = nn.MultiheadAttention(dim, num_heads=heads, dropout=dropout, batch_first=True)

        # Position-wise FFN.
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )
        self.dropout = nn.Dropout(dropout)

    def _global_attn(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """Per-graph self-attention by chunking on the batch vector.

        Under torch.amp the attention output is fp16 even when `x` is fp32,
        so we pre-allocate the output buffer in the attention op's dtype to
        avoid an `index_put` dtype mismatch.
        """
        if batch is None:
            x_in = x.unsqueeze(0)  # [1, N, D]
            out, _ = self.attn(x_in, x_in, x_in, need_weights=False)
            return out.squeeze(0)
        num_graphs = int(batch.max().item()) + 1
        out: Optional[torch.Tensor] = None
        for g in range(num_graphs):
            node_mask = batch == g
            idx = node_mask.nonzero(as_tuple=True)[0]
            if idx.numel() == 0:
                continue
            x_g = x[idx].unsqueeze(0)  # [1, n_g, D]
            attended, _ = self.attn(x_g, x_g, x_g, need_weights=False)
            attended = attended.squeeze(0)
            if out is None:
                out = torch.zeros_like(x, dtype=attended.dtype)
            out[idx] = attended
        return out if out is not None else x

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: Optional[torch.Tensor],
    ) -> torch.Tensor:
        # Local branch (with pre-norm residual).
        h_local = self.local(self.norm_local(x), edge_index, edge_attr)
        h_local = self.dropout(h_local)
        # Global branch (with pre-norm residual).
        h_global = self._global_attn(self.norm_global(x), batch)
        h_global = self.dropout(h_global)
        x = x + h_local + h_global
        # FFN (pre-norm).
        x = x + self.dropout(self.ffn(self.norm_ffn(x)))
        return x


# ---------------------------------------------------------------------------
# Full big backbone
# ---------------------------------------------------------------------------


class GraphGPSWithSTABackbone(nn.Module):
    """Big backbone: directional STA prior in parallel with a GraphGPS stack.

    Forward returns the same dict shape as TimingPropagationBackbone so the
    existing task heads, training loop, and inference pipeline work unchanged.
    """

    def __init__(
        self,
        config: NetSTAConfig,
        gps_dim: int = 256,
        gps_layers: int = 6,
        gps_heads: int = 8,
        pe_k: int = 16,
    ):
        super().__init__()
        self.config = config
        self.pe_k = pe_k

        # STA prior branch (unchanged module — proven).
        self.sta = TimingPropagationBackbone(config)
        sta_out_dim = self.sta.node_emb_dim  # 2 * hidden_sta

        # GraphGPS branch.
        self.feat_proj = nn.Linear(config.node_feature_dim, gps_dim)
        self.pe_proj = nn.Linear(pe_k, gps_dim)
        self.gps_blocks = nn.ModuleList(
            [
                GraphGPSBlock(gps_dim, heads=gps_heads, dropout=config.dropout,
                              edge_dim=config.edge_feature_dim)
                for _ in range(gps_layers)
            ]
        )
        self.gps_norm = nn.LayerNorm(gps_dim)

        # Fusion: project both branches into a shared embedding the heads read.
        # Width = sta_out_dim so the existing AT/RT half-slicing in the heads
        # still aligns with the STA prior's [AT, RT] halves on the first
        # sta_out_dim columns. GPS contribution is added in pre-fusion.
        fusion_in = sta_out_dim + gps_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, sta_out_dim * 2), nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(sta_out_dim * 2, sta_out_dim),
        )

        # Match the small-model interface so the existing heads slot in.
        self.node_emb_dim = sta_out_dim
        self.graph_emb_dim = 2 * self.node_emb_dim

    def set_soft_temperature(self, beta: float) -> None:
        """Forward to the STA prior so the existing training loop can anneal."""
        self.sta.set_soft_temperature(beta)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # 1. STA prior branch.
        sta_out = self.sta(x, edge_index, edge_attr=edge_attr, batch=batch)
        sta_emb = sta_out["node_emb"]            # [N, sta_out_dim]
        clock_logit = sta_out.get("clock_period_logit")

        # 2. GraphGPS branch.
        pe = _batched_lap_pe(edge_index, batch, x.size(0), self.pe_k)
        h = self.feat_proj(x) + self.pe_proj(pe)
        for block in self.gps_blocks:
            h = block(h, edge_index, edge_attr if edge_attr is not None else x.new_zeros(0, 1), batch)
        h = self.gps_norm(h)                     # [N, gps_dim]

        # 3. Fuse.
        node_emb = self.fusion(torch.cat([sta_emb, h], dim=-1))  # [N, node_emb_dim]
        graph_emb = torch.cat(
            [global_mean_pool(node_emb, batch), global_max_pool(node_emb, batch)],
            dim=-1,
        )
        return {
            "node_emb": node_emb,
            "graph_emb": graph_emb,
            **({"clock_period_logit": clock_logit} if clock_logit is not None else {}),
        }

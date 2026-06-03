"""
Configuration objects for the NetSTA model.

A single dataclass collects every knob the backbone, task heads, and
optimizer need so callers can change behaviour without editing model code.
"""

from dataclasses import dataclass, field
from typing import Dict, Tuple


# Default training targets all three timing quantities the directional
# backbone is structured around. Auxiliary AT/RT supervision anchors the
# backbone halves to the physical quantities they represent; the slack head
# then derives slack compositionally as RT_pred - AT_pred. critical_path is
# available on demand via --tasks but is no longer in the default set —
# ablation shows the BCE gradient interferes with slack regression.
DEFAULT_TASK_WEIGHTS: Dict[str, float] = {
    "slack": 1.0,
    "arrival_time": 0.3,
    "required_time": 0.3,
}

DEFAULT_ACTIVE_TASKS: Tuple[str, ...] = (
    "slack", "arrival_time", "required_time",
)


@dataclass
class NetSTAConfig:
    """Hyperparameters for the NetSTA backbone, heads, and optimizer."""

    # Feature dimensions -- node_feature_dim is data-dependent and must be
    # set by the caller before model construction.
    node_feature_dim: int = 0
    # Default reflects the current graph_builder output:
    # [wire_delay, manhattan_distance, net_fanout] = 3 dims.
    edge_feature_dim: int = 3

    # Backbone
    hidden_dim: int = 64
    num_layers: int = 4
    num_heads: int = 4
    dropout: float = 0.1

    # Optimizer
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    # Multiplier applied per-layer from the head backwards: param_lr =
    # base_lr * (layer_decay ** (num_layers - layer_idx - 1)). 1.0 disables.
    layer_decay: float = 1.0

    # Task selection. Only tasks listed in `active_tasks` are constructed;
    # `task_weights` must contain a weight for every active task.
    active_tasks: Tuple[str, ...] = DEFAULT_ACTIVE_TASKS
    task_weights: Dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_TASK_WEIGHTS)
    )

    # Class-imbalance cap for the critical-path BCE pos_weight.
    critical_pos_weight_cap: float = 10.0

    # Train-set summary stats baked into each timing head so its forward
    # returns predictions in absolute ns. Filled in by train.py from the train
    # split; checkpoints round-trip them via the head's register_buffer slots.
    # Slack stats are also used by the compositional SlackHead path (see
    # NetSTAModel) which produces slack as the difference of standardized RT
    # and AT predictions, scaled back to ns through these stats.
    slack_mean: float = 0.0
    slack_std: float = 1.0
    arrival_time_mean: float = 0.0
    arrival_time_std: float = 1.0
    required_time_mean: float = 0.0
    required_time_std: float = 1.0

    # Ablation flags. Defaults reproduce the original GATv2 architecture
    # exactly (so the existing ablation table still measures what it claims).
    use_residual: bool = True
    use_attention: bool = True  # False -> GCNConv (uniform message passing)
    # When True, the backbone uses SymmetryAwareAttention instead of GATv2Conv.
    # Needed for analog circuits where edge_feature_dim must include a
    # matching_constraint indicator at the trailing edge-feature column.
    use_symmetry_attention: bool = False

    # Backbone selector:
    #   "gatv2"  -> GATv2 stack with mean+max pooling (original architecture)
    #   "timing" -> directional STA-aware propagation: a forward pass with
    #               max-aggregation models arrival-time accumulation; a
    #               backward pass with min-aggregation models required-time
    #               propagation. Each pass iterates `num_layers` times so
    #               messages can reach gates `num_layers` hops deep, matching
    #               the longest path in the typical training-set circuit
    #               (depth 1-14 for 40-gate circuits).
    backbone_kind: str = "timing"

    def validate(self) -> None:
        if self.node_feature_dim <= 0:
            raise ValueError("node_feature_dim must be set to a positive value")
        if self.edge_feature_dim <= 0:
            raise ValueError("edge_feature_dim must be positive")
        missing = [t for t in self.active_tasks if t not in self.task_weights]
        if missing:
            raise ValueError(
                f"task_weights missing entries for active tasks: {missing}"
            )

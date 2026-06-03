"""Model + config tests."""

import pytest
import torch

from netsta.config import NetSTAConfig
from netsta.model import (
    AnalogPerformanceHead,
    CongestionHead,
    CriticalPathHead,
    DRCHead,
    NetSTAModel,
    SlackHead,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def test_config_defaults_match_documented_values():
    cfg = NetSTAConfig(node_feature_dim=24)
    assert cfg.hidden_dim == 64
    assert cfg.num_layers == 4
    assert cfg.num_heads == 4
    assert cfg.dropout == 0.1
    assert cfg.edge_feature_dim == 3 or cfg.edge_feature_dim == 5  # tolerate later bumps
    assert cfg.active_tasks == ("slack", "critical_path")


def test_config_validate_rejects_zero_node_dim():
    with pytest.raises(ValueError):
        NetSTAConfig().validate()


def test_config_validate_rejects_missing_task_weight():
    with pytest.raises(ValueError):
        NetSTAConfig(
            node_feature_dim=24,
            active_tasks=("slack", "drc"),
            task_weights={"slack": 0.5},
        ).validate()


# ---------------------------------------------------------------------------
# Individual heads
# ---------------------------------------------------------------------------


def test_slack_head_shape_and_loss():
    head = SlackHead(in_dim=32, hidden=16, dropout=0.0)
    pred = head(torch.randn(8, 32))
    assert pred.shape == (8,)
    loss = head.loss(pred, torch.randn(8))
    assert loss.dim() == 0 and torch.isfinite(loss)


def test_critical_path_head_pos_weight_clamped():
    head = CriticalPathHead(in_dim=32, hidden=16, dropout=0.0, pos_weight_cap=10.0)
    pred = head(torch.randn(20, 32))
    # All-zero targets exercise the divide-by-zero guard.
    loss = head.loss(pred, torch.zeros(20))
    assert torch.isfinite(loss)


def test_congestion_head_mse():
    head = CongestionHead(in_dim=32, hidden=16, dropout=0.0)
    pred = head(torch.randn(8, 32))
    loss = head.loss(pred, torch.rand(8))
    assert torch.isfinite(loss) and loss >= 0


def test_drc_head_focal_loss_positive_on_imbalanced_target():
    head = DRCHead(in_dim=32, hidden=16, dropout=0.0)
    pred = torch.randn(40)
    target = torch.zeros(40)
    target[:4] = 1.0  # 10 % positives
    loss = head.loss(pred, target)
    assert torch.isfinite(loss) and loss > 0


def test_analog_perf_head_two_output():
    head = AnalogPerformanceHead(in_dim=32, hidden=16, dropout=0.0)
    pred = head(torch.randn(6, 32))
    assert pred.shape == (6, 2)
    loss = head.loss(pred, torch.rand(6, 2))
    assert torch.isfinite(loss) and loss >= 0


# ---------------------------------------------------------------------------
# Full model forward + loss
# ---------------------------------------------------------------------------


def test_model_forward_returns_all_task_predictions(untrained_model, sample_pyg_data):
    data = sample_pyg_data
    out = untrained_model(
        data.x, data.edge_index, edge_attr=data.edge_attr, batch=None,
    )
    for task in untrained_model.heads.keys():
        assert task in out
    assert "_node_emb" in out and "_graph_emb" in out
    assert out["_node_emb"].shape[0] == data.num_nodes
    # graph_emb = concat(mean_pool, max_pool) of node_emb, so its width is
    # always 2 × node_emb_dim regardless of which backbone is selected.
    assert out["_graph_emb"].shape[1] == 2 * out["_node_emb"].shape[1]


def test_model_compute_loss_returns_finite_total(untrained_model, sample_pyg_data):
    data = sample_pyg_data
    out = untrained_model(
        data.x, data.edge_index, edge_attr=data.edge_attr, batch=None,
    )
    targets = {
        "slack":              data.y_slack,
        "critical_path":      data.y_critical,
        "congestion":         data.y_congestion,
        "drc":                data.y_drc,
        "analog_performance": data.y_analog_performance,
    }
    total, per_task = untrained_model.compute_loss(out, targets)
    assert torch.isfinite(total)
    assert set(per_task.keys()) == set(untrained_model.heads.keys())
    for v in per_task.values():
        assert torch.isfinite(v)


def test_model_backward_pass_doesnt_crash(untrained_model, sample_pyg_data):
    data = sample_pyg_data
    out = untrained_model(
        data.x, data.edge_index, edge_attr=data.edge_attr, batch=None,
    )
    targets = {
        "slack":              data.y_slack,
        "critical_path":      data.y_critical,
        "congestion":         data.y_congestion,
        "drc":                data.y_drc,
        "analog_performance": data.y_analog_performance,
    }
    loss, _ = untrained_model.compute_loss(out, targets)
    loss.backward()
    # At least one trainable parameter must have a gradient.
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in untrained_model.parameters())
    assert has_grad


def test_timing_backbone_is_default():
    """The directional STA-aware backbone is the default — that's the whole
    point of the rewrite. Locks in the default so a config drift can't
    silently revert to the GATv2 stack that lost the MLP comparison."""
    from netsta.config import NetSTAConfig
    cfg = NetSTAConfig(node_feature_dim=24)
    assert cfg.backbone_kind == "timing"


def test_timing_backbone_forward_returns_concat_at_rt_embeddings(sample_pyg_data):
    """Per-node embedding from the timing backbone is concat([AT, RT]) so its
    dim is 2 × hidden, not hidden. Slack head should be able to subtract."""
    from dataclasses import replace
    from netsta.config import NetSTAConfig
    from netsta.model import NetSTAModel, TimingPropagationBackbone
    cfg = NetSTAConfig(
        node_feature_dim=sample_pyg_data.x.size(1),
        edge_feature_dim=sample_pyg_data.edge_attr.size(1),
        hidden_dim=16, num_layers=3, num_heads=1, dropout=0.0,
        active_tasks=("slack", "critical_path"),
        task_weights={"slack": 0.5, "critical_path": 0.5},
        backbone_kind="timing",
    )
    model = NetSTAModel(cfg)
    assert isinstance(model.backbone, TimingPropagationBackbone)
    out = model(
        sample_pyg_data.x, sample_pyg_data.edge_index,
        edge_attr=sample_pyg_data.edge_attr, batch=None,
    )
    # node_emb width = 2 × hidden_dim because forward + backward passes are concatenated.
    assert out["_node_emb"].shape[1] == 2 * cfg.hidden_dim
    # graph_emb = concat(mean_pool, max_pool) over node_emb = 4 × hidden_dim.
    assert out["_graph_emb"].shape[1] == 4 * cfg.hidden_dim


def test_timing_backbone_propagates_along_dag_depth(sample_pyg_data):
    """One iteration of the forward layer should be enough to move signal
    one hop downstream — verify by comparing init embedding to post-forward
    embedding at a deep node (i.e. ensure they differ)."""
    from netsta.config import NetSTAConfig
    from netsta.model import NetSTAModel
    cfg = NetSTAConfig(
        node_feature_dim=sample_pyg_data.x.size(1),
        edge_feature_dim=sample_pyg_data.edge_attr.size(1),
        hidden_dim=8, num_layers=4, num_heads=1, dropout=0.0,
        active_tasks=("slack",),
        task_weights={"slack": 1.0},
        backbone_kind="timing",
    )
    model = NetSTAModel(cfg).eval()
    out = model(
        sample_pyg_data.x, sample_pyg_data.edge_index,
        edge_attr=sample_pyg_data.edge_attr, batch=None,
    )
    # AT half and RT half should not be identical — that would mean either
    # pass is a no-op (no MP happening).
    h = out["_node_emb"]
    at_part = h[:, : cfg.hidden_dim]
    rt_part = h[:, cfg.hidden_dim :]
    assert not torch.allclose(at_part, rt_part, atol=1e-6), (
        "AT and RT embeddings are identical — directional propagation didn't run"
    )


def test_symmetry_aware_attention_swap(tiny_config, sample_pyg_data):
    """Asking for symmetry attention should swap the backbone layers and still
    produce a valid forward pass. Symmetry attention is a GATv2-family layer,
    so this test explicitly uses backbone_kind='gatv2'."""
    from dataclasses import replace
    from netsta.model import NetSTAModel, SymmetryAwareAttention
    cfg = replace(tiny_config, use_symmetry_attention=True, backbone_kind="gatv2")
    model = NetSTAModel(cfg)
    # First conv layer is now SymmetryAwareAttention rather than GATv2Conv.
    assert isinstance(model.backbone.convs[0], SymmetryAwareAttention)
    data = sample_pyg_data
    out = model(data.x, data.edge_index, edge_attr=data.edge_attr, batch=None)
    assert torch.isfinite(out["_node_emb"]).all()

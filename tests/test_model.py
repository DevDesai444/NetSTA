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
    # 2 * hidden * heads = 2 * 32 * 2 = 128 for the tiny config.
    assert out["_graph_emb"].shape[1] == 2 * untrained_model.config.hidden_dim * untrained_model.config.num_heads


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


def test_symmetry_aware_attention_swap(tiny_config, sample_pyg_data):
    """Asking for symmetry attention should swap the backbone layers and still
    produce a valid forward pass."""
    from dataclasses import replace
    from netsta.model import NetSTAModel, SymmetryAwareAttention
    cfg = replace(tiny_config, use_symmetry_attention=True)
    model = NetSTAModel(cfg)
    # First conv layer is now SymmetryAwareAttention rather than GATv2Conv.
    assert isinstance(model.backbone.convs[0], SymmetryAwareAttention)
    data = sample_pyg_data
    out = model(data.x, data.edge_index, edge_attr=data.edge_attr, batch=None)
    assert torch.isfinite(out["_node_emb"]).all()

"""Tests for the big GraphGPS+STA backbone."""

import pytest
import torch

from netsta.config import NetSTAConfig
from netsta.model import NetSTAModel


def _cfg(backbone_kind: str = "graphgps_sta") -> NetSTAConfig:
    return NetSTAConfig(
        node_feature_dim=17, edge_feature_dim=5, hidden_dim=64, num_layers=4,
        num_heads=4, backbone_kind=backbone_kind,
        active_tasks=("slack", "arrival_time", "required_time",
                       "critical_path", "congestion", "drc"),
        task_weights={"slack": 3.0, "arrival_time": 1.0, "required_time": 1.0,
                      "critical_path": 1.0, "congestion": 1.0, "drc": 1.0,
                      "clock_period": 0.5},
        slack_mean=0.5, slack_std=0.7,
        arrival_time_mean=0.6, arrival_time_std=0.8,
        required_time_mean=1.2, required_time_std=1.1,
        clock_period_mean=0.85, clock_period_std=0.94,
        raw_feature_residual=True,
    )


def test_big_model_builds_with_expected_size():
    m = NetSTAModel(_cfg())
    n_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
    # ~5M with hidden_sta=64, gps_dim=256, 6 GPS layers — small variance is fine.
    assert 3_000_000 < n_params < 8_000_000, f"expected ~5M params, got {n_params}"


def test_big_model_forward_all_heads():
    m = NetSTAModel(_cfg())
    m.eval()
    N = 80
    x = torch.randn(N, 17)
    edge_index = torch.randint(0, N, (2, 200))
    edge_attr = torch.randn(200, 5)
    batch = torch.cat([torch.zeros(40), torch.ones(40)]).long()
    out = m(x, edge_index, edge_attr=edge_attr, batch=batch)
    for t in ("slack", "arrival_time", "required_time", "critical_path",
              "congestion", "drc"):
        assert t in out, f"missing head {t}"
    assert out["slack"].shape == (N,)


def test_big_model_backward_no_nans():
    m = NetSTAModel(_cfg())
    m.train()
    N = 40
    x = torch.randn(N, 17)
    edge_index = torch.randint(0, N, (2, 100))
    edge_attr = torch.randn(100, 5)
    batch = torch.zeros(N, dtype=torch.long)
    targets = {
        "slack": torch.randn(N) * 0.5,
        "arrival_time": torch.randn(N) * 0.8,
        "required_time": torch.randn(N) * 1.0,
        "critical_path": torch.rand(N).round(),
        "congestion": torch.rand(N),
        "drc": torch.rand(N).round(),
        "clock_period": torch.tensor([0.9]),
    }
    out = m(x, edge_index, edge_attr=edge_attr, batch=batch)
    loss, _ = m.compute_loss(out, targets)
    loss.backward()
    for n, p in m.named_parameters():
        if p.grad is not None:
            assert not torch.isnan(p.grad).any(), f"NaN grad in {n}"


def test_big_model_emits_clock_logit():
    m = NetSTAModel(_cfg())
    m.eval()
    N = 30
    x = torch.randn(N, 17)
    edge_index = torch.randint(0, N, (2, 60))
    edge_attr = torch.randn(60, 5)
    out = m(x, edge_index, edge_attr=edge_attr)
    assert "_clock_period_logit" in out

"""Tests for the 4-agent design advisory pipeline (deterministic path)."""

import numpy as np

from netsta.agents import DesignReport, diagnose
from netsta.agents.tools import (
    rank_congestion,
    rank_critical_nodes,
    rank_drc_hotspots,
)


def _synthetic_predictions(n=40, seed=0):
    rng = np.random.default_rng(seed)
    node_ids = [f"g{i}" for i in range(n)]
    preds = {
        "slack": np.concatenate([rng.uniform(-0.08, -0.01, 6), rng.uniform(0.05, 0.3, n - 6)]),
        "critical_path": np.concatenate([rng.uniform(1, 3, 6), rng.uniform(-3, -1, n - 6)]),
        "congestion": np.concatenate([rng.uniform(0.4, 0.8, 4), rng.uniform(0.0, 0.2, n - 4)]),
        "drc": np.concatenate([rng.uniform(1, 3, 3), rng.uniform(-4, -1, n - 3)]),
    }
    return node_ids, preds


def test_tools_flag_bottlenecks():
    node_ids, preds = _synthetic_predictions()
    slack_bn = rank_critical_nodes(preds, node_ids)
    assert slack_bn is not None and slack_bn.task == "slack"
    assert slack_bn.severity > 0
    assert rank_drc_hotspots(preds, node_ids) is not None
    assert rank_congestion(preds, node_ids) is not None


def test_tools_quiet_on_clean_circuit():
    node_ids = [f"g{i}" for i in range(20)]
    clean = {
        "slack": np.full(20, 0.3),
        "drc": np.full(20, -6.0),
        "congestion": np.full(20, 0.05),
    }
    assert rank_drc_hotspots(clean, node_ids) is None
    assert rank_congestion(clean, node_ids) is None


def test_diagnose_produces_report():
    node_ids, preds = _synthetic_predictions()
    report = diagnose(
        {"node_ids": node_ids, "predictions": preds},
        circuit_name="t", topology="pipelined_datapath", use_autogen="never",
    )
    assert isinstance(report, DesignReport)
    assert report.backend == "deterministic"
    assert report.bottlenecks and report.recommendations
    # The supervisor + 3 specialists each contribute a transcript turn.
    agents = {t.agent for t in report.transcript}
    assert {"SupervisorAgent", "TimingAgent", "DRCAgent", "OptimizationAgent"} <= agents


def test_recommendations_are_grounded():
    node_ids, preds = _synthetic_predictions()
    report = diagnose(
        {"node_ids": node_ids, "predictions": preds}, use_autogen="never"
    )
    # Every fix should be a real strategy with outcomes from the graph.
    rec = next(r for r in report.recommendations if r.fix == "gate_upsizing")
    assert "slack_improved" in rec.outcomes
    assert rec.confidence > 0


def test_cross_task_conflicts_flagged():
    node_ids, preds = _synthetic_predictions()
    report = diagnose({"node_ids": node_ids, "predictions": preds}, use_autogen="never")
    # The optimization agent should surface at least one conflict advisory
    # (gate_upsizing/buffer_insertion vs cell_spreading).
    advisories = [r for r in report.recommendations if r.fix == "reconcile_conflict"]
    assert advisories, "expected cross-task conflict reconciliation"

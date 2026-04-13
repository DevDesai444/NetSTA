"""
Digital STA tests.

The STA in this repo is a simplified arrival-time / required-time engine.
Tests therefore check *invariants* (arrival monotonicity, slack/critical
consistency) rather than exact ns values, since exact values depend on the
Nangate45 cell-delay model.
"""

import math

import pytest

from timingnet.circuit_gen import generate_circuit
from timingnet.sta import run_sta


def test_run_sta_returns_required_fields(sample_digital_circuit):
    r = run_sta(sample_digital_circuit)
    assert "node_timing" in r
    assert "clock_period_ns" in r
    assert "min_slack_ns" in r
    assert "num_critical_nodes" in r
    assert len(r["node_timing"]) == len(sample_digital_circuit.nodes)


def test_node_timing_keys_match_circuit_nodes(sample_digital_circuit):
    r = run_sta(sample_digital_circuit)
    assert set(r["node_timing"].keys()) == set(sample_digital_circuit.nodes.keys())


def test_pi_arrival_time_is_zero_or_tiny(sample_digital_circuit):
    r = run_sta(sample_digital_circuit)
    for pi in sample_digital_circuit.primary_inputs:
        at = r["node_timing"][pi].get("arrival_time", 0.0)
        # PIs are sources; arrival time should be 0 (or numerically tiny).
        assert at < 1e-6, f"PI {pi} has nonzero arrival_time={at}"


def test_arrival_time_monotonic_along_signal_path(sample_digital_circuit):
    """arrival_time at any sink ≥ arrival_time at the net's driver."""
    c = sample_digital_circuit
    r = run_sta(c)
    nt = r["node_timing"]
    for net in c.nets.values():
        drv_at = nt[net.driver].get("arrival_time", 0.0)
        for sink in net.sinks:
            sink_at = nt[sink].get("arrival_time", 0.0)
            # Tiny floating-point slack acceptable.
            assert sink_at + 1e-9 >= drv_at, (
                f"arrival regression: {net.driver}@{drv_at} -> {sink}@{sink_at}"
            )


def test_critical_nodes_match_min_slack_reported_by_engine(sample_digital_circuit):
    """The STA flags is_critical = (slack <= min_slack + 0.3*range).

    For circuits where only a few nodes lie on a reachable PI→PO path, this
    threshold collapses to ~min_slack and every flagged node's slack should
    match the engine's reported `min_slack_ns` (within numerical tolerance).
    """
    r = run_sta(sample_digital_circuit)
    nt = r["node_timing"]
    crit = [t for t in nt.values() if t.get("is_critical")]
    if not crit:
        pytest.skip("circuit has no critical nodes at this clock period")
    reported_min = r["min_slack_ns"]
    for t in crit:
        assert math.isfinite(t["slack"])
        # Critical slack should be at or near the reported minimum (the
        # threshold is min + 30% of slack range — small for our circuits).
        assert t["slack"] <= reported_min + 1e-3


def test_num_critical_nodes_matches_node_timing(sample_digital_circuit):
    r = run_sta(sample_digital_circuit)
    counted = sum(1 for t in r["node_timing"].values() if t.get("is_critical"))
    assert r["num_critical_nodes"] == counted


def test_sta_on_small_circuit_returns_consistent_fields():
    """Smallest meaningful test: every node carries arrival_time, slack, and
    is_critical fields; PIs and POs land at non-negative arrival times.
    (This STA is a simplified ordering+slack engine — it doesn't propagate
    real delays — so we assert finiteness rather than strict positivity.)
    """
    c = generate_circuit(num_inputs=2, num_gates=5, num_outputs=1, seed=99)
    r = run_sta(c)
    nt = r["node_timing"]
    for nid in c.primary_inputs + c.gate_ids + c.primary_outputs:
        t = nt[nid]
        for key in ("arrival_time", "slack", "is_critical"):
            assert key in t, f"{nid} missing {key}"
        assert math.isfinite(t["arrival_time"]) and t["arrival_time"] >= 0
        assert math.isfinite(t["slack"])
        assert isinstance(t["is_critical"], bool)

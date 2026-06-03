"""
Digital STA tests.

The STA in this repo is a simplified arrival-time / required-time engine.
Tests therefore check *invariants* (arrival monotonicity, slack/critical
consistency) rather than exact ns values, since exact values depend on the
Nangate45 cell-delay model.
"""

import math

import pytest

from netsta.circuit_gen import generate_circuit
from netsta.sta import run_sta


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


def test_critical_nodes_have_slack_at_or_below_threshold(sample_digital_circuit):
    """is_critical is set when slack <= CRITICAL_SLACK_THRESHOLD_NS (fixed
    absolute threshold). Every flagged node's slack must satisfy that bound.
    """
    from netsta.sta import CRITICAL_SLACK_THRESHOLD_NS
    r = run_sta(sample_digital_circuit)
    nt = r["node_timing"]
    crit = [t for t in nt.values() if t.get("is_critical")]
    if not crit:
        pytest.skip("circuit has no critical nodes at this clock period")
    for t in crit:
        assert math.isfinite(t["slack"])
        assert t["slack"] <= CRITICAL_SLACK_THRESHOLD_NS + 1e-9, (
            f"flagged critical but slack={t['slack']:.6f} > threshold"
        )


def test_num_critical_nodes_matches_node_timing(sample_digital_circuit):
    r = run_sta(sample_digital_circuit)
    counted = sum(1 for t in r["node_timing"].values() if t.get("is_critical"))
    assert r["num_critical_nodes"] == counted


def test_sta_on_small_circuit_returns_consistent_fields():
    """Smallest meaningful test: every node carries arrival_time, slack, and
    is_critical fields; PIs and POs land at non-negative arrival times."""
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


def test_sta_actually_propagates_arrival_time_through_dag():
    """Regression test for the old topological_sort double-increment bug
    that left arrival_time stuck at 0 for every gate, which then silently
    masked the broken STA behind the leaked-feature MLP. After the fix,
    arrival_time at gates deep in the DAG must be strictly positive and
    logical_depth must reflect the actual DAG depth (> 1).
    """
    c = generate_circuit(num_inputs=8, num_gates=40, num_outputs=4, seed=0)
    r = run_sta(c)
    nt = r["node_timing"]

    assert r["max_arrival_time_ns"] > 0.01, (
        f"max_arrival_time={r['max_arrival_time_ns']} — AT didn't propagate"
    )
    gate_ats = [nt[g]["arrival_time"] for g in c.gate_ids]
    assert max(gate_ats) > 0.01, "every gate has AT=0 — propagation broken"

    gate_depths = [nt[g]["logical_depth"] for g in c.gate_ids]
    assert max(gate_depths) > 1, (
        f"max logical_depth={max(gate_depths)} — depth not propagating"
    )

    # Slack must show real spread, not collapse to a single value.
    gate_slacks = [nt[g]["slack"] for g in c.gate_ids]
    spread = max(gate_slacks) - min(gate_slacks)
    assert spread > 0.01, f"slack collapsed to a point (spread={spread})"


def test_graph_builder_emits_at_rt_labels_in_ns():
    """y_arrival_time and y_required_time must mirror the STA's per-node
    AT/RT values exactly (in ns), so the auxiliary heads can supervise them.
    Slack must equal RT - AT at every node within float tolerance.
    """
    from netsta.graph_builder import circuit_to_pyg
    c = generate_circuit(num_inputs=4, num_gates=12, num_outputs=2, seed=11)
    sta = run_sta(c)
    data = circuit_to_pyg(c, sta)

    node_order = c.primary_inputs + c.gate_ids + c.primary_outputs
    for i, nid in enumerate(node_order):
        t = sta["node_timing"][nid]
        assert abs(float(data.y_arrival_time[i]) - t["arrival_time"]) < 1e-6
        assert abs(float(data.y_required_time[i]) - t["required_time"]) < 1e-6
        # STA identity: slack = RT - AT, exactly.
        assert abs(float(data.y_slack[i]) - (t["required_time"] - t["arrival_time"])) < 1e-6
        assert int(data.y_logical_depth[i]) == t["logical_depth"]


def test_critical_path_label_is_fixed_absolute_threshold():
    """The is_critical label must be `slack <= CRITICAL_SLACK_THRESHOLD_NS`,
    NOT a per-graph quantile. Two circuits with identical slack values must
    flag identical critical-path nodes regardless of what the rest of each
    circuit's slack distribution looks like.
    """
    from netsta.sta import CRITICAL_SLACK_THRESHOLD_NS
    c = generate_circuit(num_inputs=6, num_gates=30, num_outputs=3, seed=7)
    r = run_sta(c)
    for nid, t in r["node_timing"].items():
        expected = t["slack"] <= CRITICAL_SLACK_THRESHOLD_NS
        assert t["is_critical"] == expected, (
            f"{nid}: slack={t['slack']:.4f}, threshold={CRITICAL_SLACK_THRESHOLD_NS}, "
            f"is_critical={t['is_critical']} (expected {expected})"
        )

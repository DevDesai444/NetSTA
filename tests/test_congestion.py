"""RUDY congestion + DRC labelling tests."""

import numpy as np

from timingnet.circuit_gen import generate_circuit
from timingnet.congestion import compute_demand_grid, compute_rudy_congestion
from timingnet.drc import HOTSPOT_FACTOR, TRACKS_PER_CELL, compute_drc_labels


def test_compute_demand_grid_matches_grid_size(sample_digital_circuit):
    demand, (gw, gh) = compute_demand_grid(sample_digital_circuit)
    assert len(demand) == gw
    assert all(len(row) == gh for row in demand)


def test_rudy_congestion_normalized_in_unit_range(sample_digital_circuit):
    cong, _ = compute_rudy_congestion(sample_digital_circuit)
    assert cong, "expected at least one node"
    arr = np.array(list(cong.values()))
    assert (arr >= 0).all()
    assert (arr <= 1.0 + 1e-9).all()
    # Maximum should hit 1.0 since we normalize by the per-circuit peak.
    assert abs(arr.max() - 1.0) < 1e-6


def test_rudy_empty_positions_returns_zero():
    """A circuit with no placement information yields zero congestion."""
    c = generate_circuit(num_inputs=2, num_gates=2, num_outputs=1, seed=1)
    c.positions = {}             # clear placement
    c.grid_size = (0, 0)
    cong, demand = compute_rudy_congestion(c)
    assert all(v == 0.0 for v in cong.values())
    assert demand == [[0.0]]


def test_drc_labels_are_binary(sample_digital_circuit):
    labels = compute_drc_labels(sample_digital_circuit)
    for v in labels.values():
        assert v in (0.0, 1.0)


def test_drc_threshold_consistent_with_demand():
    """Manually constructed: any cell whose raw demand > 0.9 * tracks_per_cell
    should be a hotspot; cells at or below shouldn't."""
    c = generate_circuit(num_inputs=4, num_gates=20, num_outputs=2, seed=11)
    _, demand = compute_rudy_congestion(c)
    threshold = TRACKS_PER_CELL * HOTSPOT_FACTOR
    labels = compute_drc_labels(c, demand_grid=demand)
    # Spot-check: every flagged node sits in a cell whose demand exceeds the threshold.
    for nid, lab in labels.items():
        if lab != 1.0:
            continue
        x, y = c.positions[nid]
        assert demand[x][y] > threshold, (
            f"{nid} flagged but demand={demand[x][y]} <= {threshold}"
        )


def test_rudy_zero_when_all_nets_single_pin():
    """Single-pin nets (driver only, no sinks) contribute no routing demand."""
    c = generate_circuit(num_inputs=2, num_gates=2, num_outputs=1, seed=2)
    # Forcibly strip sinks from every net to reduce them to 1-pin.
    for net in c.nets.values():
        net.sinks = []
    _, demand = compute_rudy_congestion(c)
    flat = [v for row in demand for v in row]
    assert max(flat) == 0.0

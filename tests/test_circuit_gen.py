"""Digital + analog circuit generation tests."""

import pytest

from netsta.analog_circuit_gen import ANALOG_TOPOLOGIES, generate_analog_circuit
from netsta.circuit_gen import Circuit, generate_circuit


# ---------------------------------------------------------------------------
# Digital
# ---------------------------------------------------------------------------


def test_digital_circuit_counts(sample_digital_circuit):
    c = sample_digital_circuit
    assert len(c.primary_inputs) == 4
    assert len(c.gate_ids) == 10
    assert len(c.primary_outputs) == 2
    assert len(c.nodes) == 16
    assert all(c.nodes[nid].node_type == "PI" for nid in c.primary_inputs)
    assert all(c.nodes[nid].node_type == "PO" for nid in c.primary_outputs)


def test_digital_circuit_nets_well_formed(sample_digital_circuit):
    c = sample_digital_circuit
    for net in c.nets.values():
        assert net.driver in c.nodes, f"net {net.name} has driver not in nodes"
        for sink in net.sinks:
            assert sink in c.nodes, f"net {net.name} has sink {sink} not in nodes"
        # No duplicate sinks per net.
        assert len(net.sinks) == len(set(net.sinks))


def test_digital_positions_within_grid(sample_digital_circuit):
    c = sample_digital_circuit
    gw, gh = c.grid_size
    assert gw > 0 and gh > 0
    for nid, (x, y) in c.positions.items():
        assert 0 <= x < gw, f"{nid} x={x} outside grid width {gw}"
        assert 0 <= y < gh, f"{nid} y={y} outside grid height {gh}"


def test_digital_circuit_is_acyclic(sample_digital_circuit):
    """DAG check: each gate's output net should not reach itself transitively."""
    c = sample_digital_circuit
    out_net = {nid: c.nodes[nid].output_net for nid in c.nodes}
    in_nets = {nid: set(c.nodes[nid].input_nets) for nid in c.nodes}
    # Walk forward via nets and ensure no node visited twice.
    for start in c.gate_ids:
        seen = {start}
        frontier = [start]
        while frontier:
            cur = frontier.pop()
            net = out_net.get(cur)
            if not net or net not in c.nets:
                continue
            for sink in c.nets[net].sinks:
                if sink == start:
                    pytest.fail(f"cycle found through {start}")
                if sink not in seen:
                    seen.add(sink)
                    frontier.append(sink)


def test_digital_circuit_seed_determinism():
    a = generate_circuit(num_inputs=4, num_gates=8, num_outputs=2, seed=7)
    b = generate_circuit(num_inputs=4, num_gates=8, num_outputs=2, seed=7)
    assert list(a.nodes.keys()) == list(b.nodes.keys())
    assert a.positions == b.positions


# ---------------------------------------------------------------------------
# Analog
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("topo_fn", ANALOG_TOPOLOGIES, ids=lambda f: f.__name__)
def test_each_analog_topology_produces_valid_circuit(topo_fn):
    c = generate_analog_circuit(seed=42, topology=topo_fn.__name__)
    assert isinstance(c, Circuit)
    assert c.is_analog is True
    assert len(c.nodes) > 0
    assert len(c.nets) > 0
    # Every node has a position; grid encloses them.
    gw, gh = c.grid_size
    assert gw > 0 and gh > 0
    for nid, (x, y) in c.positions.items():
        assert 0 <= x < gw and 0 <= y < gh
    # Every device has analog parameters.
    for gid in c.gate_ids:
        assert gid in c.analog_params, f"{gid} missing analog_params"


def test_diff_pair_has_symmetry_groups(sample_analog_circuit):
    """Diff pair must mark M1/M2 (or R1/R2) as matched."""
    groups = sample_analog_circuit.symmetry_groups
    assert groups, "diff_pair must define at least one symmetry group"
    # At least one group has ≥2 members.
    by_group = {}
    for nid, g in groups.items():
        by_group.setdefault(g, []).append(nid)
    assert any(len(members) >= 2 for members in by_group.values())


def test_invalid_topology_raises():
    with pytest.raises(ValueError):
        generate_analog_circuit(topology="not_a_real_topo")

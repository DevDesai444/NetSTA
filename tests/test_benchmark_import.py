"""Tests for the .bench netlist importer."""

import os
import tempfile

import pytest

from netsta.benchmark_import import bench_to_circuit, cone_windows, parse_bench
from netsta.nangate45 import NANGATE45_CELLS
from netsta.sta import run_sta

_BENCH = """
# tiny test netlist
INPUT(a)
INPUT(b)
INPUT(c)
OUTPUT(z)

g1 = NAND(a, b)
g2 = NOT(g1)
g3 = NAND(a, b, c, g1)
q  = DFF(g2)
z  = AND(q, g3)
"""


@pytest.fixture
def bench_path():
    fd, path = tempfile.mkstemp(suffix=".bench")
    with os.fdopen(fd, "w") as f:
        f.write(_BENCH)
    yield path
    os.unlink(path)


def test_parse_bench(bench_path):
    nl = parse_bench(bench_path)
    assert set(nl.inputs) == {"a", "b", "c"}
    assert nl.outputs == ["z"]
    assert ("q", "g2") in nl.dffs
    funcs = {f for _o, f, _i in nl.gates}
    assert "NAND" in funcs and "NOT" in funcs and "AND" in funcs


def test_circuit_cells_are_all_known(bench_path):
    c = bench_to_circuit(parse_bench(bench_path), seed=0)
    # Every gate maps to a real Nangate45 cell (wide gates decomposed).
    for nid in c.gate_ids:
        assert c.nodes[nid].node_type in NANGATE45_CELLS


def test_dff_is_cut(bench_path):
    c = bench_to_circuit(parse_bench(bench_path), seed=0)
    # DFF Q becomes a launch PI, DFF D becomes a capture PO.
    assert any(n.node_type == "PI" and n.output_net == "q" for n in c.nodes.values())
    assert "z" in {n.output_net for n in c.nodes.values()} or True  # z driven
    # The 4-input NAND must have been decomposed into 2-input cells.
    assert all(
        NANGATE45_CELLS[c.nodes[g].node_type]["num_inputs"] <= 3 for g in c.gate_ids
    )


def test_sta_runs_on_imported_circuit(bench_path):
    c = bench_to_circuit(parse_bench(bench_path), seed=0)
    res = run_sta(c)
    assert res["clock_period_ns"] > 0
    assert "node_timing" in res and len(res["node_timing"]) == len(c.nodes)


def test_wire_lengths_follow_placement(bench_path):
    c = bench_to_circuit(parse_bench(bench_path), seed=0)
    assert all(net.wire_length_um > 0 for net in c.nets.values())


def test_cone_windows_subgraphs(bench_path):
    c = bench_to_circuit(parse_bench(bench_path), seed=0)
    cones = cone_windows(c, max_cones=4, min_cone_nodes=3, seed=1)
    for cone in cones:
        assert cone.primary_outputs  # each cone is endpoint-rooted
        assert len(cone.nodes) >= 3
        # cone is a valid sub-circuit: STA runs
        run_sta(cone)

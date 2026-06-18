"""Tests for the EPFL flattened-assign Verilog reader."""

import os
import tempfile

import pytest

from netsta.benchmark_import import (
    _looks_like_epfl, bench_to_circuit, load_netlist, parse_epfl_verilog,
)
from netsta.nangate45 import NANGATE45_CELLS
from netsta.sta import run_sta


_EPFL_TINY = """
// tiny EPFL-style design
module tiny ( \\a[0] , \\a[1] , \\a[2] , \\f[0] , \\f[1] );
  input \\a[0] , \\a[1] , \\a[2] ;
  output \\f[0] , \\f[1] ;
  assign n1 = ~\\a[0]  & ~\\a[1] ;
  assign n2 = \\a[0]  & \\a[1] ;
  assign \\f[0]  = n1 | n2;
  assign \\f[1]  = ~\\a[2] ;
endmodule
"""


@pytest.fixture
def tiny_path():
    fd, path = tempfile.mkstemp(suffix=".v")
    with os.fdopen(fd, "w") as f:
        f.write(_EPFL_TINY)
    yield path
    os.unlink(path)


def test_parse_epfl_extracts_io(tiny_path):
    nl = parse_epfl_verilog(tiny_path)
    assert set(nl.inputs) == {"a_0", "a_1", "a_2"}
    assert set(nl.outputs) == {"f_0", "f_1"}


def test_parse_epfl_decomposes_assigns(tiny_path):
    nl = parse_epfl_verilog(tiny_path)
    # `~a & ~b` becomes: two NOT gates + one AND -> 3 gates per such assign
    # plus the OR (1 gate) + one extra NOT for f_1.
    funcs = {f for _o, f, _i in nl.gates}
    assert "AND" in funcs and "OR" in funcs and "NOT" in funcs


def test_epfl_circuit_uses_known_cells(tiny_path):
    c = bench_to_circuit(parse_epfl_verilog(tiny_path), seed=0)
    for nid in c.gate_ids:
        assert c.nodes[nid].node_type in NANGATE45_CELLS


def test_epfl_circuit_runs_sta(tiny_path):
    c = load_netlist(tiny_path, seed=0)
    r = run_sta(c)
    assert r["clock_period_ns"] > 0


def test_auto_detect_epfl_vs_gate_primitive(tiny_path):
    assert _looks_like_epfl(tiny_path) is True

"""
Import real gate-level benchmark netlists (ISCAS-85/'89, ITC'99, and other
.bench-format suites) into the project's Circuit model.

The synthetic generator makes random DAGs, which lack the long-range path
structure real circuits have — so a graph-blind MLP keeps pace with the GNN on
them. Real netlists have genuine depth and reconvergence, which is where
message passing earns its keep. We parse the standard .bench format, map each
logic gate onto the Nangate45 cell library (decomposing wide gates into 2-input
trees, since the library tops out at 3 inputs), cut sequential elements at
their boundaries, place the result, and hand it to the existing STA /
congestion / DRC labelers — so real structure flows through the exact same
schema-v9 pipeline as the synthetic data.

.bench grammar (ISCAS / ITC'99):
    INPUT(a)
    OUTPUT(z)
    g = NAND(a, b)
    q = DFF(d)

DFFs are cut for static timing: the Q output becomes a pseudo-PI launch point
and the D input becomes a pseudo-PO capture point — exactly how STA times a
sequential design between registers. fanin_cone() carves a large circuit into
many endpoint-rooted sub-circuits so one netlist yields many training graphs.
"""

import os
import random
import re
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .circuit_gen import Circuit, Net, Node, _assign_positions


_IO_RE = re.compile(r"^(INPUT|OUTPUT)\s*\(\s*([^)\s]+)\s*\)$", re.I)
_ASSIGN_RE = re.compile(r"^([^=\s]+)\s*=\s*([A-Za-z_][A-Za-z0-9_]*)\s*\(([^)]*)\)$")

# Wire length model: base + PITCH * manhattan(driver, sink) once placed, so the
# wire-delay edge feature actually correlates with the layout.
_WIRE_BASE_UM = 2.0
_WIRE_PITCH_UM = 2.0


@dataclass
class BenchNetlist:
    """Parsed .bench intermediate representation (pre-cell-mapping)."""
    name: str
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    gates: List[Tuple[str, str, List[str]]] = field(default_factory=list)  # (out, func, ins)
    dffs: List[Tuple[str, str]] = field(default_factory=list)              # (q, d)

    def stats(self) -> dict:
        return {
            "name": self.name,
            "inputs": len(self.inputs),
            "outputs": len(self.outputs),
            "gates": len(self.gates),
            "dffs": len(self.dffs),
        }


def parse_bench(path: str) -> BenchNetlist:
    """Parse one .bench file into a BenchNetlist. Robust to comments/dialects."""
    nl = BenchNetlist(name=os.path.splitext(os.path.basename(path))[0])
    with open(path, "r", errors="ignore") as f:
        for raw in f:
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            m = _IO_RE.match(line)
            if m:
                bucket = nl.inputs if m.group(1).upper() == "INPUT" else nl.outputs
                bucket.append(m.group(2))
                continue
            m = _ASSIGN_RE.match(line)
            if m:
                out, func, args = m.group(1), m.group(2).upper(), m.group(3)
                ins = [a.strip() for a in args.split(",") if a.strip()]
                if func.startswith("DFF"):
                    if ins:
                        nl.dffs.append((out, ins[0]))
                elif ins:
                    nl.gates.append((out, func, ins))
                # zero-arg assigns (constant ties) are dropped
                continue
            # Unrecognized construct — ignore so one odd line doesn't kill a parse.
    return nl


# ---------------------------------------------------------------------------
# Gate decomposition: map a .bench logic function onto Nangate45 cells.
# The library has 1-input (INV/BUF), 2-input, and a couple of 3-input cells;
# anything wider becomes a balanced 2-input tree (+ an inverter for the
# negated NAND/NOR/XNOR families). Exact boolean equivalence is irrelevant —
# only the timing graph (delays, depth, fanout) matters for the STA labels.
# ---------------------------------------------------------------------------

_DIRECT_CELL = {
    ("NAND", 2): "NAND2_X1", ("NAND", 3): "NAND3_X1",
    ("NOR", 2): "NOR2_X1", ("NOR", 3): "NOR3_X1",
    ("AND", 2): "AND2_X1", ("OR", 2): "OR2_X1",
    ("XOR", 2): "XOR2_X1", ("XNOR", 2): "XNOR2_X1",
}


def _emit_gate(func, ins, out_sig, add_gate, tmp, pending):
    func = func.upper()
    k = len(ins)
    if func in ("NOT", "INV"):
        pending.append((ins[0], add_gate("INV_X1", out_sig)))
        return
    if func in ("BUF", "BUFF", "BUFFER"):
        pending.append((ins[0], add_gate("BUF_X1", out_sig)))
        return
    if k == 1:
        # Degenerate single-input AND/OR/... — a buffer preserves the timing arc.
        pending.append((ins[0], add_gate("BUF_X1", out_sig)))
        return
    if (func, k) in _DIRECT_CELL:
        gid = add_gate(_DIRECT_CELL[(func, k)], out_sig)
        for s in ins:
            pending.append((s, gid))
        return

    # Wide gate -> balanced 2-input tree, inverting for the negated families.
    if func in ("AND", "NAND"):
        base, invert = "AND2_X1", func == "NAND"
    elif func in ("OR", "NOR"):
        base, invert = "OR2_X1", func == "NOR"
    elif func in ("XOR", "XNOR"):
        base, invert = "XOR2_X1", func == "XNOR"
    else:
        base, invert = "AND2_X1", False  # unknown function -> AND tree (timing only)

    target = tmp() if invert else out_sig
    _tree(base, ins, target, add_gate, tmp, pending)
    if invert:
        pending.append((target, add_gate("INV_X1", out_sig)))


def _tree(cell, ins, out_sig, add_gate, tmp, pending):
    """Reduce `ins` to `out_sig` through a balanced tree of 2-input `cell`."""
    level = list(ins)
    while len(level) > 2:
        nxt = []
        i = 0
        while i < len(level):
            if i + 1 < len(level):
                t = tmp()
                gid = add_gate(cell, t)
                pending.append((level[i], gid))
                pending.append((level[i + 1], gid))
                nxt.append(t)
                i += 2
            else:
                nxt.append(level[i])
                i += 1
        level = nxt
    gid = add_gate(cell, out_sig)
    pending.append((level[0], gid))
    pending.append((level[1], gid))


# ---------------------------------------------------------------------------
# Netlist -> Circuit
# ---------------------------------------------------------------------------


def bench_to_circuit(nl: BenchNetlist, name: Optional[str] = None, seed: int = 0) -> Circuit:
    """Build a placed Circuit from a parsed BenchNetlist.

    DFFs are cut: each Q becomes a pseudo-PI (launch) and each D feeds a
    pseudo-PO (capture), exposing the inter-register combinational graph the
    way STA times a sequential design.
    """
    rng = random.Random(seed)
    c = Circuit(name=name or nl.name)
    gctr, tctr, poctr = [0], [0], [0]
    pending: List[Tuple[str, str]] = []  # (signal, consumer_node_id), wired in pass 2

    def add_net(sig, driver_id):
        # First driver wins (guards against malformed multi-driven nets).
        c.nets.setdefault(sig, Net(name=sig, driver=driver_id))

    def add_pi(sig):
        nid = f"PI_{sig}"
        if nid in c.nodes:
            return
        c.nodes[nid] = Node(node_id=nid, node_type="PI", output_net=sig)
        c.primary_inputs.append(nid)
        add_net(sig, nid)

    def add_gate(cell, out_net):
        gid = f"g{gctr[0]}"
        gctr[0] += 1
        c.nodes[gid] = Node(node_id=gid, node_type=cell, output_net=out_net)
        c.gate_ids.append(gid)
        add_net(out_net, gid)
        return gid

    def tmp():
        s = f"__t{tctr[0]}"
        tctr[0] += 1
        return s

    def add_po(sig):
        pid = f"PO{poctr[0]}_{sig}"
        poctr[0] += 1
        c.nodes[pid] = Node(node_id=pid, node_type="PO")
        c.primary_outputs.append(pid)
        pending.append((sig, pid))

    # 1. PIs + DFF-Q launch points.
    for sig in nl.inputs:
        add_pi(sig)
    for q, _d in nl.dffs:
        add_pi(q)
    # 2. Gates (decomposed to known cells).
    for out_sig, func, ins in nl.gates:
        _emit_gate(func, ins, out_sig, add_gate, tmp, pending)
    # 3. Capture points: primary outputs + DFF-D inputs.
    for sig in nl.outputs:
        add_po(sig)
    for _q, d in nl.dffs:
        add_po(d)
    # 4. Wire consumers to driver nets (second pass resolves forward refs).
    for sig, cons in pending:
        net = c.nets.get(sig)
        if net is None:
            continue  # dangling / constant reference
        net.sinks.append(cons)
        node = c.nodes.get(cons)
        if node is not None:
            node.input_nets.append(sig)
    # 5. Place, then derive wire lengths from placement distance.
    c.positions, c.grid_size = _assign_positions(c, rng)
    _assign_wire_lengths(c)
    return c


def _assign_wire_lengths(c: Circuit) -> None:
    pos = c.positions
    for net in c.nets.values():
        dpos = pos.get(net.driver)
        if dpos is None or not net.sinks:
            net.wire_length_um = _WIRE_BASE_UM
            continue
        dists = [
            abs(dpos[0] - sp[0]) + abs(dpos[1] - sp[1])
            for s in net.sinks
            if (sp := pos.get(s)) is not None
        ]
        net.wire_length_um = _WIRE_BASE_UM + _WIRE_PITCH_UM * (max(dists) if dists else 0)


def load_bench_circuit(path: str, name: Optional[str] = None, seed: int = 0) -> Circuit:
    """parse_bench + bench_to_circuit convenience."""
    return bench_to_circuit(parse_bench(path), name=name, seed=seed)


# ---------------------------------------------------------------------------
# Structural-Verilog netlists (ISCAS-85 ships as gate-primitive .v)
# ---------------------------------------------------------------------------

_VPRIM = {
    "and": "AND", "nand": "NAND", "or": "OR", "nor": "NOR",
    "not": "NOT", "buf": "BUF", "xor": "XOR", "xnor": "XNOR",
}
# `nand NAND2_1 (out, a, b);` — instance name optional; first arg is the output.
_VGATE_RE = re.compile(
    r"\b(and|nand|or|nor|not|buf|xor|xnor)\b\s+(?:\w+\s*)?\(([^)]*)\)\s*;", re.I
)
_VIO_RE = re.compile(r"\b(input|output)\b\s+([^;]+);", re.I)


def parse_verilog(path: str) -> BenchNetlist:
    """Parse a gate-primitive structural Verilog netlist into a BenchNetlist."""
    with open(path, "r", errors="ignore") as f:
        text = f.read()
    text = re.sub(r"//[^\n]*", "", text)            # line comments
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)  # block comments

    nl = BenchNetlist(name=os.path.splitext(os.path.basename(path))[0])
    for kind, body in _VIO_RE.findall(text):
        names = [s.strip() for s in body.replace("\n", " ").split(",") if s.strip()]
        (nl.inputs if kind.lower() == "input" else nl.outputs).extend(names)
    for prim, args in _VGATE_RE.findall(text):
        toks = [a.strip() for a in args.replace("\n", " ").split(",") if a.strip()]
        if len(toks) < 2:
            continue
        out, ins = toks[0], toks[1:]
        nl.gates.append((out, _VPRIM[prim.lower()], ins))
    return nl


# EPFL benchmarks ship as `assign w = ~a & b;` style. We parse each RHS into a
# series of synthetic 2-input gates so the timing graph still captures the
# right depth/fanout structure (exact boolean equivalence isn't the goal here;
# the GNN learns to predict our STA's output on this graph).
_EPFL_ASSIGN_RE = re.compile(r"\bassign\s+([^=;]+?)\s*=\s*([^;]+);", re.S)
# Escape names: `\foo[3] ` — backslash, then non-space, terminated by whitespace.
_ESCAPE_NAME_RE = re.compile(r"\\(\S+?)\s")


def _normalize_signal(s: str) -> str:
    """Strip whitespace and collapse escape-name spaces -> underscore form."""
    s = s.strip()
    if s.startswith("\\"):
        s = s[1:].rstrip()
    # `a[3]` -> `a_3`; `b[10]` -> `b_10`; preserves identifier sanity downstream.
    return s.replace("[", "_").replace("]", "").replace(" ", "")


def _epfl_strip_escapes(text: str) -> str:
    """Convert `\\name ` (backslash, name, trailing whitespace) to plain `name`."""
    return _ESCAPE_NAME_RE.sub(lambda m: m.group(1) + " ", text)


def _parse_epfl_rhs(rhs: str, out_sig: str, gates: list, tmp_counter: list) -> None:
    """Parse a single assign RHS into gate tuples that produce out_sig.

    Recognized operators (binary, no precedence — EPFL's outputs are already
    pre-flattened to 2-operand assignments by their ABC export):
        a & b   -> AND
        a | b   -> OR
        ~a      -> NOT (unary; emits an intermediate)
        a       -> BUF
    Anything weirder (parens, &&, etc.) falls back to a BUF on the first symbol;
    we keep the timing graph well-formed even if the boolean isn't exact.
    """
    rhs = rhs.strip()
    # NOT prefix on operands inlined first: `~a & ~b` -> a NOT-tree then AND.
    # Split on the first top-level `&` or `|`.
    op = None
    depth = 0
    split_at = -1
    for i, ch in enumerate(rhs):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif depth == 0 and ch in "&|" and (i == 0 or rhs[i - 1] != ch):
            op = ch
            split_at = i
            break
    if split_at == -1:
        # Single operand (possibly with leading ~).
        token = rhs.strip().lstrip("(").rstrip(")")
        if token.startswith("~"):
            inner = _normalize_signal(token[1:])
            gates.append((out_sig, "NOT", [inner]))
        else:
            gates.append((out_sig, "BUF", [_normalize_signal(token)]))
        return

    left = rhs[:split_at].strip()
    right = rhs[split_at + 1 :].strip()

    def _operand(expr: str) -> str:
        """Return a signal name for an operand, emitting a NOT gate if inverted."""
        expr = expr.strip().lstrip("(").rstrip(")").strip()
        if expr.startswith("~"):
            inner = _normalize_signal(expr[1:])
            tmp = f"__epfl_t{tmp_counter[0]}"
            tmp_counter[0] += 1
            gates.append((tmp, "NOT", [inner]))
            return tmp
        return _normalize_signal(expr)

    a = _operand(left)
    b = _operand(right)
    func = "AND" if op == "&" else "OR"
    gates.append((out_sig, func, [a, b]))


def parse_epfl_verilog(path: str) -> BenchNetlist:
    """Parse an EPFL-style flattened-assign Verilog netlist into a BenchNetlist.

    EPFL benchmark format: every signal is defined by `assign w = expr;` where
    expr is at most one binary op of operands (possibly negated). Output sees
    a clean DAG of AND/OR/NOT gates, ready for the same downstream pipeline.
    """
    with open(path, "r", errors="ignore") as f:
        text = f.read()
    text = re.sub(r"//[^\n]*", "", text)
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    text = _epfl_strip_escapes(text)

    nl = BenchNetlist(name=os.path.splitext(os.path.basename(path))[0])
    for kind, body in _VIO_RE.findall(text):
        names = [_normalize_signal(s) for s in body.replace("\n", " ").split(",") if s.strip()]
        (nl.inputs if kind.lower() == "input" else nl.outputs).extend(names)

    tmp_counter = [0]
    for lhs, rhs in _EPFL_ASSIGN_RE.findall(text):
        out_sig = _normalize_signal(lhs)
        _parse_epfl_rhs(rhs, out_sig, nl.gates, tmp_counter)
    return nl


def _looks_like_epfl(path: str) -> bool:
    """Heuristic: file uses `assign` and no gate primitives anywhere.

    EPFL multiplier's port list alone is ~700 lines — looking only at the head
    misses the body. Scan the whole file (it's still ~1ms even for big files).
    """
    try:
        with open(path, "r", errors="ignore") as f:
            text = f.read()
    except Exception:
        return False
    # Strip comments before checking for `assign` and gate primitives.
    text = re.sub(r"//[^\n]*", "", text)
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return "assign " in text and not _VGATE_RE.search(text)


def load_netlist(path: str, name: Optional[str] = None, seed: int = 0) -> Circuit:
    """Load a .bench or structural-Verilog (.v) netlist into a Circuit.

    Dispatches between three readers:
        - parse_bench       for `.bench` (ITC'99, OpenABC, ISCAS-89 .bench)
        - parse_epfl_verilog for EPFL's `assign`-based Verilog
        - parse_verilog     for ISCAS-85 gate-primitive Verilog
    """
    lp = path.lower()
    if lp.endswith(".v"):
        parser = parse_epfl_verilog if _looks_like_epfl(path) else parse_verilog
    else:
        parser = parse_bench
    return bench_to_circuit(parser(path), name=name, seed=seed)


# ---------------------------------------------------------------------------
# Fan-in cone windowing: one big netlist -> many real-structure sub-circuits.
# ---------------------------------------------------------------------------


def _predecessors(circuit: Circuit) -> Dict[str, List[str]]:
    preds: Dict[str, List[str]] = {nid: [] for nid in circuit.nodes}
    for net in circuit.nets.values():
        for sink in net.sinks:
            if sink in preds:
                preds[sink].append(net.driver)
    return preds


def fanin_cone(
    circuit: Circuit,
    target_id: str,
    max_nodes: int = 6000,
    name: Optional[str] = None,
    seed: int = 0,
    _preds: Optional[Dict[str, List[str]]] = None,
) -> Optional[Circuit]:
    """Return the transitive fan-in cone of `target_id` as a standalone Circuit.

    The cone is predecessor-closed, so every internal node keeps all its
    drivers; original PIs (and DFF-Q launch points) inside the cone stay PIs,
    and `target_id` becomes the sole PO. Returns None for trivially small cones
    (nothing for the GNN to learn from). The sub-circuit is re-placed so its
    congestion/DRC labels are self-consistent.
    """
    preds = _preds if _preds is not None else _predecessors(circuit)
    seen = {target_id}
    queue = deque([target_id])
    while queue and len(seen) < max_nodes:
        n = queue.popleft()
        for p in preds.get(n, []):
            if p not in seen:
                seen.add(p)
                queue.append(p)

    if len(seen) < 4:
        return None

    rng = random.Random(seed)
    sub = Circuit(name=name or f"{circuit.name}__cone_{target_id}")
    for nid in seen:
        src = circuit.nodes[nid]
        sub.nodes[nid] = Node(
            node_id=nid, node_type=src.node_type, output_net=src.output_net
        )
        if nid == target_id:
            sub.primary_outputs.append(nid)
        elif src.node_type == "PI":
            sub.primary_inputs.append(nid)
        else:
            sub.gate_ids.append(nid)

    for net in circuit.nets.values():
        if net.driver in seen:
            sinks_in = [s for s in net.sinks if s in seen]
            if sinks_in:
                sub.nets[net.name] = Net(
                    name=net.name, driver=net.driver, sinks=sinks_in,
                )

    if not sub.primary_inputs or not any(net.sinks for net in sub.nets.values()):
        return None

    sub.positions, sub.grid_size = _assign_positions(sub, rng)
    _assign_wire_lengths(sub)
    return sub


def cone_windows(
    circuit: Circuit,
    max_cones: int = 40,
    min_cone_nodes: int = 8,
    max_cone_nodes: int = 6000,
    seed: int = 0,
) -> List[Circuit]:
    """Carve a circuit into up to `max_cones` endpoint-rooted sub-circuits.

    Endpoints are the circuit's POs (real outputs + DFF-D capture points),
    shuffled so we sample a spread of endpoints rather than just the first few.
    Overlapping cones from one circuit are fine — they all land in the same
    train/test split (we split by source circuit), so this is augmentation, not
    leakage. Used to turn ~100 real netlists into thousands of training graphs.
    """
    preds = _predecessors(circuit)
    endpoints = list(circuit.primary_outputs)
    rng = random.Random(seed)
    rng.shuffle(endpoints)

    out: List[Circuit] = []
    for i, ep in enumerate(endpoints):
        if len(out) >= max_cones:
            break
        cone = fanin_cone(
            circuit, ep, max_nodes=max_cone_nodes,
            name=f"{circuit.name}__c{i}", seed=seed + i, _preds=preds,
        )
        if cone is None or len(cone.nodes) < min_cone_nodes:
            continue
        out.append(cone)
    return out

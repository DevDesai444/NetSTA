"""
Classical Static Timing Analysis (STA) engine.

Computes arrival times (AT), required times (RT), and slack for every node
in a combinational circuit using topological traversal. These values are the
ground-truth labels the GNN learns to predict.

Forward pass:  AT[node] = max over input drivers of
                          (AT[driver] + gate_delay(driver) + wire_delay)
Backward pass: RT[node] = min over output sinks of
                          (RT[sink] - gate_delay(sink) - wire_delay)
Slack:         RT - AT (positive = timing met, negative = violation)

Clock period defaults to 1.2 × max PO arrival time (20 % margin) when not
provided, so the worst slack is ~0 and tight paths land just above it.
"""

from collections import defaultdict, deque
from typing import Optional, List, Dict, Tuple

from .circuit_gen import Circuit
from .nangate45 import (
    NANGATE45_CELLS,
    WIRE_CAP_PER_UM,
    compute_gate_delay,
    compute_wire_delay,
)


# Absolute slack (ns) at or below which a node is on the critical path. Picked
# so the digital STA's typical slack distribution puts ~10–25 % of nodes below
# the threshold across the dataset's circuit size range, giving non-degenerate
# class balance for both small and large graphs. Used by the critical-path
# classification head's label.
CRITICAL_SLACK_THRESHOLD_NS = 0.05

# Capacitance assumption for primary output pads (fF).
PO_LOAD_CAP_FF = 2.0


# ---------------------------------------------------------------------------
# DAG plumbing
# ---------------------------------------------------------------------------


def _build_dag(circuit: Circuit) -> Tuple[Dict[str, List[str]], Dict[str, int]]:
    """Return (successors, in_degree) over the full node set.

    Each unique (driver, sink) pair contributes exactly one edge. PIs, gates,
    and POs are all keys in both dicts so the topological sort can start from
    any zero-in-degree source (typically the PIs).
    """
    nodes = circuit.primary_inputs + circuit.gate_ids + circuit.primary_outputs
    succ: Dict[str, List[str]] = {nid: [] for nid in nodes}
    in_degree: Dict[str, int] = {nid: 0 for nid in nodes}

    seen_edges = set()  # guard against duplicate edges in malformed nets
    for net in circuit.nets.values():
        driver = net.driver
        if driver not in succ:
            continue
        for sink in net.sinks:
            if sink not in in_degree:
                continue
            edge = (driver, sink)
            if edge in seen_edges:
                continue
            seen_edges.add(edge)
            succ[driver].append(sink)
            in_degree[sink] += 1

    return succ, in_degree


def topological_sort(circuit: Circuit) -> List[str]:
    """Return ALL circuit nodes in topological order (Kahn's algorithm).

    PIs are sources (in_degree 0), POs sinks. If the graph contains a cycle
    the trailing nodes are dropped — callers should treat a short return as
    a malformed circuit.
    """
    succ, in_degree = _build_dag(circuit)
    queue = deque(nid for nid, d in in_degree.items() if d == 0)
    order: List[str] = []
    while queue:
        node = queue.popleft()
        order.append(node)
        for child in succ[node]:
            in_degree[child] -= 1
            if in_degree[child] == 0:
                queue.append(child)
    return order


# ---------------------------------------------------------------------------
# Per-node helpers
# ---------------------------------------------------------------------------


def _node_load_cap(circuit: Circuit, node_id: str) -> float:
    """Total load capacitance (fF) on the node's output net."""
    node = circuit.nodes[node_id]
    if not node.output_net or node.output_net not in circuit.nets:
        return 0.0
    net = circuit.nets[node.output_net]
    total = 0.0
    for sink_id in net.sinks:
        sink_node = circuit.nodes.get(sink_id)
        if sink_node is None:
            continue
        if sink_node.node_type == "PO":
            total += PO_LOAD_CAP_FF
        elif sink_node.node_type in NANGATE45_CELLS:
            total += NANGATE45_CELLS[sink_node.node_type]["input_cap"]
    total += WIRE_CAP_PER_UM * net.wire_length_um
    return total


def _gate_delay(circuit: Circuit, node_id: str, load_cap: float) -> float:
    """Delay through the gate driving `node_id`'s output net.

    PIs and POs have no gate delay (they're pseudo-pins). Anything else looks
    up its Nangate45 cell. Unknown cell types contribute zero rather than
    crashing — keeps `run_sta` robust to lightly malformed test circuits.
    """
    node = circuit.nodes[node_id]
    if node.node_type in ("PI", "PO"):
        return 0.0
    if node.node_type not in NANGATE45_CELLS:
        return 0.0
    return compute_gate_delay(node.node_type, load_cap)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_sta(circuit: Circuit, clock_period_ns: Optional[float] = None) -> dict:
    """
    Run static timing analysis on a combinational circuit.

    Returns:
      {
        "node_timing": {node_id: {arrival_time, required_time, slack,
                                   is_critical, load_cap, gate_delay,
                                   logical_depth}},
        "clock_period_ns": float,
        "max_arrival_time_ns": float,
        "min_slack_ns": float,
        "num_critical_nodes": int,
      }

    `is_critical` is set when `slack <= CRITICAL_SLACK_THRESHOLD_NS` —
    a fixed absolute threshold, not a per-graph quantile. The previous
    quantile-based label was trivially predictable from in-graph depth
    ranking, which is why even Linear Regression scored AUC ≈ 0.94 on it.
    """
    topo = topological_sort(circuit)

    # Pre-compute per-node load_cap once — referenced in both passes.
    load_cap: Dict[str, float] = {nid: _node_load_cap(circuit, nid) for nid in topo}

    # Pre-compute per-(driver, sink) wire delay using the sink's input cap so
    # the delay represents the physical net loading at that pin.
    wire_delay: Dict[Tuple[str, str], float] = {}
    for net in circuit.nets.values():
        if net.driver not in circuit.nodes:
            continue
        for sink_id in net.sinks:
            if sink_id not in circuit.nodes:
                continue
            sink_node = circuit.nodes[sink_id]
            if sink_node.node_type == "PO":
                sink_cap = PO_LOAD_CAP_FF
            elif sink_node.node_type in NANGATE45_CELLS:
                sink_cap = NANGATE45_CELLS[sink_node.node_type]["input_cap"]
            else:
                sink_cap = PO_LOAD_CAP_FF
            wire_delay[(net.driver, sink_id)] = compute_wire_delay(
                net.wire_length_um, sink_cap
            )

    # Reverse adjacency: sink -> [(driver, net), ...] so the forward pass can
    # iterate input drivers without re-scanning every net per node.
    inputs_of: Dict[str, List[str]] = defaultdict(list)
    for net in circuit.nets.values():
        if net.driver not in circuit.nodes:
            continue
        for sink_id in net.sinks:
            if sink_id in circuit.nodes:
                inputs_of[sink_id].append(net.driver)

    succ, _ = _build_dag(circuit)

    # --- Forward: arrival_time + logical_depth ---
    arrival_time: Dict[str, float] = {nid: 0.0 for nid in topo}
    logical_depth: Dict[str, int] = {nid: 0 for nid in topo}
    gate_delay_used: Dict[str, float] = {nid: 0.0 for nid in topo}

    for nid in topo:
        if nid in circuit.primary_inputs:
            arrival_time[nid] = 0.0
            logical_depth[nid] = 0
            gate_delay_used[nid] = 0.0
            continue

        node = circuit.nodes[nid]
        best_at = 0.0
        best_depth = 0
        for drv in inputs_of[nid]:
            drv_at = arrival_time[drv]
            drv_gate_delay = _gate_delay(circuit, drv, load_cap[drv])
            wd = wire_delay.get((drv, nid), 0.0)
            pin_at = drv_at + drv_gate_delay + wd
            if pin_at > best_at:
                best_at = pin_at
            if logical_depth[drv] + 1 > best_depth:
                best_depth = logical_depth[drv] + 1

        arrival_time[nid] = best_at
        logical_depth[nid] = best_depth
        # gate_delay at this node is its OWN delay (used by callers as
        # gate-intrinsic timing info, distinct from incoming wire delay).
        gate_delay_used[nid] = _gate_delay(circuit, nid, load_cap[nid])

    max_at = max(
        (arrival_time[po] for po in circuit.primary_outputs),
        default=max(arrival_time.values(), default=0.0),
    )
    if clock_period_ns is None:
        # 20 % margin so the worst-case slack lands at ~0.2 × max_AT.
        clock_period_ns = max(max_at * 1.2, 1e-3)

    # --- Backward: required_time ---
    required_time: Dict[str, float] = {nid: clock_period_ns for nid in topo}

    for nid in reversed(topo):
        children = succ[nid]
        if nid in circuit.primary_outputs:
            required_time[nid] = clock_period_ns
            continue
        if not children:
            # Floating node (no fanout, not a PO). RT defaults to clock so its
            # slack reflects how much margin its arrival leaves before the
            # clock edge — same convention industry tools use for unconstrained
            # endpoints.
            required_time[nid] = clock_period_ns
            continue

        best_rt = float("inf")
        for child in children:
            child_rt = required_time[child]
            child_gate_delay = _gate_delay(circuit, child, load_cap[child])
            wd = wire_delay.get((nid, child), 0.0)
            rt_at_node = child_rt - child_gate_delay - wd
            if rt_at_node < best_rt:
                best_rt = rt_at_node
        required_time[nid] = best_rt if best_rt < float("inf") else clock_period_ns

    # --- Slack + critical-path label ---
    slack = {nid: required_time[nid] - arrival_time[nid] for nid in topo}
    min_slack = min(slack.values()) if slack else 0.0
    is_critical = {nid: slack[nid] <= CRITICAL_SLACK_THRESHOLD_NS for nid in topo}

    # Cover any nodes missing from topo (cycle survivors) with safe defaults.
    for nid in circuit.nodes:
        arrival_time.setdefault(nid, 0.0)
        required_time.setdefault(nid, clock_period_ns)
        slack.setdefault(nid, clock_period_ns)
        is_critical.setdefault(nid, False)
        logical_depth.setdefault(nid, 0)
        gate_delay_used.setdefault(nid, 0.0)
        load_cap.setdefault(nid, 0.0)

    node_timing = {
        nid: {
            "arrival_time": arrival_time[nid],
            "required_time": required_time[nid],
            "slack": slack[nid],
            "is_critical": is_critical[nid],
            "load_cap": load_cap[nid],
            "gate_delay": gate_delay_used[nid],
            "logical_depth": logical_depth[nid],
        }
        for nid in circuit.nodes
    }

    return {
        "node_timing": node_timing,
        "clock_period_ns": float(clock_period_ns),
        "max_arrival_time_ns": float(max_at),
        "min_slack_ns": float(min_slack),
        "num_critical_nodes": int(sum(1 for v in is_critical.values() if v)),
    }

"""
Classical Static Timing Analysis (STA) engine.

Computes arrival times (AT), required times (RT), and slack for
every node in a combinational circuit using topological traversal.
This serves as ground truth for GNN training.
"""

from collections import deque
from typing import Optional, List, Dict

from .circuit_gen import Circuit
from .nangate45 import (
    NANGATE45_CELLS,
    compute_gate_delay,
    compute_wire_delay,
)


def topological_sort(circuit: Circuit) -> List[str]:
    """Return gate and PO node IDs in topological order."""
    in_degree = {}
    adj = {}  # node_id -> list of successor node_ids

    all_nodes = circuit.gate_ids + circuit.primary_outputs
    for nid in all_nodes:
        in_degree[nid] = 0
        adj[nid] = []

    # Build adjacency from nets
    for net in circuit.nets.values():
        driver = net.driver
        for sink in net.sinks:
            if sink in in_degree:
                in_degree[sink] += 1
            if driver in adj:
                adj[driver].append(sink)
            elif driver in [p for p in circuit.primary_inputs]:
                # PI drives into the circuit
                if driver not in adj:
                    adj[driver] = []
                adj[driver].append(sink)
                # sink gets an extra in-degree from PI
                if sink in in_degree:
                    in_degree[sink] += 1

    # Add PIs with zero in-degree as sources
    queue = deque()
    for pi in circuit.primary_inputs:
        queue.append(pi)

    for nid in all_nodes:
        if in_degree.get(nid, 0) == 0 and nid not in circuit.primary_inputs:
            queue.append(nid)

    order = []
    visited = set()
    while queue:
        node = queue.popleft()
        if node in visited:
            continue
        visited.add(node)
        order.append(node)
        for succ in adj.get(node, []):
            in_degree[succ] -= 1
            if in_degree[succ] == 0:
                queue.append(succ)

    return order


def _compute_load_cap(circuit: Circuit, node_id: str) -> float:
    """Compute total load capacitance on a node's output net (fF)."""
    node = circuit.nodes[node_id]
    if not node.output_net:
        return 0.0

    net = circuit.nets[node.output_net]
    total_cap = 0.0

    for sink_id in net.sinks:
        sink_node = circuit.nodes[sink_id]
        if sink_node.node_type == "PO":
            total_cap += 2.0  # output pad capacitance estimate
        elif sink_node.node_type in NANGATE45_CELLS:
            total_cap += NANGATE45_CELLS[sink_node.node_type]["input_cap"]

    # Add wire capacitance
    from .nangate45 import WIRE_CAP_PER_UM
    total_cap += WIRE_CAP_PER_UM * net.wire_length_um

    return total_cap


def run_sta(circuit: Circuit, clock_period_ns: Optional[float] = None) -> dict:
    """
    Run static timing analysis on a combinational circuit.

    Returns a dict with per-node timing:
      {node_id: {"arrival_time", "required_time", "slack", "is_critical",
                  "load_cap", "gate_delay", "logical_depth"}}

    If clock_period_ns is None, it is set to 1.2x the maximum arrival time
    (giving 20% timing margin).
    """
    topo_order = topological_sort(circuit)

    arrival_time = {}
    gate_delay = {}
    load_cap = {}
    logical_depth = {}

    # --- Forward pass: compute arrival times ---
    # Initialize PIs
    for pi in circuit.primary_inputs:
        arrival_time[pi] = 0.0
        gate_delay[pi] = 0.0
        load_cap[pi] = _compute_load_cap(circuit, pi)
        logical_depth[pi] = 0

    for node_id in topo_order:
        if node_id in circuit.primary_inputs:
            continue

        node = circuit.nodes[node_id]

        if node.node_type == "PO":
            # PO arrival = max arrival of its input
            max_at = 0.0
            max_depth = 0
            for net_name in node.input_nets:
                net = circuit.nets[net_name]
                driver_id = net.driver
                driver_at = arrival_time.get(driver_id, 0.0)
                driver_load = _compute_load_cap(circuit, driver_id)

                # Wire delay from driver to this PO
                w_delay = compute_wire_delay(net.wire_length_um, 2.0)

                # Gate delay of driver
                driver_node = circuit.nodes[driver_id]
                if driver_node.node_type in NANGATE45_CELLS:
                    g_delay = compute_gate_delay(driver_node.node_type, driver_load)
                else:
                    g_delay = 0.0

                total_at = driver_at + g_delay + w_delay
                if total_at > max_at:
                    max_at = total_at
                max_depth = max(max_depth, logical_depth.get(driver_id, 0))

            arrival_time[node_id] = max_at
            gate_delay[node_id] = 0.0
            load_cap[node_id] = 0.0
            logical_depth[node_id] = max_depth + 1
        else:
            # Logic gate
            cell_name = node.node_type
            node_load = _compute_load_cap(circuit, node_id)
            g_delay = compute_gate_delay(cell_name, node_load)

            max_at = 0.0
            max_depth = 0
            for net_name in node.input_nets:
                net = circuit.nets[net_name]
                driver_id = net.driver
                driver_at = arrival_time.get(driver_id, 0.0)
                driver_load = _compute_load_cap(circuit, driver_id)

                # Driver gate delay
                driver_node = circuit.nodes[driver_id]
                if driver_node.node_type in NANGATE45_CELLS:
                    d_delay = compute_gate_delay(driver_node.node_type, driver_load)
                else:
                    d_delay = 0.0

                # Wire delay
                w_delay = compute_wire_delay(net.wire_length_um, node_load)

                pin_at = driver_at + d_delay + w_delay
                if pin_at > max_at:
                    max_at = pin_at
                max_depth = max(max_depth, logical_depth.get(driver_id, 0))

            arrival_time[node_id] = max_at
            gate_delay[node_id] = g_delay
            load_cap[node_id] = node_load
            logical_depth[node_id] = max_depth + 1

    # Determine clock period
    max_at = max(arrival_time.get(po, 0.0) for po in circuit.primary_outputs)
    if clock_period_ns is None:
        clock_period_ns = max_at * 1.2 if max_at > 0 else 1.0

    # --- Backward pass: compute required times ---
    required_time = {}

    # Initialize PO required times
    for po in circuit.primary_outputs:
        required_time[po] = clock_period_ns

    # Reverse topological order
    for node_id in reversed(topo_order):
        if node_id in [p for p in circuit.primary_outputs]:
            continue
        if node_id in circuit.primary_inputs:
            continue

        node = circuit.nodes[node_id]
        output_net_name = node.output_net
        if not output_net_name:
            required_time[node_id] = clock_period_ns
            continue

        net = circuit.nets[output_net_name]
        min_rt = float("inf")

        for sink_id in net.sinks:
            sink_rt = required_time.get(sink_id, clock_period_ns)
            sink_node = circuit.nodes[sink_id]

            # Subtract this gate's delay and wire delay to sink
            if sink_node.node_type in NANGATE45_CELLS:
                sink_load = _compute_load_cap(circuit, sink_id)
                s_delay = compute_gate_delay(sink_node.node_type, sink_load)
            else:
                s_delay = 0.0

            w_delay = compute_wire_delay(
                net.wire_length_um,
                NANGATE45_CELLS[sink_node.node_type]["input_cap"]
                if sink_node.node_type in NANGATE45_CELLS
                else 2.0,
            )

            rt_at_node = sink_rt - s_delay - w_delay
            if rt_at_node < min_rt:
                min_rt = rt_at_node

        required_time[node_id] = min_rt if min_rt < float("inf") else clock_period_ns

    # Also set PI required times
    for pi in circuit.primary_inputs:
        node = circuit.nodes[pi]
        net = circuit.nets[node.output_net]
        min_rt = float("inf")
        for sink_id in net.sinks:
            sink_rt = required_time.get(sink_id, clock_period_ns)
            sink_node = circuit.nodes[sink_id]
            if sink_node.node_type in NANGATE45_CELLS:
                sink_load = _compute_load_cap(circuit, sink_id)
                s_delay = compute_gate_delay(sink_node.node_type, sink_load)
            else:
                s_delay = 0.0
            w_delay = compute_wire_delay(
                net.wire_length_um,
                NANGATE45_CELLS[sink_node.node_type]["input_cap"]
                if sink_node.node_type in NANGATE45_CELLS
                else 2.0,
            )
            rt_at_node = sink_rt - s_delay - w_delay
            if rt_at_node < min_rt:
                min_rt = rt_at_node
        required_time[pi] = min_rt if min_rt < float("inf") else clock_period_ns

    # --- Compute slack ---
    slack = {}
    for node_id in arrival_time:
        at = arrival_time[node_id]
        rt = required_time.get(node_id, clock_period_ns)
        slack[node_id] = rt - at

    # Critical path: nodes within 30% of the minimum slack
    min_slack = min(slack.values()) if slack else 0.0
    slack_range = max(slack.values()) - min_slack if slack else 1.0
    threshold = min_slack + 0.30 * slack_range if slack_range > 0 else min_slack

    is_critical = {nid: slack[nid] <= threshold for nid in slack}

    # Build results (gates only, excluding PIs and POs for GNN training)
    results = {}
    for node_id in circuit.gate_ids:
        results[node_id] = {
            "arrival_time": arrival_time.get(node_id, 0.0),
            "required_time": required_time.get(node_id, clock_period_ns),
            "slack": slack.get(node_id, 0.0),
            "is_critical": is_critical.get(node_id, False),
            "load_cap": load_cap.get(node_id, 0.0),
            "gate_delay": gate_delay.get(node_id, 0.0),
            "logical_depth": logical_depth.get(node_id, 0),
        }

    # Also include PIs and POs for completeness
    for node_id in circuit.primary_inputs + circuit.primary_outputs:
        results[node_id] = {
            "arrival_time": arrival_time.get(node_id, 0.0),
            "required_time": required_time.get(node_id, clock_period_ns),
            "slack": slack.get(node_id, 0.0),
            "is_critical": is_critical.get(node_id, False),
            "load_cap": load_cap.get(node_id, 0.0),
            "gate_delay": gate_delay.get(node_id, 0.0),
            "logical_depth": logical_depth.get(node_id, 0),
        }

    return {
        "node_timing": results,
        "clock_period_ns": clock_period_ns,
        "max_arrival_time_ns": max_at,
        "min_slack_ns": min_slack,
        "num_critical_nodes": sum(1 for v in is_critical.values() if v),
    }

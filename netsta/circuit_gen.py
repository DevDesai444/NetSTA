"""
Synthetic combinational circuit generator using Nangate45 cells.

Generates random gate-level netlists as directed acyclic graphs (DAGs)
with realistic topology, fanout distribution, and cell selection.
"""

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .nangate45 import NANGATE45_CELLS, CELL_NAMES, DEFAULT_WIRE_LENGTH_UM


@dataclass
class Net:
    """A wire connecting a driver to one or more sinks."""
    name: str
    driver: str
    sinks: List[str] = field(default_factory=list)
    wire_length_um: float = DEFAULT_WIRE_LENGTH_UM


@dataclass
class Node:
    """A gate or primary I/O in the circuit."""
    node_id: str
    node_type: str
    input_nets: List[str] = field(default_factory=list)
    output_net: str = ""


@dataclass
class Circuit:
    """A combinational circuit netlist with synthetic placement.

    Supports both digital (Nangate45 cells) and analog (BSIM-like devices)
    via the `is_analog` flag plus per-node `analog_params` and
    `symmetry_groups` side-tables. Digital circuits leave those empty.
    """
    name: str
    nodes: dict = field(default_factory=dict)
    nets: dict = field(default_factory=dict)
    primary_inputs: List[str] = field(default_factory=list)
    primary_outputs: List[str] = field(default_factory=list)
    gate_ids: List[str] = field(default_factory=list)
    # 2D grid placement: node_id -> (x, y). Populated by generate_circuit.
    positions: Dict[str, Tuple[int, int]] = field(default_factory=dict)
    grid_size: Tuple[int, int] = (0, 0)
    # Analog-only side tables.
    is_analog: bool = False
    analog_params: Dict[str, dict] = field(default_factory=dict)
    # Matched-device groups: node_id -> group_id (0 = unmatched).
    symmetry_groups: Dict[str, int] = field(default_factory=dict)


def _assign_positions(circuit: Circuit, rng: random.Random) -> Tuple[Dict[str, Tuple[int, int]], Tuple[int, int]]:
    """Place each node on an integer 2D grid.

    PIs occupy column x=0, POs column x=grid_w-1, gates fill the interior.
    Grid size scales with total node count so density stays roughly constant.
    """
    n_total = len(circuit.nodes)
    side = max(6, int(math.sqrt(n_total) * 1.6))
    grid_w = grid_h = side

    positions: Dict[str, Tuple[int, int]] = {}

    # PIs evenly along the left edge.
    n_pis = max(1, len(circuit.primary_inputs))
    for i, pi_id in enumerate(circuit.primary_inputs):
        y = min(grid_h - 1, int(i * grid_h / n_pis))
        positions[pi_id] = (0, y)

    # POs evenly along the right edge.
    n_pos = max(1, len(circuit.primary_outputs))
    for i, po_id in enumerate(circuit.primary_outputs):
        y = min(grid_h - 1, int(i * grid_h / n_pos))
        positions[po_id] = (grid_w - 1, y)

    # Gates randomly in the interior columns.
    interior = [(x, y) for x in range(1, grid_w - 1) for y in range(grid_h)]
    rng.shuffle(interior)
    for i, gate_id in enumerate(circuit.gate_ids):
        positions[gate_id] = interior[i % len(interior)]

    return positions, (grid_w, grid_h)


def generate_circuit(
    num_inputs: int = 8,
    num_gates: int = 30,
    num_outputs: int = 4,
    max_fanout: int = 4,
    seed: Optional[int] = None,
    name: str = "circuit",
    allowed_cells: Optional[List[str]] = None,
) -> Circuit:
    """
    Generate a random combinational circuit.

    The circuit is built layer by layer. Each gate randomly selects
    its inputs from available signals (PIs + outputs of earlier gates),
    ensuring a valid DAG structure.

    Pass `allowed_cells` to restrict gate selection to a subset of the
    Nangate45 cell library (used by topology-generalization benchmarks).
    """
    if seed is not None:
        random.seed(seed)
    cell_pool = allowed_cells if allowed_cells else CELL_NAMES

    circuit = Circuit(name=name)
    available_signals = []  # (node_id, net_name) pairs of available driver outputs
    net_counter = 0
    fanout_count = {}  # net_name -> current fanout

    # Create primary inputs
    for i in range(num_inputs):
        pi_id = f"PI_{i}"
        net_name = f"n{net_counter}"
        net_counter += 1

        node = Node(node_id=pi_id, node_type="PI", output_net=net_name)
        net = Net(name=net_name, driver=pi_id)

        circuit.nodes[pi_id] = node
        circuit.nets[net_name] = net
        circuit.primary_inputs.append(pi_id)
        available_signals.append((pi_id, net_name))
        fanout_count[net_name] = 0

    # Create gates layer by layer
    for i in range(num_gates):
        # Pick a random cell type
        cell_name = random.choice(cell_pool)
        cell_info = NANGATE45_CELLS[cell_name]
        n_inputs = cell_info["num_inputs"]

        gate_id = f"G_{i}"
        output_net_name = f"n{net_counter}"
        net_counter += 1

        # Select input signals (prefer signals with low fanout for realism)
        if len(available_signals) < n_inputs:
            # Not enough signals; reuse what we have
            chosen = [random.choice(available_signals) for _ in range(n_inputs)]
        else:
            # Weighted selection: prefer lower fanout signals
            weights = []
            for _, net_name in available_signals:
                fo = fanout_count.get(net_name, 0)
                weights.append(max(0.1, max_fanout - fo))

            chosen = []
            candidates = list(range(len(available_signals)))
            for _ in range(n_inputs):
                if not candidates:
                    candidates = list(range(len(available_signals)))
                w = [weights[c] for c in candidates]
                total = sum(w)
                w = [x / total for x in w]
                idx = random.choices(candidates, weights=w, k=1)[0]
                chosen.append(available_signals[idx])
                candidates.remove(idx)

        input_nets = []
        for _, net_name in chosen:
            input_nets.append(net_name)
            circuit.nets[net_name].sinks.append(gate_id)
            fanout_count[net_name] = fanout_count.get(net_name, 0) + 1

        # Create gate node and output net
        node = Node(
            node_id=gate_id,
            node_type=cell_name,
            input_nets=input_nets,
            output_net=output_net_name,
        )
        wire_len = random.uniform(5.0, 50.0)  # random wire length in um
        net = Net(
            name=output_net_name,
            driver=gate_id,
            wire_length_um=wire_len,
        )

        circuit.nodes[gate_id] = node
        circuit.nets[output_net_name] = net
        circuit.gate_ids.append(gate_id)
        available_signals.append((gate_id, output_net_name))
        fanout_count[output_net_name] = 0

    # Create primary outputs from the last few gate outputs
    # Pick outputs from the later gates for realism
    late_gates = circuit.gate_ids[max(0, len(circuit.gate_ids) - num_outputs * 3):]
    output_sources = random.sample(late_gates, min(num_outputs, len(late_gates)))

    # If we need more outputs, pick from remaining gates
    while len(output_sources) < num_outputs:
        output_sources.append(random.choice(circuit.gate_ids))

    for i, src_gate_id in enumerate(output_sources):
        po_id = f"PO_{i}"
        src_net = circuit.nodes[src_gate_id].output_net
        circuit.nets[src_net].sinks.append(po_id)

        node = Node(
            node_id=po_id,
            node_type="PO",
            input_nets=[src_net],
        )
        circuit.nodes[po_id] = node
        circuit.primary_outputs.append(po_id)

    # Synthetic placement on a 2D grid; positions drive routing congestion
    # (RUDY) and manhattan-distance edge features.
    positions, grid = _assign_positions(circuit, random)
    circuit.positions = positions
    circuit.grid_size = grid

    return circuit


def circuit_to_dict(circuit: Circuit) -> dict:
    """Serialize circuit to a plain dictionary for JSON storage."""
    return {
        "name": circuit.name,
        "primary_inputs": circuit.primary_inputs,
        "primary_outputs": circuit.primary_outputs,
        "gate_ids": circuit.gate_ids,
        "nodes": {
            nid: {
                "node_id": n.node_id,
                "node_type": n.node_type,
                "input_nets": n.input_nets,
                "output_net": n.output_net,
            }
            for nid, n in circuit.nodes.items()
        },
        "nets": {
            nname: {
                "name": net.name,
                "driver": net.driver,
                "sinks": net.sinks,
                "wire_length_um": net.wire_length_um,
            }
            for nname, net in circuit.nets.items()
        },
        "positions": {nid: list(p) for nid, p in circuit.positions.items()},
        "grid_size": list(circuit.grid_size),
    }


def dict_to_circuit(d: dict) -> Circuit:
    """Deserialize circuit from a plain dictionary."""
    circuit = Circuit(
        name=d["name"],
        primary_inputs=d["primary_inputs"],
        primary_outputs=d["primary_outputs"],
        gate_ids=d["gate_ids"],
    )
    for nid, nd in d["nodes"].items():
        circuit.nodes[nid] = Node(**nd)
    for nname, netd in d["nets"].items():
        circuit.nets[nname] = Net(**netd)
    if "positions" in d:
        circuit.positions = {nid: tuple(p) for nid, p in d["positions"].items()}
    if "grid_size" in d:
        circuit.grid_size = tuple(d["grid_size"])
    return circuit

"""
Analog circuit topology generator.

Produces small representative analog topologies as DAGs over device-level
nodes with placement coordinates that respect basic analog layout rules:
matched devices placed adjacent in the same row, current mirrors clustered,
signal flow left-to-right.

Topologies:
  - current_mirror      (2 matched NMOS)
  - common_source_amp   (1 NMOS + resistive load)
  - diff_pair           (2 matched NMOS + tail + 2 matched R loads)
  - two_stage_opamp     (diff-pair stage + CS stage + Miller cap)
  - folded_cascode      (folded-cascode single stage)

Each topology populates Circuit.is_analog / analog_params / symmetry_groups
and assigns grid positions consistent with the digital placement scheme so
graph_builder can compute per-cell congestion uniformly.
"""

import random
from typing import Callable, Dict, List, Optional

from .analog_library import DEFAULT_DEVICE_PARAMS, device_params
from .circuit_gen import Circuit, Net, Node


class _AnalogBuilder:
    """Stateful helper for incrementally building an analog Circuit."""

    def __init__(self, name: str):
        self.circuit = Circuit(name=name)
        self.circuit.is_analog = True
        self._net_counter = 0

    def add_port(self, name: str, kind: str, position):
        node = Node(node_id=name, node_type=kind)
        self.circuit.nodes[name] = node
        if kind == "PI":
            self.circuit.primary_inputs.append(name)
        elif kind == "PO":
            self.circuit.primary_outputs.append(name)
        self.circuit.positions[name] = tuple(position)
        return name

    def add_device(self, kind: str, node_id: str, position,
                   params: Optional[dict] = None, symmetry_group: int = 0):
        node = Node(node_id=node_id, node_type=kind)
        self.circuit.nodes[node_id] = node
        self.circuit.gate_ids.append(node_id)
        merged = device_params(kind)
        if params:
            merged.update(params)
        self.circuit.analog_params[node_id] = merged
        self.circuit.positions[node_id] = tuple(position)
        if symmetry_group:
            self.circuit.symmetry_groups[node_id] = symmetry_group
        return node_id

    def add_net(self, driver_id: str, sink_ids, wire_length_um: float = 5.0):
        net_name = f"n{self._net_counter}"
        self._net_counter += 1
        net = Net(
            name=net_name,
            driver=driver_id,
            sinks=list(sink_ids),
            wire_length_um=wire_length_um,
        )
        self.circuit.nets[net_name] = net
        # Hook driver's output_net (if a device) and append to sinks' input_nets.
        if driver_id in self.circuit.nodes:
            self.circuit.nodes[driver_id].output_net = net_name
        for s in sink_ids:
            if s in self.circuit.nodes:
                self.circuit.nodes[s].input_nets.append(net_name)
        return net_name

    def finalize(self, grid_w: Optional[int] = None, grid_h: Optional[int] = None) -> Circuit:
        # Auto grid sizing if not provided.
        xs = [p[0] for p in self.circuit.positions.values()]
        ys = [p[1] for p in self.circuit.positions.values()]
        gw = grid_w if grid_w is not None else (max(xs) + 2 if xs else 4)
        gh = grid_h if grid_h is not None else (max(ys) + 2 if ys else 4)
        self.circuit.grid_size = (gw, gh)
        return self.circuit


# ---------------------------------------------------------------------------
# Topologies — each takes `seed` for parameter jitter and returns Circuit.
# ---------------------------------------------------------------------------


def _jitter(rng: random.Random, base: float, frac: float = 0.2) -> float:
    return base * rng.uniform(1.0 - frac, 1.0 + frac)


def current_mirror(seed: int, name: str = "current_mirror") -> Circuit:
    rng = random.Random(seed)
    b = _AnalogBuilder(name)
    iref = b.add_port("IREF", "PI", (0, 0))
    iout = b.add_port("IOUT", "PO", (3, 0))
    p = {"W": _jitter(rng, 5e-6), "L": 130e-9}
    m1 = b.add_device("NMOS", "M1", (1, 0), params=p, symmetry_group=1)
    m2 = b.add_device("NMOS", "M2", (2, 0), params=p, symmetry_group=1)
    b.add_net(iref, [m1])
    b.add_net(m1, [m2])  # gate-tie between matched devices
    b.add_net(m2, [iout])
    return b.finalize()


def common_source_amp(seed: int, name: str = "cs_amp") -> Circuit:
    rng = random.Random(seed)
    b = _AnalogBuilder(name)
    vin = b.add_port("VIN", "PI", (0, 1))
    vout = b.add_port("VOUT", "PO", (3, 1))
    m1 = b.add_device(
        "NMOS", "M1", (1, 1),
        params={"W": _jitter(rng, 4e-6), "L": 130e-9},
    )
    rload = b.add_device(
        "R", "R1", (2, 1),
        params={"R": _jitter(rng, 20e3)},
    )
    b.add_net(vin, [m1])
    b.add_net(m1, [rload])
    b.add_net(rload, [vout])
    return b.finalize()


def diff_pair(seed: int, name: str = "diff_pair") -> Circuit:
    rng = random.Random(seed)
    b = _AnalogBuilder(name)
    vinp = b.add_port("VINP", "PI", (0, 0))
    vinm = b.add_port("VINM", "PI", (0, 2))
    voutp = b.add_port("VOUTP", "PO", (4, 0))
    voutm = b.add_port("VOUTM", "PO", (4, 2))
    vbias = b.add_port("VBIAS", "PI", (0, 1))

    pair_p = {"W": _jitter(rng, 4e-6), "L": 130e-9}
    m1 = b.add_device("NMOS", "M1", (2, 0), params=pair_p, symmetry_group=1)
    m2 = b.add_device("NMOS", "M2", (2, 2), params=pair_p, symmetry_group=1)
    mtail = b.add_device(
        "NMOS", "MTAIL", (2, 1),
        params={"W": _jitter(rng, 8e-6), "L": 200e-9},
    )
    load_p = {"R": _jitter(rng, 25e3)}
    r1 = b.add_device("R", "R1", (3, 0), params=load_p, symmetry_group=2)
    r2 = b.add_device("R", "R2", (3, 2), params=load_p, symmetry_group=2)

    b.add_net(vinp, [m1])
    b.add_net(vinm, [m2])
    b.add_net(vbias, [mtail])
    b.add_net(mtail, [m1, m2])      # tail feeds both diff devices
    b.add_net(m1, [r1])
    b.add_net(m2, [r2])
    b.add_net(r1, [voutp])
    b.add_net(r2, [voutm])
    return b.finalize()


def two_stage_opamp(seed: int, name: str = "two_stage_opamp") -> Circuit:
    rng = random.Random(seed)
    b = _AnalogBuilder(name)
    vinp = b.add_port("VINP", "PI", (0, 0))
    vinm = b.add_port("VINM", "PI", (0, 4))
    vbias = b.add_port("VBIAS", "PI", (0, 2))
    vout = b.add_port("VOUT", "PO", (6, 2))

    # Stage 1: diff pair (M1/M2) + mirror load (M3/M4) + tail (M5).
    pair_p = {"W": _jitter(rng, 4e-6), "L": 130e-9}
    mirror_p = {"W": _jitter(rng, 6e-6), "L": 200e-9}
    m1 = b.add_device("NMOS", "M1", (2, 1), params=pair_p, symmetry_group=1)
    m2 = b.add_device("NMOS", "M2", (2, 3), params=pair_p, symmetry_group=1)
    m3 = b.add_device("PMOS", "M3", (3, 1), params=mirror_p, symmetry_group=2)
    m4 = b.add_device("PMOS", "M4", (3, 3), params=mirror_p, symmetry_group=2)
    m5 = b.add_device(
        "NMOS", "M5", (2, 2),
        params={"W": _jitter(rng, 10e-6), "L": 200e-9},
    )

    # Stage 2: CS amp M6 + active load M7.
    m6 = b.add_device(
        "NMOS", "M6", (4, 3),
        params={"W": _jitter(rng, 8e-6), "L": 130e-9},
    )
    m7 = b.add_device(
        "PMOS", "M7", (4, 1),
        params={"W": _jitter(rng, 16e-6), "L": 200e-9},
    )

    # Miller compensation cap between stage-2 output and stage-1 output.
    cc = b.add_device(
        "C", "Cc", (5, 2),
        params={"C": _jitter(rng, 1e-12)},
    )

    # Wire it up.
    b.add_net(vinp, [m1])
    b.add_net(vinm, [m2])
    b.add_net(vbias, [m5])
    b.add_net(m5, [m1, m2])
    b.add_net(m1, [m3])
    b.add_net(m2, [m4, m6])           # M4 drain is stage-1 output; feeds M6 gate
    b.add_net(m3, [m4])               # mirror gate-tie
    b.add_net(m6, [m7, cc])           # stage-2 output node
    b.add_net(m7, [vout])
    b.add_net(cc, [vout])
    return b.finalize()


def folded_cascode(seed: int, name: str = "folded_cascode") -> Circuit:
    rng = random.Random(seed)
    b = _AnalogBuilder(name)
    vinp = b.add_port("VINP", "PI", (0, 1))
    vinm = b.add_port("VINM", "PI", (0, 3))
    vbias = b.add_port("VBIAS", "PI", (0, 2))
    vout = b.add_port("VOUT", "PO", (5, 2))

    pair_p = {"W": _jitter(rng, 4e-6), "L": 130e-9}
    cascode_p = {"W": _jitter(rng, 6e-6), "L": 200e-9}

    m1 = b.add_device("NMOS", "M1", (2, 1), params=pair_p, symmetry_group=1)
    m2 = b.add_device("NMOS", "M2", (2, 3), params=pair_p, symmetry_group=1)
    # PMOS cascode legs (matched).
    m3 = b.add_device("PMOS", "M3", (3, 1), params=cascode_p, symmetry_group=2)
    m4 = b.add_device("PMOS", "M4", (3, 3), params=cascode_p, symmetry_group=2)
    # NMOS cascode bias loads (matched).
    m5 = b.add_device("NMOS", "M5", (4, 1), params=cascode_p, symmetry_group=3)
    m6 = b.add_device("NMOS", "M6", (4, 3), params=cascode_p, symmetry_group=3)
    mtail = b.add_device(
        "NMOS", "MTAIL", (2, 2),
        params={"W": _jitter(rng, 10e-6), "L": 200e-9},
    )

    b.add_net(vinp, [m1])
    b.add_net(vinm, [m2])
    b.add_net(vbias, [mtail])
    b.add_net(mtail, [m1, m2])
    b.add_net(m1, [m3])
    b.add_net(m2, [m4])
    b.add_net(m3, [m5])
    b.add_net(m4, [m6, vout])
    b.add_net(m5, [m6])
    return b.finalize()


ANALOG_TOPOLOGIES: List[Callable[[int], Circuit]] = [
    current_mirror,
    common_source_amp,
    diff_pair,
    two_stage_opamp,
    folded_cascode,
]


def generate_analog_circuit(
    seed: int = 0,
    topology: Optional[str] = None,
    name: Optional[str] = None,
) -> Circuit:
    """Generate an analog circuit. Random topology unless `topology` is set."""
    rng = random.Random(seed)
    topo_map = {fn.__name__: fn for fn in ANALOG_TOPOLOGIES}
    if topology is None:
        fn = rng.choice(ANALOG_TOPOLOGIES)
    elif topology in topo_map:
        fn = topo_map[topology]
    else:
        raise ValueError(f"Unknown analog topology '{topology}'. "
                         f"Available: {list(topo_map)}")
    circuit = fn(seed, name=name or f"{fn.__name__}_{seed}")
    return circuit

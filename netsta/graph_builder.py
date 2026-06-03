"""
Build PyG Data tensors from Circuits — unified digital + analog schema.

Node features (24 dims) — topology-free identity only:
  [0 : NUM_GATE_TYPES+2]              digital gate-type one-hot (PI/PO/funcs)
  [13 : 13 + NUM_DEVICE_TYPES]        analog device-type one-hot
                                        (NMOS, PMOS, R, C, CURRENT_MIRROR, DIFF_PAIR)
  [19]                                W_over_L (analog)
  [20]                                operating_region (analog;
                                        sat=1, triode=0, off=-1)
  [21]                                symmetry_group_norm (analog;
                                        normalized group id, 0 if unmatched)
  [22..23]                            [is_digital, is_analog]

We deliberately do NOT include logical_depth or load_cap — those are STA
outputs, so feeding them in as inputs short-circuits the regression. We also
do NOT include precomputed 1-hop aggregates (fanout count, pin density,
net degree, mean bbox area) — those are quantities message passing should
*derive* from the graph; baking them into node features hides the work the
GNN is supposed to do, which is why the MLP baseline matched the GNN under
the old schema.

Topology and placement signal reach the model via the edge features:

Edge features (5 dims):
  [0] wire_delay_norm
  [1] manhattan_distance_norm
  [2] net_fanout_norm                  (per-edge, derived from the driving net)
  [3] coupling_capacitance_norm        (analog only; digital=0)
  [4] matching_constraint              (analog only; 1 if endpoints share
                                        symmetry_group, else 0; digital=0)

Labels:
  y_slack            per-node slack in ns (digital STA output)
  y_arrival_time     per-node AT in ns (auxiliary supervision for the
                     backbone's forward / AT half)
  y_required_time    per-node RT in ns (auxiliary supervision for the
                     backbone's backward / RT half)
  y_logical_depth    per-node depth from the deepest PI; not a target,
                     used by evaluate.py to stratify R^2 per depth
  y_critical         binary, slack <= CRITICAL_SLACK_THRESHOLD_NS
  y_congestion       per-node congestion (RUDY)
  y_drc              binary DRC hotspot
  y_analog_performance: [N, 2] = (gbw_score, parasitic_impact)
"""

import torch
from torch_geometric.data import Data

from .analog_library import (
    DEFAULT_DEVICE_PARAMS,
    DEVICE_TO_IDX,
    NUM_DEVICE_TYPES,
)
from .circuit_gen import Circuit
from .congestion import compute_rudy_congestion
from .drc import compute_drc_labels
from .nangate45 import (
    NANGATE45_CELLS,
    GATE_TYPE_TO_IDX,
    NUM_GATE_TYPES,
    compute_wire_delay,
)


GATE_TYPE_DIM = NUM_GATE_TYPES + 2     # +PI +PO
PI_IDX = NUM_GATE_TYPES
PO_IDX = NUM_GATE_TYPES + 1
# Total node feature dim: gate-type(13) + analog(9) + type-flags(2) = 24
# No precomputed STA outputs or 1-hop aggregates — message passing derives those.
ANALOG_BLOCK_DIM = NUM_DEVICE_TYPES + 3  # device-type one-hot + (W/L, op_region, sym)
NODE_FEAT_DIM = GATE_TYPE_DIM + ANALOG_BLOCK_DIM + 2
EDGE_FEAT_DIM = 5


def _max_pos(vals, default=1.0):
    return max((v for v in vals if v is not None and v > 0), default=default)


def circuit_to_pyg(circuit: Circuit, sta_results: dict) -> Data:
    """Build a PyG Data with the unified feature + label schema.

    `sta_results` shape matches whichever STA was run — digital
    `run_sta(circuit)` or analog `run_analog_sta(circuit)`. Both expose a
    `node_timing` dict keyed by node id.
    """
    is_analog = bool(getattr(circuit, "is_analog", False))
    node_timing = sta_results.get("node_timing", {})

    node_order = circuit.primary_inputs + circuit.gate_ids + circuit.primary_outputs
    node_to_idx = {nid: i for i, nid in enumerate(node_order)}
    num_nodes = len(node_order)

    # Always compute placement-based features (works for analog and digital).
    congestion_map, demand_grid = compute_rudy_congestion(circuit)
    drc_map = compute_drc_labels(circuit, demand_grid=demand_grid)
    positions = circuit.positions
    analog_params = circuit.analog_params or {}
    symmetry_groups = circuit.symmetry_groups or {}

    # ----- Raw scalar collection.
    # Only analog device W/L survives here; topology aggregates and STA-internal
    # quantities (depth, load_cap) are intentionally excluded — see module
    # docstring.
    raw_wl = []  # W/L for analog MOS only

    for nid in node_order:
        node = circuit.nodes[nid]
        if is_analog and node.node_type in ("NMOS", "PMOS"):
            p = analog_params.get(nid, {})
            W = p.get("W") or 1e-6
            L = p.get("L") or 130e-9
            raw_wl.append(W / max(L, 1e-12))
        else:
            raw_wl.append(0.0)

    max_wl = _max_pos(raw_wl, 1.0)
    max_group = _max_pos(symmetry_groups.values(), 1.0) if symmetry_groups else 1.0

    # ----- Per-node feature tensor.
    x = torch.zeros(num_nodes, NODE_FEAT_DIM)
    analog_base = GATE_TYPE_DIM  # analog block starts immediately after gate-type one-hot

    for i, nid in enumerate(node_order):
        node = circuit.nodes[nid]

        # Gate-type one-hot region (digital). PIs/POs are common to both;
        # analog device nodes leave gate-type bits zero.
        if node.node_type == "PI":
            x[i, PI_IDX] = 1.0
        elif node.node_type == "PO":
            x[i, PO_IDX] = 1.0
        elif (not is_analog) and node.node_type in NANGATE45_CELLS:
            func = NANGATE45_CELLS[node.node_type]["function"]
            x[i, GATE_TYPE_TO_IDX[func]] = 1.0

        # Analog device features (zeros for digital).
        if is_analog and node.node_type in DEVICE_TO_IDX:
            x[i, analog_base + DEVICE_TO_IDX[node.node_type]] = 1.0
            # W/L
            x[i, analog_base + NUM_DEVICE_TYPES + 0] = raw_wl[i] / max_wl
            # Operating region: assume saturation if MOS with bias, off otherwise.
            if node.node_type in ("NMOS", "PMOS"):
                x[i, analog_base + NUM_DEVICE_TYPES + 1] = 1.0  # sat
            elif node.node_type in ("R", "C"):
                x[i, analog_base + NUM_DEVICE_TYPES + 1] = 0.0  # passive
            else:
                x[i, analog_base + NUM_DEVICE_TYPES + 1] = -1.0
            # Symmetry group (normalized).
            grp = symmetry_groups.get(nid, 0)
            x[i, analog_base + NUM_DEVICE_TYPES + 2] = grp / max_group

        # Circuit-type indicator at the tail.
        x[i, NODE_FEAT_DIM - 2] = 0.0 if is_analog else 1.0  # is_digital
        x[i, NODE_FEAT_DIM - 1] = 1.0 if is_analog else 0.0  # is_analog

    # ----- Edges.
    src_list, dst_list = [], []
    raw_wd, raw_md, raw_nf, raw_coup, raw_match = [], [], [], [], []

    for net in circuit.nets.values():
        if net.driver not in node_to_idx:
            continue
        src_idx = node_to_idx[net.driver]
        net_fanout = len(net.sinks)
        driver_pos = positions.get(net.driver, (0, 0))

        # Driver Cgd (analog) for coupling estimate.
        driver_params = analog_params.get(net.driver, {}) if is_analog else {}
        driver_cgd = driver_params.get("Cgd", DEFAULT_DEVICE_PARAMS.get(
            circuit.nodes[net.driver].node_type, {}
        ).get("Cgd", 0.0))

        for sink_id in net.sinks:
            if sink_id not in node_to_idx:
                continue
            dst_idx = node_to_idx[sink_id]
            src_list.append(src_idx); dst_list.append(dst_idx)

            sink_node = circuit.nodes[sink_id]
            # Wire delay: digital uses Nangate cell input cap; analog uses
            # device input_capacitance proxy with the same compute function.
            if not is_analog and sink_node.node_type in NANGATE45_CELLS:
                sink_cap = NANGATE45_CELLS[sink_node.node_type]["input_cap"]
            else:
                sink_params = analog_params.get(sink_id, {})
                sink_cap = sink_params.get(
                    "input_capacitance",
                    DEFAULT_DEVICE_PARAMS.get(sink_node.node_type, {}).get(
                        "input_capacitance", 2.0,
                    ),
                )
                # Digital compute_wire_delay expects pF-scale capacitance; the
                # analog values are ~1e-15 F. Scale up so the math is sane.
                if is_analog:
                    sink_cap = sink_cap * 1e15
            raw_wd.append(compute_wire_delay(net.wire_length_um, sink_cap))

            sink_pos = positions.get(sink_id, (0, 0))
            raw_md.append(
                abs(driver_pos[0] - sink_pos[0]) + abs(driver_pos[1] - sink_pos[1])
            )
            raw_nf.append(net_fanout)
            raw_coup.append(driver_cgd if is_analog else 0.0)
            # matching_constraint: 1 if both endpoints share a symmetry group
            if is_analog and symmetry_groups:
                g1 = symmetry_groups.get(net.driver, 0)
                g2 = symmetry_groups.get(sink_id, 0)
                raw_match.append(1.0 if (g1 != 0 and g1 == g2) else 0.0)
            else:
                raw_match.append(0.0)

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)

    if raw_wd:
        max_wd = max(max(raw_wd), 1e-6)
        max_md = max(max(raw_md), 1)
        max_nf = max(max(raw_nf), 1)
        max_coup = max(max(raw_coup), 1e-30)
        edge_attr = torch.tensor(
            [
                [
                    wd / max_wd,
                    md / max_md,
                    nf / max_nf,
                    coup / max_coup,
                    match,
                ]
                for wd, md, nf, coup, match in zip(
                    raw_wd, raw_md, raw_nf, raw_coup, raw_match,
                )
            ],
            dtype=torch.float,
        )
    else:
        edge_attr = torch.zeros((0, EDGE_FEAT_DIM), dtype=torch.float)

    # ----- Labels.
    # All three timing labels (slack, arrival_time, required_time) are stored
    # as raw nanoseconds. Per-task mean/std come from the train split via
    # netsta.stats and are carried inside each head as register_buffer slots.
    # The auxiliary arrival_time and required_time labels let the timing
    # backbone's two halves be supervised directly on the quantities they are
    # structured around, rather than via the indirect gradient from slack alone.
    if is_analog:
        # No digital STA — zero-fill so mixed-mode training can still batch.
        y_slack = torch.zeros(num_nodes, dtype=torch.float)
        y_arrival_time = torch.zeros(num_nodes, dtype=torch.float)
        y_required_time = torch.zeros(num_nodes, dtype=torch.float)
        y_logical_depth = torch.zeros(num_nodes, dtype=torch.float)
        y_critical = torch.zeros(num_nodes, dtype=torch.float)
        y_drc = torch.tensor(
            [drc_map.get(nid, 0.0) for nid in node_order], dtype=torch.float
        )
        gbw_col = [node_timing.get(nid, {}).get("gbw_score", 0.0) for nid in node_order]
        par_col = [node_timing.get(nid, {}).get("parasitic_impact", 0.0) for nid in node_order]
        y_analog = torch.tensor(list(zip(gbw_col, par_col)), dtype=torch.float)
    else:
        slacks = [node_timing.get(nid, {}).get("slack", 0.0) for nid in node_order]
        ats = [node_timing.get(nid, {}).get("arrival_time", 0.0) for nid in node_order]
        rts = [node_timing.get(nid, {}).get("required_time", 0.0) for nid in node_order]
        depths = [node_timing.get(nid, {}).get("logical_depth", 0) for nid in node_order]
        y_slack = torch.tensor(slacks, dtype=torch.float)
        y_arrival_time = torch.tensor(ats, dtype=torch.float)
        y_required_time = torch.tensor(rts, dtype=torch.float)
        # y_logical_depth is metadata, not a training target — used by
        # evaluate.py for per-depth R^2 stratification.
        y_logical_depth = torch.tensor(depths, dtype=torch.float)
        y_critical = torch.tensor(
            [1.0 if node_timing.get(nid, {}).get("is_critical", False) else 0.0
             for nid in node_order],
            dtype=torch.float,
        )
        y_drc = torch.tensor(
            [drc_map.get(nid, 0.0) for nid in node_order], dtype=torch.float
        )
        y_analog = torch.zeros((num_nodes, 2), dtype=torch.float)

    y_congestion = torch.tensor(
        [congestion_map.get(nid, 0.0) for nid in node_order], dtype=torch.float,
    )

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y_slack=y_slack,
        y_arrival_time=y_arrival_time,
        y_required_time=y_required_time,
        y_logical_depth=y_logical_depth,
        y_critical=y_critical,
        y_congestion=y_congestion,
        y_drc=y_drc,
        y_analog_performance=y_analog,
        num_nodes=num_nodes,
    )

    data.is_analog = bool(is_analog)
    data.clock_period = sta_results.get("clock_period_ns", 0.0)
    data.circuit_name = circuit.name
    return data

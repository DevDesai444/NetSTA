"""
Build PyG Data tensors from Circuits — unified digital + analog schema.

Node features (15 dims) — Liberty scalars + identity flags:
  [0]                                 is_pi          (1 if primary input)
  [1]                                 is_po          (1 if primary output)
  [2..10]                             analog block (9 dims; see below)
  [11]                                is_digital
  [12]                                is_analog
  [13]                                intrinsic_delay_norm
                                        (Liberty intrinsic_delay / 0.1 ns)
  [14]                                input_cap_norm
                                        (Liberty input_cap / 5.0 fF)
  [15]                                output_res_norm
                                        (Liberty output_res / 500 ohm)
  [16]                                clock_period_norm
                                        (graph-level clock_period / 2 ns,
                                        broadcast to every node)

The 13-dim gate-type one-hot was dropped at v9. The ablation showed that
once intrinsic_delay is exposed as a scalar feature, the one-hot is
redundant — removing it actually moved R^2 by +0.030. The two additional
Liberty scalars (input_cap, output_res) carry the remaining cell-specific
timing information that intrinsic_delay alone does not: the gate's
contribution to downstream load and its own resistance scaling. Together
with intrinsic_delay these three uniquely identify every Nangate45 cell
the model will see, in 3 dense scalars instead of 13 sparse columns.

Analog block (9 dims) at [2..10]:
  [2 : 2 + NUM_DEVICE_TYPES]          analog device-type one-hot
                                        (NMOS, PMOS, R, C, CURRENT_MIRROR, DIFF_PAIR)
  [8]                                 W_over_L (analog)
  [9]                                 operating_region (sat=1, triode=0, off=-1)
  [10]                                symmetry_group_norm
                                        (analog group id, 0 if unmatched)

We deliberately do NOT include logical_depth or load_cap — those are STA
outputs, so feeding them in as inputs short-circuits the regression. We also
do NOT include precomputed 1-hop aggregates (fanout count, pin density,
net degree, mean bbox area) — those are quantities message passing should
*derive* from the graph; baking them into node features hides the work the
GNN is supposed to do, which is why the MLP baseline matched the GNN under
the old schema.

Topology and placement signal reach the model via the edge features:

Edge features (5 dims):
  [0] wire_delay_norm                  (ns / WIRE_DELAY_REF_NS)
  [1] manhattan_distance_norm          (cells / MANHATTAN_REF_CELLS)
  [2] net_fanout_norm                  (per-edge, derived from the driving net)
  [3] coupling_capacitance_norm        (analog only; digital=0)
  [4] matching_constraint              (analog only; 1 if endpoints share
                                        symmetry_group, else 0; digital=0)

Normalization is by FIXED dataset-wide reference constants (below) rather
than per-graph max. Per-graph normalization meant "0.5" was a different
absolute ns in every circuit, so the model could not learn the wire_delay ->
slack contribution consistently across graphs — which is why ablating edge
features barely moved the metric. Fixed constants preserve absolute scale.

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
    compute_wire_delay,
)


# Schema v9 layout:
#   PI/PO flags (2) + analog block (9) + is_digital/is_analog (2)
#   + Liberty scalars: intrinsic_delay, input_cap, output_res, clock_period (4)
# Total: 2 + 9 + 2 + 4 = 17 dims.
# No precomputed STA outputs or 1-hop aggregates — message passing derives those.
ANALOG_BLOCK_DIM = NUM_DEVICE_TYPES + 3  # device-type one-hot + (W/L, op_region, sym)
PI_PO_DIM = 2
TYPE_FLAG_DIM = 2
LIBERTY_SCALAR_DIM = 4  # intrinsic_delay, input_cap, output_res, clock_period
NODE_FEAT_DIM = PI_PO_DIM + ANALOG_BLOCK_DIM + TYPE_FLAG_DIM + LIBERTY_SCALAR_DIM
EDGE_FEAT_DIM = 5

# ---------------------------------------------------------------------------
# Fixed dataset-wide reference constants used to normalize the raw physical
# quantities below into dimensionless features. These are NOT learned and
# NOT per-graph — they're absolute scales picked to put the typical value
# range near 1.0 across the Nangate45 + synthetic-DAG distribution we train
# on. Keeping them fixed is what lets the model learn a consistent mapping
# from edge_attr -> ns delay across all circuits in the dataset.
#   wire_delay        ~ 0.001-0.05 ns  -> ref 0.1 ns
#   manhattan_dist    ~ 1-30 cells     -> ref 30
#   net_fanout        ~ 1-10           -> ref 10
#   coupling_cap      ~ 0-1e-15 F      -> ref 1e-15 F
#   intrinsic_delay   ~ 0.01-0.07 ns   -> ref 0.1 ns
#   input_cap         ~ 0.5-5 fF       -> ref 5 fF
#   output_res        ~ 100-500 ohm    -> ref 500 ohm
#   clock_period      ~ 0.3-5.0 ns     -> ref 2.0 ns
# Picked once on the v7 dataset distribution and held constant since.
WIRE_DELAY_REF_NS = 0.1
MANHATTAN_REF_CELLS = 30.0
FANOUT_REF = 10.0
COUPLING_REF_F = 1e-15
INTRINSIC_DELAY_REF_NS = 0.1
INPUT_CAP_REF_FF = 5.0
OUTPUT_RES_REF_OHM = 500.0
CLOCK_PERIOD_REF_NS = 2.0


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
    # Layout (15 dims): [is_pi, is_po, analog_block(9), is_digital, is_analog,
    #                    intrinsic_delay, input_cap, output_res, clock_period]
    x = torch.zeros(num_nodes, NODE_FEAT_DIM)
    IS_PI_IDX = 0
    IS_PO_IDX = 1
    ANALOG_BASE = 2
    IS_DIGITAL_IDX = ANALOG_BASE + ANALOG_BLOCK_DIM         # 11
    IS_ANALOG_IDX = IS_DIGITAL_IDX + 1                      # 12
    LIBERTY_BASE = IS_ANALOG_IDX + 1                        # 13
    INTRINSIC_DELAY_IDX = LIBERTY_BASE + 0                  # 13
    INPUT_CAP_IDX = LIBERTY_BASE + 1                        # 14
    OUTPUT_RES_IDX = LIBERTY_BASE + 2                       # 15
    CLOCK_PERIOD_IDX = LIBERTY_BASE + 3                     # 16

    # Graph-level clock period broadcast to every node (boundary condition for
    # the RT pass; required-time at the POs IS this value).
    clock_period_ns = float(sta_results.get("clock_period_ns", 0.0) or 0.0)
    clock_period_norm = clock_period_ns / CLOCK_PERIOD_REF_NS

    for i, nid in enumerate(node_order):
        node = circuit.nodes[nid]

        # PI / PO flags. Common to both digital and analog circuits — these
        # are pseudo-pins, not cell instances.
        if node.node_type == "PI":
            x[i, IS_PI_IDX] = 1.0
        elif node.node_type == "PO":
            x[i, IS_PO_IDX] = 1.0

        # Analog device features (zeros for digital).
        if is_analog and node.node_type in DEVICE_TO_IDX:
            x[i, ANALOG_BASE + DEVICE_TO_IDX[node.node_type]] = 1.0
            # W/L
            x[i, ANALOG_BASE + NUM_DEVICE_TYPES + 0] = raw_wl[i] / max_wl
            # Operating region: assume saturation if MOS with bias, off otherwise.
            if node.node_type in ("NMOS", "PMOS"):
                x[i, ANALOG_BASE + NUM_DEVICE_TYPES + 1] = 1.0  # sat
            elif node.node_type in ("R", "C"):
                x[i, ANALOG_BASE + NUM_DEVICE_TYPES + 1] = 0.0  # passive
            else:
                x[i, ANALOG_BASE + NUM_DEVICE_TYPES + 1] = -1.0
            # Symmetry group (normalized).
            grp = symmetry_groups.get(nid, 0)
            x[i, ANALOG_BASE + NUM_DEVICE_TYPES + 2] = grp / max_group

        # Circuit-type indicator.
        x[i, IS_DIGITAL_IDX] = 0.0 if is_analog else 1.0
        x[i, IS_ANALOG_IDX] = 1.0 if is_analog else 0.0

        # Liberty scalars for digital gates. PIs / POs / analog devices keep
        # zero — the analog block carries the analog-equivalent information.
        if (not is_analog) and node.node_type in NANGATE45_CELLS:
            cell = NANGATE45_CELLS[node.node_type]
            x[i, INTRINSIC_DELAY_IDX] = cell["intrinsic_delay"] / INTRINSIC_DELAY_REF_NS
            x[i, INPUT_CAP_IDX] = cell["input_cap"] / INPUT_CAP_REF_FF
            x[i, OUTPUT_RES_IDX] = cell["output_res"] / OUTPUT_RES_REF_OHM

        # Clock period (graph-level, broadcast). Same value on every node of
        # the same circuit; the model uses it as RT's boundary value.
        x[i, CLOCK_PERIOD_IDX] = clock_period_norm

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
        # Fixed dataset-wide normalization — the previous per-graph max made
        # "0.5 wire delay" mean a different absolute ns in every circuit, so
        # the model never saw a consistent physical scale and ablating edge
        # features did not move the metric. Constants are defined at module
        # level and held fixed across the entire dataset.
        edge_attr = torch.tensor(
            [
                [
                    wd / WIRE_DELAY_REF_NS,
                    md / MANHATTAN_REF_CELLS,
                    nf / FANOUT_REF,
                    coup / COUPLING_REF_F,
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

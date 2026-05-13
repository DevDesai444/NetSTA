"""
Simplified small-signal "STA" for analog circuits.

For each device node we estimate:
  - gm and intrinsic gain from W, L, and a default bias current
  - total node capacitance C_load = own input_capacitance + sum of input_cap
    of devices on this node's output net
  - per-node GBW ≈ gm / (2π · C_load)
  - parasitic_impact ≈ coupling-cap energy at the node, normalized

Outputs (per-node dict) mirror the shape of the digital STA so graph_builder
can fold them into PyG labels via a common path.

This is intentionally a coarse model — real analog timing needs SPICE.
Outputs are still deterministic and topology-driven so the model has signal
to learn.
"""

import math
from typing import Dict

from .analog_library import (
    DEFAULT_DEVICE_PARAMS,
    estimate_cgs,
    estimate_gm,
    estimate_intrinsic_gain,
)


def _device_input_cap(node_type: str, params: dict) -> float:
    if node_type in ("NMOS", "PMOS"):
        return params.get("Cgs") or estimate_cgs(node_type, params.get("W", 1e-6), params.get("L", 130e-9))
    sheet = DEFAULT_DEVICE_PARAMS.get(node_type, {})
    return params.get("input_capacitance", sheet.get("input_capacitance", 1e-15))


def run_analog_sta(circuit, default_bias_ua: float = 50.0) -> dict:
    """Compute per-node small-signal labels for an analog circuit.

    Returns a dict shaped like the digital STA:
      {
        "node_timing": {node_id: {gbw_score, bandwidth_limited,
                                  parasitic_impact, gm, ...}},
        "max_gbw": float,
        "max_parasitic": float,
      }
    """
    node_timing: Dict[str, dict] = {}
    raw_gbw: Dict[str, float] = {}
    raw_parasitic: Dict[str, float] = {}

    # Pre-compute per-node load capacitance: own Cgs + sum of input_cap on
    # output-net sinks.
    bias_a = default_bias_ua * 1e-6
    for nid, node in circuit.nodes.items():
        params = (circuit.analog_params or {}).get(nid, {})
        node_type = node.node_type
        W = params.get("W", 1e-6)
        L = params.get("L", 130e-9)

        # Own gm + intrinsic gain.
        if node_type in ("NMOS", "PMOS"):
            gm = params.get("gm") or estimate_gm(node_type, W, L, bias_a)
            gain = params.get("gain") or estimate_intrinsic_gain(node_type, W, L, bias_a)
        else:
            gm = 0.0
            gain = 0.0

        # Load cap = own self-cap + sum of sinks' input caps on output net.
        own_self_cap = _device_input_cap(node_type, params)
        sink_cap_sum = 0.0
        coupling_sum = 0.0
        if node.output_net and node.output_net in circuit.nets:
            net = circuit.nets[node.output_net]
            for sink_id in net.sinks:
                sink_node = circuit.nodes.get(sink_id)
                if sink_node is None:
                    continue
                sink_params = (circuit.analog_params or {}).get(sink_id, {})
                sink_cap_sum += _device_input_cap(sink_node.node_type, sink_params)
                # Coupling cap proxy: Cgd of the driving device contributes
                # parasitic feedthrough to each sink.
                coupling_sum += sink_params.get("Cgd", DEFAULT_DEVICE_PARAMS.get(
                    sink_node.node_type, {}).get("Cgd", 0.0))

        c_load = own_self_cap + sink_cap_sum + 1e-18  # avoid /0

        # Per-node "GBW": ω_t ≈ gm / C_load. Convert to Hz, log-compress.
        if gm > 0.0:
            wt = gm / c_load
            f_t = wt / (2.0 * math.pi)
            gbw_log = math.log10(max(f_t, 1.0))  # log10 Hz, in roughly [0, 12]
        else:
            gbw_log = 0.0

        raw_gbw[nid] = gbw_log
        raw_parasitic[nid] = coupling_sum + own_self_cap

        node_timing[nid] = {
            "gm": gm,
            "intrinsic_gain": gain,
            "c_load": c_load,
            "gbw_log10_hz": gbw_log,
            "coupling_cap": coupling_sum,
        }

    # Normalize GBW and parasitic to [0, 1] and derive bandwidth_limited.
    max_gbw = max(max(raw_gbw.values(), default=1.0), 1.0)
    max_parasitic = max(max(raw_parasitic.values(), default=1e-18), 1e-18)

    for nid, t in node_timing.items():
        gbw_score = raw_gbw[nid] / max_gbw
        parasitic_impact = raw_parasitic[nid] / max_parasitic
        # bandwidth-limited if parasitic load dominates gm capability
        bandwidth_limited = (
            t["gm"] > 0.0 and parasitic_impact > 0.5 and gbw_score < 0.5
        )
        t["gbw_score"] = gbw_score
        t["parasitic_impact"] = parasitic_impact
        t["bandwidth_limited"] = bool(bandwidth_limited)

    return {
        "node_timing": node_timing,
        "max_gbw": max_gbw,
        "max_parasitic": max_parasitic,
        # Mimic digital STA's contract so graph_builder can read it uniformly.
        "clock_period_ns": 0.0,
        "max_arrival_time_ns": 0.0,
        "min_slack_ns": 0.0,
        "num_critical_nodes": 0,
    }

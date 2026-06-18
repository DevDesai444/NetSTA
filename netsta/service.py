"""
Reusable analysis service: circuit -> GNN predictions -> agent design report.

One core that the CLI, the FastAPI backend, and any other client sit on top of.
Given a circuit (digital, analog, imported netlist, or NL spec) it runs STA +
the GNN (when a checkpoint exists, else it falls back to the STA ground truth so
the demo still works), then the 4-agent advisory pipeline — returning a single
JSON-serialisable dict with the graph, the per-node predictions, and the report.
"""

import os
from typing import Any, Dict, List, Optional

import numpy as np

from .agents import diagnose
from .circuit_gen import Circuit, generate_circuit
from .graph_builder import circuit_to_pyg
from .sta import run_sta

DEFAULT_CKPT = os.environ.get("NETSTA_CHECKPOINT", "checkpoints_real/best_model.pt")


def _f(x) -> float:
    return float(x)


def _logit(label: np.ndarray, scale: float = 10.0) -> np.ndarray:
    """Map a 0/1 label array to pseudo-logits so sigmoid recovers ~0/1."""
    return (np.asarray(label, dtype=float) - 0.5) * scale


# ---------------------------------------------------------------------------
# Circuit builders
# ---------------------------------------------------------------------------


def build_digital(num_inputs=8, num_gates=30, num_outputs=4, seed=42, name="digital") -> Circuit:
    return generate_circuit(
        num_inputs=num_inputs, num_gates=num_gates,
        num_outputs=num_outputs, seed=seed, name=name,
    )


def build_analog(topology="two_stage_opamp", seed=42) -> Circuit:
    from .analog_circuit_gen import generate_analog_circuit
    return generate_analog_circuit(seed=seed, topology=topology, name=topology)


def build_from_nl(query: str):
    """NL -> CircuitSpec -> analog circuit (returns (circuit, spec, backend))."""
    from .rag.circuit_parser import generate_from_spec, parse_to_spec
    spec, backend = parse_to_spec(query)
    circuit = generate_from_spec(spec)
    return circuit, spec, backend


def build_from_bench(path: str, seed=0) -> Circuit:
    from .benchmark_import import load_bench_circuit
    return load_bench_circuit(path, seed=seed)


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def _run_predictions(circuit, checkpoint_path):
    """Return (node_ids, predictions, source, graph_embedding). GNN if a
    checkpoint loads, else STA ground truth as a stand-in."""
    is_analog = bool(getattr(circuit, "is_analog", False))
    if is_analog:
        from .analog_sta import run_analog_sta
        sta = run_analog_sta(circuit)
    else:
        sta = run_sta(circuit)
    data = circuit_to_pyg(circuit, sta)
    node_ids = circuit.primary_inputs + circuit.gate_ids + circuit.primary_outputs

    if checkpoint_path and os.path.exists(checkpoint_path):
        try:
            from .predict import load_model, predict_circuit
            model = load_model(checkpoint_path, device="cpu")
            out = predict_circuit(model, circuit, device="cpu")
            return out["node_ids"], out["predictions"], "gnn", out.get("graph_emb"), data, sta
        except Exception as exc:
            print(f"[service] GNN unavailable ({exc!r}); using STA ground truth.")

    # Fallback: STA/labeling ground truth in the shape the agents expect.
    preds = {
        "slack": data.y_slack.numpy(),
        "critical_path": _logit(data.y_critical.numpy()),
        "congestion": data.y_congestion.numpy(),
        "drc": _logit(data.y_drc.numpy()),
    }
    return node_ids, preds, "ground_truth", None, data, sta


def _graph_payload(circuit, node_ids, preds) -> Dict[str, Any]:
    idx = {nid: i for i, nid in enumerate(node_ids)}
    slack = np.asarray(preds.get("slack", []), dtype=float).reshape(-1)
    crit = 1.0 / (1.0 + np.exp(-np.asarray(preds.get("critical_path", []), dtype=float).reshape(-1)))
    cong = np.asarray(preds.get("congestion", []), dtype=float).reshape(-1)
    drc = 1.0 / (1.0 + np.exp(-np.asarray(preds.get("drc", []), dtype=float).reshape(-1)))

    def at(arr, i):
        return _f(arr[i]) if i < arr.size else None

    nodes = []
    for i, nid in enumerate(node_ids):
        node = circuit.nodes.get(nid)
        pos = circuit.positions.get(nid, (0, 0))
        nodes.append({
            "id": nid,
            "type": node.node_type if node else "?",
            "x": int(pos[0]), "y": int(pos[1]),
            "slack": at(slack, i), "critical": at(crit, i),
            "congestion": at(cong, i), "drc": at(drc, i),
        })
    edges = []
    for net in circuit.nets.values():
        if net.driver in idx:
            for sink in net.sinks:
                if sink in idx:
                    edges.append([idx[net.driver], idx[sink]])
    return {"nodes": nodes, "edges": edges}


def analyze_circuit(
    circuit: Circuit,
    checkpoint_path: Optional[str] = DEFAULT_CKPT,
    topology: Optional[str] = None,
    process_node: str = "45nm",
    use_autogen: str = "auto",
) -> Dict[str, Any]:
    """Full pipeline on one circuit -> JSON-serialisable result dict."""
    node_ids, preds, source, _emb, data, sta = _run_predictions(circuit, checkpoint_path)
    report = diagnose(
        {"node_ids": node_ids, "predictions": preds},
        circuit_name=getattr(circuit, "name", "circuit"),
        topology=topology, process_node=process_node, use_autogen=use_autogen,
    )
    return {
        "circuit_name": getattr(circuit, "name", "circuit"),
        "prediction_source": source,
        "num_nodes": len(node_ids),
        "num_edges": int(data.edge_index.size(1)),
        "clock_period_ns": _f(sta.get("clock_period_ns", 0.0)),
        "graph": _graph_payload(circuit, node_ids, preds),
        "report": report.model_dump(),
    }

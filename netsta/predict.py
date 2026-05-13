"""
Inference utilities for NetSTA.

`predict_circuit` returns one prediction per active task in the loaded model,
plus the PyG Data tensor (which carries ground-truth labels and circuit
metadata) so callers can render any task without re-running the pipeline.
"""

import numpy as np
import torch

from .config import NetSTAConfig
from .model import NetSTAModel
from .circuit_gen import Circuit
from .sta import run_sta
from .graph_builder import circuit_to_pyg
from .train import TARGET_KEY


def load_model(checkpoint_path: str, device: str = "cpu") -> NetSTAModel:
    """Load a trained NetSTAModel from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg_data = dict(checkpoint["config"])
    if isinstance(cfg_data.get("active_tasks"), list):
        cfg_data["active_tasks"] = tuple(cfg_data["active_tasks"])
    config = NetSTAConfig(**cfg_data)
    model = NetSTAModel(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def predict_circuit(
    model: NetSTAModel,
    circuit: Circuit,
    device: str = "cpu",
) -> dict:
    """Run all active task heads on one circuit.

    Returns a dict with:
      - node_ids: ordered list of node identifiers
      - active_tasks: tasks the loaded model produces predictions for
      - predictions / ground_truth: {task_name: np.ndarray} per active task
      - sta_results: full STA results
      - data: the PyG Data tensor (carries y_* labels, max_slack, etc.)
      - node_emb / graph_emb: backbone outputs (numpy)
      - Backwards-compatible aliases for slack / critical_path (used by the
        existing Streamlit timing-analysis path).
    """
    sta_results = run_sta(circuit)
    data = circuit_to_pyg(circuit, sta_results)
    data = data.to(device)

    preds = model(data.x, data.edge_index, edge_attr=data.edge_attr)

    node_order = circuit.primary_inputs + circuit.gate_ids + circuit.primary_outputs
    active_tasks = list(model.heads.keys())

    predictions = {t: preds[t].detach().cpu().numpy() for t in active_tasks}
    ground_truth = {}
    for t in active_tasks:
        key = TARGET_KEY[t]
        if hasattr(data, key):
            ground_truth[t] = getattr(data, key).detach().cpu().numpy()
        else:
            ground_truth[t] = np.zeros_like(predictions[t])

    out = {
        "node_ids": node_order,
        "active_tasks": active_tasks,
        "predictions": predictions,
        "ground_truth": ground_truth,
        "sta_results": sta_results,
        "data": data,
        "node_emb": preds["_node_emb"].detach().cpu().numpy(),
        "graph_emb": preds["_graph_emb"].detach().cpu().numpy(),
        "max_slack": float(getattr(data, "max_slack", 1.0)),
    }

    # Backwards-compat fields consumed by older streamlit/predict callers.
    if "slack" in predictions:
        out["predicted_slack"] = predictions["slack"]
        out["ground_truth_slack"] = ground_truth["slack"]
    if "critical_path" in predictions:
        crit_prob = 1.0 / (1.0 + np.exp(-predictions["critical_path"]))
        out["predicted_critical"] = crit_prob
        out["predicted_critical_binary"] = (crit_prob > 0.85).astype(int)
        out["ground_truth_critical"] = ground_truth["critical_path"].astype(int)
    return out

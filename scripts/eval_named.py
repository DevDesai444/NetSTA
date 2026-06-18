"""
Evaluate a trained NetSTA checkpoint on famous held-out benchmark circuits.

These circuits are excluded from training (see build_real_dataset --exclude) so
this is a clean per-benchmark generalization check. Each circuit is windowed
into fan-in cones (same as the training distribution), labelled under an
aggressive clock so the critical-path label is non-trivial, and the model's
per-node predictions are pooled across cones for R²/AUC.

    python3 scripts/eval_named.py --checkpoint checkpoints_real/circuit/best_model.pt
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from netsta.benchmark_import import cone_windows, load_netlist
from netsta.evaluate import classification_metrics, regression_metrics
from netsta.graph_builder import circuit_to_pyg
from netsta.predict import load_model
from netsta.sta import run_sta

NAMED = {
    "c6288": "benchmarks/iscas/ISCAS85/c6288/c6288.v",   # ISCAS-85 16x16 multiplier
    "b19": "benchmarks/itc99/i99t/b19/b19.bench",        # ITC'99 large design
}

_REG = ("slack", "arrival_time", "required_time")
_CLS = ("critical_path", "drc")
_TARGET = {
    "slack": "y_slack", "arrival_time": "y_arrival_time",
    "required_time": "y_required_time", "critical_path": "y_critical", "drc": "y_drc",
}


@torch.no_grad()
def eval_circuit(model, circuit, clock_factor=0.95, max_cones=40):
    cones = cone_windows(circuit, max_cones=max_cones, min_cone_nodes=8, seed=7)
    if not cones:
        cones = [circuit]
    pooled = {t: ([], []) for t in _REG + _CLS}
    for cone in cones:
        base = run_sta(cone)
        clock = max(base["max_arrival_time_ns"] * clock_factor, 1e-3)
        res = run_sta(cone, clock_period_ns=clock)
        data = circuit_to_pyg(cone, res)
        out = model(data.x, data.edge_index, edge_attr=data.edge_attr)
        for t in _REG + _CLS:
            key = _TARGET[t]
            if t in out and hasattr(data, key):
                pooled[t][0].append(out[t].cpu().numpy())
                pooled[t][1].append(getattr(data, key).cpu().numpy())

    metrics = {}
    for t in _REG:
        if pooled[t][0]:
            pr = np.concatenate(pooled[t][0])
            tg = np.concatenate(pooled[t][1])
            m = regression_metrics(pr, tg)
            metrics[t] = {"r2": round(m["r2"], 4), "mae": round(m["mae"], 4)}
    for t in _CLS:
        if pooled[t][0]:
            pr = np.concatenate(pooled[t][0])
            tg = np.concatenate(pooled[t][1])
            m, _ = classification_metrics(pr, tg)
            metrics[t] = {"auc_roc": round(m["auc_roc"], 4), "f1": round(m["f1"], 4)}
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints_real/best_model.pt")
    ap.add_argument("--out", default="results/named_benchmarks.json")
    args = ap.parse_args()

    model = load_model(args.checkpoint, device="cpu")
    results = {}
    for name, path in NAMED.items():
        if not os.path.exists(path):
            print(f"[skip] {name}: {path} not found")
            continue
        circuit = load_netlist(path, name=name)
        print(f"Evaluating {name} ({len(circuit.nodes)} nodes)...")
        results[name] = eval_circuit(model, circuit)
        print(f"  {results[name]}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()

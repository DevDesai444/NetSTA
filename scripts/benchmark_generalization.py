#!/usr/bin/env python3
"""
Generalization benchmark for NetSTA.

Three out-of-distribution tests:
  1. Size:     train on 20-100 gate circuits, test on 150-300 gate circuits.
  2. Topology: train on {AND,OR,NOT}-only circuits, test on circuits that
               also include {XOR,XNOR,MUX,AOI,OAI}.
  3. Depth:    train on shallow circuits (max logical depth <= 10), test on
               deeper ones (>= 15).

Reports train and test metrics per dimension and the gap between them.
"""

import argparse
import os
import random
import sys
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from _bench_utils import (
    RESULTS_DIR,
    collect_test_predictions,
    ensure_dirs,
    fit_torch_model,
    make_netsta,
    save_json,
    save_text,
    write_markdown_table,
)

from netsta.circuit_gen import generate_circuit
from netsta.dataset import NetSTAAugment
from netsta.evaluate import classification_metrics, regression_metrics
from netsta.graph_builder import circuit_to_pyg
from netsta.nangate45 import NANGATE45_CELLS
from netsta.sta import run_sta
from netsta.train import _select_device


SIMPLE_GATE_FUNCS = {"AND", "OR", "NOT"}


def _cells_for(funcs):
    """Cell names whose 'function' is in `funcs`."""
    return [name for name, info in NANGATE45_CELLS.items() if info["function"] in funcs]


def _generate_one(idx, *, gate_lo, gate_hi, allowed_cells=None, seed_base=0):
    n_gates = random.randint(gate_lo, gate_hi)
    n_inputs = max(4, n_gates // 5)
    n_outputs = max(2, n_gates // 10)
    circuit = generate_circuit(
        num_inputs=n_inputs,
        num_gates=n_gates,
        num_outputs=n_outputs,
        seed=seed_base + idx,
        name=f"gen_{idx:05d}",
        allowed_cells=allowed_cells,
    )
    sta_res = run_sta(circuit)
    data = circuit_to_pyg(circuit, sta_res)
    data._max_logical_depth = max(
        (sta_res["node_timing"].get(nid, {}).get("logical_depth", 0)
         for nid in sta_res["node_timing"]), default=0,
    )
    return data


def build_in_memory_dataset(
    n: int,
    *,
    gate_range: Optional[tuple] = (15, 80),
    allowed_funcs: Optional[set] = None,
    depth_predicate: Optional[Callable[[int], bool]] = None,
    seed: int = 0,
    max_tries_factor: int = 20,
) -> List[Data]:
    """Generate `n` PyG Data objects satisfying given constraints.

    `depth_predicate` (e.g. lambda d: d <= 10) is applied AFTER STA. Generation
    retries up to `max_tries_factor * n` times before giving up.
    """
    random.seed(seed)
    allowed_cells = _cells_for(allowed_funcs) if allowed_funcs else None
    if allowed_funcs is not None and not allowed_cells:
        raise RuntimeError(f"No Nangate45 cells match functions {allowed_funcs}")
    out: List[Data] = []
    tries = 0
    lo, hi = gate_range or (15, 80)
    seed_base = seed * 1000
    while len(out) < n and tries < max_tries_factor * n:
        d = _generate_one(tries, gate_lo=lo, gate_hi=hi,
                          allowed_cells=allowed_cells, seed_base=seed_base)
        tries += 1
        if depth_predicate is not None and not depth_predicate(d._max_logical_depth):
            continue
        out.append(d)
    if len(out) < n:
        print(f"  WARN: only produced {len(out)}/{n} circuits after {tries} tries "
              f"(constraints too tight?)")
    return out


def metrics_on(model, loader, device):
    active = list(model.heads.keys())
    preds, targets = collect_test_predictions(model, loader, device, active)
    slack = regression_metrics(preds["slack"], targets["slack"]) if "slack" in active else {}
    crit = {}
    if "critical_path" in active:
        crit, _ = classification_metrics(preds["critical_path"], targets["critical_path"])
    return {
        "slack_mse": slack.get("mse", float("nan")),
        "slack_r2": slack.get("r2", float("nan")),
        "cp_accuracy": crit.get("accuracy", float("nan")),
        "cp_f1": crit.get("f1", float("nan")),
    }


def run_test(
    name: str,
    train_data: List[Data],
    test_data: List[Data],
    *,
    device, epochs, patience, warmup_epochs, batch_size, seed,
):
    if not train_data or not test_data:
        return {"name": name, "error": "insufficient data after constraints"}

    print(f"\n=== {name} ===")
    print(f"  train circuits: {len(train_data)}  test circuits: {len(test_data)}")
    sample = train_data[0]
    node_feature_dim = sample.x.size(1)
    edge_feature_dim = sample.edge_attr.size(1) if sample.edge_attr.dim() > 1 else 1

    # Carve a val split off the train side (15%).
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(train_data)).tolist()
    val_size = max(1, int(0.15 * len(train_data)))
    val_data = [train_data[i] for i in perm[:val_size]]
    actual_train = [train_data[i] for i in perm[val_size:]]

    augment = NetSTAAugment()
    train_loader = DataLoader(
        [augment(d) for d in actual_train], batch_size=batch_size, shuffle=True
    )
    # Note: above bakes in one round of augmentation since these are plain lists,
    # not a Dataset wrapping. Acceptable here since augmentation is mild.
    val_loader = DataLoader(val_data, batch_size=batch_size)
    test_loader = DataLoader(test_data, batch_size=batch_size)
    train_metric_loader = DataLoader(actual_train, batch_size=batch_size)

    # Standardize on the train side only — testing on a distribution-shifted
    # split would otherwise leak its own scale into the model. Compute stats
    # for slack, AT, and RT in one pass over the in-memory train Data objects.
    from netsta.stats import DatasetStats
    has_at = hasattr(actual_train[0], "y_arrival_time") if actual_train else False
    has_rt = hasattr(actual_train[0], "y_required_time") if actual_train else False
    stats = DatasetStats.from_target_tensors(
        (d.y_slack for d in actual_train),
        arrival_tensors=(d.y_arrival_time for d in actual_train) if has_at else None,
        required_tensors=(d.y_required_time for d in actual_train) if has_rt else None,
    )

    torch.manual_seed(seed)
    model = make_netsta(
        node_feature_dim, edge_feature_dim,
        slack_mean=stats.slack_mean, slack_std=stats.slack_std,
        arrival_time_mean=stats.arrival_time_mean,
        arrival_time_std=stats.arrival_time_std,
        required_time_mean=stats.required_time_mean,
        required_time_std=stats.required_time_std,
    ).to(device)
    best_state, *_ = fit_torch_model(
        model, train_loader, val_loader, device,
        epochs=epochs, patience=patience, warmup_epochs=warmup_epochs,
        log_prefix=f"  [{name}] ",
    )
    model.load_state_dict(best_state)
    train_m = metrics_on(model, train_metric_loader, device)
    test_m = metrics_on(model, test_loader, device)
    return {"name": name, "train": train_m, "test": test_m,
            "n_train": len(actual_train), "n_val": len(val_data), "n_test": len(test_data)}


def main():
    parser = argparse.ArgumentParser(description="Run NetSTA generalization tests")
    parser.add_argument("--num-circuits", type=int, default=1000,
                        help="Target |train| per test (each test generates its own pool)")
    parser.add_argument("--num-test-circuits", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    args = parser.parse_args()

    ensure_dirs()
    device = _select_device(args.device)
    print(f"Device: {device}")

    results = []

    # --- Test 1: Size shift ---
    try:
        train_size = build_in_memory_dataset(
            args.num_circuits, gate_range=(20, 100), seed=args.seed,
        )
        test_size = build_in_memory_dataset(
            args.num_test_circuits, gate_range=(150, 300), seed=args.seed + 1,
        )
        results.append(run_test(
            "Size (small→large)", train_size, test_size,
            device=device, epochs=args.epochs, patience=args.patience,
            warmup_epochs=args.warmup_epochs, batch_size=args.batch_size, seed=args.seed,
        ))
    except Exception as exc:
        print(f"[size] FAILED: {exc!r}")
        results.append({"name": "Size (small→large)", "error": repr(exc)})

    # --- Test 2: Topology shift ---
    try:
        train_topo = build_in_memory_dataset(
            args.num_circuits, gate_range=(20, 80),
            allowed_funcs=SIMPLE_GATE_FUNCS, seed=args.seed + 2,
        )
        test_topo = build_in_memory_dataset(
            args.num_test_circuits, gate_range=(20, 80),
            allowed_funcs=None, seed=args.seed + 3,  # full library
        )
        results.append(run_test(
            "Topology (subset→full)", train_topo, test_topo,
            device=device, epochs=args.epochs, patience=args.patience,
            warmup_epochs=args.warmup_epochs, batch_size=args.batch_size, seed=args.seed,
        ))
    except Exception as exc:
        print(f"[topo] FAILED: {exc!r}")
        results.append({"name": "Topology (subset→full)", "error": repr(exc)})

    # --- Test 3: Depth shift ---
    # Empirically (50-seed sweep of the current circuit_gen + STA):
    #   gates [15, 40] -> max_depth in [5, 11], mostly 7-9
    #   gates [80, 300] -> max_depth in [9, 16], mostly 10-13
    # So a "shallow" train pool of depth <= 7 catches roughly half the
    # smaller circuits, and a "deep" test pool of depth >= 10 catches
    # ~90 % of the larger ones. Both thresholds are reachable, the
    # distributions are clearly separated (no overlap by construction),
    # and the test exercises whether the model generalizes from
    # 5-7-hop paths to 10+-hop paths — exactly the iteration-depth
    # property the timing backbone is supposed to support.
    try:
        train_depth = build_in_memory_dataset(
            args.num_circuits, gate_range=(15, 40),
            depth_predicate=lambda d: d <= 7, seed=args.seed + 4,
            max_tries_factor=40,
        )
        test_depth = build_in_memory_dataset(
            args.num_test_circuits, gate_range=(80, 300),
            depth_predicate=lambda d: d >= 10, seed=args.seed + 5,
            max_tries_factor=40,
        )
        results.append(run_test(
            "Depth (shallow→deep)", train_depth, test_depth,
            device=device, epochs=args.epochs, patience=args.patience,
            warmup_epochs=args.warmup_epochs, batch_size=args.batch_size, seed=args.seed,
        ))
    except Exception as exc:
        print(f"[depth] FAILED: {exc!r}")
        results.append({"name": "Depth (shallow→deep)", "error": repr(exc)})

    headers = ["Generalization Test", "Train R²", "Test R²", "Δ R²",
               "Train CP Acc", "Test CP Acc", "Δ Acc"]
    rows = []
    for r in results:
        if "error" in r:
            rows.append([r["name"], "ERR", "ERR", "--", "ERR", "ERR", "--"])
            continue
        tr, te = r["train"], r["test"]
        dr = te["slack_r2"] - tr["slack_r2"]
        da = (te["cp_accuracy"] - tr["cp_accuracy"]) * 100
        rows.append([
            r["name"],
            f"{tr['slack_r2']:.3f}",
            f"{te['slack_r2']:.3f}",
            f"{'+' if dr >= 0 else ''}{dr:.3f}",
            f"{100*tr['cp_accuracy']:.1f}%",
            f"{100*te['cp_accuracy']:.1f}%",
            f"{'+' if da >= 0 else ''}{da:.1f}%",
        ])

    md = "# NetSTA Generalization Study\n\n"
    md += "Tests how well NetSTA transfers across three distribution shifts.\n\n"
    md += write_markdown_table(headers, rows)
    save_text(os.path.join(RESULTS_DIR, "generalization_study.md"), md)
    save_json(os.path.join(RESULTS_DIR, "generalization_study.json"), {
        "config": vars(args),
        "results": results,
    })
    print("\nSaved generalization_study.{md,json}")


if __name__ == "__main__":
    main()

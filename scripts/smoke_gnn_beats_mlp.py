#!/usr/bin/env python3
"""
Quick CPU smoke test: did the rewrite actually flip the GNN-vs-MLP outcome?

Trains NetSTA (timing backbone) and MLPBaseline on a small dataset for a
modest number of epochs, both with the new label/feature schema. Prints
slack R^2, MSE (ns^2), and critical-path F1 / AUC for each. Pass criterion:

    NetSTA.slack_r2 > MLPBaseline.slack_r2 - 0.02

The 0.02 cushion lets reasonable-but-narrow-loss differences pass while
still flagging an outright regression. If this fails, do not pay for the
Kaggle benchmark run -- iterate locally first.

Runtime: ~5 min on a macbook CPU at the defaults below.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from _bench_utils import (  # noqa: E402
    collect_test_predictions,
    compute_target_stats,
    fit_torch_model,
    make_netsta,
    split_indices,
)

from netsta.baselines import MLPBaseline  # noqa: E402
from netsta.dataset import NetSTAAugment, NetSTADataset, TransformSubset  # noqa: E402
from netsta.evaluate import classification_metrics, regression_metrics  # noqa: E402
from netsta.train import _select_device  # noqa: E402


def run_one(label, model, train_loader, val_loader, test_loader, device,
            epochs, patience):
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[{label}] params={n_params:,}")
    t0 = time.perf_counter()
    best_state, best_val_loss, best_epoch, _ = fit_torch_model(
        model, train_loader, val_loader, device,
        epochs=epochs, patience=patience, warmup_epochs=3,
        log_prefix=f"  [{label}] ",
    )
    train_secs = time.perf_counter() - t0
    model.load_state_dict(best_state)
    active = list(model.heads.keys())
    preds, targets = collect_test_predictions(model, test_loader, device, active)
    out = {
        "params": n_params,
        "train_time_s": train_secs,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
    }
    for task in ("slack", "arrival_time", "required_time"):
        if task in preds:
            out[task] = regression_metrics(preds[task], targets[task])
    if "critical_path" in preds:
        m, _ = classification_metrics(preds["critical_path"], targets["critical_path"])
        out["critical_path"] = m
    return out


def main():
    parser = argparse.ArgumentParser(description="GNN-beats-MLP CPU smoke test")
    parser.add_argument("--data-dir", default="data_smoke")
    parser.add_argument("--num-circuits", type=int, default=120)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--output", default="results/smoke_gnn_vs_mlp.json")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = _select_device("auto")
    print(f"Device: {device}")

    dataset = NetSTADataset(
        root=args.data_dir, num_circuits=args.num_circuits, seed=args.seed,
    )
    train_idx, val_idx, test_idx = split_indices(len(dataset), args.seed)
    print(f"Train/Val/Test: {len(train_idx)}/{len(val_idx)}/{len(test_idx)}")

    sample = dataset[0]
    node_dim = sample.x.size(1)
    edge_dim = sample.edge_attr.size(1)
    stats = compute_target_stats(dataset, train_idx)
    print(f"node_feat_dim={node_dim}  edge_feat_dim={edge_dim}  "
          f"slack mean={stats.slack_mean:.4f}ns std={stats.slack_std:.4f}ns | "
          f"AT mean={stats.arrival_time_mean:.4f}/std={stats.arrival_time_std:.4f} "
          f"RT mean={stats.required_time_mean:.4f}/std={stats.required_time_std:.4f}")

    augment = NetSTAAugment()
    train_ds = TransformSubset(dataset, train_idx, transform=augment)
    val_ds = TransformSubset(dataset, val_idx)
    test_ds = TransformSubset(dataset, test_idx)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)

    netsta = make_netsta(
        node_dim, edge_dim,
        hidden_dim=args.hidden,
        num_layers=args.num_layers,
        slack_mean=stats.slack_mean, slack_std=stats.slack_std,
        arrival_time_mean=stats.arrival_time_mean,
        arrival_time_std=stats.arrival_time_std,
        required_time_mean=stats.required_time_mean,
        required_time_std=stats.required_time_std,
    ).to(device)
    mlp = MLPBaseline(
        node_dim, hidden=args.hidden * 2,
        slack_mean=stats.slack_mean, slack_std=stats.slack_std,
    ).to(device)

    results = {
        "config": vars(args),
        "slack_stats": {"mean": stats.slack_mean, "std": stats.slack_std},
        "arrival_time_stats": {
            "mean": stats.arrival_time_mean, "std": stats.arrival_time_std,
        },
        "required_time_stats": {
            "mean": stats.required_time_mean, "std": stats.required_time_std,
        },
        "node_feature_dim": node_dim,
        "edge_feature_dim": edge_dim,
    }
    results["NetSTA"] = run_one(
        "NetSTA", netsta, train_loader, val_loader, test_loader,
        device, args.epochs, args.patience,
    )
    results["MLP"] = run_one(
        "MLP", mlp, train_loader, val_loader, test_loader,
        device, args.epochs, args.patience,
    )

    gnn_r2 = results["NetSTA"]["slack"]["r2"]
    mlp_r2 = results["MLP"]["slack"]["r2"]
    gnn_mse = results["NetSTA"]["slack"]["mse"]
    mlp_mse = results["MLP"]["slack"]["mse"]
    gnn_cp = results["NetSTA"].get("critical_path", {})
    mlp_cp = results["MLP"].get("critical_path", {})

    at_r2 = results["NetSTA"].get("arrival_time", {}).get("r2", float("nan"))
    rt_r2 = results["NetSTA"].get("required_time", {}).get("r2", float("nan"))

    print("\n=== Summary ===")
    print(f"  NetSTA  slack R^2 = {gnn_r2:.4f}   MSE (ns^2) = {gnn_mse:.6f}")
    print(f"  MLP     slack R^2 = {mlp_r2:.4f}   MSE (ns^2) = {mlp_mse:.6f}")
    if "arrival_time" in results["NetSTA"]:
        print(f"  NetSTA  AT R^2    = {at_r2:.4f}   RT R^2 = {rt_r2:.4f}")
    if gnn_cp and mlp_cp:
        print(f"  NetSTA  CP F1 = {gnn_cp['f1']:.4f}   AUC = {gnn_cp['auc_roc']:.4f}")
        print(f"  MLP     CP F1 = {mlp_cp['f1']:.4f}   AUC = {mlp_cp['auc_roc']:.4f}")

    slack_passes = gnn_r2 > mlp_r2 - 0.02
    # Auxiliary supervision sanity check: AT and RT predictions must beat the
    # mean baseline (R^2 > 0). If they don't, the backbone halves aren't
    # learning their intended quantities and the compositional slack head is
    # built on a faulty foundation.
    aux_passes = (
        (not (at_r2 == at_r2) or at_r2 > 0.0)  # tolerate NaN if AT not active
        and (not (rt_r2 == rt_r2) or rt_r2 > 0.0)
    )
    passed = slack_passes and aux_passes
    results["pass_criterion"] = (
        "gnn_r2 > mlp_r2 - 0.02 AND (AT_r2 > 0 if active) AND (RT_r2 > 0 if active)"
    )
    results["passed"] = bool(passed)
    if passed:
        print("\n  PASS")
    elif not slack_passes:
        print("\n  FAIL — NetSTA did not beat MLP on slack R^2")
    else:
        print("\n  FAIL — auxiliary AT/RT predictions are below the mean baseline")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  results -> {args.output}")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Multi-seed robustness benchmark.

Trains NetSTA with N different seeds (default 5) and reports
mean ± std of test-set metrics. Failures in individual seeds are caught,
logged, and excluded from the aggregate without stopping the run.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from _bench_utils import (
    RESULTS_DIR,
    build_loaders,
    collect_test_predictions,
    ensure_dirs,
    fit_torch_model,
    compute_target_stats,
    make_netsta,
    save_json,
    save_text,
    split_indices,
    write_markdown_table,
)

from netsta.dataset import NetSTAAugment, NetSTADataset
from netsta.evaluate import classification_metrics, regression_metrics
from netsta.train import _select_device


def _parse_seeds(arg: str):
    seeds = [int(s.strip()) for s in arg.split(",") if s.strip()]
    if not seeds:
        raise argparse.ArgumentTypeError("--seeds must contain at least one integer")
    return seeds


def run_seed(seed, dataset, node_feature_dim, edge_feature_dim,
             device, epochs, patience, warmup_epochs, batch_size):
    torch.manual_seed(seed)
    train_idx, val_idx, test_idx = split_indices(len(dataset), seed)
    train_loader, val_loader, test_loader, *_ = build_loaders(
        dataset, train_idx, val_idx, test_idx, batch_size,
        train_transform=NetSTAAugment(),
    )

    stats = compute_target_stats(dataset, train_idx)
    model = make_netsta(
        node_feature_dim, edge_feature_dim,
        slack_mean=stats.slack_mean, slack_std=stats.slack_std,
        arrival_time_mean=stats.arrival_time_mean,
        arrival_time_std=stats.arrival_time_std,
        required_time_mean=stats.required_time_mean,
        required_time_std=stats.required_time_std,
    ).to(device)
    best_state, best_val_loss, best_epoch, _ = fit_torch_model(
        model, train_loader, val_loader, device,
        epochs=epochs, patience=patience, warmup_epochs=warmup_epochs,
        log_prefix=f"[seed={seed}] ",
    )
    model.load_state_dict(best_state)
    active = list(model.heads.keys())
    preds, targets = collect_test_predictions(model, test_loader, device, active)

    slack = regression_metrics(preds["slack"], targets["slack"]) if "slack" in active else {}
    crit = {}
    if "critical_path" in active:
        crit, _ = classification_metrics(preds["critical_path"], targets["critical_path"])
    return {
        "slack_mse": slack.get("mse", float("nan")),
        "slack_r2": slack.get("r2", float("nan")),
        "cp_accuracy": crit.get("accuracy", float("nan")),
        "cp_f1": crit.get("f1", float("nan")),
        "cp_auc_roc": crit.get("auc_roc", float("nan")),
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
    }


def main():
    parser = argparse.ArgumentParser(description="Run NetSTA multi-seed robustness analysis")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--num-circuits", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--seeds", type=_parse_seeds, default=[42, 123, 456, 789, 1024])
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    args = parser.parse_args()

    ensure_dirs()
    device = _select_device(args.device)
    print(f"Device: {device} | seeds: {args.seeds}")

    dataset = NetSTADataset(root=args.data_dir, num_circuits=args.num_circuits, seed=42)
    sample = dataset[0]
    node_feature_dim = sample.x.size(1)
    edge_feature_dim = sample.edge_attr.size(1) if sample.edge_attr.dim() > 1 else 1

    per_seed = {}
    for seed in args.seeds:
        print(f"\n=== seed {seed} ===")
        try:
            per_seed[seed] = run_seed(
                seed, dataset, node_feature_dim, edge_feature_dim,
                device, args.epochs, args.patience, args.warmup_epochs, args.batch_size,
            )
        except Exception as exc:
            print(f"[seed={seed}] FAILED: {exc!r}")
            per_seed[seed] = {"error": repr(exc)}

    # Aggregate
    metric_keys = ["slack_mse", "slack_r2", "cp_accuracy", "cp_f1", "cp_auc_roc"]
    pretty = {
        "slack_mse": "Slack MSE",
        "slack_r2": "Slack R²",
        "cp_accuracy": "CP Accuracy",
        "cp_f1": "CP F1",
        "cp_auc_roc": "CP AUC-ROC",
    }
    valid_seeds = [s for s in args.seeds if "error" not in per_seed[s]]
    agg = {}
    for k in metric_keys:
        vals = [per_seed[s][k] for s in valid_seeds if not np.isnan(per_seed[s].get(k, np.nan))]
        agg[k] = (float(np.mean(vals)) if vals else float("nan"),
                  float(np.std(vals)) if len(vals) > 1 else 0.0)

    # Table
    headers = ["Metric"] + [f"Seed {s}" for s in args.seeds] + ["Mean ± Std"]
    rows = []
    for k in metric_keys:
        is_pct = k in ("cp_accuracy",)
        per = []
        for s in args.seeds:
            v = per_seed[s].get(k)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                per.append("ERR")
            elif is_pct:
                per.append(f"{100*v:.1f}%")
            elif k in ("slack_mse",):
                per.append(f"{v:.4f}")
            else:
                per.append(f"{v:.3f}")
        mean, std = agg[k]
        if is_pct:
            mean_std = f"{100*mean:.1f}% ± {100*std:.1f}%"
        elif k in ("slack_mse",):
            mean_std = f"{mean:.4f} ± {std:.4f}"
        else:
            mean_std = f"{mean:.3f} ± {std:.3f}"
        rows.append([pretty[k]] + per + [mean_std])

    md = "# NetSTA Robustness Analysis\n\n"
    md += f"_Trained NetSTA across {len(args.seeds)} seeds: {args.seeds}._\n"
    md += f"_Each: {args.num_circuits} circuits, {args.epochs} epochs (early-stop), seed-specific split._\n\n"
    if len(valid_seeds) < len(args.seeds):
        failed = [s for s in args.seeds if s not in valid_seeds]
        md += f"> Failed seeds (excluded from aggregates): {failed}\n\n"
    md += write_markdown_table(headers, rows)
    save_text(os.path.join(RESULTS_DIR, "robustness_analysis.md"), md)
    save_json(os.path.join(RESULTS_DIR, "robustness_analysis.json"), {
        "config": vars(args),
        "per_seed": per_seed,
        "aggregate": agg,
    })
    print("\nSaved:")
    print(f"  {RESULTS_DIR}/robustness_analysis.md")
    print(f"  {RESULTS_DIR}/robustness_analysis.json")


if __name__ == "__main__":
    main()

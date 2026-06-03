#!/usr/bin/env python3
"""
Architecture and feature ablation study for NetSTA.

Trains 9 variants on the same dataset and split, then reports per-task
deltas from the full reference model.
"""

import argparse
import copy
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from _bench_utils import (
    PLOTS_DIR,
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


# Node feature layout (see netsta/graph_builder.py):
#   [0 : GATE_TYPE_DIM]                gate-type one-hot incl. PI/PO  (13 dims)
#   [GATE_TYPE_DIM : analog_end]       analog device-type + W/L + op_region + sym
#   [last two cols]                    [is_digital, is_analog]
# The `analog_device` ablation zeros every dim from GATE_TYPE_DIM onward
# (analog block + circuit-type flags). On a pure-digital dataset that block is
# all zero anyway, so this ablation only bites for mixed/analog runs.
def _make_feature_mask(node_feature_dim: int, drop: str):
    from netsta.graph_builder import GATE_TYPE_DIM
    mask = torch.ones(node_feature_dim)
    if drop == "gate_type":
        mask[:GATE_TYPE_DIM] = 0.0
    elif drop == "analog_device":
        mask[GATE_TYPE_DIM:] = 0.0
    return mask


def _transform_factory(mask=None, drop_edge_attr=False):
    def fn(data):
        data = data.clone()
        if mask is not None:
            data.x = data.x * mask
        if drop_edge_attr:
            data.edge_attr = torch.zeros_like(data.edge_attr)
        # Augmentation on top (training only) is added at loader-build time.
        return data
    return fn


class _ChainTransform:
    def __init__(self, *fns):
        self.fns = [f for f in fns if f is not None]
    def __call__(self, data):
        for f in self.fns:
            data = f(data)
        return data


ABLATIONS = [
    ("Full Model (reference)", {}),
    ("No edge features", {"drop_edge": True}),
    ("No residual connections", {"use_residual": False}),
    ("No attention", {"use_attention": False}),
    ("2 layers", {"num_layers": 2}),
    ("6 layers", {"num_layers": 6}),
    ("No gate type features", {"drop_feat": "gate_type"}),
    ("Single task (slack only)", {"tasks": ("slack",)}),
]


def run_ablation(name, ablation, dataset, node_feature_dim, edge_feature_dim,
                 device, epochs, patience, warmup_epochs, batch_size, seed):
    print(f"\n--- {name} ---")
    torch.manual_seed(seed)
    train_idx, val_idx, test_idx = split_indices(len(dataset), seed)

    feature_mask = None
    if "drop_feat" in ablation:
        feature_mask = _make_feature_mask(node_feature_dim, ablation["drop_feat"])
    drop_edge = ablation.get("drop_edge", False)

    base_transform = _transform_factory(feature_mask, drop_edge)
    train_transform = _ChainTransform(base_transform, NetSTAAugment())
    eval_transform = base_transform  # mask edges/features on val+test too

    from netsta.dataset import TransformSubset
    from torch_geometric.loader import DataLoader as PygLoader
    train_ds = TransformSubset(dataset, train_idx, transform=train_transform)
    val_ds = TransformSubset(dataset, val_idx, transform=eval_transform)
    test_ds = TransformSubset(dataset, test_idx, transform=eval_transform)
    train_loader = PygLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = PygLoader(val_ds, batch_size=batch_size)
    test_loader = PygLoader(test_ds, batch_size=batch_size)

    stats = compute_target_stats(dataset, train_idx)
    model_kwargs = dict(
        hidden_dim=64,
        num_layers=ablation.get("num_layers", 4),
        num_heads=4,
        use_residual=ablation.get("use_residual", True),
        use_attention=ablation.get("use_attention", True),
        tasks=ablation.get("tasks", ("slack", "arrival_time", "required_time")),
        slack_mean=stats.slack_mean,
        slack_std=stats.slack_std,
        arrival_time_mean=stats.arrival_time_mean,
        arrival_time_std=stats.arrival_time_std,
        required_time_mean=stats.required_time_mean,
        required_time_std=stats.required_time_std,
    )
    model = make_netsta(node_feature_dim, edge_feature_dim, **model_kwargs).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  params={n_params:,}  config={model_kwargs}")

    best_state, best_val_loss, best_epoch, _ = fit_torch_model(
        model, train_loader, val_loader, device,
        epochs=epochs, patience=patience, warmup_epochs=warmup_epochs,
        log_prefix="  ",
    )
    model.load_state_dict(best_state)
    active = list(model.heads.keys())
    preds, targets = collect_test_predictions(model, test_loader, device, active)

    slack = regression_metrics(preds["slack"], targets["slack"]) if "slack" in active else {}
    crit = {}
    if "critical_path" in active:
        crit, _ = classification_metrics(preds["critical_path"], targets["critical_path"])
    return {
        "name": name,
        "config": model_kwargs,
        "params": n_params,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "slack_mse": slack.get("mse", float("nan")),
        "slack_r2": slack.get("r2", float("nan")),
        "cp_accuracy": crit.get("accuracy", float("nan")),
        "cp_f1": crit.get("f1", float("nan")),
    }


def _delta(reference, val, *, lower_better=False):
    if reference is None or val is None:
        return "--"
    if np.isnan(reference) or np.isnan(val):
        return "--"
    d = val - reference
    sign = "+" if d > 0 else ""
    return f"{sign}{d:.4f}" if abs(d) < 1 else f"{sign}{d:.3f}"


def _delta_pct(reference, val):
    if reference is None or val is None:
        return "--"
    if np.isnan(reference) or np.isnan(val):
        return "--"
    d = (val - reference) * 100
    sign = "+" if d > 0 else ""
    return f"{sign}{d:.1f}%"


def _matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def main():
    parser = argparse.ArgumentParser(description="Run NetSTA ablation study")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--num-circuits", type=int, default=1000)
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
    dataset = NetSTADataset(root=args.data_dir, num_circuits=args.num_circuits, seed=args.seed)
    sample = dataset[0]
    node_feature_dim = sample.x.size(1)
    edge_feature_dim = sample.edge_attr.size(1) if sample.edge_attr.dim() > 1 else 1

    results = []
    for name, ablation in ABLATIONS:
        try:
            results.append(run_ablation(
                name, ablation, dataset, node_feature_dim, edge_feature_dim,
                device, args.epochs, args.patience, args.warmup_epochs,
                args.batch_size, args.seed,
            ))
        except Exception as exc:
            print(f"[{name}] FAILED: {exc!r}")
            results.append({"name": name, "error": repr(exc)})

    # Reference is the first non-failed run.
    ref = next((r for r in results if "error" not in r), None)
    if ref is None:
        save_text(os.path.join(RESULTS_DIR, "ablation_study.md"), "# Ablation study\n\nAll runs failed.\n")
        return

    headers = ["Configuration", "Slack MSE", "Δ MSE", "Slack R²", "Δ R²",
               "CP Acc", "Δ Acc", "CP F1", "Δ F1"]
    rows = []
    for r in results:
        if "error" in r:
            rows.append([r["name"], "ERR", "--", "--", "--", "--", "--", "--", "--"])
            continue
        is_ref = (r is ref)
        rows.append([
            r["name"],
            f"{r['slack_mse']:.4f}",
            "—" if is_ref else _delta(ref["slack_mse"], r["slack_mse"]),
            f"{r['slack_r2']:.3f}",
            "—" if is_ref else _delta(ref["slack_r2"], r["slack_r2"]),
            f"{100*r['cp_accuracy']:.1f}%" if not np.isnan(r["cp_accuracy"]) else "--",
            "—" if is_ref else (_delta_pct(ref["cp_accuracy"], r["cp_accuracy"])
                                if not np.isnan(r["cp_accuracy"]) else "--"),
            f"{r['cp_f1']:.3f}" if not np.isnan(r["cp_f1"]) else "--",
            "—" if is_ref else (_delta(ref["cp_f1"], r["cp_f1"])
                                if not np.isnan(r["cp_f1"]) else "--"),
        ])

    md = "# NetSTA Ablation Study\n\n"
    md += f"_Reference: {ref['name']}. {args.num_circuits} circuits, seed {args.seed}._\n\n"
    md += write_markdown_table(headers, rows)
    md += "\n![ablation R² delta](plots/ablation_chart.png)\n"
    save_text(os.path.join(RESULTS_DIR, "ablation_study.md"), md)
    save_json(os.path.join(RESULTS_DIR, "ablation_study.json"), {
        "config": vars(args),
        "reference": ref["name"],
        "results": results,
    })

    # Bar chart of R² delta vs reference (skip reference itself).
    plt = _matplotlib()
    others = [r for r in results if "error" not in r and r is not ref]
    if others:
        labels = [r["name"] for r in others]
        deltas = [r["slack_r2"] - ref["slack_r2"] for r in others]
        colors = ["tab:red" if d < 0 else "tab:green" for d in deltas]
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.barh(labels, deltas, color=colors)
        ax.axvline(0, color="black", linewidth=0.5)
        ax.set_xlabel("Δ Slack R² vs reference")
        ax.set_title("Ablation R² impact (negative = worse than full model)")
        ax.invert_yaxis()
        ax.grid(True, axis="x", alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(PLOTS_DIR, "ablation_chart.png"), dpi=120)
        plt.close(fig)

    print("\nSaved ablation_study.{md,json} and plots/ablation_chart.png")


if __name__ == "__main__":
    main()

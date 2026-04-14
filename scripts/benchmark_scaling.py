#!/usr/bin/env python3
"""
Scaling benchmark for NetSTA.

Two analyses:
  - Data scaling: train NetSTA on increasing dataset sizes and plot test
    metrics vs |train|.
  - Inference scaling: time GNN inference vs classical STA on freshly-
    generated circuits of varying gate counts, then report a speedup table.
"""

import argparse
import os
import sys
import time
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
    make_netsta,
    save_json,
    save_text,
    split_indices,
    write_markdown_table,
)

from timingnet.circuit_gen import generate_circuit
from timingnet.dataset import NetSTAAugment, TimingNetDataset
from timingnet.evaluate import classification_metrics, regression_metrics
from timingnet.graph_builder import circuit_to_pyg
from timingnet.sta import run_sta
from timingnet.train import _select_device


def _matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def train_on_subset(dataset, n_train_circuits, total_seed_circuits,
                    node_feature_dim, edge_feature_dim, device,
                    epochs, patience, warmup_epochs, batch_size, seed):
    """Use the first `n_train_circuits` for training; keep val/test at fixed fractions of total."""
    torch.manual_seed(seed)
    # Reproducibly shuffle the full pool, then take a prefix for training.
    perm = torch.randperm(total_seed_circuits,
                          generator=torch.Generator().manual_seed(seed)).tolist()
    val_size = int(0.15 * total_seed_circuits)
    test_size = int(0.15 * total_seed_circuits)
    val_idx = perm[:val_size]
    test_idx = perm[val_size : val_size + test_size]
    pool = perm[val_size + test_size :]
    train_idx = pool[:n_train_circuits]
    if len(train_idx) < n_train_circuits:
        print(f"  WARN: requested {n_train_circuits} train circuits but only "
              f"{len(train_idx)} available after val/test split")

    train_loader, val_loader, test_loader, *_ = build_loaders(
        dataset, train_idx, val_idx, test_idx, batch_size,
        train_transform=NetSTAAugment(),
    )
    model = make_netsta(node_feature_dim, edge_feature_dim).to(device)
    best_state, _, _, _ = fit_torch_model(
        model, train_loader, val_loader, device,
        epochs=epochs, patience=patience, warmup_epochs=warmup_epochs,
        log_prefix=f"  |train|={n_train_circuits} ",
    )
    model.load_state_dict(best_state)
    active = list(model.heads.keys())
    preds, targets = collect_test_predictions(model, test_loader, device, active)
    slack = regression_metrics(preds["slack"], targets["slack"]) if "slack" in active else {}
    crit = {}
    if "critical_path" in active:
        crit, _ = classification_metrics(preds["critical_path"], targets["critical_path"])
    return {
        "n_train": n_train_circuits,
        "slack_mse": slack.get("mse", float("nan")),
        "slack_r2": slack.get("r2", float("nan")),
        "cp_accuracy": crit.get("accuracy", float("nan")),
        "cp_f1": crit.get("f1", float("nan")),
        "cp_auc_roc": crit.get("auc_roc", float("nan")),
    }


@torch.no_grad()
def inference_time_one_circuit(model, data, device, repeats=5):
    """Mean ms over `repeats` forward passes after a small warmup."""
    data = data.to(device)
    # Warmup
    for _ in range(2):
        model(data.x, data.edge_index, edge_attr=data.edge_attr)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeats):
        model(data.x, data.edge_index, edge_attr=data.edge_attr)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / repeats * 1000.0


def sta_time_one_circuit(circuit, repeats=3):
    t0 = time.perf_counter()
    for _ in range(repeats):
        run_sta(circuit)
    return (time.perf_counter() - t0) / repeats * 1000.0


def main():
    parser = argparse.ArgumentParser(description="Run scaling analysis")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--max-circuits", type=int, default=2000,
                        help="Pool size to generate; sizes are sub-sampled from this.")
    parser.add_argument("--data-sizes", default="100,250,500,1000,1500,2000")
    parser.add_argument("--gate-sizes", default="20,50,100,200,500,1000")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    args = parser.parse_args()
    data_sizes = [int(x) for x in args.data_sizes.split(",") if x.strip()]
    gate_sizes = [int(x) for x in args.gate_sizes.split(",") if x.strip()]

    ensure_dirs()
    device = _select_device(args.device)
    print(f"Device: {device}")

    pool_size = max(args.max_circuits, max(data_sizes) + int(0.3 * args.max_circuits))
    print(f"Generating dataset pool: {pool_size} circuits")
    dataset = TimingNetDataset(root=args.data_dir, num_circuits=pool_size, seed=args.seed)
    sample = dataset[0]
    node_feature_dim = sample.x.size(1)
    edge_feature_dim = sample.edge_attr.size(1) if sample.edge_attr.dim() > 1 else 1

    # --- Data scaling ---
    print("\n=== Data scaling ===")
    data_rows = []
    for n in data_sizes:
        if n > pool_size - int(0.3 * pool_size):
            print(f"  skipping |train|={n}: not enough circuits in pool")
            data_rows.append({"n_train": n, "error": "insufficient pool"})
            continue
        try:
            row = train_on_subset(
                dataset, n, pool_size, node_feature_dim, edge_feature_dim,
                device, args.epochs, args.patience, args.warmup_epochs,
                args.batch_size, args.seed,
            )
            print(f"  |train|={n}: slack_r2={row['slack_r2']:.3f} cp_f1={row['cp_f1']:.3f}")
            data_rows.append(row)
        except Exception as exc:
            print(f"  |train|={n} FAILED: {exc!r}")
            data_rows.append({"n_train": n, "error": repr(exc)})

    # --- Inference scaling ---
    print("\n=== Inference scaling ===")
    # Train a reference model on the full pool's standard split for the timing model.
    train_idx, val_idx, test_idx = split_indices(len(dataset), args.seed)
    train_loader, val_loader, test_loader, *_ = build_loaders(
        dataset, train_idx, val_idx, test_idx, args.batch_size,
        train_transform=NetSTAAugment(),
    )
    timing_model = make_netsta(node_feature_dim, edge_feature_dim).to(device)
    print("Training timing reference model...")
    best_state, *_ = fit_torch_model(
        timing_model, train_loader, val_loader, device,
        epochs=args.epochs, patience=args.patience, warmup_epochs=args.warmup_epochs,
        log_prefix="  [timing-ref] ",
    )
    timing_model.load_state_dict(best_state)
    timing_model.eval()

    timing_rows = []
    for n_gates in gate_sizes:
        try:
            # Generate a representative circuit of this size.
            circuit = generate_circuit(
                num_inputs=max(4, n_gates // 8),
                num_gates=n_gates,
                num_outputs=max(2, n_gates // 16),
                seed=args.seed + n_gates,
                name=f"scale_{n_gates}",
            )
            sta_res = run_sta(circuit)
            data = circuit_to_pyg(circuit, sta_res)
            sta_ms = sta_time_one_circuit(circuit)
            gnn_ms = inference_time_one_circuit(timing_model, data, device)
            speedup = sta_ms / gnn_ms if gnn_ms > 0 else float("nan")
            print(f"  gates={n_gates}: STA={sta_ms:.2f}ms GNN={gnn_ms:.2f}ms ({speedup:.1f}x)")
            timing_rows.append({
                "n_gates": n_gates,
                "sta_ms": sta_ms,
                "gnn_ms": gnn_ms,
                "speedup": speedup,
            })
        except Exception as exc:
            print(f"  gates={n_gates} FAILED: {exc!r}")
            timing_rows.append({"n_gates": n_gates, "error": repr(exc)})

    # --- Plots ---
    plt = _matplotlib()

    valid_data = [r for r in data_rows if "error" not in r]
    if valid_data:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        ns = [r["n_train"] for r in valid_data]
        r2s = [r["slack_r2"] for r in valid_data]
        f1s = [r["cp_f1"] for r in valid_data]
        axes[0].plot(ns, r2s, marker="o")
        axes[0].set_xlabel("Training circuits")
        axes[0].set_ylabel("Slack R²")
        axes[0].set_title("Data scaling: Slack R²")
        axes[0].grid(True, alpha=0.3)
        axes[1].plot(ns, f1s, marker="o", color="tab:orange")
        axes[1].set_xlabel("Training circuits")
        axes[1].set_ylabel("Critical-Path F1")
        axes[1].set_title("Data scaling: CP F1")
        axes[1].grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(PLOTS_DIR, "data_scaling.png"), dpi=120)
        plt.close(fig)

    valid_timing = [r for r in timing_rows if "error" not in r]
    if valid_timing:
        fig, ax = plt.subplots(figsize=(6, 4))
        ns = [r["n_gates"] for r in valid_timing]
        stas = [r["sta_ms"] for r in valid_timing]
        gnns = [r["gnn_ms"] for r in valid_timing]
        ax.plot(ns, stas, marker="o", label="Classical STA")
        ax.plot(ns, gnns, marker="s", label="NetSTA GNN")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Circuit size (gates)")
        ax.set_ylabel("Time per circuit (ms)")
        ax.set_title("Inference scaling")
        ax.legend()
        ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(PLOTS_DIR, "inference_scaling.png"), dpi=120)
        plt.close(fig)

    # --- Markdown ---
    md = "# NetSTA Scaling Analysis\n\n## Data scaling\n\n"
    md += write_markdown_table(
        ["|train|", "Slack MSE", "Slack R²", "CP Accuracy", "CP F1", "CP AUC"],
        [
            [r["n_train"],
             f"{r['slack_mse']:.4f}" if "error" not in r else "ERR",
             f"{r['slack_r2']:.3f}" if "error" not in r else "ERR",
             f"{100*r['cp_accuracy']:.1f}%" if "error" not in r else "ERR",
             f"{r['cp_f1']:.3f}" if "error" not in r else "ERR",
             f"{r['cp_auc_roc']:.3f}" if "error" not in r else "ERR"]
            for r in data_rows
        ],
    )
    md += "\n![data scaling](plots/data_scaling.png)\n\n"
    md += "## Inference scaling: classical STA vs NetSTA GNN\n\n"
    md += write_markdown_table(
        ["Circuit Size (gates)", "Classical STA (ms)", "NetSTA GNN (ms)", "Speedup"],
        [
            [r["n_gates"],
             f"{r['sta_ms']:.2f}" if "error" not in r else "ERR",
             f"{r['gnn_ms']:.2f}" if "error" not in r else "ERR",
             f"{r['speedup']:.1f}x" if "error" not in r else "ERR"]
            for r in timing_rows
        ],
    )
    md += "\n![inference scaling](plots/inference_scaling.png)\n"
    save_text(os.path.join(RESULTS_DIR, "scaling_analysis.md"), md)
    save_json(os.path.join(RESULTS_DIR, "scaling_analysis.json"), {
        "config": vars(args),
        "data_scaling": data_rows,
        "inference_scaling": timing_rows,
    })
    print("\nSaved scaling_analysis.{md,json} and plots/data_scaling.png, plots/inference_scaling.png")


if __name__ == "__main__":
    main()

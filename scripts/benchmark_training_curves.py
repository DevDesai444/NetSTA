#!/usr/bin/env python3
"""
Train one reference NetSTA model and visualize its convergence.

The existing `netsta.train.train(...)` already logs full per-epoch history
into checkpoint_dir/results.json -- we run training, read that history, and
plot loss curves, per-task loss curves, and the LR schedule.
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from _bench_utils import PLOTS_DIR, RESULTS_DIR, ensure_dirs, save_json

from netsta.train import train as run_train


def _matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_curves(history, best_epoch, out_path):
    plt = _matplotlib()
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train"]["loss"] for h in history]
    val_loss = [h["val"]["loss"] for h in history]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(epochs, train_loss, label="train")
    ax.plot(epochs, val_loss, label="val")
    if best_epoch:
        ax.axvline(best_epoch, color="green", linestyle="--", alpha=0.6,
                   label=f"best epoch ({best_epoch})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("NetSTA training curves")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_per_task(history, out_path):
    plt = _matplotlib()
    epochs = [h["epoch"] for h in history]
    # Discover task names from the first entry's per-task loss keys.
    first = history[0]
    tasks = sorted(k.replace("_loss", "") for k in first["train"] if k.endswith("_loss"))
    if not tasks:
        return
    fig, axes = plt.subplots(1, len(tasks), figsize=(5 * len(tasks), 4), sharey=False)
    if len(tasks) == 1:
        axes = [axes]
    for ax, t in zip(axes, tasks):
        key = f"{t}_loss"
        train_y = [h["train"][key] for h in history]
        val_y = [h["val"][key] for h in history]
        ax.plot(epochs, train_y, label="train")
        ax.plot(epochs, val_y, label="val")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(f"Task: {t}")
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_lr(history, out_path):
    plt = _matplotlib()
    epochs = [h["epoch"] for h in history]
    lrs = [h["lr"] for h in history]
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(epochs, lrs)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning rate")
    ax.set_yscale("log")
    ax.set_title("LR schedule (warmup + cosine)")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Train one model and plot convergence")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--checkpoint-dir", default="checkpoints/training_curves")
    parser.add_argument("--num-circuits", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--tasks", default="slack,critical_path")
    args = parser.parse_args()
    tasks = tuple(t.strip() for t in args.tasks.split(",") if t.strip())

    ensure_dirs()
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    print(f"Training reference model ({args.num_circuits} circuits, {args.epochs} epochs)...")

    run_train(
        data_dir=args.data_dir,
        checkpoint_dir=args.checkpoint_dir,
        num_circuits=args.num_circuits,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        tasks=tasks,
        seed=args.seed,
        patience=args.patience,
        warmup_epochs=args.warmup_epochs,
        device=args.device,
        load_cached=True,
        augment=True,
    )

    results_path = os.path.join(args.checkpoint_dir, "results.json")
    with open(results_path) as f:
        results = json.load(f)
    history = results["history"]
    best_epoch = results.get("best_epoch")

    # Copy the training log into results/ for the report compiler.
    save_json(os.path.join(RESULTS_DIR, "training_log.json"), results)

    plot_curves(history, best_epoch, os.path.join(PLOTS_DIR, "training_curves.png"))
    plot_per_task(history, os.path.join(PLOTS_DIR, "per_task_curves.png"))
    plot_lr(history, os.path.join(PLOTS_DIR, "lr_schedule.png"))
    print("\nSaved:")
    print(f"  {RESULTS_DIR}/training_log.json")
    print(f"  {PLOTS_DIR}/training_curves.png")
    print(f"  {PLOTS_DIR}/per_task_curves.png")
    print(f"  {PLOTS_DIR}/lr_schedule.png")


if __name__ == "__main__":
    main()

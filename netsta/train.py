"""
Training pipeline for NetSTA.

Features:
  - Configurable active tasks via --tasks (slack, critical_path, congestion)
  - Linear warmup (5 epochs) -> cosine annealing LR schedule
  - Mixed-precision training on CUDA, plain fp32 fallback on CPU/MPS
  - Early stopping on validation loss (configurable patience)
  - Per-epoch logging of train/val totals + per-task losses + LR
  - Best checkpoint selected by validation loss
"""

import argparse
import contextlib
import json
import os
from dataclasses import asdict

import torch
from torch_geometric.loader import DataLoader

from .config import NetSTAConfig, DEFAULT_TASK_WEIGHTS
from .dataset import (
    AnalogCircuitDataset,
    MixedCircuitDataset,
    NetSTAAugment,
    NetSTADataset,
    TransformSubset,
)
from .model import NetSTAModel
from .stats import DatasetStats, STATS_FILENAME


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


TARGET_KEY = {
    "slack": "y_slack",
    "arrival_time": "y_arrival_time",
    "required_time": "y_required_time",
    "critical_path": "y_critical",
    "congestion": "y_congestion",
    "drc": "y_drc",
    "analog_performance": "y_analog_performance",
}

# Expands `--tasks all` to every task supported by the model registry.
ALL_TASKS = (
    "slack", "arrival_time", "required_time",
    "critical_path", "congestion", "drc", "analog_performance",
)


def _targets_for_batch(data, active_tasks):
    targets = {}
    for t in active_tasks:
        key = TARGET_KEY[t]
        if not hasattr(data, key):
            raise RuntimeError(
                f"Active task '{t}' requires field '{key}' on the dataset, "
                f"but it is missing. Regenerate the dataset (force-regenerate "
                f"or bump schema) or drop the task from --tasks."
            )
        targets[t] = getattr(data, key)
    return targets


def _select_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        # MPS is intentionally skipped from auto-detection: PyG's scatter_reduce
        # ops aren't implemented on MPS as of torch 2.3, so attempting MPS
        # raises NotImplementedError. Pass --device mps explicitly (with
        # PYTORCH_ENABLE_MPS_FALLBACK=1) if you want the slow fallback.
        return torch.device("cpu")
    return torch.device(name)


def _build_scheduler(optimizer, total_epochs: int, warmup_epochs: int):
    """Linear warmup for `warmup_epochs`, then cosine to 0 for the remainder."""
    warmup_epochs = max(0, min(warmup_epochs, total_epochs - 1))
    if warmup_epochs == 0:
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs)
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, total_epochs - warmup_epochs)
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer, [warmup, cosine], milestones=[warmup_epochs]
    )


def _amp_context(use_amp: bool):
    return torch.amp.autocast("cuda") if use_amp else contextlib.nullcontext()


@torch.no_grad()
def _compute_test_metrics(model, loader, device, tasks):
    """Per-task evaluation metrics on the test loader.

    Returns {task: {metric_name: value}}. Late-imports the metric helpers
    from evaluate.py so the train module stays importable on its own.
    """
    import numpy as np
    from .evaluate import classification_metrics, regression_metrics

    model.eval()
    preds = {t: [] for t in tasks}
    targs = {t: [] for t in tasks}
    for data in loader:
        data = data.to(device)
        out = model(
            data.x, data.edge_index, edge_attr=data.edge_attr, batch=data.batch,
        )
        for t in tasks:
            if t not in out or not hasattr(data, TARGET_KEY[t]):
                continue
            preds[t].append(out[t].detach().cpu().numpy())
            targs[t].append(getattr(data, TARGET_KEY[t]).detach().cpu().numpy())

    out_metrics = {}
    for t in tasks:
        if not preds[t]:
            continue
        pr = np.concatenate(preds[t], axis=0)
        tg = np.concatenate(targs[t], axis=0)
        if t in ("slack", "arrival_time", "required_time", "congestion"):
            out_metrics[t] = regression_metrics(pr, tg)
        elif t in ("critical_path", "drc"):
            m, _ = classification_metrics(pr, tg)
            out_metrics[t] = m
        elif t == "analog_performance":
            # 2-output regression: report metrics per column + averaged.
            gbw_m = regression_metrics(pr[:, 0], tg[:, 0])
            par_m = regression_metrics(pr[:, 1], tg[:, 1])
            out_metrics[t] = {
                "gbw_mse": gbw_m["mse"], "gbw_r2": gbw_m["r2"],
                "parasitic_mse": par_m["mse"], "parasitic_r2": par_m["r2"],
            }
    return out_metrics


# ---------------------------------------------------------------------------
# Train / eval steps
# ---------------------------------------------------------------------------


def train_epoch(model, loader, optimizer, device, scaler, use_amp):
    model.train()
    active = list(model.heads.keys())
    total_loss = 0.0
    per_task_totals = {t: 0.0 for t in active}
    num_batches = 0

    for data in loader:
        data = data.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        targets = _targets_for_batch(data, active)

        with _amp_context(use_amp):
            preds = model(
                data.x, data.edge_index, edge_attr=data.edge_attr, batch=data.batch
            )
            loss, per_task = model.compute_loss(preds, targets)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item()
        for t in active:
            per_task_totals[t] += per_task[t].item()
        num_batches += 1

    return {
        "loss": total_loss / max(num_batches, 1),
        **{f"{t}_loss": v / max(num_batches, 1) for t, v in per_task_totals.items()},
    }


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    active = list(model.heads.keys())
    total_loss = 0.0
    per_task_totals = {t: 0.0 for t in active}
    num_batches = 0

    for data in loader:
        data = data.to(device, non_blocking=True)
        targets = _targets_for_batch(data, active)
        preds = model(
            data.x, data.edge_index, edge_attr=data.edge_attr, batch=data.batch
        )
        loss, per_task = model.compute_loss(preds, targets)
        total_loss += loss.item()
        for t in active:
            per_task_totals[t] += per_task[t].item()
        num_batches += 1

    return {
        "loss": total_loss / max(num_batches, 1),
        **{f"{t}_loss": v / max(num_batches, 1) for t, v in per_task_totals.items()},
    }


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------


def _resolve_task_weights(active_tasks, task_weights):
    """Build a weight dict for active_tasks.

    Resolution order per task:
      1. user-supplied value in `task_weights` if present
      2. DEFAULT_TASK_WEIGHTS for that task name
      3. uniform 1/N fallback for the residual

    The previous behavior of returning a flat 1/N dict whenever
    task_weights was None bypassed DEFAULT_TASK_WEIGHTS entirely, so the
    carefully chosen default ratios (slack=0.5, AT=1.0, RT=1.0 — auxiliary
    supervision balanced against the slack constraint to prevent collapse
    of the compositional readout) never reached the model.
    """
    user = task_weights or {}
    fallback = 1.0 / len(active_tasks)
    out = {}
    for t in active_tasks:
        if t in user:
            out[t] = user[t]
        elif t in DEFAULT_TASK_WEIGHTS:
            out[t] = DEFAULT_TASK_WEIGHTS[t]
        else:
            out[t] = fallback
    return out


def train(
    data_dir: str = "data",
    checkpoint_dir: str = "checkpoints",
    num_circuits: int = 500,
    epochs: int = 100,
    batch_size: int = 16,
    lr: float = 1e-3,
    hidden_channels: int = 64,
    num_layers: int = 4,
    heads: int = 4,
    tasks=("slack", "arrival_time", "required_time"),
    task_weights=None,
    seed: int = 42,
    patience: int = 40,
    warmup_epochs: int = 5,
    device: str = "auto",
    load_cached: bool = True,
    augment: bool = True,
    circuit_type: str = "digital",
):
    torch.manual_seed(seed)
    dev = _select_device(device)
    print(f"Using device: {dev}")

    if circuit_type == "all":
        tasks = ALL_TASKS
    tasks = tuple(tasks)

    print(f"Preparing dataset ({num_circuits} circuits, type={circuit_type}, "
          f"load_cached={load_cached})...")
    if circuit_type == "digital":
        dataset = NetSTADataset(
            root=data_dir, num_circuits=num_circuits, seed=seed,
            force_regenerate=not load_cached,
        )
    elif circuit_type == "analog":
        dataset = AnalogCircuitDataset(
            root=data_dir or "data_analog",
            num_circuits=num_circuits, seed=seed,
            force_regenerate=not load_cached,
        )
    elif circuit_type == "mixed":
        dataset = MixedCircuitDataset(
            root=data_dir or "data_mixed",
            num_circuits=num_circuits, seed=seed,
            force_regenerate=not load_cached,
        )
    else:
        raise ValueError(
            f"Unknown circuit_type '{circuit_type}'. "
            "Use one of: digital, analog, mixed."
        )

    # Split indices reproducibly
    n = len(dataset)
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)
    indices = torch.randperm(n, generator=torch.Generator().manual_seed(seed)).tolist()
    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]

    augmenter = NetSTAAugment() if augment else None
    train_ds = TransformSubset(dataset, train_idx, transform=augmenter)
    val_ds = TransformSubset(dataset, val_idx)
    test_ds = TransformSubset(dataset, test_idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    # Validate dataset has fields for all active tasks before model build.
    sample = dataset[0]
    in_channels = sample.x.size(1)
    edge_dim = sample.edge_attr.size(1) if sample.edge_attr.dim() > 1 else 1
    _targets_for_batch(sample, tasks)  # raises with a clear error if missing

    print(f"Input features: {in_channels}, Edge features: {edge_dim}")
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    print(f"Active tasks: {list(tasks)}")

    # Compute train-split stats for every active timing regression task
    # (slack, arrival_time, required_time) so each head can standardize its
    # own loss while keeping forward outputs in ns.
    has_at = any(hasattr(dataset[i], "y_arrival_time") for i in train_idx[:1])
    has_rt = any(hasattr(dataset[i], "y_required_time") for i in train_idx[:1])
    at_iter = (dataset[i].y_arrival_time for i in train_idx) if has_at else None
    rt_iter = (dataset[i].y_required_time for i in train_idx) if has_rt else None
    stats = DatasetStats.from_target_tensors(
        (dataset[i].y_slack for i in train_idx),
        arrival_tensors=at_iter,
        required_tensors=rt_iter,
    )
    stats_path = os.path.join(checkpoint_dir, STATS_FILENAME)
    os.makedirs(checkpoint_dir, exist_ok=True)
    stats.save(stats_path)
    print(
        f"Train-split stats (ns):  "
        f"slack mean={stats.slack_mean:.4f}/std={stats.slack_std:.4f}  "
        f"AT mean={stats.arrival_time_mean:.4f}/std={stats.arrival_time_std:.4f}  "
        f"RT mean={stats.required_time_mean:.4f}/std={stats.required_time_std:.4f}  "
        f"-> {stats_path}"
    )

    use_symmetry = circuit_type in ("analog", "mixed")
    config = NetSTAConfig(
        node_feature_dim=in_channels,
        edge_feature_dim=edge_dim,
        hidden_dim=hidden_channels,
        num_layers=num_layers,
        num_heads=heads,
        dropout=0.1,
        learning_rate=lr,
        task_weights=_resolve_task_weights(tasks, task_weights),
        active_tasks=tuple(tasks),
        use_symmetry_attention=use_symmetry,
        slack_mean=stats.slack_mean,
        slack_std=stats.slack_std,
        arrival_time_mean=stats.arrival_time_mean,
        arrival_time_std=stats.arrival_time_std,
        required_time_mean=stats.required_time_mean,
        required_time_std=stats.required_time_std,
    )
    model = NetSTAModel(config).to(dev)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")

    optimizer = torch.optim.AdamW(
        model.get_param_groups(), weight_decay=config.weight_decay
    )
    scheduler = _build_scheduler(optimizer, epochs, warmup_epochs)

    use_amp = dev.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    if use_amp:
        print("Mixed-precision training enabled (torch.amp on CUDA).")
    else:
        print(f"Mixed precision unavailable on {dev.type}; running in fp32.")

    os.makedirs(checkpoint_dir, exist_ok=True)
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_no_improve = 0
    history = []

    for epoch in range(1, epochs + 1):
        train_metrics = train_epoch(model, train_loader, optimizer, dev, scaler, use_amp)
        val_metrics = validate(model, val_loader, dev)
        scheduler.step()
        current_lr = optimizer.param_groups[-1]["lr"]

        is_best = val_metrics["loss"] < best_val_loss
        marker = " *best*" if is_best else ""
        per_task_str = " ".join(
            f"{t}={train_metrics[f'{t}_loss']:.4f}/{val_metrics[f'{t}_loss']:.4f}"
            for t in tasks
        )
        print(
            f"Epoch {epoch:3d}/{epochs} | LR {current_lr:.2e} | "
            f"Train {train_metrics['loss']:.4f} | Val {val_metrics['loss']:.4f} | "
            f"[{per_task_str}]{marker}"
        )

        history.append(
            {
                "epoch": epoch,
                "lr": current_lr,
                "train": train_metrics,
                "val": val_metrics,
            }
        )

        if is_best:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": asdict(config),
                    "best_val_loss": best_val_loss,
                    "epoch": epoch,
                },
                os.path.join(checkpoint_dir, "best_model.pt"),
            )
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch} (no improvement in {patience})")
                break

    # Final test eval on the best checkpoint.
    checkpoint = torch.load(
        os.path.join(checkpoint_dir, "best_model.pt"), weights_only=False
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = validate(model, test_loader, dev)

    print("\n--- Test (best checkpoint) ---")
    print("Per-task losses:")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.6f}")

    # Per-task metrics (R²/AUC/etc.) on top of the losses.
    rich_metrics = _compute_test_metrics(model, test_loader, dev, tasks)
    if rich_metrics:
        print("Per-task metrics:")
        for task, m in rich_metrics.items():
            line = "  " + task + ":  " + "  ".join(
                f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                for k, v in m.items()
            )
            print(line)
        test_metrics["per_task_metrics"] = rich_metrics

    results = {
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "test_metrics": test_metrics,
        "num_params": num_params,
        "config": asdict(config),
        "history": history,
        "cli": {
            "num_circuits": num_circuits,
            "epochs": epochs,
            "batch_size": batch_size,
            "patience": patience,
            "warmup_epochs": warmup_epochs,
            "device": str(dev),
        },
    }
    with open(os.path.join(checkpoint_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    return test_metrics


def _parse_tasks(arg: str):
    tasks = [t.strip() for t in arg.split(",") if t.strip()]
    if tasks == ["all"]:
        return tuple(ALL_TASKS)
    valid = set(TARGET_KEY.keys())
    bad = [t for t in tasks if t not in valid]
    if bad:
        raise argparse.ArgumentTypeError(
            f"Unknown task(s): {bad}. Valid: {sorted(valid)} or 'all'"
        )
    if not tasks:
        raise argparse.ArgumentTypeError("--tasks must contain at least one task")
    return tuple(tasks)


def main():
    parser = argparse.ArgumentParser(description="Train NetSTA")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--num-circuits", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument(
        "--tasks",
        type=_parse_tasks,
        default=("slack", "critical_path"),
        help="Comma-separated active task heads (slack,critical_path,congestion)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument(
        "--device", default="auto", choices=["auto", "cuda", "mps", "cpu"]
    )
    parser.add_argument(
        "--load-cached",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse cached dataset on disk (default). Use --no-load-cached to regenerate.",
    )
    parser.add_argument(
        "--augment",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply train-time augmentation (default on).",
    )
    parser.add_argument(
        "--circuit-type",
        default="digital",
        choices=["digital", "analog", "mixed"],
        help="Which circuit family to train on. `mixed` concatenates both.",
    )
    args = parser.parse_args()

    train(
        data_dir=args.data_dir,
        checkpoint_dir=args.checkpoint_dir,
        num_circuits=args.num_circuits,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden_channels=args.hidden_channels,
        num_layers=args.num_layers,
        heads=args.heads,
        tasks=args.tasks,
        seed=args.seed,
        patience=args.patience,
        warmup_epochs=args.warmup_epochs,
        device=args.device,
        load_cached=args.load_cached,
        augment=args.augment,
        circuit_type=args.circuit_type,
    )


if __name__ == "__main__":
    main()

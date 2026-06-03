"""
Shared helpers for the NetSTA benchmark scripts.

Centralizes:
  - results/plots directory setup
  - reproducible train/val/test split
  - tiny generic train loop usable by both NetSTA and the GNN baselines
  - inference-time measurement
  - markdown table emitter
"""

from __future__ import annotations

import json
import os
import time
from contextlib import nullcontext
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
from torch_geometric.loader import DataLoader

from netsta.config import NetSTAConfig
from netsta.dataset import NetSTADataset, TransformSubset
from netsta.model import NetSTAModel
from netsta.train import (
    TARGET_KEY,
    _build_scheduler,
    _resolve_task_weights,
    _select_device,
    _targets_for_batch,
    train_epoch,
    validate,
)


RESULTS_DIR = "results"
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")


def ensure_dirs():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)


def split_indices(n: int, seed: int) -> Tuple[List[int], List[int], List[int]]:
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)
    indices = torch.randperm(n, generator=torch.Generator().manual_seed(seed)).tolist()
    return indices[:n_train], indices[n_train : n_train + n_val], indices[n_train + n_val :]


def build_loaders(dataset, train_idx, val_idx, test_idx, batch_size, train_transform=None):
    train_ds = TransformSubset(dataset, train_idx, transform=train_transform)
    val_ds = TransformSubset(dataset, val_idx)
    test_ds = TransformSubset(dataset, test_idx)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size),
        DataLoader(test_ds, batch_size=batch_size),
        train_ds,
        val_ds,
        test_ds,
    )


def fit_torch_model(
    model,
    train_loader,
    val_loader,
    device,
    epochs: int,
    warmup_epochs: int = 5,
    patience: int = 25,
    weight_decay: float = 1e-4,
    log_prefix: str = "",
) -> Tuple[Dict, float, int, List[Dict]]:
    """Generic torch training loop with warmup + cosine + ES on val loss.

    Returns (best_state_dict, best_val_loss, best_epoch, history).
    """
    optimizer = torch.optim.AdamW(model.get_param_groups(), weight_decay=weight_decay)
    scheduler = _build_scheduler(optimizer, epochs, warmup_epochs)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_no_improve = 0
    history: List[Dict] = []

    for epoch in range(1, epochs + 1):
        tr = train_epoch(model, train_loader, optimizer, device, scaler, use_amp)
        va = validate(model, val_loader, device)
        scheduler.step()
        current_lr = optimizer.param_groups[-1]["lr"]
        history.append({"epoch": epoch, "lr": current_lr, "train": tr, "val": va})

        if va["loss"] < best_val_loss:
            best_val_loss = va["loss"]
            best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"{log_prefix}early stop @ epoch {epoch} (no val improvement in {patience})")
                break

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"{log_prefix}epoch {epoch:3d}/{epochs} | lr {current_lr:.2e} | "
                f"train {tr['loss']:.4f} | val {va['loss']:.4f}"
                + ("  *best*" if best_epoch == epoch else "")
            )

    return best_state, best_val_loss, best_epoch, history


@torch.no_grad()
def measure_inference_time(
    model, loader, device, warmup_batches: int = 2
) -> Tuple[float, float]:
    """Return (mean_ms_per_circuit, total_seconds_on_test_set)."""
    model.eval()
    # Warm up so JIT/cudnn caches don't pollute the first call.
    seen_warmup = 0
    for batch in loader:
        if seen_warmup >= warmup_batches:
            break
        batch = batch.to(device)
        model(batch.x, batch.edge_index, edge_attr=batch.edge_attr, batch=batch.batch)
        seen_warmup += 1
    total = 0.0
    circuits = 0
    if device.type == "cuda":
        torch.cuda.synchronize()
    for batch in loader:
        batch = batch.to(device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        model(batch.x, batch.edge_index, edge_attr=batch.edge_attr, batch=batch.batch)
        if device.type == "cuda":
            torch.cuda.synchronize()
        total += time.perf_counter() - t0
        circuits += int(batch.batch.max().item()) + 1 if hasattr(batch, "batch") and batch.batch is not None else 1
    return (total / max(circuits, 1)) * 1000.0, total


@torch.no_grad()
def collect_test_predictions(model, loader, device, active_tasks: Sequence[str]):
    """Concatenate per-task predictions and targets over the loader. CPU numpy."""
    import numpy as np

    model.eval()
    preds = {t: [] for t in active_tasks}
    targets = {t: [] for t in active_tasks}
    for batch in loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index, edge_attr=batch.edge_attr, batch=batch.batch)
        for t in active_tasks:
            preds[t].append(out[t].detach().cpu().numpy())
            targets[t].append(getattr(batch, TARGET_KEY[t]).detach().cpu().numpy())
    return (
        {t: np.concatenate(v) for t, v in preds.items()},
        {t: np.concatenate(v) for t, v in targets.items()},
    )


def write_markdown_table(
    headers: Sequence[str], rows: Sequence[Sequence[str]], align: str = "|"
) -> str:
    sep = ["---"] * len(headers)
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(sep) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(out) + "\n"


def save_json(path: str, data) -> None:
    ensure_dirs()
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def save_text(path: str, text: str) -> None:
    ensure_dirs()
    with open(path, "w") as f:
        f.write(text)


def make_netsta(
    node_feature_dim: int,
    edge_feature_dim: int,
    *,
    hidden_dim: int = 64,
    num_layers: int = 4,
    num_heads: int = 4,
    dropout: float = 0.1,
    lr: float = 1e-3,
    tasks: Sequence[str] = ("slack", "arrival_time", "required_time"),
    task_weights=None,
    use_residual: bool = True,
    use_attention: bool = True,
    slack_mean: float = 0.0,
    slack_std: float = 1.0,
    arrival_time_mean: float = 0.0,
    arrival_time_std: float = 1.0,
    required_time_mean: float = 0.0,
    required_time_std: float = 1.0,
) -> NetSTAModel:
    config = NetSTAConfig(
        node_feature_dim=node_feature_dim,
        edge_feature_dim=edge_feature_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout,
        learning_rate=lr,
        task_weights=_resolve_task_weights(tasks, task_weights),
        active_tasks=tuple(tasks),
        use_residual=use_residual,
        use_attention=use_attention,
        slack_mean=slack_mean,
        slack_std=slack_std,
        arrival_time_mean=arrival_time_mean,
        arrival_time_std=arrival_time_std,
        required_time_mean=required_time_mean,
        required_time_std=required_time_std,
    )
    return NetSTAModel(config)


def compute_slack_stats(dataset, train_idx):
    """Mean/std of y_slack across the training subset (ns).

    Lives here so every benchmark script reuses the same stat — comparing GNN
    vs MLP vs GCN with different slack standardizations would be apples-to-
    oranges. Returns (slack_mean, slack_std). std is floored at 1e-3.
    """
    from netsta.stats import DatasetStats
    stats = DatasetStats.from_slack_tensors(
        dataset[i].y_slack for i in train_idx
    )
    return stats.slack_mean, stats.slack_std


def compute_target_stats(dataset, train_idx):
    """Mean/std of y_slack, y_arrival_time, y_required_time across the train
    subset (ns). Used by every benchmark script so comparisons share the same
    standardization. Returns a DatasetStats dataclass.
    """
    from netsta.stats import DatasetStats
    sample = dataset[train_idx[0]] if len(train_idx) else None
    has_at = sample is not None and hasattr(sample, "y_arrival_time")
    has_rt = sample is not None and hasattr(sample, "y_required_time")
    return DatasetStats.from_target_tensors(
        (dataset[i].y_slack for i in train_idx),
        arrival_tensors=(dataset[i].y_arrival_time for i in train_idx) if has_at else None,
        required_tensors=(dataset[i].y_required_time for i in train_idx) if has_rt else None,
    )

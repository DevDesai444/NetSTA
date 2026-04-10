"""
Evaluation pipeline for NetSTA.

Loads a trained checkpoint, runs inference on the held-out test split, and:
  - Computes per-task metrics (regression: MSE/MAE/R^2/Pearson; classification:
    accuracy/precision/recall/F1/AUC-ROC with optimal threshold search)
  - Saves plots (scatter for regression, ROC + confusion matrix for
    classification)
  - Dumps everything to a JSON file
"""

import argparse
import json
import os
from typing import Dict, Tuple

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from .config import NetSTAConfig
from .dataset import TimingNetDataset, TransformSubset
from .model import NetSTAModel
from .train import TARGET_KEY, _select_device


# ---------------------------------------------------------------------------
# Metric primitives (numpy, no sklearn)
# ---------------------------------------------------------------------------


def _safe_div(num: float, denom: float, default=float("nan")) -> float:
    return float(num / denom) if denom > 0 else float(default)


def regression_metrics(pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    pred = pred.astype(np.float64).ravel()
    target = target.astype(np.float64).ravel()
    diff = pred - target
    mse = float((diff ** 2).mean())
    mae = float(np.abs(diff).mean())
    ss_res = float((diff ** 2).sum())
    ss_tot = float(((target - target.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    if pred.std() > 0 and target.std() > 0:
        pearson = float(np.corrcoef(pred, target)[0, 1])
    else:
        pearson = float("nan")
    return {"mse": mse, "mae": mae, "r2": float(r2), "pearson": pearson}


def roc_curve_auc(
    y_true: np.ndarray, y_score: np.ndarray
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Return (auc, fpr, tpr, thresholds). NaN AUC when one class is absent."""
    y_true = y_true.astype(np.float64).ravel()
    y_score = y_score.astype(np.float64).ravel()
    P = float(y_true.sum())
    N = float(len(y_true) - P)
    if P == 0 or N == 0:
        return float("nan"), np.array([0.0, 1.0]), np.array([0.0, 1.0]), np.array([])

    order = np.argsort(-y_score)
    y_sorted = y_true[order]
    s_sorted = y_score[order]
    tps = np.cumsum(y_sorted)
    fps = np.cumsum(1.0 - y_sorted)
    tpr = tps / P
    fpr = fps / N
    tpr = np.concatenate([[0.0], tpr])
    fpr = np.concatenate([[0.0], fpr])
    thresholds = np.concatenate([[np.inf], s_sorted])
    auc = float(np.trapezoid(tpr, fpr)) if hasattr(np, "trapezoid") else float(
        np.trapz(tpr, fpr)
    )
    return auc, fpr, tpr, thresholds


def classification_metrics(
    logits: np.ndarray, target: np.ndarray
) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
    score = 1.0 / (1.0 + np.exp(-logits.astype(np.float64).ravel()))
    target = target.astype(np.float64).ravel()
    auc, fpr, tpr, thresholds = roc_curve_auc(target, score)

    # Youden's J = TPR - FPR. thresholds[0] is +inf (no positives predicted);
    # skip it so the chosen threshold is real.
    if thresholds.size > 1:
        j = tpr[1:] - fpr[1:]
        best_idx = int(np.argmax(j))
        best_threshold = float(thresholds[best_idx + 1])
    else:
        best_threshold = 0.5

    pred = (score >= best_threshold).astype(np.float64)
    tp = float(((pred == 1) & (target == 1)).sum())
    fp = float(((pred == 1) & (target == 0)).sum())
    fn = float(((pred == 0) & (target == 1)).sum())
    tn = float(((pred == 0) & (target == 0)).sum())

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall) if precision and recall else 0.0
    accuracy = (tp + tn) / max(len(target), 1)

    metrics = {
        "accuracy": float(accuracy),
        "precision": precision,
        "recall": recall,
        "f1": float(f1),
        "auc_roc": float(auc),
        "best_threshold": best_threshold,
    }
    extras = {
        "fpr": fpr,
        "tpr": tpr,
        "confusion_matrix": np.array([[tn, fp], [fn, tp]]),
        "pred": pred,
        "score": score,
    }
    return metrics, extras


# ---------------------------------------------------------------------------
# Inference collector
# ---------------------------------------------------------------------------


@torch.no_grad()
def collect_predictions(model, loader, device, active_tasks):
    """Return ({task: pred_array}, {task: target_array})."""
    model.eval()
    preds = {t: [] for t in active_tasks}
    targets = {t: [] for t in active_tasks}
    for data in loader:
        data = data.to(device)
        out = model(
            data.x, data.edge_index, edge_attr=data.edge_attr, batch=data.batch
        )
        for t in active_tasks:
            preds[t].append(out[t].detach().cpu().numpy())
            targets[t].append(getattr(data, TARGET_KEY[t]).detach().cpu().numpy())
    preds = {t: np.concatenate(v) for t, v in preds.items()}
    targets = {t: np.concatenate(v) for t, v in targets.items()}
    return preds, targets


# ---------------------------------------------------------------------------
# Plotting (matplotlib, headless-safe)
# ---------------------------------------------------------------------------


def _scatter_plot(pred, target, title, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(target, pred, alpha=0.3, s=8)
    lo = float(min(target.min(), pred.min()))
    hi = float(max(target.max(), pred.max()))
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _roc_plot(fpr, tpr, auc, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Critical Path ROC")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _confusion_plot(cm, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["pred 0", "pred 1"])
    ax.set_yticklabels(["true 0", "true 1"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{int(cm[i, j])}", ha="center", va="center", color="black")
    ax.set_title("Confusion matrix")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def evaluate(
    checkpoint_path: str,
    data_dir: str,
    output_dir: str,
    num_circuits: int,
    seed: int,
    batch_size: int,
    device: str,
):
    os.makedirs(output_dir, exist_ok=True)
    dev = _select_device(device)
    print(f"Using device: {dev}")

    checkpoint = torch.load(checkpoint_path, map_location=dev, weights_only=False)
    cfg_data = dict(checkpoint["config"])
    if isinstance(cfg_data.get("active_tasks"), list):
        cfg_data["active_tasks"] = tuple(cfg_data["active_tasks"])
    config = NetSTAConfig(**cfg_data)
    model = NetSTAModel(config).to(dev)
    model.load_state_dict(checkpoint["model_state_dict"])

    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch')} "
          f"(val_loss={checkpoint.get('best_val_loss')})")
    print(f"Active tasks: {list(config.active_tasks)}")

    # Recreate the same train/val/test split used in train.py
    dataset = TimingNetDataset(root=data_dir, num_circuits=num_circuits, seed=seed)
    n = len(dataset)
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)
    indices = torch.randperm(n, generator=torch.Generator().manual_seed(seed)).tolist()
    test_idx = indices[n_train + n_val :]
    test_ds = TransformSubset(dataset, test_idx)
    loader = DataLoader(test_ds, batch_size=batch_size)
    print(f"Test set size: {len(test_ds)} circuits")

    preds, targets = collect_predictions(model, loader, dev, config.active_tasks)

    all_metrics: Dict[str, Dict[str, float]] = {}

    for task in config.active_tasks:
        if task in ("slack", "congestion"):
            m = regression_metrics(preds[task], targets[task])
            all_metrics[task] = m
            print(f"\n[{task}]  MSE={m['mse']:.6f}  MAE={m['mae']:.6f}  "
                  f"R^2={m['r2']:.4f}  Pearson={m['pearson']:.4f}")
            _scatter_plot(
                preds[task], targets[task],
                f"{task} predictions vs actual",
                os.path.join(output_dir, f"{task}_scatter.png"),
            )
        elif task in ("critical_path", "drc"):
            m, extras = classification_metrics(preds[task], targets[task])
            all_metrics[task] = m
            print(f"\n[{task}]  Acc={m['accuracy']:.4f}  Prec={m['precision']:.4f}  "
                  f"Rec={m['recall']:.4f}  F1={m['f1']:.4f}  AUC={m['auc_roc']:.4f}  "
                  f"(thr={m['best_threshold']:.3f})")
            _roc_plot(extras["fpr"], extras["tpr"], m["auc_roc"],
                      os.path.join(output_dir, f"{task}_roc.png"))
            _confusion_plot(extras["confusion_matrix"],
                            os.path.join(output_dir, f"{task}_confusion.png"))

    out = {
        "checkpoint": os.path.abspath(checkpoint_path),
        "epoch": checkpoint.get("epoch"),
        "best_val_loss": checkpoint.get("best_val_loss"),
        "active_tasks": list(config.active_tasks),
        "metrics": all_metrics,
        "test_set_size": len(test_ds),
    }
    out_path = os.path.join(output_dir, "metrics.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved metrics to {out_path}")
    return out


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained NetSTA model")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="evaluation")
    parser.add_argument("--num-circuits", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--device", default="auto", choices=["auto", "cuda", "mps", "cpu"]
    )
    args = parser.parse_args()
    evaluate(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        num_circuits=args.num_circuits,
        seed=args.seed,
        batch_size=args.batch_size,
        device=args.device,
    )


if __name__ == "__main__":
    main()

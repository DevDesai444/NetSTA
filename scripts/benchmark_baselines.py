#!/usr/bin/env python3
"""
Baseline comparison benchmark for NetSTA.

Trains MLP, GCN, GraphSAGE, RandomForest, LinearRegression baselines, plus
NetSTA itself, on the same dataset and split. Reports a single comparison
table covering slack metrics, critical-path metrics, training walltime,
inference-time-per-circuit, and parameter count.
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
    compute_slack_stats,
    compute_target_stats,
    ensure_dirs,
    fit_torch_model,
    make_netsta,
    measure_inference_time,
    save_json,
    save_text,
    split_indices,
    write_markdown_table,
)

from netsta.baselines import GCNBaseline, GraphSAGEBaseline, MLPBaseline
from netsta.dataset import NetSTADataset
from netsta.evaluate import classification_metrics, regression_metrics
from netsta.train import _select_device


def fmt(v, digits=4):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "--"
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


def fmt_pct(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "--"
    return f"{100 * v:.1f}%"


def hand_crafted_node_features(dataset, indices):
    """Per-node hand-engineered feature matrix for the sklearn baselines.

    Composes the model's input vector (identity/device one-hot only) with
    hand-engineered per-node scalars the GNN must derive via message passing:
    fanin, fanout, and graph-level summaries (node count, max fanout). The
    sklearn baselines exist to answer "can a non-GNN match the GNN given
    classical hand engineering?" — so we feed them exactly that.

    Crucially, we do NOT include STA outputs (logical_depth, load_cap, slack-
    derived quantities). That was the leakage path under the old schema.
    Returns (X, y_slack, y_critical).
    """
    X_chunks: list = []
    y_slack: list = []
    y_crit: list = []
    for idx in indices:
        data = dataset[idx]
        n = data.x.size(0)
        x_np = data.x.cpu().numpy()

        # Per-node topology aggregates derived from the edge_index — the GNN
        # would compute these via message passing; sklearn gets them precomputed.
        edge_index = data.edge_index.cpu().numpy()
        src, dst = edge_index[0], edge_index[1]
        fanout = np.bincount(src, minlength=n).astype(np.float32)
        fanin = np.bincount(dst, minlength=n).astype(np.float32)
        per_node = np.stack([fanin, fanout], axis=1)

        graph_max_fanout = float(fanout.max()) if fanout.size else 0.0
        graph_avg_fanin = float(fanin.mean()) if fanin.size else 0.0
        graph_n = float(n)
        graph_summary = np.tile(
            [graph_max_fanout, graph_avg_fanin, graph_n], (n, 1)
        ).astype(np.float32)

        X_chunks.append(np.concatenate([x_np, per_node, graph_summary], axis=1))
        y_slack.append(data.y_slack.cpu().numpy())
        y_crit.append(data.y_critical.cpu().numpy())
    return (
        np.concatenate(X_chunks, axis=0),
        np.concatenate(y_slack, axis=0),
        np.concatenate(y_crit, axis=0),
    )


def evaluate_per_task(preds, targets):
    out = {}
    if "slack" in preds:
        out["slack"] = regression_metrics(preds["slack"], targets["slack"])
    if "critical_path" in preds:
        m, _ = classification_metrics(preds["critical_path"], targets["critical_path"])
        out["critical_path"] = m
    return out


def run_torch_model(label, build_fn, train_loader, val_loader, test_loader,
                    device, epochs, patience, warmup_epochs):
    model = build_fn().to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[{label}] params={n_params:,}")

    t0 = time.perf_counter()
    best_state, best_val_loss, best_epoch, _ = fit_torch_model(
        model, train_loader, val_loader, device,
        epochs=epochs, patience=patience, warmup_epochs=warmup_epochs,
        log_prefix=f"[{label}] ",
    )
    train_time = time.perf_counter() - t0
    model.load_state_dict(best_state)

    active = list(model.heads.keys())
    preds, targets = collect_test_predictions(model, test_loader, device, active)
    metrics = evaluate_per_task(preds, targets)
    infer_ms, _ = measure_inference_time(model, test_loader, device)
    return {
        "params": n_params,
        "train_time_s": train_time,
        "infer_ms_per_circuit": infer_ms,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "metrics": metrics,
    }


def run_sklearn_baseline(label, model_factory_reg, model_factory_clf,
                         X_train, y_slack_train, y_crit_train,
                         X_test, y_slack_test, y_crit_test):
    """Train a sklearn regressor (slack) + classifier (crit). Return uniform stats."""
    t0 = time.perf_counter()
    reg = model_factory_reg()
    reg.fit(X_train, y_slack_train)
    clf = model_factory_clf()
    clf.fit(X_train, y_crit_train)
    train_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    slack_pred = reg.predict(X_test)
    if hasattr(clf, "predict_proba"):
        crit_score = clf.predict_proba(X_test)[:, 1]
    elif hasattr(clf, "decision_function"):
        crit_score = clf.decision_function(X_test)
    else:
        crit_score = clf.predict(X_test).astype(float)
    infer_time = time.perf_counter() - t1
    n_circuits_estimate = max(1, int(round(len(y_slack_test) / 60.0)))  # ~60 nodes/circuit
    infer_ms_per = (infer_time / n_circuits_estimate) * 1000.0

    # crit_score for our classification_metrics expects logits; convert prob->logit safely.
    eps = 1e-6
    p = np.clip(crit_score, eps, 1 - eps)
    logits = np.log(p / (1 - p))
    metrics = {
        "slack": regression_metrics(slack_pred, y_slack_test),
        "critical_path": classification_metrics(logits, y_crit_test)[0],
    }
    return {
        "params": None,
        "train_time_s": train_time,
        "infer_ms_per_circuit": infer_ms_per,
        "metrics": metrics,
    }


def main():
    parser = argparse.ArgumentParser(description="Run baseline comparison benchmark")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--num-circuits", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument(
        "--skip",
        default="",
        help="Comma-separated baselines to skip (mlp,gcn,graphsage,rf,linear,netsta)",
    )
    args = parser.parse_args()
    skip = set(s.strip() for s in args.skip.split(",") if s.strip())

    ensure_dirs()
    torch.manual_seed(args.seed)
    device = _select_device(args.device)
    print(f"Device: {device}")

    dataset = NetSTADataset(root=args.data_dir, num_circuits=args.num_circuits, seed=args.seed)
    train_idx, val_idx, test_idx = split_indices(len(dataset), args.seed)
    train_loader, val_loader, test_loader, *_ = build_loaders(
        dataset, train_idx, val_idx, test_idx, args.batch_size
    )
    sample = dataset[0]
    node_feature_dim = sample.x.size(1)
    edge_feature_dim = sample.edge_attr.size(1) if sample.edge_attr.dim() > 1 else 1
    print(f"Node features: {node_feature_dim}, Edge features: {edge_feature_dim}")
    print(f"Train/Val/Test: {len(train_idx)}/{len(val_idx)}/{len(test_idx)}")

    # Compute target stats once from the train split; pass to every torch
    # model so the loss is computed on the same z-scale. Sklearn baselines
    # regress against raw ns and don't need standardization. Baselines only
    # train against slack — the AT/RT supervision is part of NetSTA's design.
    stats = compute_target_stats(dataset, train_idx)
    print(
        f"Train-split stats (ns):  "
        f"slack mean={stats.slack_mean:.4f}/std={stats.slack_std:.4f}  "
        f"AT mean={stats.arrival_time_mean:.4f}/std={stats.arrival_time_std:.4f}  "
        f"RT mean={stats.required_time_mean:.4f}/std={stats.required_time_std:.4f}"
    )

    results = {}

    # --- Torch baselines ---
    torch_specs = [
        ("MLP", "mlp",
         lambda: MLPBaseline(node_feature_dim, hidden=128,
                             slack_mean=stats.slack_mean, slack_std=stats.slack_std)),
        ("GCN", "gcn",
         lambda: GCNBaseline(node_feature_dim, hidden=256, num_layers=4,
                             slack_mean=stats.slack_mean, slack_std=stats.slack_std)),
        ("GraphSAGE", "graphsage",
         lambda: GraphSAGEBaseline(node_feature_dim, hidden=256, num_layers=4,
                                   slack_mean=stats.slack_mean, slack_std=stats.slack_std)),
        ("NetSTA", "netsta",
         lambda: make_netsta(node_feature_dim, edge_feature_dim,
                             slack_mean=stats.slack_mean, slack_std=stats.slack_std,
                             arrival_time_mean=stats.arrival_time_mean,
                             arrival_time_std=stats.arrival_time_std,
                             required_time_mean=stats.required_time_mean,
                             required_time_std=stats.required_time_std)),
    ]
    for label, key, factory in torch_specs:
        if key in skip:
            print(f"[{label}] skipped")
            continue
        try:
            print(f"\n=== {label} ===")
            results[label] = run_torch_model(
                label, factory, train_loader, val_loader, test_loader,
                device, args.epochs, args.patience, args.warmup_epochs,
            )
        except Exception as exc:
            print(f"[{label}] FAILED: {exc!r}")
            results[label] = {"error": repr(exc)}

    # --- Sklearn baselines ---
    if "rf" not in skip or "linear" not in skip:
        print("\n=== Building hand-crafted feature matrices ===")
        X_train, y_slack_train, y_crit_train = hand_crafted_node_features(dataset, train_idx)
        X_test, y_slack_test, y_crit_test = hand_crafted_node_features(dataset, test_idx)
        print(f"  shapes: X_train={X_train.shape}, X_test={X_test.shape}")

        if "rf" not in skip:
            try:
                from sklearn.ensemble import (
                    RandomForestClassifier,
                    RandomForestRegressor,
                )
                print("\n=== Random Forest ===")
                results["Random Forest"] = run_sklearn_baseline(
                    "RF",
                    lambda: RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=args.seed),
                    lambda: RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=args.seed, class_weight="balanced"),
                    X_train, y_slack_train, y_crit_train,
                    X_test, y_slack_test, y_crit_test,
                )
            except Exception as exc:
                print(f"[RF] FAILED: {exc!r}")
                results["Random Forest"] = {"error": repr(exc)}

        if "linear" not in skip:
            try:
                from sklearn.linear_model import LinearRegression, LogisticRegression
                print("\n=== Linear Regression ===")
                results["Linear Regression"] = run_sklearn_baseline(
                    "LR",
                    lambda: LinearRegression(),
                    lambda: LogisticRegression(max_iter=1000, class_weight="balanced"),
                    X_train, y_slack_train, y_crit_train,
                    X_test, y_slack_test, y_crit_test,
                )
            except Exception as exc:
                print(f"[LR] FAILED: {exc!r}")
                results["Linear Regression"] = {"error": repr(exc)}

    # --- Build comparison table ---
    order = ["Linear Regression", "Random Forest", "MLP", "GCN", "GraphSAGE", "NetSTA"]
    headers = [
        "Model", "Params", "Slack MSE ↓", "Slack R² ↑",
        "CP Accuracy ↑", "CP F1 ↑", "CP AUC ↑",
        "Train Time", "Infer Time/circuit",
    ]
    rows = []
    for name in order:
        r = results.get(name)
        if not r or "error" in r:
            rows.append([name, "ERR", "--", "--", "--", "--", "--", "--", "--"])
            continue
        m = r["metrics"]
        slack = m.get("slack", {})
        crit = m.get("critical_path", {})
        params = "--" if r.get("params") is None else f"{r['params'] / 1000:.0f}K"
        marker_open = "**" if name == "NetSTA" else ""
        marker_close = "**" if name == "NetSTA" else ""
        label = f"{marker_open}{name}{marker_close}"
        rows.append([
            label,
            params,
            fmt(slack.get("mse")),
            fmt(slack.get("r2"), 3),
            fmt_pct(crit.get("accuracy")),
            fmt(crit.get("f1"), 3),
            fmt(crit.get("auc_roc"), 3),
            f"{r['train_time_s']:.1f}s",
            f"{r['infer_ms_per_circuit']:.2f}ms",
        ])
    md = "# NetSTA Baseline Comparison\n\n"
    md += f"_Dataset: {args.num_circuits} circuits, seed {args.seed}, "
    md += f"{len(train_idx)}/{len(val_idx)}/{len(test_idx)} split._\n\n"
    md += write_markdown_table(headers, rows)
    save_text(os.path.join(RESULTS_DIR, "baseline_comparison.md"), md)
    save_json(os.path.join(RESULTS_DIR, "baseline_comparison.json"), {
        "config": vars(args),
        "results": results,
    })
    print("\nSaved:")
    print(f"  {RESULTS_DIR}/baseline_comparison.md")
    print(f"  {RESULTS_DIR}/baseline_comparison.json")


if __name__ == "__main__":
    main()

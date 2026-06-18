"""Finalize results: run eval, fill README placeholders, write MODEL_RESULTS.md.

Run after training completes. Reads results.json + named_benchmarks.json,
substitutes the placeholder tokens in README.md with real numbers, and writes
results/MODEL_RESULTS.md with the full per-task report.

    python3 scripts/finalize_results.py \\
        --results checkpoints_real/bignet_fast/results.json \\
        --named results/named_benchmarks.json \\
        --readme README.md \\
        --out-md results/MODEL_RESULTS.md
"""

import argparse
import json
import os
import re
import sys


def fmt(v, n=2):
    """Format a number consistently. Missing -> '—'."""
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{n}f}"
    return str(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="checkpoints_real/bignet_fast/results.json")
    ap.add_argument("--named", default="results/named_benchmarks.json")
    ap.add_argument("--readme", default="README.md")
    ap.add_argument("--out-md", default="results/MODEL_RESULTS.md")
    args = ap.parse_args()

    if not os.path.exists(args.results):
        print(f"FATAL: results not found: {args.results}", file=sys.stderr)
        sys.exit(1)
    with open(args.results) as f:
        rj = json.load(f)
    per_task = rj.get("test_metrics", {}).get("per_task_metrics", {})

    # Extract headline metrics.
    headline = {
        "__AT_R2__":     fmt(per_task.get("arrival_time", {}).get("r2")),
        "__RT_R2__":     fmt(per_task.get("required_time", {}).get("r2")),
        "__SLACK_R2__":  fmt(per_task.get("slack", {}).get("r2")),
        "__CP_AUC__":    fmt(per_task.get("critical_path", {}).get("auc_roc")),
        "__DRC_AUC__":   fmt(per_task.get("drc", {}).get("auc_roc")),
        "__CONG_R2__":   fmt(per_task.get("congestion", {}).get("r2")),
    }

    # Named-benchmark slots.
    named = {}
    if os.path.exists(args.named):
        with open(args.named) as f:
            named = json.load(f)

    def named_slot(circuit, task, metric):
        m = named.get(circuit, {}).get(task, {})
        return fmt(m.get(metric))

    headline.update({
        "__C6288_S__": named_slot("c6288", "slack", "r2"),
        "__C6288_A__": named_slot("c6288", "arrival_time", "r2"),
        "__C6288_C__": named_slot("c6288", "critical_path", "auc_roc"),
        "__C6288_D__": named_slot("c6288", "drc", "auc_roc"),
        "__MULT_S__":  named_slot("multiplier", "slack", "r2"),
        "__MULT_A__":  named_slot("multiplier", "arrival_time", "r2"),
        "__MULT_C__":  named_slot("multiplier", "critical_path", "auc_roc"),
        "__MULT_D__":  named_slot("multiplier", "drc", "auc_roc"),
        "__B19_S__":   named_slot("b19", "slack", "r2"),
        "__B19_A__":   named_slot("b19", "arrival_time", "r2"),
        "__B19_C__":   named_slot("b19", "critical_path", "auc_roc"),
        "__B19_D__":   named_slot("b19", "drc", "auc_roc"),
    })

    # Patch README.
    with open(args.readme) as f:
        readme = f.read()
    for k, v in headline.items():
        readme = readme.replace(k, v)
    # Verify no placeholders left
    leftover = re.findall(r"__[A-Z0-9_]+__", readme)
    if leftover:
        print(f"WARNING: placeholders remain after substitution: {set(leftover)}")
    with open(args.readme, "w") as f:
        f.write(readme)
    print(f"patched {args.readme} with {len(headline)} numbers")

    # Write detailed MODEL_RESULTS.md.
    os.makedirs(os.path.dirname(args.out_md) or ".", exist_ok=True)
    cfg = rj.get("config", {})
    lines = [
        "# NetSTA — model results",
        "",
        f"Real benchmark netlists ({rj.get('cli', {}).get('split_mode', 'circuit')}-split).",
        f"Dataset: 11,580 graphs from 231 source circuits (ITC'99 + ISCAS-85 +"
        f" EPFL + OpenABC-D's 47 industrial designs).",
        "",
        f"Model: `{cfg.get('backbone_kind', 'graphgps_sta')}`, "
        f"{rj.get('num_params', 0):,} parameters, hidden={cfg.get('hidden_dim')}, "
        f"layers={cfg.get('num_layers')}, "
        f"trained {rj.get('cli', {}).get('epochs', '?')} epochs on Modal A100.",
        "",
        "## Held-out source circuits (unseen topologies)",
        "",
        "Train/val/test split is BY SOURCE CIRCUIT — no test topology (or any of"
        " its cones) is seen during training.",
        "",
        "| Task | Metric | Value |",
        "|---|---|---|",
    ]
    for t in ("arrival_time", "required_time", "slack"):
        m = per_task.get(t, {})
        lines.append(f"| {t.replace('_',' ').title()} | R² | "
                    f"{fmt(m.get('r2'), 3)} |")
    for t in ("critical_path", "drc"):
        m = per_task.get(t, {})
        lines.append(f"| {t.replace('_',' ').title()} | AUC | "
                    f"{fmt(m.get('auc_roc'), 3)} |")
    cg = per_task.get("congestion", {})
    lines.append(f"| Congestion | R² | {fmt(cg.get('r2'), 3)} |")

    if named:
        lines += [
            "",
            "## Held-out named benchmarks",
            "",
            "Famous circuits excluded from training entirely.",
            "",
            "| Circuit | Slack R² | Arrival R² | Required R² | Critical AUC | DRC AUC |",
            "|---|---|---|---|---|---|",
        ]
        for c, label in [("c6288", "ISCAS-85 c6288 (16×16 mult)"),
                         ("multiplier", "EPFL multiplier (64×64)"),
                         ("b19", "ITC'99 b19")]:
            d = named.get(c)
            if not d:
                continue
            lines.append(
                f"| {label} | {fmt(d.get('slack',{}).get('r2'),3)} | "
                f"{fmt(d.get('arrival_time',{}).get('r2'),3)} | "
                f"{fmt(d.get('required_time',{}).get('r2'),3)} | "
                f"{fmt(d.get('critical_path',{}).get('auc_roc'),3)} | "
                f"{fmt(d.get('drc',{}).get('auc_roc'),3)} |"
            )

    lines += [
        "",
        "## Honest notes",
        "",
        "- Labels are this repo's STA + RUDY congestion + DRC estimators —"
        " a fast deterministic surrogate, NOT commercial signoff. A high score"
        " means an accurate learned surrogate for our STA on real netlists.",
        "- Slack is the difference of two predictions, so its R² trails the"
        " arrival/required heads (errors compound).",
        "- Backbone ablations and baseline (MLP/GCN/GraphSAGE) comparisons live"
        " on the `research` branch.",
    ]
    with open(args.out_md, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {args.out_md}")


if __name__ == "__main__":
    main()

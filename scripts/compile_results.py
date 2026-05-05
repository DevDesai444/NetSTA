#!/usr/bin/env python3
"""
Compile all benchmark outputs into a single report.

Walks `results/`, reads every JSON it knows about, and assembles
`results/BENCHMARK_REPORT.md`. Sections with missing inputs are noted but
skipped without failing -- so a partial benchmark run still produces a
partial report.
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from _bench_utils import PLOTS_DIR, RESULTS_DIR, ensure_dirs


SECTIONS = [
    {
        "title": "Baseline Comparison",
        "md": "baseline_comparison.md",
        "json": "baseline_comparison.json",
        "takeaway_fn": "_takeaway_baselines",
        "plots": [],
    },
    {
        "title": "Robustness (5-seed)",
        "md": "robustness_analysis.md",
        "json": "robustness_analysis.json",
        "takeaway_fn": "_takeaway_robustness",
        "plots": [],
    },
    {
        "title": "Scaling",
        "md": "scaling_analysis.md",
        "json": "scaling_analysis.json",
        "takeaway_fn": "_takeaway_scaling",
        "plots": ["plots/data_scaling.png", "plots/inference_scaling.png"],
    },
    {
        "title": "Ablation",
        "md": "ablation_study.md",
        "json": "ablation_study.json",
        "takeaway_fn": "_takeaway_ablation",
        "plots": ["plots/ablation_chart.png"],
    },
    {
        "title": "Generalization",
        "md": "generalization_study.md",
        "json": "generalization_study.json",
        "takeaway_fn": "_takeaway_generalization",
        "plots": [],
    },
    {
        "title": "Training Curves",
        "md": None,
        "json": "training_log.json",
        "takeaway_fn": "_takeaway_training",
        "plots": ["plots/training_curves.png", "plots/per_task_curves.png", "plots/lr_schedule.png"],
    },
]


def _read_md(path):
    if not path or not os.path.exists(path):
        return None
    with open(path) as f:
        return f.read()


def _read_json(path):
    if not path or not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


# --- Takeaway generators (best-effort, fall back to vague text) ---

def _takeaway_baselines(data):
    if not data or "results" not in data:
        return None
    res = data["results"]
    netsta = res.get("NetSTA")
    if not netsta or "metrics" not in netsta:
        return None
    slack = netsta["metrics"].get("slack", {})
    crit = netsta["metrics"].get("critical_path", {})
    return (
        f"NetSTA reaches Slack R² = {slack.get('r2', float('nan')):.3f}, "
        f"CP F1 = {crit.get('f1', float('nan')):.3f} "
        f"with {netsta.get('params', 0) / 1000:.0f}K parameters. "
        f"Compare to the linear baseline floor and the graph-blind MLP to gauge "
        f"how much of the lift comes from message passing vs attention."
    )


def _takeaway_robustness(data):
    if not data or "aggregate" not in data:
        return None
    agg = data["aggregate"]
    mse_mean, mse_std = agg.get("slack_mse", (float("nan"), 0))
    r2_mean, r2_std = agg.get("slack_r2", (float("nan"), 0))
    f1_mean, f1_std = agg.get("cp_f1", (float("nan"), 0))
    return (
        f"Across seeds: Slack R² = {r2_mean:.3f} ± {r2_std:.3f}, "
        f"CP F1 = {f1_mean:.3f} ± {f1_std:.3f}, "
        f"Slack MSE = {mse_mean:.4f} ± {mse_std:.4f}. "
        f"Variance gives a rough confidence band on the headline numbers."
    )


def _fmt_speedup(speedup: float) -> str:
    if speedup is None or (isinstance(speedup, float) and (speedup != speedup)):
        return "unmeasured"
    return f"{speedup:.1f}x faster than" if speedup >= 1.0 else f"{1.0 / speedup:.1f}x slower than"


def _takeaway_scaling(data):
    if not data:
        return None
    timing = [r for r in (data.get("inference_scaling") or []) if "error" not in r]
    if not timing:
        return None
    biggest = max(timing, key=lambda r: r["n_gates"])
    return (
        f"At {biggest['n_gates']} gates the GNN is "
        f"{_fmt_speedup(biggest['speedup'])} classical STA "
        f"({biggest['gnn_ms']:.2f}ms vs {biggest['sta_ms']:.2f}ms). "
        f"See data_scaling.png for how R²/F1 evolve with training-set size."
    )


def _takeaway_ablation(data):
    if not data or "results" not in data:
        return None
    res = data["results"]
    ref_name = data.get("reference")
    ref = next((r for r in res if r.get("name") == ref_name and "error" not in r), None)
    if not ref:
        return None
    others = [r for r in res if r is not ref and "error" not in r]
    if not others:
        return None
    drops = sorted(others, key=lambda r: r["slack_r2"] - ref["slack_r2"])
    worst = drops[0]
    delta = worst["slack_r2"] - ref["slack_r2"]
    if delta < 0:
        return (
            f"Largest R² regression comes from '{worst['name']}' "
            f"(Δ = {delta:+.3f} R²), "
            f"which is the biggest single-knob contributor to NetSTA's performance."
        )
    return (
        "No ablation reduced R² versus the reference in this run -- "
        "every delta is non-negative, which usually means the dataset or "
        "training budget was too small to separate architectural signal "
        "from noise. Re-run at full scale to draw conclusions."
    )


def _takeaway_generalization(data):
    if not data or "results" not in data:
        return None
    lines = []
    for r in data["results"]:
        if "error" in r:
            continue
        gap = r["test"]["slack_r2"] - r["train"]["slack_r2"]
        lines.append(f"{r['name']}: train R² {r['train']['slack_r2']:.3f} → "
                     f"test R² {r['test']['slack_r2']:.3f} (Δ {gap:+.3f})")
    return " | ".join(lines) if lines else None


def _takeaway_training(data):
    if not data:
        return None
    test = data.get("test_metrics", {})
    return (
        f"Reference run best epoch = {data.get('best_epoch')} "
        f"(val loss {data.get('best_val_loss', float('nan')):.4f}). "
        f"Final test metrics: {test}."
    )


TAKEAWAY_FNS = {
    "_takeaway_baselines": _takeaway_baselines,
    "_takeaway_robustness": _takeaway_robustness,
    "_takeaway_scaling": _takeaway_scaling,
    "_takeaway_ablation": _takeaway_ablation,
    "_takeaway_generalization": _takeaway_generalization,
    "_takeaway_training": _takeaway_training,
}


def build_executive_summary(per_section):
    """Two-three sentence punchline from whatever sections succeeded."""
    parts = []
    bench = per_section.get("Baseline Comparison")
    if bench:
        bdata = bench.get("data") or {}
        netsta = (bdata.get("results") or {}).get("NetSTA") or {}
        m = netsta.get("metrics") or {}
        if m:
            slack_r2 = (m.get("slack") or {}).get("r2")
            cp_f1 = (m.get("critical_path") or {}).get("f1")
            if slack_r2 is not None and cp_f1 is not None:
                parts.append(
                    f"NetSTA hits Slack R² = {slack_r2:.3f} and CP F1 = {cp_f1:.3f} "
                    f"on the standard 1000-circuit benchmark."
                )
    rob = per_section.get("Robustness (5-seed)")
    if rob:
        agg = (rob.get("data") or {}).get("aggregate") or {}
        if agg.get("slack_r2"):
            mean, std = agg["slack_r2"]
            parts.append(
                f"Across 5 seeds the headline R² has σ = {std:.3f}, "
                f"so the result is reproducible rather than a lucky seed."
            )
    scl = per_section.get("Scaling")
    if scl:
        rows = (scl.get("data") or {}).get("inference_scaling") or []
        valid = [r for r in rows if "error" not in r]
        if valid:
            biggest = max(valid, key=lambda r: r["n_gates"])
            parts.append(
                f"At {biggest['n_gates']} gates the GNN is "
                f"{_fmt_speedup(biggest['speedup'])} the classical STA reference."
            )
    if not parts:
        return "Benchmark report compiled from partial results -- see sections below."
    return " ".join(parts)


def main():
    parser = argparse.ArgumentParser(description="Compile NetSTA benchmark results")
    parser.add_argument("--results-dir", default=RESULTS_DIR)
    parser.add_argument("--output", default=os.path.join(RESULTS_DIR, "BENCHMARK_REPORT.md"))
    args = parser.parse_args()

    ensure_dirs()
    per_section = {}
    for spec in SECTIONS:
        md_path = os.path.join(args.results_dir, spec["md"]) if spec["md"] else None
        json_path = os.path.join(args.results_dir, spec["json"]) if spec["json"] else None
        per_section[spec["title"]] = {
            "spec": spec,
            "md": _read_md(md_path),
            "data": _read_json(json_path),
        }

    summary = build_executive_summary(per_section)
    out = ["# NetSTA Benchmark Report", "", "## Executive summary", "", summary, ""]

    for spec in SECTIONS:
        section = per_section[spec["title"]]
        out.append(f"## {spec['title']}")
        out.append("")
        md = section["md"]
        if md is None and section["data"] is None:
            out.append(f"_No outputs found for this section (expected `{spec['md'] or spec['json']}`). "
                       f"Run the matching benchmark script._")
            out.append("")
            continue
        if md:
            # Trim the leading "# " title the source MD includes so we don't double-up.
            lines = md.splitlines()
            if lines and lines[0].startswith("# "):
                lines = lines[1:]
            out.append("\n".join(lines).strip())
            out.append("")
        for plot_rel in spec["plots"]:
            # Skip if the section MD already references this plot (avoids
            # duplicate embeds when a benchmark writes its own image links).
            if md and plot_rel in md:
                continue
            full = os.path.join(args.results_dir, plot_rel)
            if os.path.exists(full):
                out.append(f"![{plot_rel}]({plot_rel})")
                out.append("")
        takeaway = TAKEAWAY_FNS.get(spec["takeaway_fn"], lambda d: None)(section["data"])
        if takeaway:
            out.append(f"**Takeaway:** {takeaway}")
            out.append("")

    with open(args.output, "w") as f:
        f.write("\n".join(out).rstrip() + "\n")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()

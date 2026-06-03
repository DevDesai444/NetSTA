#!/usr/bin/env python3
"""
Kaggle-ready entry point for the full NetSTA benchmark suite.

This script is designed to be the *only* cell you need to run inside a Kaggle
notebook (Python 3, GPU T4 x1 is enough). It:

  1. Verifies the GPU is visible and prints the torch/PyG versions.
  2. Optionally installs torch-geometric if the Kaggle image does not have
     a compatible build pinned for the current torch version.
  3. Sets a deterministic working directory.
  4. Runs the six benchmark scripts in order with the same arguments
     `scripts/run_all_benchmarks.sh` uses (1000 circuits, 200 epochs, the
     standard 5-seed robustness sweep).
  5. Compiles results/BENCHMARK_REPORT.md and prints a summary at the end.

USAGE (Kaggle):

  !git clone https://github.com/<user>/NetSTA.git /kaggle/working/NetSTA
  %cd /kaggle/working/NetSTA
  !python scripts/kaggle_run_benchmarks.py

USAGE (local GPU box, for reproducibility checks):

  python scripts/kaggle_run_benchmarks.py --num-circuits 1000 --epochs 200

Total wall time: ~3-5 hours on a T4. The headline `baseline_comparison` and
`robustness_analysis` come first so even if a long-running task at the end
times out, the most important numbers are already on disk.

Output:
  - results/baseline_comparison.{json,md}
  - results/robustness_analysis.{json,md}
  - results/scaling_analysis.{json,md}
  - results/ablation_study.{json,md}
  - results/generalization_study.{json,md}
  - results/training_log.json
  - results/BENCHMARK_REPORT.md   (compiled summary)
  - results/plots/*.png

Each individual step also writes a per-step progress log under
results/master_run.log so you can tail the run from another cell.
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
LOG_PATH = RESULTS_DIR / "master_run.log"

STEPS = [
    # (label, script, default-args)
    ("baselines",        "benchmark_baselines.py",        ["--num-circuits", "{nc}", "--epochs", "{ep}"]),
    ("robustness",       "benchmark_robustness.py",       ["--num-circuits", "{nc}", "--epochs", "{ep}",
                                                          "--seeds", "42,123,456,789,1024"]),
    ("scaling",          "benchmark_scaling.py",          ["--epochs", "{ep}"]),
    ("ablation",         "benchmark_ablation.py",         ["--num-circuits", "{nc}", "--epochs", "{ep}"]),
    ("generalization",   "benchmark_generalization.py",   ["--num-circuits", "{nc}", "--epochs", "{ep}"]),
    ("training_curves",  "benchmark_training_curves.py",  ["--num-circuits", "{nc}", "--epochs", "{ep}"]),
]


def _check_environment():
    """Print torch / PyG / CUDA versions; fail loudly if no GPU is visible."""
    import torch
    print(f"torch         : {torch.__version__}")
    print(f"cuda available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"cuda device   : {torch.cuda.get_device_name(0)}")
        print(f"cuda memory   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    try:
        import torch_geometric
        print(f"torch_geometric: {torch_geometric.__version__}")
    except ImportError as exc:
        print(f"torch_geometric NOT INSTALLED: {exc}")
        print("Run: pip install torch_geometric")
        sys.exit(2)
    if not torch.cuda.is_available():
        print("\nWARNING: no CUDA device. Continuing on CPU — the full suite will take ~24h+.")


def _run_step(label, script, args_template, num_circuits, epochs, dry_run):
    """Run one benchmark step and tee its output to results/master_run.log."""
    script_path = REPO_ROOT / "scripts" / script
    args = [a.format(nc=num_circuits, ep=epochs) for a in args_template]
    cmd = [sys.executable, str(script_path), *args]

    header = f"\n=== [{label}] {' '.join(cmd)} ==="
    print(header)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a") as log:
        log.write(header + "\n")

    if dry_run:
        print("  (dry-run, not executing)")
        return True

    t0 = time.perf_counter()
    try:
        with open(LOG_PATH, "a") as log:
            # subprocess.run with stdout=log streams output as it's produced
            # via the child process — but stdout=log inherits writes only
            # when the child flushes. Use Popen with line-buffered piping so
            # we can both tee live to stdout and persist to disk.
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                cwd=REPO_ROOT, text=True, bufsize=1,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                print(line, end="")
                log.write(line)
                log.flush()
            rc = proc.wait()
        elapsed = time.perf_counter() - t0
        print(f"  [{label}] completed in {elapsed:.1f}s (rc={rc})")
        if rc != 0:
            print(f"  [{label}] FAILED with rc={rc} — see {LOG_PATH}")
            return False
        return True
    except Exception as exc:
        print(f"  [{label}] EXCEPTION: {exc!r}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run the full NetSTA benchmark suite on a GPU.")
    parser.add_argument("--num-circuits", type=int, default=1000,
                        help="Circuits per benchmark (default 1000 — matches the published numbers).")
    parser.add_argument("--epochs", type=int, default=200,
                        help="Max epochs per training run (early-stop kicks in well before this).")
    parser.add_argument("--skip", default="",
                        help="Comma-separated step labels to skip "
                             "(baselines,robustness,scaling,ablation,generalization,training_curves)")
    parser.add_argument("--only", default="",
                        help="Comma-separated step labels to run exclusively (overrides --skip).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would run, do not execute.")
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True,
                        help="Compile results/BENCHMARK_REPORT.md after all steps finish (default on).")
    args = parser.parse_args()

    _check_environment()

    only = {s.strip() for s in args.only.split(",") if s.strip()}
    skip = {s.strip() for s in args.skip.split(",") if s.strip()}

    # Fresh master log per run so we don't keep stitching together old output.
    if LOG_PATH.exists():
        LOG_PATH.unlink()

    print(f"\nWriting per-step output to {LOG_PATH}\n")
    failures = []
    for label, script, tpl in STEPS:
        if only and label not in only:
            print(f"--- [{label}] skipped (not in --only) ---")
            continue
        if not only and label in skip:
            print(f"--- [{label}] skipped (in --skip) ---")
            continue
        ok = _run_step(label, script, tpl, args.num_circuits, args.epochs, args.dry_run)
        if not ok:
            failures.append(label)

    if args.compile and not args.dry_run:
        print("\n=== compile_results.py ===")
        rc = subprocess.call(
            [sys.executable, str(REPO_ROOT / "scripts" / "compile_results.py")],
            cwd=REPO_ROOT,
        )
        if rc != 0:
            failures.append("compile_results")

    print("\n=== Summary ===")
    if not failures:
        print("All steps completed successfully.")
        print(f"Headline report: {RESULTS_DIR / 'BENCHMARK_REPORT.md'}")
    else:
        print(f"{len(failures)} step(s) failed: {failures}")
        print(f"See {LOG_PATH} for details.")
    sys.exit(0 if not failures else 1)


if __name__ == "__main__":
    main()

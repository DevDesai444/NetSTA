"""Build the real-netlist graph dataset from .bench files.

Walks a benchmark tree, parses every .bench into the Nangate45 Circuit model,
windows each into fan-in cones, labels with STA/congestion/DRC under sampled
clock targets, and saves a single-file artifact (graphs + per-graph source
circuit) for training.

    python3 scripts/build_real_dataset.py --bench-root benchmarks --out data_real/graphs.pt

Named benchmarks reserved for a clean held-out evaluation are excluded from the
training pool via --exclude (default: b19).
"""

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from netsta.real_dataset import build_real_graphs, save_dataset, source_base, summarize


def find_bench_files(root: str) -> list:
    files = sorted(glob.glob(os.path.join(root, "**", "*.bench"), recursive=True))
    # Our Verilog reader handles ISCAS-85's gate-primitive netlists. EPFL .v use
    # behavioural `assign`, and ISCAS-89 .v use dff sub-modules — both skipped.
    vfiles = sorted(glob.glob(os.path.join(root, "**", "*.v"), recursive=True))
    files += [f for f in vfiles if "ISCAS85" in f]
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench-root", default="benchmarks")
    ap.add_argument("--out", default="data_real/graphs.pt")
    ap.add_argument("--cones", type=int, default=28, help="max fan-in cones per circuit")
    ap.add_argument("--max-whole-nodes", type=int, default=8000)
    ap.add_argument("--max-cone-nodes", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--exclude", default="b19",
        help="comma-separated source bases to hold out (named-benchmark eval)",
    )
    ap.add_argument("--limit", type=int, default=0, help="cap #files (0=all) for quick tests")
    args = ap.parse_args()

    excluded = {b.strip() for b in args.exclude.split(",") if b.strip()}
    files = find_bench_files(args.bench_root)
    files = [
        f for f in files
        if source_base(os.path.splitext(os.path.basename(f))[0]) not in excluded
    ]
    if args.limit:
        files = files[: args.limit]

    print(f"Found {len(files)} .bench files (excluded bases: {sorted(excluded)})")
    graphs, sources = build_real_graphs(
        files,
        cones_per_circuit=args.cones,
        max_whole_nodes=args.max_whole_nodes,
        max_cone_nodes=args.max_cone_nodes,
        seed=args.seed,
    )
    summary = summarize(graphs, sources)
    print("\n=== dataset summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    save_dataset(args.out, graphs, sources, meta={"summary": summary, "excluded": sorted(excluded)})
    print(f"\nSaved {len(graphs)} graphs -> {args.out}")


if __name__ == "__main__":
    main()

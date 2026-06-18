"""Build the real-netlist graph dataset from .bench / .v files.

Walks a benchmark tree, parses every file into the Nangate45 Circuit model,
windows each into fan-in cones, labels with STA / RUDY congestion / DRC under
sampled clock targets, and saves a single-file artifact for training.

    python3 scripts/build_real_dataset.py --bench-root benchmarks --out data_real/graphs.pt

Sources covered:
  - ITC'99       (b01..b22 .bench)
  - ISCAS-85     (c432..c7552 gate-primitive .v)
  - EPFL         (assign-based .v — needs the EPFL parser)
  - OpenABC-D    (47 large industrial AIG .bench: AES, ethernet, JPEG, FPU,
                  RISC-V cores) + 2k+ post-synthesis Verilog netlists

Held-out by default: c6288, EPFL multiplier, b19.
"""

import argparse
import glob
import multiprocessing as mp
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from netsta.real_dataset import save_dataset, source_base, summarize


# Constants tunable by CLI flags.
DEFAULT_EXCLUDE = "b19,c6288,multiplier"


def find_files(root: str) -> list:
    files = sorted(glob.glob(os.path.join(root, "**", "*.bench"), recursive=True))
    vfiles = sorted(glob.glob(os.path.join(root, "**", "*.v"), recursive=True))
    # ISCAS-89 .v ships with a dff sub-module we don't model — skip those.
    vfiles = [f for f in vfiles if "ISCAS89" not in f]
    files += vfiles
    return files


def _process_one(args):
    """Top-level worker: parse one netlist, window into cones, label each cone.

    Heavy circuits (> max_whole_nodes) skip the whole-circuit graph and emit
    only cones. We sample multiple clock targets per cone so the slack/critical
    label distribution covers timing-met and timing-violated regimes.
    """
    (path, fi, seed, cones_per_circuit, max_whole_nodes, max_cone_nodes,
     min_nodes, clock_lo, clock_hi, n_clocks) = args
    try:
        from netsta.benchmark_import import (
            _looks_like_epfl, bench_to_circuit, cone_windows,
            parse_bench, parse_epfl_verilog, parse_verilog,
        )
        from netsta.graph_builder import circuit_to_pyg
        from netsta.sta import run_sta
        import random
        if path.lower().endswith(".v"):
            parser = parse_epfl_verilog if _looks_like_epfl(path) else parse_verilog
        else:
            parser = parse_bench
        nl = parser(path)
        whole = bench_to_circuit(nl, seed=seed + fi)
    except Exception as exc:
        return path, [], f"parse_failed: {exc!r}"
    base = source_base(nl.name)
    rng = random.Random(seed + fi * 7919)

    circuits = []
    if min_nodes <= len(whole.nodes) <= max_whole_nodes:
        circuits.append(whole)
    circuits.extend(
        cone_windows(
            whole, max_cones=cones_per_circuit,
            min_cone_nodes=min_nodes, max_cone_nodes=max_cone_nodes,
            seed=seed + fi,
        )
    )

    graphs = []
    for c in circuits:
        try:
            base_sta = run_sta(c)
            max_at = base_sta["max_arrival_time_ns"]
        except Exception:
            continue
        for _k in range(n_clocks):
            factor = rng.uniform(clock_lo, clock_hi)
            clock = max(max_at * factor, 1e-3)
            try:
                res = run_sta(c, clock_period_ns=clock)
                graphs.append((base, circuit_to_pyg(c, res)))
            except Exception:
                continue
    msg = f"{nl.name}: whole={len(whole.nodes)}n -> {len(graphs)} graphs"
    return path, graphs, msg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench-root", default="benchmarks")
    ap.add_argument("--out", default="data_real/graphs.pt")
    ap.add_argument("--cones", type=int, default=24, help="max fan-in cones per circuit")
    ap.add_argument("--max-whole-nodes", type=int, default=8000)
    ap.add_argument("--max-cone-nodes", type=int, default=6000)
    ap.add_argument("--min-nodes", type=int, default=8)
    ap.add_argument("--clock-lo", type=float, default=0.85)
    ap.add_argument("--clock-hi", type=float, default=1.15)
    ap.add_argument("--n-clocks", type=int, default=2,
                    help="number of clock targets sampled per cone")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--exclude", default=DEFAULT_EXCLUDE,
                    help="comma-separated source bases to hold out")
    ap.add_argument("--limit", type=int, default=0, help="cap #files (0=all)")
    ap.add_argument("--workers", type=int, default=max(2, mp.cpu_count() - 2))
    ap.add_argument("--max-graphs", type=int, default=0,
                    help="overall cap on graphs (0=no cap); stops once reached")
    args = ap.parse_args()

    excluded = {b.strip() for b in args.exclude.split(",") if b.strip()}
    files = find_files(args.bench_root)
    files = [
        f for f in files
        if source_base(os.path.splitext(os.path.basename(f))[0]) not in excluded
    ]
    if args.limit:
        files = files[: args.limit]

    print(f"Found {len(files)} files (excluded: {sorted(excluded)})")
    print(f"Workers: {args.workers}  cones/circuit: {args.cones}  clocks/cone: {args.n_clocks}")

    job_args = [
        (f, i, args.seed, args.cones, args.max_whole_nodes, args.max_cone_nodes,
         args.min_nodes, args.clock_lo, args.clock_hi, args.n_clocks)
        for i, f in enumerate(files)
    ]

    t0 = time.time()
    graphs = []
    sources = []
    done = 0
    with mp.Pool(processes=args.workers) as pool:
        for path, gs, msg in pool.imap_unordered(_process_one, job_args, chunksize=1):
            done += 1
            for base, data in gs:
                graphs.append(data)
                sources.append(base)
            elapsed = time.time() - t0
            rate = done / max(elapsed, 0.1)
            eta = (len(files) - done) / max(rate, 1e-3)
            print(f"[{done}/{len(files)} | {len(graphs)} graphs | "
                  f"{rate:.1f} files/s | eta {eta/60:.1f}min]  {msg}")
            if args.max_graphs and len(graphs) >= args.max_graphs:
                print(f"Hit max-graphs cap ({args.max_graphs}); stopping.")
                pool.terminate()
                break

    summary = summarize(graphs, sources)
    print("\n=== dataset summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"Build wall-clock: {(time.time() - t0)/60:.1f} min")

    save_dataset(args.out, graphs, sources,
                 meta={"summary": summary, "excluded": sorted(excluded)})
    print(f"\nSaved {len(graphs)} graphs -> {args.out}")


if __name__ == "__main__":
    main()

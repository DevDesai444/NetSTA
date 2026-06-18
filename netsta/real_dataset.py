"""
Dataset of real benchmark netlists for NetSTA.

Turns a set of .bench files (ITC'99 / ISCAS) into PyG graphs through the exact
schema-v9 pipeline used for synthetic data: parse -> Nangate45 Circuit -> STA /
congestion / DRC labels -> circuit_to_pyg. Two things make it a *large, real*
dataset rather than a handful of circuits:

  1. Fan-in cone windowing — each netlist is carved into many endpoint-rooted
     sub-circuits, so ~100 source files yield thousands of real-structure
     graphs of varied size and depth.
  2. Per-graph clock targets — each graph gets a clock sampled around its own
     critical path, so the slack distribution (and the critical-path label)
     spans timing-met and timing-violated regimes across all sizes. This is
     what fixes the all-negative critical-path label a fixed ns threshold gives
     on large circuits.

Splitting is by *source circuit* (e.g. all cones/variants of `b14` land in one
split) so test topologies are genuinely unseen — the honest generalization
test, not a leaky random split over overlapping cones.
"""

import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from .benchmark_import import (
    _looks_like_epfl,
    bench_to_circuit,
    cone_windows,
    parse_bench,
    parse_epfl_verilog,
    parse_verilog,
)
from .graph_builder import circuit_to_pyg
from .sta import run_sta


def source_base(name: str) -> str:
    """Group key for splitting: strip variant/cone suffixes.

    `b21_opt`, `b21_C`, `b21__c3` all map to `b21` so no variant or cone of a
    circuit can straddle the train/test boundary.
    """
    return name.split("_")[0].split("__")[0]


def _label_with_clock(circuit, rng: random.Random, clock_factor_range):
    """Run STA under a sampled clock target and build the PyG graph.

    A first STA pass gives max arrival time; the clock is then set to a sampled
    multiple of it (aggressive multiples create real timing violations), and a
    second pass produces slack/AT/RT/critical under that constraint.
    """
    base = run_sta(circuit)
    max_at = base["max_arrival_time_ns"]
    factor = rng.uniform(*clock_factor_range)
    clock = max(max_at * factor, 1e-3)
    res = run_sta(circuit, clock_period_ns=clock)
    return circuit_to_pyg(circuit, res)


def build_real_graphs(
    bench_paths: Sequence[str],
    cones_per_circuit: int = 28,
    max_whole_nodes: int = 8000,
    max_cone_nodes: int = 6000,
    min_nodes: int = 8,
    clock_factor_range: Tuple[float, float] = (0.85, 1.15),
    seed: int = 42,
    verbose: bool = True,
) -> Tuple[List, List[str]]:
    """Build (graphs, sources) from a list of .bench files."""
    graphs: List = []
    sources: List[str] = []
    rng = random.Random(seed)

    for fi, path in enumerate(bench_paths):
        try:
            if path.lower().endswith(".v"):
                parser = parse_epfl_verilog if _looks_like_epfl(path) else parse_verilog
            else:
                parser = parse_bench
            nl = parser(path)
            whole = bench_to_circuit(nl, seed=seed + fi)
        except Exception as exc:  # keep one bad file from killing the build
            if verbose:
                print(f"  [skip] {path}: {exc!r}")
            continue
        base = source_base(nl.name)

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

        added = 0
        for c in circuits:
            try:
                data = _label_with_clock(c, rng, clock_factor_range)
            except Exception:
                continue
            graphs.append(data)
            sources.append(base)
            added += 1
        if verbose:
            print(f"  {nl.name}: {len(whole.nodes)} nodes -> {added} graphs")

    return graphs, sources


def circuit_level_split(
    sources: Sequence[str],
    seed: int = 42,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
) -> Tuple[List[int], List[int], List[int]]:
    """Assign whole source circuits to train/val/test (no topology leakage)."""
    bases = sorted(set(sources))
    rng = random.Random(seed)
    rng.shuffle(bases)
    n = len(bases)
    n_test = max(1, int(round(test_frac * n)))
    n_val = max(1, int(round(val_frac * n)))
    test_b = set(bases[:n_test])
    val_b = set(bases[n_test : n_test + n_val])
    train_idx, val_idx, test_idx = [], [], []
    for i, s in enumerate(sources):
        if s in test_b:
            test_idx.append(i)
        elif s in val_b:
            val_idx.append(i)
        else:
            train_idx.append(i)
    return train_idx, val_idx, test_idx


class InMemoryGraphDataset:
    """Minimal list-backed dataset matching the NetSTADataset interface."""

    def __init__(self, graphs: Sequence):
        self.graphs = list(graphs)

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx]


def save_dataset(path: str, graphs: List, sources: List[str], meta: Optional[dict] = None):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({"graphs": graphs, "sources": sources, "meta": meta or {}}, path)


def load_dataset(path: str):
    blob = torch.load(path, weights_only=False)
    return blob["graphs"], blob["sources"], blob.get("meta", {})


def summarize(graphs: Sequence, sources: Sequence[str]) -> dict:
    """Quick stats for logging/sanity: sizes, critical-rate, source spread."""
    import numpy as np

    sizes = [int(g.x.size(0)) for g in graphs]
    crit_rate = []
    for g in graphs:
        if hasattr(g, "y_critical") and g.y_critical.numel():
            crit_rate.append(float(g.y_critical.float().mean()))
    return {
        "num_graphs": len(graphs),
        "num_sources": len(set(sources)),
        "nodes_min": min(sizes) if sizes else 0,
        "nodes_max": max(sizes) if sizes else 0,
        "nodes_mean": float(np.mean(sizes)) if sizes else 0.0,
        "total_nodes": int(sum(sizes)),
        "critical_rate_mean": float(np.mean(crit_rate)) if crit_rate else 0.0,
    }

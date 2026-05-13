"""
CLI: index circuits with a trained model and run a similarity query.

Examples:
    python3 -m netsta.search --circuit-type digital --top-k 3
    python3 -m netsta.search --circuit-type analog --top-k 5 --num-circuits 30

The script will:
  1. Auto-locate a compatible checkpoint under checkpoints/.
  2. Build (or reuse) a small dataset of the requested type.
  3. Embed every circuit and persist to ./circuit_embeddb/.
  4. Pick one anchor circuit, query the index, and print the top-k.
"""

import argparse
import os
import sys
from typing import Optional

import torch

from .dataset import (
    AnalogCircuitDataset,
    MixedCircuitDataset,
    NetSTADataset,
)
from .predict import load_model
from .similarity.circuit_index import CircuitIndex
from .similarity.search import find_similar


_CHECKPOINT_CANDIDATES = {
    "digital": [
        "checkpoints/mixed/best_model.pt",   # unified-schema digital ✓
        "checkpoints/best_model.pt",         # might be legacy schema
    ],
    "analog": [
        "checkpoints/analog/best_model.pt",
        "checkpoints/mixed/best_model.pt",
    ],
    "mixed": [
        "checkpoints/mixed/best_model.pt",
        "checkpoints/analog/best_model.pt",
    ],
}


def _find_checkpoint(circuit_type: str) -> Optional[str]:
    for path in _CHECKPOINT_CANDIDATES.get(circuit_type, []):
        if os.path.exists(path):
            return path
    # Last resort: any .pt under checkpoints/.
    if os.path.isdir("checkpoints"):
        for root, _, files in os.walk("checkpoints"):
            for f in files:
                if f.endswith(".pt"):
                    return os.path.join(root, f)
    return None


def _build_dataset(circuit_type: str, num_circuits: int, data_dir: Optional[str], seed: int):
    if circuit_type == "digital":
        return NetSTADataset(
            root=data_dir or "data", num_circuits=num_circuits, seed=seed,
        )
    if circuit_type == "analog":
        return AnalogCircuitDataset(
            root=data_dir or "data_analog", num_circuits=num_circuits, seed=seed,
        )
    if circuit_type == "mixed":
        return MixedCircuitDataset(
            root=data_dir or "data_mixed", num_circuits=num_circuits, seed=seed,
        )
    raise ValueError(f"Unknown --circuit-type '{circuit_type}'")


def _print_neighbour_row(rank: int, hit: dict) -> None:
    m = hit.get("metadata") or {}
    sim = hit.get("similarity")
    sim_str = f"{sim:+.4f}" if isinstance(sim, float) else "  --  "
    print(
        f"  #{rank}: id={hit.get('id')}  sim={sim_str}  "
        f"type={m.get('circuit_type', '-')}  "
        f"nodes={m.get('num_gates', '-')}  "
        f"max_cong={m.get('max_congestion', 0):.3f}  "
        f"cp_len={m.get('critical_path_length', 0)}  "
        f"avg_slack={m.get('avg_slack', 0):.3f}  "
        f"name={m.get('circuit_name', '')}"
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Similarity search over circuit embeddings.",
    )
    parser.add_argument(
        "--circuit-type", default="digital",
        choices=["digital", "analog", "mixed"],
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--num-circuits", type=int, default=20,
                        help="Index size (default 20 — smoke-friendly).")
    parser.add_argument("--data-dir", default=None,
                        help="Override dataset root.")
    parser.add_argument("--checkpoint", default=None,
                        help="Override checkpoint path.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--anchor-index", type=int, default=0,
                        help="Which circuit in the dataset to use as query.")
    parser.add_argument("--rebuild", action="store_true",
                        help="Force-rebuild the embedding index.")
    # Metadata filter flags. Each maps to a $gte / $lte clause on Chroma's
    # `where` filter. --min-gain expects either a 0..1 score (gbw_score) or
    # a dB-style number >= 1.0 which is divided by 100 (60 -> 0.6).
    parser.add_argument("--min-gain", type=float, default=None,
                        help="Minimum avg_gbw_score (0..1) or a dB-style "
                             ">=1 number that is interpreted as N/100.")
    parser.add_argument("--max-congestion", type=float, default=None,
                        help="Maximum max_congestion (0..1).")
    parser.add_argument("--min-congestion", type=float, default=None,
                        help="Minimum max_congestion (0..1).")
    parser.add_argument("--min-nodes", type=int, default=None,
                        help="Minimum num_gates.")
    parser.add_argument("--max-nodes", type=int, default=None,
                        help="Maximum num_gates.")
    parser.add_argument("--min-slack", type=float, default=None,
                        help="Minimum avg_slack.")
    parser.add_argument("--max-slack", type=float, default=None,
                        help="Maximum avg_slack.")
    args = parser.parse_args(argv)

    ckpt = args.checkpoint or _find_checkpoint(args.circuit_type)
    if ckpt is None:
        print(f"No checkpoint available for --circuit-type {args.circuit_type}. "
              "Train one first, e.g.:\n"
              "  python3 -m netsta.train --num-circuits 200 --epochs 10")
        return 0  # graceful no-op for smoke tests

    print(f"Loading checkpoint: {ckpt}")
    try:
        model = load_model(ckpt, device="cpu")
    except Exception as exc:
        print(f"Checkpoint load failed: {exc!r}")
        print("Likely a feature-schema mismatch. Retrain with the current schema.")
        return 1
    print(f"  active tasks: {list(model.heads.keys())}")
    print(f"  node_feature_dim: {model.config.node_feature_dim}")

    print(f"\nBuilding dataset (type={args.circuit_type}, n={args.num_circuits})...")
    try:
        dataset = _build_dataset(args.circuit_type, args.num_circuits, args.data_dir, args.seed)
    except Exception as exc:
        print(f"Dataset construction failed: {exc!r}")
        return 1
    print(f"  dataset size: {len(dataset)}")

    # Confirm feature-dim compatibility between model and dataset.
    sample = dataset[0]
    if sample.x.size(1) != model.config.node_feature_dim:
        print(f"Schema mismatch: dataset node_feature_dim={sample.x.size(1)} "
              f"but checkpoint expects {model.config.node_feature_dim}. "
              "Retrain with the current schema (DATA_SCHEMA_VERSION) or "
              "delete the stale dataset cache.")
        return 1

    print("\nBuilding embedding index (ChromaDB at ./circuit_embeddb/)...")
    # Namespace the collection per circuit-type so switching --circuit-type
    # doesn't query a stale index built for a different type.
    collection = f"circuit_embeddings_{args.circuit_type}"
    index = CircuitIndex(
        model=model, device="cpu", collection_name=collection,
    )
    if args.rebuild:
        index.reset()
    index.build(dataset, force=args.rebuild)

    # Collect metadata-range filters into a target_specs dict for find_similar.
    target_specs: dict = {}
    if args.min_gain is not None:
        gain = args.min_gain / 100.0 if args.min_gain >= 1.0 else args.min_gain
        target_specs["avg_gbw_score"] = {"min": gain}
    if args.max_congestion is not None or args.min_congestion is not None:
        bound = {}
        if args.min_congestion is not None:
            bound["min"] = args.min_congestion
        if args.max_congestion is not None:
            bound["max"] = args.max_congestion
        target_specs["max_congestion"] = bound
    if args.min_nodes is not None or args.max_nodes is not None:
        bound = {}
        if args.min_nodes is not None:
            bound["min"] = args.min_nodes
        if args.max_nodes is not None:
            bound["max"] = args.max_nodes
        target_specs["num_gates"] = bound
    if args.min_slack is not None or args.max_slack is not None:
        bound = {}
        if args.min_slack is not None:
            bound["min"] = args.min_slack
        if args.max_slack is not None:
            bound["max"] = args.max_slack
        target_specs["avg_slack"] = bound
    if target_specs:
        print(f"Applying metadata filters: {target_specs}")

    anchor = dataset[min(args.anchor_index, len(dataset) - 1)]
    anchor_name = getattr(anchor, "circuit_name", f"index_{args.anchor_index}")
    print(f"\nAnchor circuit: idx={args.anchor_index} name={anchor_name}")
    print(f"Top-{args.top_k} similar circuits:")
    hits = find_similar(
        anchor, model, index, top_k=args.top_k,
        where=target_specs if target_specs else None,
    )
    if not hits:
        print("  (no matches — index may be empty)")
        return 0
    for rank, hit in enumerate(hits, start=1):
        _print_neighbour_row(rank, hit)
    return 0


if __name__ == "__main__":
    sys.exit(main())

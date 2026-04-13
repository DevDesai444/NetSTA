"""Generate training data for TimingNet."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from timingnet.dataset import TimingNetDataset


def main():
    print("Generating TimingNet dataset...")
    dataset = TimingNetDataset(
        root="data",
        num_circuits=500,
        min_gates=15,
        max_gates=80,
        seed=42,
        force_regenerate=True,
    )
    print(f"Generated {len(dataset)} circuits in data/processed/")

    sample = dataset[0]
    print(f"Sample graph: {sample.num_nodes} nodes, {sample.edge_index.size(1)} edges")
    print(f"Node features: {sample.x.size(1)} dims")
    print(f"Critical nodes: {int(sample.y_critical.sum())}/{sample.num_nodes}")


if __name__ == "__main__":
    main()

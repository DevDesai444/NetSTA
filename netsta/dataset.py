"""
PyTorch Geometric dataset for NetSTA.

Generates or loads synthetic circuits with STA ground truth. Provides:
  - NetSTADataset: cached per-circuit dataset, schema-versioned so stale
    caches built before y_congestion auto-regenerate.
  - NetSTAAugment: random per-graph node-feature noise + edge dropout for
    training-time augmentation.
  - TransformSubset: index/transform wrapper for train/val/test splits.
"""

import json
import os
from typing import Callable, Optional, Sequence

import torch

from .circuit_gen import generate_circuit, circuit_to_dict
from .sta import run_sta
from .graph_builder import circuit_to_pyg


# Bump when the on-disk PyG Data schema changes (e.g. new label field).
# v5: drop logical_depth, load_cap, and 5 1-hop topology aggregates from
#     node features (31 -> 24 dims). Forces GNN to derive topology via MP.
# v6: switch y_slack from per-graph normalized [-1, 1] to raw absolute ns.
#     Standardization happens inside SlackHead via persisted train-set stats.
#     y_critical now uses fixed CRITICAL_SLACK_THRESHOLD_NS, not a per-graph
#     quantile. Also: STA propagation rewrite means slack values themselves
#     change — caches MUST regenerate.
DATA_SCHEMA_VERSION = 6


class NetSTAAugment:
    """Per-graph augmentation: gaussian node-feature noise + edge dropout.

    Designed to be applied inside __getitem__ so each epoch sees a fresh
    sample. Operates on a clone so the cached Data object stays clean.
    """

    def __init__(self, node_noise_sigma: float = 0.01, edge_dropout_p: float = 0.05):
        self.node_noise_sigma = node_noise_sigma
        self.edge_dropout_p = edge_dropout_p

    def __call__(self, data):
        data = data.clone()
        if self.node_noise_sigma > 0:
            data.x = data.x + torch.randn_like(data.x) * self.node_noise_sigma
        if self.edge_dropout_p > 0 and data.edge_index.size(1) > 0:
            keep = torch.rand(data.edge_index.size(1)) > self.edge_dropout_p
            data.edge_index = data.edge_index[:, keep]
            if data.edge_attr is not None:
                data.edge_attr = data.edge_attr[keep]
        return data


class TransformSubset:
    """Index/transform wrapper for splitting a dataset into train/val/test.

    Applies its transform on every __getitem__, so training-time augmentation
    is re-sampled each epoch rather than baked in once.
    """

    def __init__(
        self,
        base,
        indices: Sequence[int],
        transform: Optional[Callable] = None,
    ):
        self.base = base
        self.indices = list(indices)
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        data = self.base[self.indices[i]]
        if self.transform is not None:
            data = self.transform(data)
        return data


class NetSTADataset:
    """
    Dataset of synthetic circuits with STA ground truth.

    Each sample is a PyG Data object representing one circuit, cached as a
    .pt file under `root/processed/`. The cache is keyed on `num_circuits`
    and `DATA_SCHEMA_VERSION`; bumping the schema or changing the count
    triggers regeneration.
    """

    def __init__(
        self,
        root="data",
        num_circuits=500,
        min_gates=15,
        max_gates=80,
        min_inputs=4,
        max_inputs=16,
        min_outputs=2,
        max_outputs=8,
        seed=42,
        force_regenerate=False,
    ):
        self.num_circuits = num_circuits
        self.min_gates = min_gates
        self.max_gates = max_gates
        self.min_inputs = min_inputs
        self.max_inputs = max_inputs
        self.min_outputs = min_outputs
        self.max_outputs = max_outputs
        self.seed = seed
        self.root = root

        self.processed_dir = os.path.join(root, "processed")
        os.makedirs(self.processed_dir, exist_ok=True)

        metadata_path = os.path.join(self.processed_dir, "metadata.json")
        needs_regen = force_regenerate or not os.path.exists(metadata_path)
        if not needs_regen:
            with open(metadata_path) as f:
                meta = json.load(f)
            if (
                meta.get("num_circuits") != num_circuits
                or meta.get("schema_version") != DATA_SCHEMA_VERSION
            ):
                needs_regen = True
        if needs_regen:
            self._generate()

    def _generate(self):
        import random
        random.seed(self.seed)

        for i in range(self.num_circuits):
            n_inputs = random.randint(self.min_inputs, self.max_inputs)
            n_gates = random.randint(self.min_gates, self.max_gates)
            n_outputs = random.randint(self.min_outputs, self.max_outputs)

            circuit = generate_circuit(
                num_inputs=n_inputs,
                num_gates=n_gates,
                num_outputs=n_outputs,
                seed=self.seed + i,
                name=f"circuit_{i:04d}",
            )

            sta_results = run_sta(circuit)
            data = circuit_to_pyg(circuit, sta_results)

            torch.save(data, os.path.join(self.processed_dir, f"circuit_{i:04d}.pt"))

            circuit_json = circuit_to_dict(circuit)
            circuit_json["sta_summary"] = {
                "clock_period_ns": sta_results["clock_period_ns"],
                "max_arrival_time_ns": sta_results["max_arrival_time_ns"],
                "min_slack_ns": sta_results["min_slack_ns"],
                "num_critical_nodes": sta_results["num_critical_nodes"],
            }
            with open(
                os.path.join(self.processed_dir, f"circuit_{i:04d}.json"), "w"
            ) as f:
                json.dump(circuit_json, f, indent=2)

        with open(os.path.join(self.processed_dir, "metadata.json"), "w") as f:
            json.dump(
                {
                    "num_circuits": self.num_circuits,
                    "seed": self.seed,
                    "min_gates": self.min_gates,
                    "max_gates": self.max_gates,
                    "schema_version": DATA_SCHEMA_VERSION,
                },
                f,
            )

    def __len__(self):
        return self.num_circuits

    def __getitem__(self, idx):
        path = os.path.join(self.processed_dir, f"circuit_{idx:04d}.pt")
        return torch.load(path, weights_only=False)


# ---------------------------------------------------------------------------
# Analog + mixed datasets
# ---------------------------------------------------------------------------


class AnalogCircuitDataset:
    """Cached analog-circuit dataset.

    Generates and stores one analog topology per circuit (cycling through
    the five available templates) with parameter jitter per seed.
    """

    def __init__(self, root="data_analog", num_circuits=200, seed=42,
                 force_regenerate=False):
        self.root = root
        self.num_circuits = num_circuits
        self.seed = seed
        self.processed_dir = os.path.join(root, "processed")
        os.makedirs(self.processed_dir, exist_ok=True)

        metadata_path = os.path.join(self.processed_dir, "metadata.json")
        needs_regen = force_regenerate or not os.path.exists(metadata_path)
        if not needs_regen:
            with open(metadata_path) as f:
                meta = json.load(f)
            if (
                meta.get("num_circuits") != num_circuits
                or meta.get("schema_version") != DATA_SCHEMA_VERSION
                or meta.get("kind") != "analog"
            ):
                needs_regen = True
        if needs_regen:
            self._generate()

    def _generate(self):
        from .analog_circuit_gen import ANALOG_TOPOLOGIES, generate_analog_circuit
        from .analog_sta import run_analog_sta
        from .graph_builder import circuit_to_pyg

        for i in range(self.num_circuits):
            topo_fn = ANALOG_TOPOLOGIES[i % len(ANALOG_TOPOLOGIES)]
            circuit = generate_analog_circuit(
                seed=self.seed + i, topology=topo_fn.__name__,
                name=f"acircuit_{i:04d}",
            )
            sta_results = run_analog_sta(circuit)
            data = circuit_to_pyg(circuit, sta_results)
            torch.save(data, os.path.join(self.processed_dir, f"circuit_{i:04d}.pt"))

        with open(os.path.join(self.processed_dir, "metadata.json"), "w") as f:
            json.dump(
                {
                    "num_circuits": self.num_circuits,
                    "seed": self.seed,
                    "schema_version": DATA_SCHEMA_VERSION,
                    "kind": "analog",
                },
                f,
            )

    def __len__(self):
        return self.num_circuits

    def __getitem__(self, idx):
        path = os.path.join(self.processed_dir, f"circuit_{idx:04d}.pt")
        return torch.load(path, weights_only=False)


class MixedCircuitDataset:
    """Concatenates a digital and an analog dataset for joint training.

    First `n_digital` indices are digital, then analog. Both datasets share
    the unified feature schema so PyG batching works without further glue.
    """

    def __init__(self, root="data_mixed", num_circuits=200, seed=42,
                 force_regenerate=False, analog_fraction: float = 0.5):
        n_analog = max(1, int(num_circuits * analog_fraction))
        n_digital = max(1, num_circuits - n_analog)
        self.digital = NetSTADataset(
            root=os.path.join(root, "digital"),
            num_circuits=n_digital, seed=seed,
            force_regenerate=force_regenerate,
        )
        self.analog = AnalogCircuitDataset(
            root=os.path.join(root, "analog"),
            num_circuits=n_analog, seed=seed + 7919,
            force_regenerate=force_regenerate,
        )
        self.n_digital = n_digital
        self.n_analog = n_analog

    def __len__(self):
        return self.n_digital + self.n_analog

    def __getitem__(self, idx):
        if idx < self.n_digital:
            return self.digital[idx]
        return self.analog[idx - self.n_digital]

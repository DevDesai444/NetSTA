"""
Train-set summary statistics persisted alongside model checkpoints.

The slack regression target is the absolute STA slack in nanoseconds. Its
distribution depends on the circuit-size mix in the train split, so a model
trained on small circuits needs different mean/std than one trained on large
ones. We compute these once from the train split, persist them with the
checkpoint, and bake them into the SlackHead's standardization layer so the
forward pass returns predictions directly in ns at every callsite.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from typing import Iterable

import torch


@dataclass
class DatasetStats:
    """Per-task summary statistics derived from the train split.

    For now: slack only. Add critical_path_positive_rate, congestion_mean
    here if/when additional heads need similar normalization.
    """
    slack_mean: float = 0.0
    slack_std: float = 1.0

    @classmethod
    def from_slack_tensors(cls, slack_tensors: Iterable[torch.Tensor]) -> "DatasetStats":
        """Compute mean/std across concatenated per-node slack values.

        std is floored at 1e-3 so a constant-target degenerate split doesn't
        produce a divide-by-zero in the head.
        """
        flat = torch.cat([t.flatten() for t in slack_tensors if t.numel() > 0])
        if flat.numel() == 0:
            return cls(slack_mean=0.0, slack_std=1.0)
        mean = float(flat.mean().item())
        std = float(flat.std(unbiased=False).item())
        if not math.isfinite(std) or std < 1e-3:
            std = 1e-3
        return cls(slack_mean=mean, slack_std=std)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "DatasetStats":
        with open(path) as f:
            data = json.load(f)
        return cls(**data)


STATS_FILENAME = "dataset_stats.json"

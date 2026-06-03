"""
Train-set summary statistics persisted alongside model checkpoints.

The timing regression targets (slack, arrival_time, required_time) are stored
as absolute nanoseconds. Their distributions depend on the circuit-size mix in
the train split, so a model trained on small circuits needs different mean/std
than one trained on large ones. We compute these once from the train split,
persist them with the checkpoint, and bake them into each head's
standardization layer so the forward pass returns predictions directly in ns
at every callsite.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from typing import Iterable, Tuple

import torch


@dataclass
class DatasetStats:
    """Per-task summary statistics derived from the train split.

    Slack stats remain as the primary fields for backwards compatibility with
    earlier checkpoints; arrival_time and required_time stats are populated
    when the auxiliary heads are active. A stats file written by an older
    training run will still load — the new fields fall back to safe defaults
    that produce identity scaling.
    """
    slack_mean: float = 0.0
    slack_std: float = 1.0
    arrival_time_mean: float = 0.0
    arrival_time_std: float = 1.0
    required_time_mean: float = 0.0
    required_time_std: float = 1.0

    @classmethod
    def from_slack_tensors(cls, slack_tensors: Iterable[torch.Tensor]) -> "DatasetStats":
        """Backwards-compatible single-target factory for slack only.

        Existing callers expecting the slack-only API still work; the new
        AT/RT fields stay at their default identity values.
        """
        mean, std = _mean_std(slack_tensors)
        return cls(slack_mean=mean, slack_std=std)

    @classmethod
    def from_target_tensors(
        cls,
        slack_tensors: Iterable[torch.Tensor],
        arrival_tensors: Iterable[torch.Tensor] | None = None,
        required_tensors: Iterable[torch.Tensor] | None = None,
    ) -> "DatasetStats":
        """Compute mean/std for every available target stream.

        Pass `None` for AT/RT when those labels are not yet populated on the
        dataset (e.g. v6-and-older caches loaded against new code) — the
        defaults keep the head's standardization as an identity transform.
        """
        slack_m, slack_s = _mean_std(slack_tensors)
        out = cls(slack_mean=slack_m, slack_std=slack_s)
        if arrival_tensors is not None:
            out.arrival_time_mean, out.arrival_time_std = _mean_std(arrival_tensors)
        if required_tensors is not None:
            out.required_time_mean, out.required_time_std = _mean_std(required_tensors)
        return out

    def save(self, path: str) -> None:
        if os.path.dirname(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "DatasetStats":
        with open(path) as f:
            data = json.load(f)
        # Drop unknown keys so an older or newer file still loads cleanly.
        valid = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in valid})


def _mean_std(tensors: Iterable[torch.Tensor]) -> Tuple[float, float]:
    flat = torch.cat([t.flatten() for t in tensors if t.numel() > 0])
    if flat.numel() == 0:
        return 0.0, 1.0
    mean = float(flat.mean().item())
    std = float(flat.std(unbiased=False).item())
    if not math.isfinite(std) or std < 1e-3:
        std = 1e-3
    return mean, std


STATS_FILENAME = "dataset_stats.json"

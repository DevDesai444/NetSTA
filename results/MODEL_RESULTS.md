# NetSTA — model results

Real benchmark netlists (circuit-split).
Dataset: 11,580 graphs from 231 source circuits (ITC'99 + ISCAS-85 + EPFL + OpenABC-D's 47 industrial designs).

Model: `graphgps_sta`, 4,995,726 parameters, hidden=64, layers=6, trained 150 epochs on Modal A100.

## Held-out source circuits (unseen topologies)

Train/val/test split is BY SOURCE CIRCUIT — no test topology (or any of its cones) is seen during training.

| Task | Metric | Value |
|---|---|---|
| Arrival Time | R² | 0.696 |
| Required Time | R² | 0.726 |
| Slack | R² | 0.657 |
| Critical Path | AUC | 0.803 |
| Drc | AUC | 0.888 |
| Congestion | R² | 0.339 |

## Held-out named benchmarks

Famous circuits excluded from training entirely.

| Circuit | Slack R² | Arrival R² | Required R² | Critical AUC | DRC AUC |
|---|---|---|---|---|---|
| ISCAS-85 c6288 (16×16 mult) | -0.219 | 0.475 | 0.287 | 0.492 | 0.735 |
| EPFL multiplier (64×64) | 0.132 | 0.298 | 0.234 | 0.568 | 0.613 |
| ITC'99 b19 | 0.141 | 0.504 | 0.500 | 0.474 | 0.784 |

## Honest notes

- Labels are this repo's STA + RUDY congestion + DRC estimators — a fast deterministic surrogate, NOT commercial signoff. A high score means an accurate learned surrogate for our STA on real netlists.
- Slack is the difference of two predictions, so its R² trails the arrival/required heads (errors compound).
- Backbone ablations and baseline (MLP/GCN/GraphSAGE) comparisons live on the `research` branch.

# NetSTA — model results

Real benchmark netlists (ITC'99 + ISCAS-85), schema v9. Dataset: 2,665 fan-in
cone graphs from 30 source circuits (8–7,299 nodes). Model: directional
STA-aware backbone with raw-feature residual, 6 heads, 77.6K parameters,
hidden 64, 8 relaxation iterations, trained on a single A10 GPU.

## Held-out source circuits (unseen topologies)

The strict generalization test — train/val/test split **by source circuit**, so
no test topology (or any of its cones) is seen in training. Best epoch 160.

| Task | Metric | Value |
|---|---|---|
| Arrival time | R² | **0.636** |
| Required time | R² | 0.572 |
| Slack | R² | 0.401 |
| Critical path | AUC / F1 | 0.739 / 0.386 |
| DRC hotspot | AUC / F1 | 0.813 / 0.881 |
| Congestion | R² | 0.217 |

The arrival/required-time heads carry the strongest signal — they predict a
directly-propagated quantity. Slack trails them because it's the difference of
two predictions (errors compound). DRC classification is reliable (AUC 0.81);
the critical-path head is the weakest, and as the named-benchmark numbers below
show, it does not generalize well circuit-by-circuit.

## Held-out named benchmarks

Famous circuits excluded from training entirely, evaluated with the held-out
model above (cone-windowed, aggressive clock).

| Circuit | Slack R² | Arrival R² | Required R² | Critical AUC | DRC AUC |
|---|---|---|---|---|---|
| ITC'99 `b19` (≈259K gates) | 0.379 | 0.490 | 0.489 | 0.421 | 0.793 |
| ISCAS-85 `c6288` (16×16 multiplier) | −1.125 | −0.039 | 0.217 | 0.535 | 0.718 |

`b19` is close to the ITC'99 training distribution and generalizes reasonably on
the timing-regression and DRC heads. `c6288` is a dense, highly-regular
multiplier structurally unlike anything in training — the model fails to
generalize to it (negative R² on slack/arrival), which is an honest
out-of-distribution limitation worth stating plainly rather than hiding. DRC
prediction holds up best across both (AUC 0.72–0.79).

## Notes

- A random-split (in-distribution) run early-stopped at epoch 14 with similar or
  slightly worse numbers (slack R² 0.37, congestion unstable); the
  by-circuit-split model above is both better-trained and the more defensible
  metric, so it is the reported model.
- Labels are this repository's STA / RUDY-congestion / DRC estimators — a fast,
  deterministic surrogate, not a commercial signoff flow. The numbers measure
  how well the GNN reproduces that surrogate on real netlists.
- Backbone ablations and baseline (MLP/GCN/GraphSAGE) comparisons are on the
  `research` branch.

# NetSTA Baseline Comparison

_Dataset: 1000 circuits, seed 42, 700/150/150 split._

| Model | Params | Slack MSE ↓ | Slack R² ↑ | CP Accuracy ↑ | CP F1 ↑ | CP AUC ↑ | Train Time | Infer Time/circuit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Linear Regression | -- | 0.0566 | 0.590 | 84.2% | 0.480 | 0.942 | 7.6s | 0.01ms |
| Random Forest | -- | 0.0567 | 0.590 | 84.2% | 0.479 | 0.942 | 4.4s | 0.66ms |
| MLP | 78K | 0.0425 | 0.692 | 85.8% | 0.505 | 0.948 | 143.7s | 0.12ms |
| GCN | 406K | 0.0483 | 0.650 | 86.9% | 0.523 | 0.954 | 113.3s | 0.29ms |
| GraphSAGE | 668K | 0.0486 | 0.648 | 86.3% | 0.513 | 0.951 | 51.0s | 0.19ms |
| **NetSTA** | 472K | 0.0494 | 0.642 | 85.8% | 0.505 | 0.953 | 118.6s | 0.48ms |

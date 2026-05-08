# NetSTA Robustness Analysis

_Trained NetSTA across 5 seeds: [42, 123, 456, 789, 1024]._
_Each: 1000 circuits, 200 epochs (early-stop), seed-specific split._

| Metric | Seed 42 | Seed 123 | Seed 456 | Seed 789 | Seed 1024 | Mean ± Std |
| --- | --- | --- | --- | --- | --- | --- |
| Slack MSE | 0.0480 | 0.0425 | 0.0526 | 0.0645 | 0.0474 | 0.0510 ± 0.0075 |
| Slack R² | 0.653 | 0.699 | 0.613 | 0.552 | 0.674 | 0.638 ± 0.052 |
| CP Accuracy | 86.4% | 85.1% | 87.2% | 86.7% | 85.5% | 86.2% ± 0.8% |
| CP F1 | 0.512 | 0.478 | 0.521 | 0.525 | 0.498 | 0.507 ± 0.017 |
| CP AUC-ROC | 0.952 | 0.953 | 0.955 | 0.953 | 0.952 | 0.953 ± 0.001 |

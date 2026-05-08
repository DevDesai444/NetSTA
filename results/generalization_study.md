# NetSTA Generalization Study

Tests how well NetSTA transfers across three distribution shifts.

| Generalization Test | Train R² | Test R² | Δ R² | Train CP Acc | Test CP Acc | Δ Acc |
| --- | --- | --- | --- | --- | --- | --- |
| Size (small→large) | 0.597 | 0.229 | -0.369 | 85.4% | 82.7% | -2.7% |
| Topology (subset→full) | 0.515 | 0.458 | -0.058 | 86.1% | 82.0% | -4.0% |
| Depth (shallow→deep) | 0.706 | 0.274 | -0.432 | 87.0% | 80.2% | -6.7% |

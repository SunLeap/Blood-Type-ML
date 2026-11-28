# Model performance summary

Run: `primary` | search candidates per family: 20 | runtime: not measured (notebook run)

Reported test protocol: **walk-forward** (models retrained every 4 weeks on every outcome known at that time). Static scores are shown for comparison with version 1.

## Classification (next-week risk)

Selected by time-series CV macro-F1: **XGBoost** (feature set: extended, 47 numeric features, weighting: sqrt_balanced)

| Partition | Accuracy | Macro-F1 | High recall |
|---|---:|---:|---:|
| Train | 90.1% | 0.909 | 97.9% |
| Time-series CV | 59.4% | 0.522 | 34.4% |
| Validation (static) | 58.5% | 0.540 | 42.9% |
| Validation (walk-forward) | 58.7% | 0.548 | 44.9% |
| Test (static: one model fitted before the test period, version-1 protocol) | 58.9% | 0.436 | 14.3% |
| **Test (walk-forward, standard rule)** | **56.3%** | **0.426** | **14.3%** |
| Test (high-recall alert mode, P(High) >= 0.175) | 52.1% | 0.463 | 50.0% |

Test 95% bootstrap intervals (standard mode): accuracy 50.4%-61.8%, macro-F1 0.368-0.484, High recall 3.3%-28.1%

Proposal targets: accuracy >= 80.0% -> NOT met; High recall >= 75.0% -> NOT met (standard), NOT met (high-recall mode).

### Before vs after optimization (test period)

| Model | Stage | CV macro-F1 | Test accuracy | Test macro-F1 | Test High recall |
|---|---|---:|---:|---:|---:|
| Naive persistence | naive baseline | 0.486 | 54.2% | 0.475 | 33.3% |
| Decision Tree (original settings) | before optimization | 0.470 | 48.7% | 0.401 | 28.6% |
| Decision Tree (optimized) | after optimization | 0.486 | 60.5% | 0.460 | 16.7% |
| Random Forest (original settings) | before optimization | 0.511 | 59.4% | 0.516 | 35.7% |
| Random Forest (optimized) | after optimization | 0.522 | 55.2% | 0.449 | 21.4% |
| XGBoost (original settings) | before optimization | 0.507 | 55.2% | 0.453 | 21.4% |
| XGBoost (optimized) | after optimization | 0.522 | 56.3% | 0.426 | 14.3% |
| Ensemble (RF + XGBoost) (optimized) | after optimization | 0.514 | 56.0% | 0.435 | 16.7% |

## Regression (next-week demand)

Selected by time-series CV MAE: **XGBoost** (feature set: all_external, 89 numeric features)

Test MAE 170.4 units (95% CI 151.4-191.1), RMSE 235.4, MAPE 15.34% (static protocol: MAE 169.5; walk-forward validation MAE 119.3)

| Model | Stage | CV MAE | Validation MAE | Test MAE | Test RMSE | Test MAPE % |
|---|---|---:|---:|---:|---:|---:|
| Naive persistence | naive baseline | 139.9 | 135.0 | 195.2 | 273.3 | 17.40 |
| Decision Tree (original settings) | before optimization | 143.5 | 171.3 | 201.2 | 277.7 | 17.60 |
| Decision Tree (optimized) | after optimization | 136.1 | 140.2 | 191.4 | 262.2 | 16.76 |
| Random Forest (original settings) | before optimization | 123.3 | 127.7 | 169.9 | 237.9 | 15.36 |
| Random Forest (optimized) | after optimization | 122.0 | 129.9 | 169.4 | 238.4 | 15.26 |
| XGBoost (original settings) | before optimization | 121.0 | 126.1 | 167.4 | 232.1 | 15.32 |
| XGBoost (optimized) | after optimization | 119.4 | 127.0 | 170.4 | 235.4 | 15.34 |
| Ensemble (RF + XGBoost) (optimized) | after optimization | 119.8 | 127.4 | 169.0 | 235.6 | 15.23 |

## Next-week forecast (from the last observed week)

| Blood type | Last observed | Forecast week | Observed risk | Predicted risk | P(High) | Predicted demand |
|---|---|---|---|---|---:|---:|
| A | 2024-06-17 | 2024-06-24 | Medium | Medium | 0.01 | 1,570 |
| B | 2024-06-17 | 2024-06-24 | Medium | Low | 0.02 | 1,504 |
| O | 2024-06-17 | 2024-06-24 | Medium | Medium | 0.04 | 1,933 |
| AB | 2024-06-17 | 2024-06-24 | Medium | Low | 0.00 | 420 |

Forecasts are model estimates, not observations. Results are retrospective (Zhejiang data) and do not validate operational or clinical use.
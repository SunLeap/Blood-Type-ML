# Model performance summary

Run: `corrected` | search candidates per family: 20 | runtime: 27.4 min

Reported test protocol: **walk-forward** (models retrained every 4 weeks on every outcome known at that time). Static scores are shown for comparison with version 1.

## Classification (next-week risk)

Selected by time-series CV macro-F1: **XGBoost** (feature set: original, 34 numeric features, weighting: sqrt_balanced)

| Partition | Accuracy | Macro-F1 | High recall |
|---|---:|---:|---:|
| Train | 88.6% | 0.895 | 97.6% |
| Time-series CV | 60.1% | 0.527 | 32.9% |
| Validation (static) | 60.3% | 0.563 | 49.0% |
| Validation (walk-forward) | 58.0% | 0.544 | 44.9% |
| Test (static: one model fitted before the test period, version-1 protocol) | 57.3% | 0.422 | 11.9% |
| **Test (walk-forward, standard rule)** | **58.9%** | **0.457** | **16.7%** |
| Test (high-recall alert mode, P(High) >= 0.175) | 55.0% | 0.486 | 50.0% |

Test 95% bootstrap intervals (standard mode): accuracy 52.4%-65.1%, macro-F1 0.398-0.515, High recall 6.9%-27.7%

Proposal targets: accuracy >= 80.0% -> NOT met; High recall >= 75.0% -> NOT met (standard), NOT met (high-recall mode).

### Before vs after optimization (test period)

| Model | Stage | CV macro-F1 | Test accuracy | Test macro-F1 | Test High recall |
|---|---|---:|---:|---:|---:|
| Naive persistence | naive baseline | 0.486 | 54.2% | 0.475 | 33.3% |
| Decision Tree (original settings) | before optimization | 0.470 | 48.7% | 0.401 | 28.6% |
| Decision Tree (optimized) | after optimization | 0.486 | 60.5% | 0.460 | 16.7% |
| Random Forest (original settings) | before optimization | 0.511 | 59.4% | 0.516 | 35.7% |
| Random Forest (optimized) | after optimization | 0.522 | 55.2% | 0.449 | 21.4% |
| XGBoost (original settings) | before optimization | 0.509 | 56.5% | 0.471 | 23.8% |
| XGBoost (optimized) | after optimization | 0.527 | 58.9% | 0.457 | 16.7% |
| Ensemble (RF + XGBoost) (optimized) | after optimization | 0.520 | 57.3% | 0.445 | 16.7% |

## Regression (next-week demand)

Selected by time-series CV MAE: **XGBoost** (feature set: selected, 41 numeric features)

Test MAE 168.4 units (95% CI 149.7-187.9), RMSE 232.4, MAPE 15.44% (static protocol: MAE 170.7; walk-forward validation MAE 117.8)

| Model | Stage | CV MAE | Validation MAE | Test MAE | Test RMSE | Test MAPE % |
|---|---|---:|---:|---:|---:|---:|
| Naive persistence | naive baseline | 139.9 | 135.0 | 195.2 | 273.3 | 17.40 |
| Decision Tree (original settings) | before optimization | 143.5 | 171.3 | 201.2 | 277.7 | 17.60 |
| Decision Tree (optimized) | after optimization | 136.1 | 140.2 | 191.4 | 262.2 | 16.76 |
| Random Forest (original settings) | before optimization | 123.3 | 127.7 | 169.9 | 237.9 | 15.36 |
| Random Forest (optimized) | after optimization | 122.1 | 131.6 | 170.4 | 237.6 | 15.46 |
| XGBoost (original settings) | before optimization | 120.4 | 127.5 | 165.3 | 229.3 | 15.06 |
| XGBoost (optimized) | after optimization | 119.1 | 125.3 | 168.4 | 232.4 | 15.44 |
| Ensemble (RF + XGBoost) (optimized) | after optimization | 119.5 | 127.6 | 168.9 | 233.8 | 15.40 |

## Next-week forecast (from the last observed week)

| Blood type | Last observed | Forecast week | Observed risk | Predicted risk | P(High) | Predicted demand |
|---|---|---|---|---|---:|---:|
| A | 2024-06-17 | 2024-06-24 | Medium | Medium | 0.01 | 1,576 |
| B | 2024-06-17 | 2024-06-24 | Medium | Low | 0.02 | 1,550 |
| O | 2024-06-17 | 2024-06-24 | Medium | Medium | 0.08 | 1,933 |
| AB | 2024-06-17 | 2024-06-24 | Medium | Low | 0.01 | 433 |

Forecasts are model estimates, not observations. Results are retrospective (Zhejiang data) and do not validate operational or clinical use.
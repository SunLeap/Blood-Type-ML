# Blood-Type-ML — Blood Shortage Risk Prediction (University Research Prototype) — version 2

ITM-390 Machine Learning final research project. A local, **no-backend** ML project using the real
`Zhejiang_blood_donation.csv` dataset. It compares Decision Tree, Random Forest and XGBoost for
**next-week** shortage-risk labels (High / Medium / Low) and for next-week demand, per blood type
(A / B / O / AB). A Streamlit dashboard shows forecasts, observed history and model evaluation.

**Important:** this is not a live system, not clinical decision support and not a validated inventory
predictor. The latest observation is 17 June 2024, so the demo forecast is for about 24 June 2024,
even when the app is run years later.

---

## 1. Install requirements (Windows + VS Code)

Use **Python 3.11, 3.12 or 3.13**.

```bat
REM CMD: change into the project folder (the one that contains app.py)
cd path\to\Blood-Type-ML

REM CMD: create and activate an isolated environment
py -3.12 -m venv .venv
.venv\Scripts\activate.bat

REM CMD: install every required library
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

In VS Code: *Ctrl+Shift+P → Python: Select Interpreter → .venv*. For notebooks, pick the `.venv` kernel.

macOS / Linux: `python3.12 -m venv .venv && source .venv/bin/activate`, then the same `pip` commands.

## 2. One command: prepare data, train, evaluate, report

```bat
REM CMD: full pipeline (about 20-35 minutes; walk-forward retraining is the slow part)
python run_pipeline.py

REM CMD: faster run with fewer tuning candidates (about 10 minutes)
python run_pipeline.py --quick
```

This (1) cleans the raw CSV and builds features, (2) tunes and trains the classifiers, (3) tunes and trains
the regressors, (4) evaluates them on the test period once, saves everything to `models/` and `results/`, and
prints `results/summary_report.md`.

Individual steps are still available:

```bat
python -m src.data_pipeline --clean-only        REM data/cleaned_weekly.csv only
python -m src.data_pipeline                     REM cleaned + processed features
python -m src.prediction --train-all            REM both tasks (add --quick for speed)
python -m src.prediction --train-classification
python -m src.prediction --train-regression
python -m src.prediction --predict              REM forecast from the last observed week
```

## 3. Dashboard

```bat
python -m streamlit run app.py
```

Open the URL Streamlit prints (usually http://localhost:8501). Press **Ctrl+C** to stop it.

* **Forecast cards** for all four blood types, with coloured HIGH / MEDIUM / LOW banners, P(High) bars and
  predicted demand. A summary banner at the top lists any blood type at HIGH risk.
* **Alert rule**: *Standard* (most probable class) or *High-recall alert mode* (see section 6).
* **Observed history**, kept in its own section: supply / demand / collection chart, coverage chart with the
  80% and 100% cutoffs, collection-efficiency indicators and chart, and the historical risk distribution.
* **Dataset selection**: the primary Zhejiang dataset, a **CSV upload**, or any saved experiment in `runs/`.
* **Model evaluation**: headline test scores, comparison tables, confusion matrices, observed-vs-predicted
  demand, before-vs-after tables and every saved figure.

## 4. External datasets: training vs. predicting

A compatible CSV needs `date` plus `collected_X`, `demand_X` and `supply_X` for X = A, B, O, AB. Totals,
platelet columns, `team_total` and `Individual_donate` are optional: missing totals are derived from the
per-type columns, and other missing columns are left blank for the imputer. Dates may be `M/D/YYYY` or `YYYY-MM-DD`.

| Goal | Command | What gets written |
|---|---|---|
| Check a file only | `python -m src.data_pipeline --validate other.csv` | nothing |
| **Use the existing trained models** on new data | `python run_pipeline.py --predict-only other.csv` | `results/external/<name>_<time>/` only |
| **Train new models** on a different dataset | `python run_pipeline.py --data other.csv --run-name my_test` | `runs/my_test/` only |

The original dataset, `data/processed_data.csv`, `models/` and the primary `results/` are **never overwritten**
by either external option. The dashboard offers the same two modes after an upload.
Training needs at least 60 dated rows. Prediction needs at least 5.

## 5. Notebooks

```bat
python -m jupyter lab
```

Run in order (**Run → Run All Cells**):

| Notebook | Contents |
|---|---|
| `00_Environment_Check.ipynb` | Optional: confirms the kernel and library versions |
| `01_Data_Preparation.ipynb` | Raw inspection, validation report, cleaning, EDA plots, drift, features, next-week labels, leakage proof |
| `02_Model_Training.ipynb` | Chronological split + CV folds; **A. Classification** and **B. Regression**: tuning logs, class-weighting study, feature-set trials, feature importance (retrains, a few minutes) |
| `03_Model_Evaluation.ipynb` | Train/CV/validation/test comparison, confusion matrices, per-class and per-blood-type results, High-recall analysis, bootstrap intervals, observed vs predicted demand, before vs after, forecast, external-data demo (read-only) |

## 6. Methodology (the scientific meaning is unchanged)

* **Coverage** = observed supply / observed demand, per blood type.
* **Risk**: HIGH = coverage < 0.80, MEDIUM = 0.80 to < 1.00, LOW ≥ 1.00 (unchanged).
* **Collection efficiency** = supplied / collected. It is a descriptive proxy.
* **Features (week t)**: current demand, supply, collection, platelets and totals; 1/2/4-week lags; 4-week rolling
  means; calendar features. *New candidates*: 8/12-week coverage means, coverage volatility, exponentially weighted
  coverage, collected/demand ratios, demand and supply shares, and the **calendar of the target week** (week of year,
  Spring Festival proximity). The calendar is known in advance, so it does not leak.
* **Targets (t+1)**: the next observation's risk and demand. A target is kept only if the next record is 5–9 days later.
  Targets are never features. `tests/` includes a test that rewrites every future week and checks that no earlier
  feature changes.
* **Split**: chronological by target date. The earliest 70% is train, the next 15% is validation and the final 15% is test.
  All four blood types for a date stay in the same part. There is no shuffling.
* **Tuning** (new): random search over hyper-parameters (`n_estimators`, `max_depth`, `min_samples_leaf`,
  `learning_rate`, sampling, regularisation, the XGBoost loss) **and class weighting** (`none`, `balanced`,
  `sqrt_balanced`, `balanced_high_boost`). Each candidate is scored by **4-fold expanding-window time-series CV**
  inside train + validation. The original settings are always candidate 0.
* **Feature selection** (new): permutation importance is measured on an early holdout that comes *before* all CV
  blocks. The reduced set is kept only if CV improves. Otherwise the original or extended set is used.
* **Model choice**: best CV macro-F1 (classification) and best CV MAE (regression) among Decision Tree, Random
  Forest, XGBoost and (version 2) an **average of the tuned Random Forest and XGBoost**. The test period is scored
  once per model, after every choice has been made.
* **Test protocol (version 2: walk-forward)**: in version 1 one model was fitted before the test period and then
  frozen for 20 months. In version 2 each model is **retrained every 4 weeks** (`WALK_FORWARD_STEP` in
  `src/config.py`) on every outcome known at that moment (`target_date <= forecast origin`), which is how a blood
  bank would use it. The protocol was fixed before looking at test results. Static scores are still saved
  (`test_static_*`), and walk-forward is also scored on the validation weeks (`validation_wf_*`) so its effect can
  be judged without the test period. A test checks that no refit ever sees a future outcome.
* **Stock proxy and cross-type features (version 2)**: net flow (collected − supplied) summed over 4/13/26 weeks and
  divided by recent demand (the data has no inventory column, so this is an approximation); lagged collection
  relative to demand; and same-week context from all four blood types (each type's coverage, total coverage,
  lowest type coverage, number of types below 80% now and over 4 weeks). They compete with the older feature sets
  (`original`, `extended`, `all`, importance-`selected`) by CV.
* **Forecast model (version 2)**: the saved models are refit on every labeled week (through June 2024) before
  forecasting the week after the last observation. Version 1 saved a model fitted only up to October 2022.
* **Not used**: next-week demand as an input. Demand values are round hundreds and may be orders planned in advance;
  if the data provider confirms they are known before the week starts, adding them would be legitimate and could
  help a lot. Until then it would be leakage.
* **High-recall alert mode** (new): HIGH whenever P(High) ≥ a threshold. The threshold is the highest one that reaches
  75% HIGH recall on the CV out-of-fold predictions. It is never tuned on test.
* **Baselines** (new): *naive persistence* (next week = this week) and each family's *original settings*, both
  evaluated with the same protocol.
* **Investigated and rejected**: recency-weighted training, meant to handle drift, lowered CV macro-F1 for all three
  families. Predicting next-week coverage by regression and then thresholding it gave lower High recall. Neither is used.
* **Duplicate dates** are averaged, not summed (13 dates). This is an explicit assumption. Section 7b shows most
  of them are typing errors and offers a corrected copy as a separate experiment.
* **Missing data**: NBSP, `#VALUE!` and negative values become missing. Medians are imputed inside the pipeline from
  training data only. Targets are never imputed.

## 7. Results on the Zhejiang data (`python run_pipeline.py`, 20 candidates per family)

All test scores come from the final 15% of target weeks (Oct 2022 – Jun 2024, 352 rows, 36 HIGH weeks). Version 2
reports the **walk-forward** protocol (retrained every 4 weeks); `results/summary_report.md` has every number.

**Version 1 vs version 2 (selected models)**

| | Version 1 | Version 2 |
|---|---:|---:|
| Selected classifier | XGBoost | Random Forest (v2 features, `balanced_high_boost` weighting) |
| Test accuracy | 56.5% | 56.5% |
| Test macro-F1 | 0.481 | **0.490** |
| Test HIGH recall (standard rule) | 27.8% | **38.9%** |
| Test HIGH recall (high-recall alert mode) | 52.8% (accuracy 49.1%) | 52.8% (accuracy **53.4%**) |
| Selected regressor / test MAE | XGBoost / 171.1 units | XGBoost / **169.7 units** (MAPE 15.0%) |

**Classification, version 2 (test, walk-forward)**

| Model | Stage | CV macro-F1 | Test accuracy | Test macro-F1 | Test High recall |
|---|---|---:|---:|---:|---:|
| Naive persistence | baseline | 0.501 | 55.4% | 0.485 | 36.1% |
| Decision Tree | original → optimized | 0.472 → 0.490 | 49.4% → 55.4% | 0.431 → 0.484 | 36.1% → 36.1% |
| **Random Forest (selected)** | original → optimized | 0.522 → 0.538 | 59.1% → 56.5% | 0.492 → 0.490 | 27.8% → 38.9% |
| XGBoost | original → optimized | 0.528 → 0.536 | 58.2% → 56.2% | 0.491 → 0.480 | 27.8% → 30.6% |
| Ensemble (RF + XGBoost) | optimized | 0.521 | 56.5% | 0.483 | 33.3% |

**Regression, version 2 (test, walk-forward)**

| Model | Stage | CV MAE | Test MAE | Test RMSE | Test MAPE |
|---|---|---:|---:|---:|---:|
| Naive persistence | baseline | 141.7 | 204.7 | 288.6 | 17.9% |
| Decision Tree | original → optimized | 154.7 → 136.9 | 205.7 → 180.0 | 288.5 → 257.5 | 18.0% → 15.8% |
| Random Forest | original → optimized | 128.7 → 126.9 | 175.3 → 174.8 | 248.1 → 248.6 | 15.3% → 15.3% |
| **XGBoost (selected)** | original → optimized | 128.4 → 124.5 | 169.7 → 169.7 | 238.5 → 238.2 | 15.1% → 15.0% |
| Ensemble (RF + XGBoost) | optimized | 124.7 | 171.6 | 242.4 | 15.1% |

**What the version 2 changes did**

* **Walk-forward retraining** helped on the validation weeks, which were not used for any choice it affects: for
  the selected classifier, validation accuracy rose from 52.0% (static) to 54.8%, and walk-forward validation MAE was
  119.4 vs 125.0 static. On the test weeks the two protocols are within noise (static 58.0%, walk-forward 56.5%).
  Walk-forward stays the reported protocol because it was chosen in advance as the realistic one; switching after
  seeing test scores would be selecting on the test set.
* **New features**: CV picked the `all` feature set (with stock-proxy and cross-type features) for the Random Forest
  and XGBoost classifiers. The number of blood types below 80% coverage separates the classes clearly (on average 2.1
  of 4 types before a HIGH week vs 0.3 before a LOW week).
* **Ensemble**: it did not beat the single models on CV, so it was not selected.
* **Net effect**: test accuracy did not rise. HIGH-risk recall improved from 27.8% to 38.9%, and the alert mode now
  reaches the same 52.8% HIGH recall with 4 points more accuracy. The 95% bootstrap interval for test accuracy is
  49.7%–62.8%, so differences of 1–2 points are not meaningful.

**The proposal targets (80% accuracy, 75% HIGH recall) are still not met.** The selected model scores 84.5% on
training weeks but about 55% on any later period, so the limit is the predictability of the label, not model
capacity. Likely reasons: strong temporal drift (HIGH share from about 1% to 26% by year), labels that flip on
small supply changes near the 0.80 and 1.00 cutoffs, no inventory column, and a short test period. The most promising
next step is data, not tuning: confirm whether next-week demand (round hundreds) is known in advance, and obtain
real inventory levels.

Version 1 metrics are kept in `results/previous_version_v1/`, and the original code's in `results/original_baseline_run/`.

## 7b. Data-entry errors and the corrected dataset (separate experiment)

No public dataset with the same structure (weekly collection, demand and supply by blood type) was found to merge
with: searches covered Kaggle, GitHub, Mendeley, Chinese sources, Malaysia's open data (donations only) and the
2026 Frontiers in Public Health Zhejiang study, whose supplementary data come from the same Zhejiang Provincial
Blood Center and the same weeks. Instead, a close reading of the original file found typing errors:

* **Dates**: year-end weeks typed with the next year (`12/31/2013` between 12/24/2012 and 01/07/2013, 7 cases),
  four October blocks typed as January (`01/05/2015 ... 1/26/2015` between 9/28/2015 and 11/02/2015, 18 rows), and
  single mistyped days (`04/02/2015` between 4/13 and 4/27). These created the 13 "duplicate dates" that the
  pipeline averages, which mixed January with October data and left October empty.
* **Values**: copy errors (`supply_AB` holding the O value), impossible values (`supply_A = 5`, `supply_total = 0`,
  `team_total = 26511`), and AB demand of 1,300 for four weeks in 2013 (about four times normal), which created
  fake HIGH labels.

`src/data_repair.py` fixes a date only when the weekly sequence confirms it, sets clear value errors to **missing**
(never guessed), logs every change in `results/data_repair_log.csv` (63 changes: 36 dates, 27 values), and writes a
**separate** file, `data/Zhejiang_blood_donation_corrected.csv`. The original file is never changed and remains the
primary research dataset.

```bat
python run_pipeline.py --corrected     REM repair + full training in runs/corrected/ (dashboard: "Saved experiment: corrected")
```

| Test period (walk-forward) | Original data | Corrected data |
|---|---:|---:|
| Unique weeks / repeated dates | 626 / 13 | 639 / 0 |
| Selected classifier | Random Forest | XGBoost |
| Test accuracy | 56.5% | **58.9%** |
| Accuracy gain over naive persistence | +1.1 pts | **+4.7 pts** |
| Walk-forward validation accuracy | 54.8% | **58.0%** |
| Test macro-F1 | **0.490** | 0.457 |
| Test HIGH recall (standard / alert mode) | **38.9%** / 52.8% | 16.7% / 50.0% |
| Test demand MAE | 169.7 | **168.4** units |

The corrected data gives consistently higher accuracy on CV, validation and test (about 3–5 points) and a clearer gain
over the naive baseline, but its selected model catches fewer HIGH weeks with the standard rule. The two test periods
are not identical (the corrected file has 13 more weeks, 42 vs 36 HIGH test weeks), and the HIGH-recall interval is
wide (7%–28%), so treat this as a direction, not a final verdict. Whether to adopt the corrected copy as the primary
dataset is a research decision; ideally confirm the fixes with the data provider first.

## 7c. External weekly data: holidays, school breaks, COVID restrictions (optional)

`data/Zhejiang_blood_donation_corrected_enriched.csv` is the corrected weekly data plus external weekly columns
(official public holidays and make-up workdays, approximate school breaks, COVID-19 restriction stringency) and the
quarterly Data_Sheet_2 context. Sources, column meanings and leakage notes: `data/external/DATA_SOURCES.md`.
`data/Zhejiang_blood_donation_enriched.csv` is the same on the original file (prefer the corrected one).

How the code uses it: when the input file contains the external columns, `src/data_pipeline.py` copies them to every
blood-type row, and model tuning tries one extra feature set, `all_external`. It is used only if it wins the
time-series CV (see `results/*_feature_set_trials.csv`). The quarterly GDP/health/population columns load but are not
used as features. Files without these columns, including the primary dataset, behave exactly as before.

```bat
REM Train on the enriched data as a separate experiment (results in runs\enriched\; primary results untouched)
python run_pipeline.py --data data\Zhejiang_blood_donation_corrected_enriched.csv --run-name enriched

REM Faster check first (fewer tuning candidates, about 10 minutes)
python run_pipeline.py --data data\Zhejiang_blood_donation_corrected_enriched.csv --run-name enriched_quick --quick

REM Use already-trained models on the enriched file without training
python run_pipeline.py --predict-only data\Zhejiang_blood_donation_corrected_enriched.csv
```

In the dashboard, the new run appears as "Saved experiment: enriched"; compare it with the primary and
"corrected" experiments in the Model evaluation section. Notebook `03_Model_Evaluation.ipynb` section 4c compares
original and corrected; the same cell pattern works with `ProjectPaths.for_run('enriched')`.

**Forecasting new weeks.** Append each new week's blood row to the CSV, then add that week's external columns:
copy the matching `date` row from `data/external/weekly_external_features_2012_2026.csv` (it already covers every
week to December 2026), or rebuild everything with

```bat
python tools\build_weekly_features.py <holiday_json_folder> <OxCGRT_compact_national_v1.csv> <weekly.csv> [<context.csv>] <output_folder>
```

(holiday JSON files: https://github.com/NateScarlet/holiday-cn; OxCGRT: https://github.com/OxCGRT/covid-policy-dataset).
For 2027 onward, add the new holiday file when the State Council publishes it.

## 8. Project structure

```text
Blood-Type-ML/
├── run_pipeline.py              # single entry point: data -> training -> evaluation -> report
├── app.py                       # Streamlit dashboard
├── src/
│   ├── config.py                # paths, thresholds, seeds, tuning settings (one place)
│   ├── data_pipeline.py         # load, validate, clean, features, labels (was preprocessing + feature_engineering)
│   ├── data_repair.py           # corrected copy of the raw CSV: date typos, entry errors (logged)
│   ├── prediction.py            # splits, CV tuning, training, evaluation, saving, inference, external prediction
│   └── plots.py                 # evaluation figures saved to results/figures/
├── tools/build_weekly_features.py  # builds the external weekly columns (holidays, school breaks, COVID)
├── data/
│   ├── Zhejiang_blood_donation.csv   # original research dataset (never modified)
│   ├── Zhejiang_blood_donation_corrected.csv   # generated by src/data_repair.py
│   ├── Zhejiang_blood_donation_corrected_enriched.csv   # + holidays, school breaks, COVID, quarterly context
│   └── external/                     # weekly_external_features_2012_2026.csv, DATA_SOURCES.md, Data_Sheet_2
│   ├── cleaned_weekly.csv            # generated
│   └── processed_data.csv            # generated
├── models/
│   ├── best_classifier.joblib, best_regressor.joblib
│   └── model_metadata.json           # chosen params, features, threshold, data hash, library versions
├── results/
│   ├── summary_report.md, training_config.json, next_week_forecast.csv
│   ├── classification_* / regression_*   # metrics, before/after, search logs, importance, predictions, CIs
│   ├── figures/*.png
│   ├── original_baseline_run/        # metrics from the pre-refactor code
│   └── previous_version_v1/          # metrics and summary of version 1 (for the v1 vs v2 comparison)
├── notebooks/  00_Environment_Check, 01_Data_Preparation, 02_Model_Training, 03_Model_Evaluation
├── tests/      test_data_pipeline.py, test_prediction.py
├── backup/     Blood-Type-ML_backup_original_2026-11-23.zip   # the project before refactoring
│               Blood-Type-ML_v1_2026-11-23.zip                # version 1 of the refactored project
├── runs/       # separate experiments; runs/corrected/ = training on the corrected copy
├── requirements.txt, pytest.ini, .gitignore, README.md
```

## 9. Tests

```bat
python -m pytest
```

45 tests (about 1 minute) cover: risk thresholds, cleaning (NBSP, `#VALUE!`, negatives, duplicate dates), schema
validation of external files, documented dataset properties, lag and target alignment, the future-rewrite leakage
test, chronological split and CV-fold ordering, weighting strategies, the high-recall rule, training/saving/metadata
into a temporary folder (checking that primary models are untouched), next-week prediction, external prediction,
the version 2 features, walk-forward retraining never seeing a future outcome, the ensemble average, and the
data-repair rules (date fixes confirmed by the sequence, errors set missing, original never overwritten), and the
optional external columns (same weekly value for all blood types; the extra feature set is tried only when present).

## 10. Troubleshooting

* **`ModuleNotFoundError`**: activate `.venv`, then run `python -m pip install -r requirements.txt`.
* **`No module named src`**: run commands from the project root (where `app.py` is), not from `notebooks/`.
* **`Models are missing`**: run `python run_pipeline.py`.
* **PowerShell blocks activation**: use `cmd.exe` and `.venv\Scripts\activate.bat`.
* **Model file fails to load after upgrading libraries**: retrain. Never load joblib files from untrusted sources,
  because loading them can execute code.

## 11. Recommended demonstration order

1. `01_Data_Preparation`: data quality, drift, leakage proof.
2. Explain the one-week label shift and why chronological splitting and time-series CV matter.
3. `02_Model_Training`: tuning logs, weighting study, feature-set trials.
4. `03_Model_Evaluation`: before vs after, confusion matrices, High-recall trade-off, confidence intervals.
5. Dashboard: switch blood types and alert modes, and upload a CSV.
6. Limitations: cleaning assumptions, modest classification performance, retrospective dates, no clinical validation,
   and Zhejiang ≠ Cambodia.

# Blood-Type-ML

# Blood Shortage ML — University Research Prototype

This is a complete, local, **no-backend** ML project using the real `Zhejiang_blood_donation.csv` dataset. It compares three classifiers for **next-week** shortage-risk labels (High / Medium / Low) and three regressors for next-week demand (A / B / O / AB). A Streamlit dashboard demonstrates the model comparisons, observational indicators, forecasts and risk alerts.

**Important:** Not a live system, not clinical decision support, and not a validated inventory predictor. The latest data observation is June 17, 2024; the demo forecast is for approximately June 24, 2024, even when the app is run years later.

## 1. Install requirements

Use **Python 3.11 or 3.12**, Visual Studio Code (recommended) or any terminal, and an internet connection for the initial `pip install`.

### Windows Command Prompt (CMD)

```bat
REM CMD: change into the unzipped project folder
cd path\to\blood-shortage-ml

REM CMD: create an isolated Python environment
py -3.12 -m venv .venv

REM CMD: activate the environment (Command Prompt)
.venv\Scripts\activate.bat

REM CMD: upgrade the Python package installer
python -m pip install --upgrade pip

REM CMD: install every required library
python -m pip install -r requirements.txt
```

If `py -3.12` is not installed, try `py -3.11` or `python` (ensure that `python --version` prints 3.11 or 3.12).

### macOS / Linux terminal

```bash
# CMD: enter the extracted project directory
cd /path/to/blood-shortage-ml

# CMD: create a project-specific environment
python3.12 -m venv .venv

# CMD: activate that environment
source .venv/bin/activate

# CMD: install dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If `python3.12` is unavailable, use `python3.11` or `python3` with version 3.11–3.12.

## 2. Generate data and train your models

**Run commands from the directory containing `app.py`** with your environment activated:

```bash
# CMD: optional -- create cleaned_weekly.csv for inspection
python -m src.preprocessing

# CMD: generate long-format features and leakage-safe next-week targets
python -m src.feature_engineering

# CMD: compare ALL models, evaluate them, and save the best classifier and regressor
python -m src.prediction --train-all

# CMD: print the latest-observation forecasts for blood types A, B, O, AB
python -m src.prediction --predict

# CMD: start the interactive local dashboard
python -m streamlit run app.py
```

Open the local URL printed by Streamlit (usually **http://localhost:8501**). Press **Ctrl+C** to stop the app.

You can retrain only one type of model:

```bash
# CMD: compare and save classifier models only
python -m src.prediction --train-classification

# CMD: compare and save regression models only
python -m src.prediction --train-regression
```

**Note:** Training automatically rebuilds `processed_data.csv` from the supplied source CSV. You do not need to run the first two commands each time, but doing so is helpful for understanding the process. Already-trained models and comparison CSVs are included for an immediate demonstration and can be regenerated.

## 3. Explore the notebooks

```bash
# CMD: launch JupyterLab and open the five notebooks under notebooks/
python -m jupyter lab
```

In JupyterLab, select the Python environment kernel and choose **Run → Run All Cells** in this order:

| Notebook | What it demonstrates |
|---|---|
| `01_EDA.ipynb` | Raw data inspection, data quality, historical plots |
| `02_Preprocessing.ipynb` | Cleaning, blood-type conversion, 1/2/4-week lags, 4-week rolling features, next-week labels |
| `03_Classification.ipynb` | Model comparisons, classification report, confusion matrix |
| `04_Regression.ipynb` | Demand forecasting, forecast scatter plot, errors by blood type |
| `05_Model_Evaluation.ipynb` | Full score tables, comparisons, and prediction demonstration |

If Jupyter asks for a kernel, select **Python 3 (ipykernel)** from your `.venv`. You can also use VS Code and install the Python + Jupyter extensions, then select `.venv` in the interpreter/kernel picker.

## 4. What each file does

```text
blood-shortage-ml/
├── data/
│   ├── Zhejiang_blood_donation.csv   # Your original input data
│   └── processed_data.csv           # Auto-generated wide→long feature dataset
├── notebooks/
│   ├── 01_EDA.ipynb                  # Inspect raw data and create figures
│   ├── 02_Preprocessing.ipynb        # Demonstrate cleaning and features
│   ├── 03_Classification.ipynb       # Compare shortage-risk classifiers
│   ├── 04_Regression.ipynb           # Compare demand forecasts
│   └── 05_Model_Evaluation.ipynb     # Compare final results and forecasts
├── models/
│   ├── best_classifier.joblib        # Best-by-validation saved classifier
│   └── best_regressor.joblib         # Best-by-validation saved regressor
├── results/
│   ├── classification_metrics.csv    # Test/validation classification results
│   └── regression_metrics.csv        # Test/validation regression results
├── src/
│   ├── __init__.py                  # Python package marker
│   ├── preprocessing.py             # Load and clean dataset
│   ├── feature_engineering.py       # Make next-week features and labels
│   └── prediction.py                # Train/evaluate/save/load models/predict
├── app.py                            # Streamlit interactive UI
├── requirements.txt                  # Python dependencies
├── .gitignore                        # Ignore generated/local files
└── README.md                         # This instruction guide
```

`data/cleaned_weekly.csv` is an **optional generated intermediate** created by `python -m src.preprocessing`. It is not required by the app.

## 5. Definitions and methodological decisions

- **Coverage rate** = observed supply / observed demand, computed separately by blood type.
- **Risk HIGH** = coverage under 0.80. **Risk MEDIUM** = 0.80 through less than 1.00. **Risk LOW** = 1.00 or greater.
- **Collection efficiency** = observed supplied / observed collected; a descriptive *proxy*, not a measured clinical collection yield.
- **Features (t)** = current-week demand, supply, collection, platelets, total donation activity, previous weeks' values, rolling averages and calendar information.
- **Targets (t+1)** = the *next observation's* risk classification and demand; never part of feature columns. A target is kept only when the next observation is **5–9 days** later. The final historical week has no known next-week target, but it is still available for an end-of-dataset forecast.
- **Chronological split** = earliest 70% of target dates for training, next 15% for validation, final 15% for test; all blood types on a given target date remain together. No random shuffling.
- **Model selection** = best validation **macro F1** classifier, best validation **MAE** regressor. Every compared model is refit on train+validation and evaluated against the same untouched chronological test partition. Test scores are for reporting, not for selecting the models.
- **Missing data** = numeric strings cleaned (including NBSP), invalid negatives treated as missing; only *training-period* imputation is learned by `SimpleImputer` inside each sklearn pipeline. Some invalid target records are excluded; target values are never imputed.
- **Duplicate dates** = conflicting records sharing a date are **averaged**, not summed. This is an explicit assumption because duplicate observations are not known to represent separate donation events; inspect the originals for more defensible handling if available.
- **Imbalanced classes** = balanced sample weights used for all classifiers. Report per-class recall as well as the mean.
- **No separate backend** = Streamlit reads CSV and joblib models directly. No API, SQL database, Redis or user authentication.
- **Alerts** = based on the model's predicted next-week HIGH or MEDIUM risk. There is no persistent alert history, so consecutive-week/sustained alerts are not implemented.

## 6. Notes specific to the supplied original data

- 639 rows, 29 columns; date range **2012-03-19 to 2024-06-17**, not exactly the proposal's 2014–2023 period.
- 13 dates occur more than once (before reconciliation).
- Several numeric fields include NBSP characters, blank cells and `#VALUE!`; there are two negative values in `Individual_donate`.
- Weekly dates have occasional gaps/irregular intervals. Future labels are not created for these irregular gaps.
- All reported results are **retrospective historical test results**. This dataset has no live data feed.

## 7. Model results on supplied data (initial run)

| Model | Validation metric used for selection | Held-out test result |
|---|---:|---:|
| Random Forest classifier (selected) | Macro F1 = 0.553 | Accuracy 0.631 / Macro F1 0.500 / HIGH recall 0.194 |
| XGBoost classifier | Macro F1 = 0.538 | Accuracy 0.560 / Macro F1 0.467 / HIGH recall 0.250 |
| Decision Tree classifier | Macro F1 = 0.460 | Accuracy 0.517 / Macro F1 0.459 / HIGH recall 0.417 |
| XGBoost regressor (selected) | MAE = 125.24 units | MAE 170.81 units / MAPE 15.34% |
| Random Forest regressor | MAE = 126.31 units | MAE 174.25 units / MAPE 15.26% |
| Decision Tree regressor | MAE = 147.03 units | MAE 196.78 units / MAPE 16.72% |

**Project honesty:** Classification does not meet the proposal's aspirational 80% accuracy / 75% HIGH-risk recall benchmarks. The current classification should not be promoted as reliable hospital alerts. The dashboard is a research demonstration.

## 8. Troubleshooting

**`ModuleNotFoundError`** → activate `.venv` and run `python -m pip install -r requirements.txt`.

**`No module named src`** → run the `python -m src...` commands from the project root (where `app.py` is located), not from `notebooks/`.

**`Models are missing`** → run `python -m src.prediction --train-all`.

**Windows PowerShell activation is blocked** → use `cmd.exe` and `.venv\Scripts\activate.bat`, or use PowerShell's session-scoped execution policy after checking your institution's IT rules.

**Dashboard doesn't open** → use `python -m streamlit run app.py`; look for the actual local URL printed in the terminal.

**Model file cannot be loaded after dependency upgrades** → retrain it with the current environment. Do not load joblib files from unknown/untrusted sources, as pickle/joblib loading can execute code.

## 9. Recommended demonstration order

1. Show EDA and preprocessing screenshots/notebook results.
2. Explain why prediction labels are shifted ahead one week and why chronological splitting matters.
3. Run the classifier and regressor comparisons and show validation vs test metrics.
4. Launch the Streamlit dashboard; switch between A, B, O and AB.
5. Explain limitations honestly: data cleaning assumptions, modest classification and HIGH-risk recall, retrospective dates, and no clinical validation.
>>>>>>> f4380c4 (Add Code)

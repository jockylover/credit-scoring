# Credit Risk Engine

End-to-end multiclass credit rating prediction (`Good` / `Standard` / `Poor`) with model benchmarking, portfolio stress testing, fairness diagnostics, and SHAP explainability.

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6%2B-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-2.1%2B-success)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-red)
![License](https://img.shields.io/badge/license-MIT-green)

## Overview

The Credit Risk Engine takes raw monthly customer-level credit records and produces:

1. A trained champion classifier that predicts credit rating across three classes.
2. A probability-of-default (PD) signal extracted from the `Poor` class probability, suitable for downstream risk decisions.
3. A benchmark table comparing eight common classifiers under identical preprocessing.
4. Diagnostic reports covering portfolio stress, group fairness, population stability, and SHAP-based explanations.

Originally built as a self-contained reference for retail credit scoring workflows: feature engineering, model competition, risk reporting, and an interactive demo — all reproducible from a single command.

## Tech Stack

| Layer | Tooling |
|---|---|
| Data manipulation | pandas, numpy |
| Modeling | scikit-learn, XGBoost |
| Explainability | SHAP (TreeExplainer) |
| Visualization | matplotlib, seaborn, Streamlit |
| Persistence | joblib |

## Project Structure

```
credit-scoring/
├── run.py                       # CLI: train + benchmark + diagnostics
├── app/streamlit_app.py         # Interactive single-customer scoring app
├── src/credit_risk_engine/
│   ├── constants.py             # Shared risk constants (age bins, LGD, PSI buckets, ...)
│   ├── data_prep.py             # Raw cleaning and normalization
│   ├── features.py              # Ratios, rolling behavioral signals, categorical flags
│   ├── model.py                 # Preprocessing, model zoo, benchmark, PD helper
│   ├── reject_inference.py      # Pseudo-labeled reject inference
│   ├── fairness.py              # Group approval/rejection rates and PD gaps
│   ├── stress.py                # Scenario stress + Monte Carlo portfolio loss
│   ├── stability.py             # Population Stability Index drift checks
│   └── explain.py               # SHAP summaries and risk archetypes
├── train.csv / test.csv         # Source data
├── requirements.txt
└── LICENSE
```

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

Tested on Python 3.10+ on Windows. On macOS/Linux, swap the activation line for `source .venv/bin/activate`.

## Usage

### Training pipeline

```powershell
python run.py
```

Common options:

```powershell
python run.py --train-path train.csv --artifacts-dir artifacts --label-col Credit_Score
python run.py --skip-shap
python run.py --test-size 0.2 --random-state 42 --shap-sample-size 1000
```

Outputs land in `artifacts/`:

- `model.joblib` — champion model selected by macro-F1 → AUC → accuracy
- `champion_model.joblib` — alias of the above
- `xgb_model.joblib` — XGBoost specifically (used for SHAP)
- `model_benchmark.csv` — full benchmark comparison table

### Interactive demo

```powershell
streamlit run app/streamlit_app.py
```

Point it at a trained `artifacts/model.joblib` and a CSV. Pick a customer to see predicted rating, PD, recommended decision band, class-probability breakdown, scenario-stressed PD, and the top SHAP contributors.

## Results & Metrics

Each benchmarked model reports:

| Metric | Purpose |
|---|---|
| `val_accuracy` | Overall hit rate on the validation split |
| `val_macro_f1` | Class-balanced F1 (the **champion-selection metric**) |
| `val_weighted_f1` | F1 weighted by class support |
| `val_macro_precision` / `val_macro_recall` | Per-class precision/recall, averaged |
| `val_auc_ovr_macro` / `val_auc_ovr_weighted` | One-vs-rest multiclass AUC |
| `val_poor_auc` | AUC restricted to the `Poor` (default) class |
| `val_poor_ks` | KS separation for the `Poor` signal — credit-scoring standard |

Additional diagnostics:

- **Stress test** (`stress.run_scenarios`) — expected loss, unexpected loss, VaR 95/99, ES 95 across baseline / mild recession / severe stress scenarios under a Monte Carlo loss simulation.
- **Fairness** (`fairness.group_fairness_metrics`) — approve/reject rates and mean PD by age bucket across four PD thresholds.
- **Stability** (`stability.compute_psi`) — PSI across temporal slices to detect feature drift.
- **Explainability** (`explain.compute_shap` + `archetype_profiles`) — global and cohort-level SHAP attributions for the `Poor` class probability.

## Data

The pipeline expects a CSV with monthly customer-level records and at minimum:

- `Customer_ID`, `Month_idx` — identity and temporal index
- `Age`, `Annual_Income`, `Monthly_Inhand_Salary`, `Outstanding_Debt`, `Monthly_Balance`, `Credit_Utilization_Ratio`, `Num_of_Delayed_Payment`, `Delay_from_due_date`, `Total_EMI_per_month`, `Changed_Credit_Limit`
- `Credit_Score` — target label with values in `{Good, Standard, Poor}`

`train.csv` and `test.csv` follow the public Credit Score Classification dataset schema. Replace them with your own files of the same shape to retrain.

## License

Released under the [MIT License](LICENSE).

# Credit Risk Analytics Workbench

A practical, end-to-end credit risk project for multiclass rating prediction (`Good / Standard / Poor`), model benchmarking, risk diagnostics, and explainability.

## Highlights
- Multiclass training instead of binary collapse.
- One-command benchmark across common machine learning models.
- Consistent risk diagnostics: stress testing, fairness snapshot, and stability (PSI).
- SHAP-based interpretation for the `Poor` risk signal.
- Streamlit demo for single-customer scoring.

## Project Structure
- `run.py`: training + benchmark + diagnostics pipeline (CLI supported).
- `src/credit_risk_lab/model.py`: preprocessing, model zoo, metrics, PD helper.
- `src/credit_risk_lab/features.py`: behavioral/ratio feature engineering.
- `src/credit_risk_lab/data_prep.py`: raw cleaning and normalization.
- `src/credit_risk_lab/stress.py`: scenario stress and portfolio loss simulation.
- `src/credit_risk_lab/fairness.py`: group-level approval/rejection and PD gap metrics.
- `src/credit_risk_lab/stability.py`: PSI drift checks.
- `src/credit_risk_lab/explain.py`: SHAP summary utilities.
- `app/streamlit_app.py`: interactive scoring app.

## Setup
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run Training Pipeline
Basic:
```bash
python run.py
```

Common options:
```bash
python run.py --train-path train.csv --artifacts-dir artifacts --label-col Credit_Score
python run.py --skip-shap
python run.py --test-size 0.2 --random-state 42 --shap-sample-size 1000
```

## Launch Demo
```bash
streamlit run app/streamlit_app.py
```

## Generated Outputs
After training, files are written to `artifacts/`:
- `model.joblib`: champion model (selected by macro-F1, then AUC, then accuracy).
- `champion_model.joblib`: same champion model with explicit name.
- `xgb_model.joblib`: XGBoost model for SHAP workflows.
- `model_benchmark.csv`: side-by-side validation metrics for all models.

## Data Notes
- Expected label column: `Credit_Score`.
- Expected classes: `Good`, `Standard`, `Poor`.
- Default pipeline reads `train.csv` from project root.
- Probability of default (PD) is defined as `P(Credit_Score = Poor)`.

## GitHub Upload Checklist
- Verify `.gitignore` behavior (`.venv/`, `__pycache__/`, `artifacts/` are excluded).
- Decide whether to upload `train.csv`/`test.csv` based on dataset license.
- Add a `LICENSE` file before making the repository public.
- Add repository topics and a short description in GitHub settings.

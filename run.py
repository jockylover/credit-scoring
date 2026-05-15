"""
Train the Credit Risk Analytics Workbench pipeline and generate
multiclass model benchmarks plus risk diagnostics.
"""
import argparse
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd

# Ensure local src/ is on path when running without installation.
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from credit_risk_lab import data_prep, explain, fairness, features, model, stability, stress


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run end-to-end multiclass credit risk benchmark pipeline.")
    parser.add_argument("--train-path", type=str, default="train.csv", help="Input training CSV path.")
    parser.add_argument("--label-col", type=str, default="Credit_Score", help="Target label column name.")
    parser.add_argument("--artifacts-dir", type=str, default="artifacts", help="Output folder for artifacts.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Validation split ratio.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--shap-sample-size", type=int, default=1000, help="Sample size used for SHAP summary.")
    parser.add_argument("--skip-shap", action="store_true", help="Skip SHAP summary.")
    return parser.parse_args()


def main():
    args = parse_args()
    train_path = args.train_path
    label_col = args.label_col
    artifacts_dir = Path(args.artifacts_dir)

    if not Path(train_path).exists():
        raise FileNotFoundError(f"Input file not found: {train_path}")

    print("Run config:")
    print(
        {
            "train_path": train_path,
            "label_col": label_col,
            "artifacts_dir": str(artifacts_dir),
            "test_size": args.test_size,
            "random_state": args.random_state,
            "shap_sample_size": args.shap_sample_size,
            "skip_shap": args.skip_shap,
        }
    )

    print(f"Loading {train_path} ...")
    raw = data_prep.load_dataset(train_path)
    clean = data_prep.clean_raw(raw)

    # Feature engineering + snapshot to latest record per customer.
    feat = features.prepare_feature_frame(clean)
    snapshot = features.snapshot_panel(feat, label_col=label_col)
    print(f"Snapshot shape: {snapshot.shape}")
    print("Class distribution:")
    print(snapshot[label_col].value_counts(normalize=False).sort_index())

    # Train and compare common ML models.
    print("Benchmarking common ML models (multiclass: Good / Standard / Poor) ...")
    benchmark = model.benchmark_classifiers(
        snapshot,
        label_col=label_col,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    comparison = benchmark.comparison.copy()
    print("Model comparison (sorted by macro-F1):")
    print(comparison.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    champion = benchmark.results[benchmark.champion_name]
    print(f"Champion model: {benchmark.champion_name}")
    print("Champion validation metrics:", champion.metrics)
    print(model.full_classification_report(champion.y_val, champion.y_val_pred, champion.class_labels))

    # Save artifacts.
    artifacts_dir.mkdir(exist_ok=True, parents=True)
    joblib.dump(champion.model, artifacts_dir / "model.joblib")
    joblib.dump(champion.model, artifacts_dir / "champion_model.joblib")
    if "XGBoost" in benchmark.results:
        joblib.dump(benchmark.results["XGBoost"].model, artifacts_dir / "xgb_model.joblib")
    comparison.to_csv(artifacts_dir / "model_benchmark.csv", index=False)
    print(f"Saved champion model to {artifacts_dir / 'model.joblib'}")
    print(f"Saved benchmark table to {artifacts_dir / 'model_benchmark.csv'}")

    # PD scores (defined as P(Credit_Score=Poor)).
    X_snapshot = snapshot.drop(columns=[label_col])
    pd_scores = model.predict_pd(champion.model, X_snapshot)

    # Stress testing.
    ead = snapshot["Outstanding_Debt"].fillna(snapshot["Outstanding_Debt"].median()).values
    stress_df = stress.run_scenarios(pd_scores, ead=ead)
    print("Stress testing summary:")
    print(stress_df)

    # Fairness by age buckets.
    fairness_df = fairness.group_fairness_metrics(snapshot, pd_scores, group_col="Age")
    print("Fairness (age buckets) at selected thresholds:")
    print(fairness_df.head())

    # Stability across early vs late months (using PD).
    slices = stability.time_slice(snapshot)
    psi_score = stability.compute_psi(
        slices["early"]["Credit_Utilization_Ratio"].dropna().values,
        slices["late"]["Credit_Utilization_Ratio"].dropna().values,
    )
    print(f"PSI (Credit_Utilization_Ratio early vs late): {psi_score:.4f}")

    # SHAP global summary for the Poor-class risk signal.
    if args.skip_shap:
        print("SHAP summary skipped by flag.")
    else:
        shap_model = benchmark.results.get("XGBoost", champion).model
        try:
            poor_idx = model.get_poor_class_index(shap_model)
            shap_values, feature_names = explain.compute_shap(
                shap_model,
                X_snapshot,
                sample_size=args.shap_sample_size,
                class_index=poor_idx,
            )
            global_importance = pd.Series(np.abs(shap_values).mean(axis=0), index=feature_names).sort_values(
                ascending=False
            )
            print("Top SHAP features (Poor class):")
            for feat_name, imp in global_importance.head(10).items():
                print(f"{feat_name} : {imp:.4f}")
        except Exception as exc:  # pragma: no cover - optional explainability convenience
            print(f"SHAP summary skipped: {exc}")


if __name__ == "__main__":
    main()

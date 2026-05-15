from dataclasses import dataclass
from inspect import signature
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

DEFAULT_LABEL_ORDER = ("Poor", "Standard", "Good")
DEFAULT_DROP_COLUMNS = ("ID", "Customer_ID", "Name", "SSN", "Month", "Month_idx", "Credit_History_Age")


def ks_statistic(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Kolmogorov-Smirnov separation between positives and negatives.

    Defined as max(TPR - FPR) along the ROC curve. Widely used in credit
    scoring as a single-number summary of how well a probability of default
    score separates defaulters from non-defaulters; values above ~0.30
    are typically considered acceptable for retail PD models.
    """
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.max(tpr - fpr))


def _split_columns(df: pd.DataFrame, exclude: List[str]) -> Tuple[List[str], List[str]]:
    feature_cols = [c for c in df.columns if c not in exclude]
    cat_cols = [c for c in feature_cols if str(df[c].dtype) in {"object", "category"}]
    num_cols = [c for c in feature_cols if c not in cat_cols]
    return num_cols, cat_cols


def build_preprocessor(num_cols: List[str], cat_cols: List[str]) -> ColumnTransformer:
    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", encoder),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ]
    )


@dataclass
class TrainResult:
    model_name: str
    model: Pipeline
    class_labels: List[str]
    metrics: Dict[str, float]
    y_val: np.ndarray
    y_val_pred: np.ndarray
    y_val_proba: Optional[np.ndarray]


@dataclass
class BenchmarkResult:
    comparison: pd.DataFrame
    results: Dict[str, TrainResult]
    champion_name: str


def _resolve_class_labels(y_raw: pd.Series, class_order: Optional[Sequence[str]]) -> Tuple[List[str], Dict[str, int]]:
    observed = sorted(y_raw.dropna().astype(str).unique().tolist())
    if class_order:
        ordered = [c for c in class_order if c in observed]
        ordered.extend([c for c in observed if c not in ordered])
    else:
        ordered = observed
    if len(ordered) < 2:
        raise ValueError("At least two classes are required for classification.")
    return ordered, {label: idx for idx, label in enumerate(ordered)}


def _supports_sample_weight(estimator) -> bool:
    try:
        return "sample_weight" in signature(estimator.fit).parameters
    except (TypeError, ValueError):
        return False


def _attach_label_metadata(model: Pipeline, class_labels: Sequence[str]) -> Pipeline:
    model.class_labels_ = np.asarray(class_labels)
    label_to_idx = {label: idx for idx, label in enumerate(class_labels)}
    model.label_to_index_ = label_to_idx
    model.poor_class_index_ = int(label_to_idx.get("Poor", 0))
    return model


def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray],
    poor_class_index: int,
) -> Dict[str, float]:
    metrics = {
        "val_accuracy": float(accuracy_score(y_true, y_pred)),
        "val_macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "val_weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "val_macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "val_macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    if y_proba is None or y_proba.ndim != 2:
        metrics["val_auc_ovr_macro"] = np.nan
        metrics["val_auc_ovr_weighted"] = np.nan
        metrics["val_poor_auc"] = np.nan
        metrics["val_poor_ks"] = np.nan
        return metrics

    metrics["val_auc_ovr_macro"] = float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro"))
    metrics["val_auc_ovr_weighted"] = float(
        roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted")
    )
    poor_idx = min(max(poor_class_index, 0), y_proba.shape[1] - 1)
    y_true_poor = (y_true == poor_idx).astype(int)
    y_score_poor = y_proba[:, poor_idx]
    metrics["val_poor_auc"] = float(roc_auc_score(y_true_poor, y_score_poor))
    metrics["val_poor_ks"] = float(ks_statistic(y_true_poor, y_score_poor))
    return metrics


def _prepare_xy(
    df: pd.DataFrame,
    label_col: str,
    class_order: Optional[Sequence[str]],
    drop_columns: Optional[Sequence[str]],
) -> Tuple[pd.DataFrame, np.ndarray, List[str], List[str], List[str]]:
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found.")
    y_raw = df[label_col].astype(str)
    class_labels, label_to_idx = _resolve_class_labels(y_raw, class_order=class_order)
    y = y_raw.map(label_to_idx).values

    dropped = [label_col]
    if drop_columns:
        dropped.extend([c for c in drop_columns if c in df.columns])
    X = df.drop(columns=dropped)
    all_missing_cols = [c for c in X.columns if X[c].isna().all()]
    if all_missing_cols:
        X = X.drop(columns=all_missing_cols)
    num_cols, cat_cols = _split_columns(X, exclude=[])
    return X, y, class_labels, num_cols, cat_cols


def _fit_single_model(
    model_name: str,
    estimator,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    class_labels: Sequence[str],
    num_cols: Sequence[str],
    cat_cols: Sequence[str],
    sample_weight: Optional[np.ndarray] = None,
) -> TrainResult:
    preprocessor = build_preprocessor(list(num_cols), list(cat_cols))
    clf = clone(estimator)
    pipeline = Pipeline(steps=[("preprocess", preprocessor), ("model", clf)])
    fit_kwargs = {}
    if sample_weight is not None and _supports_sample_weight(clf):
        fit_kwargs["model__sample_weight"] = sample_weight
    pipeline.fit(X_train, y_train, **fit_kwargs)
    pipeline = _attach_label_metadata(pipeline, class_labels)

    y_pred = pipeline.predict(X_val)
    y_proba = pipeline.predict_proba(X_val) if hasattr(pipeline.named_steps["model"], "predict_proba") else None
    poor_idx = int(getattr(pipeline, "poor_class_index_", 0))
    metrics = _compute_metrics(y_val, y_pred, y_proba, poor_class_index=poor_idx)
    return TrainResult(
        model_name=model_name,
        model=pipeline,
        class_labels=list(class_labels),
        metrics=metrics,
        y_val=y_val,
        y_val_pred=y_pred,
        y_val_proba=y_proba,
    )


def build_model_zoo(random_state: int = 42) -> Dict[str, object]:
    """Return the default suite of classifiers compared by ``benchmark_classifiers``.

    The returned mapping holds eight estimators spanning a linear baseline
    (LogisticRegression), tree models (DecisionTree, RandomForest,
    ExtraTrees), boosted ensembles (GradientBoosting, AdaBoost, XGBoost) and
    a distance-based learner (KNN). Hyperparameters are tuned for tabular
    credit data with mild class imbalance; pass a custom dict to
    ``benchmark_classifiers(model_zoo=...)`` to override.
    """
    return {
        "LogisticRegression": LogisticRegression(max_iter=800),
        "DecisionTree": DecisionTreeClassifier(
            max_depth=10, min_samples_leaf=20, class_weight="balanced", random_state=random_state
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=350,
            min_samples_leaf=4,
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=1,
        ),
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=350,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=1,
        ),
        "GradientBoosting": GradientBoostingClassifier(random_state=random_state),
        "AdaBoost": AdaBoostClassifier(n_estimators=220, learning_rate=0.08, random_state=random_state),
        "KNN": KNeighborsClassifier(n_neighbors=31, weights="distance", n_jobs=1),
        "XGBoost": XGBClassifier(
            n_estimators=350,
            max_depth=4,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.8,
            min_child_weight=1.0,
            reg_lambda=1.0,
            gamma=0.0,
            objective="multi:softprob",
            eval_metric="mlogloss",
            tree_method="hist",
            random_state=random_state,
        ),
    }


def benchmark_classifiers(
    df: pd.DataFrame,
    label_col: str = "Credit_Score",
    class_order: Sequence[str] = DEFAULT_LABEL_ORDER,
    drop_columns: Sequence[str] = DEFAULT_DROP_COLUMNS,
    model_zoo: Optional[Dict[str, object]] = None,
    test_size: float = 0.2,
    random_state: int = 42,
    sample_weight: Optional[np.ndarray] = None,
) -> BenchmarkResult:
    """Train every classifier in the zoo and pick a champion.

    Splits ``df`` once into train/validation using stratified sampling, fits
    each pipeline (preprocessing + estimator), and returns evaluation
    metrics for each model plus the chosen champion.

    Champion selection order: macro-F1 (descending) > one-vs-rest macro AUC
    > overall accuracy. Macro-F1 is preferred to keep the minority "Poor"
    class influential when classes are imbalanced.
    """
    X, y, class_labels, num_cols, cat_cols = _prepare_xy(
        df=df,
        label_col=label_col,
        class_order=class_order,
        drop_columns=drop_columns,
    )

    if sample_weight is not None:
        X_train, X_val, y_train, y_val, sw_train, _ = train_test_split(
            X, y, sample_weight, test_size=test_size, random_state=random_state, stratify=y
        )
    else:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        sw_train = None

    zoo = model_zoo or build_model_zoo(random_state=random_state)
    results: Dict[str, TrainResult] = {}
    rows = []
    for model_name, estimator in zoo.items():
        result = _fit_single_model(
            model_name=model_name,
            estimator=estimator,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            class_labels=class_labels,
            num_cols=num_cols,
            cat_cols=cat_cols,
            sample_weight=sw_train,
        )
        results[model_name] = result
        row = {"model": model_name}
        row.update(result.metrics)
        rows.append(row)

    comparison = pd.DataFrame(rows).sort_values(
        by=["val_macro_f1", "val_auc_ovr_macro", "val_accuracy"], ascending=False
    )
    comparison = comparison.reset_index(drop=True)
    champion_name = str(comparison.iloc[0]["model"])
    return BenchmarkResult(comparison=comparison, results=results, champion_name=champion_name)


def train_xgb_classifier(
    df: pd.DataFrame,
    label_col: str = "Credit_Score",
    positive_label: str = "Poor",
    test_size: float = 0.2,
    random_state: int = 42,
    sample_weight: Optional[np.ndarray] = None,
) -> TrainResult:
    """
    Backward-compatible wrapper. Trains multiclass XGBoost and tracks Poor-class
    probability as PD for stress/fairness modules.
    """
    _ = positive_label
    bench = benchmark_classifiers(
        df=df,
        label_col=label_col,
        model_zoo={"XGBoost": build_model_zoo(random_state=random_state)["XGBoost"]},
        test_size=test_size,
        random_state=random_state,
        sample_weight=sample_weight,
    )
    return bench.results["XGBoost"]


def decode_labels(model: Pipeline, encoded: np.ndarray) -> np.ndarray:
    encoded = np.asarray(encoded)
    if encoded.dtype.kind in {"U", "S", "O"}:
        return encoded
    class_labels = getattr(model, "class_labels_", None)
    if class_labels is None:
        estimator = model.named_steps["model"]
        if hasattr(estimator, "classes_") and estimator.classes_.dtype.kind in {"U", "S", "O"}:
            return estimator.classes_[encoded.astype(int)]
        return encoded.astype(str)
    return np.asarray(class_labels)[encoded.astype(int)]


def full_classification_report(
    y_true: np.ndarray, y_pred: np.ndarray, class_labels: Optional[Sequence[str]] = None
) -> str:
    if class_labels:
        labels = list(range(len(class_labels)))
        return classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=list(class_labels),
            digits=4,
            zero_division=0,
        )
    return classification_report(y_true, y_pred, digits=4, zero_division=0)


def predict_credit_label(model: Pipeline, df: pd.DataFrame) -> np.ndarray:
    encoded = model.predict(df)
    return decode_labels(model, encoded)


def get_poor_class_index(model: Pipeline, poor_label: str = "Poor") -> int:
    if hasattr(model, "poor_class_index_"):
        return int(model.poor_class_index_)
    class_labels = getattr(model, "class_labels_", None)
    if class_labels is not None and poor_label in class_labels:
        return int(np.where(class_labels == poor_label)[0][0])
    estimator = model.named_steps["model"]
    if hasattr(estimator, "classes_"):
        classes = np.asarray(estimator.classes_)
        if poor_label in classes:
            return int(np.where(classes == poor_label)[0][0])
        if len(classes) > 1:
            return 1
    return 0


def predict_pd(model: Pipeline, df: pd.DataFrame, poor_label: str = "Poor") -> np.ndarray:
    """Probability of default defined as P(Credit_Score = Poor)."""
    proba = model.predict_proba(df)
    if proba.ndim == 1:
        return proba
    poor_idx = min(max(get_poor_class_index(model, poor_label=poor_label), 0), proba.shape[1] - 1)
    return proba[:, poor_idx]

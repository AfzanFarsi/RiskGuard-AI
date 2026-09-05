"""
train_model.py
---------------
End-to-end training script for RiskGuard AI.

Pipeline:
    raw CSV -> filter to fraud-eligible types -> feature engineering
    -> train/test split (stratified, test set = REAL untouched distribution)
    -> training-side class-imbalance handling (undersampling of the majority
       class in the TRAINING split only -- the test split is never touched)
    -> fit preprocessing (OneHotEncoder + StandardScaler) on TRAIN ONLY
    -> train Logistic Regression (baseline), Random Forest, and
       HistGradientBoosting
    -> evaluate all three on the untouched, real-world-distribution test set
    -> select the best model by PR-AUC (average precision) on the fraud class,
       since this dataset is extremely imbalanced (~0.13% fraud) and PR-AUC
       is far more informative than accuracy or even ROC-AUC here
    -> persist the winning pipeline + metrics + comparison table to disk

Run:
    python src/train_model.py --data data/dataset.csv

Reproducibility: a fixed RANDOM_SEED is used everywhere randomness occurs.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import sys
sys.path.append(str(Path(__file__).resolve().parent))
from data_preprocessing import (
    ALL_MODEL_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGET,
    load_raw_dataset,
    prepare_model_frame,
    profile_dataset,
)

RANDOM_SEED = 42
MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
OUTPUTS_DIR = Path(__file__).resolve().parents[1] / "outputs"


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )


def undersample_majority(X: pd.DataFrame, y: pd.Series, ratio: int, seed: int):
    """Undersample the majority (legit) class in the TRAINING split only.

    ratio = number of legit examples kept per 1 fraud example.
    This is purely a training-time speed/balance measure. The held-out test
    set retains the true, untouched class distribution so all reported
    metrics reflect real-world performance.
    """
    rng = np.random.RandomState(seed)
    fraud_idx = y[y == 1].index
    legit_idx = y[y == 0].index
    n_keep = min(len(legit_idx), len(fraud_idx) * ratio)
    legit_keep = rng.choice(legit_idx, size=n_keep, replace=False)
    keep_idx = np.concatenate([fraud_idx.values, legit_keep])
    rng.shuffle(keep_idx)
    return X.loc[keep_idx], y.loc[keep_idx]


def _downsample_curve(x: np.ndarray, y: np.ndarray, max_points: int = 200):
    """Evenly downsample a curve for compact storage/plotting, always keeping
    the first and last point."""
    if len(x) <= max_points:
        return x.tolist(), y.tolist()
    idx = np.linspace(0, len(x) - 1, max_points).astype(int)
    return x[idx].tolist(), y[idx].tolist()


def evaluate(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    proba = pipeline.predict_proba(X_test)[:, 1]
    preds = (proba >= 0.5).astype(int)

    cm = confusion_matrix(y_test, preds).tolist()
    fpr, tpr, _ = roc_curve(y_test, proba)
    prec_curve, rec_curve, _ = precision_recall_curve(y_test, proba)
    fpr, tpr = _downsample_curve(fpr, tpr)
    rec_curve, prec_curve = _downsample_curve(rec_curve, prec_curve)

    return {
        "precision": float(precision_score(y_test, preds, zero_division=0)),
        "recall": float(recall_score(y_test, preds, zero_division=0)),
        "f1": float(f1_score(y_test, preds, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "pr_auc": float(average_precision_score(y_test, proba)),
        "accuracy": float((preds == y_test.values).mean()),
        "confusion_matrix": cm,
        "roc_curve": {"fpr": fpr, "tpr": tpr},
        "pr_curve": {"precision": prec_curve, "recall": rec_curve},
        "n_test": int(len(y_test)),
        "n_test_fraud": int(y_test.sum()),
    }


def main(data_path: str, undersample_ratio: int = 20):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading raw dataset...")
    raw_df = load_raw_dataset(data_path)
    profile = profile_dataset(raw_df)
    print(f"Raw dataset: {profile['n_rows']:,} rows, "
          f"fraud rate {profile['fraud_rate']:.4%}")

    print("Filtering to fraud-eligible transaction types + engineering features...")
    model_df = prepare_model_frame(raw_df)
    print(f"Modeling dataset (TRANSFER + CASH_OUT only): {len(model_df):,} rows, "
          f"fraud rate {model_df[TARGET].mean():.4%}")

    X = model_df[ALL_MODEL_FEATURES]
    y = model_df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    print(f"Train: {len(X_train):,} rows ({y_train.sum()} fraud) | "
          f"Test (untouched, real distribution): {len(X_test):,} rows ({y_test.sum()} fraud)")

    X_train_bal, y_train_bal = undersample_majority(
        X_train, y_train, ratio=undersample_ratio, seed=RANDOM_SEED
    )
    print(f"Balanced training set (undersampled legit class): {len(X_train_bal):,} rows "
          f"({y_train_bal.sum()} fraud, ratio 1:{undersample_ratio})")

    candidates = {
        "logistic_regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_SEED
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=5,
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_SEED,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=300,
            max_depth=8,
            learning_rate=0.1,
            random_state=RANDOM_SEED,
        ),
    }

    results = {}
    fitted_pipelines = {}
    for name, clf in candidates.items():
        print(f"\nTraining {name}...")
        t0 = time.time()
        pipe = Pipeline([
            ("preprocess", build_preprocessor()),
            ("clf", clf),
        ])
        pipe.fit(X_train_bal, y_train_bal)
        elapsed = time.time() - t0
        metrics = evaluate(pipe, X_test, y_test)
        metrics["train_time_seconds"] = round(elapsed, 2)
        results[name] = metrics
        fitted_pipelines[name] = pipe
        print(f"  done in {elapsed:.1f}s | PR-AUC={metrics['pr_auc']:.4f} "
              f"ROC-AUC={metrics['roc_auc']:.4f} Recall={metrics['recall']:.4f} "
              f"Precision={metrics['precision']:.4f} F1={metrics['f1']:.4f}")

    best_name = max(results, key=lambda n: results[n]["pr_auc"])
    print(f"\nBest model by PR-AUC on fraud class: {best_name}")

    best_pipeline = fitted_pipelines[best_name]
    joblib.dump(best_pipeline, MODELS_DIR / "risk_model.joblib")

    summary = {
        "best_model": best_name,
        "random_seed": RANDOM_SEED,
        "undersample_ratio": undersample_ratio,
        "dataset_profile": profile,
        "modeling_rows": int(len(model_df)),
        "modeling_fraud_rate": float(model_df[TARGET].mean()),
        "n_train": int(len(X_train)),
        "n_train_balanced": int(len(X_train_bal)),
        "n_test": int(len(X_test)),
        "results": results,
        "features_used": ALL_MODEL_FEATURES,
        "feature_names_after_encoding": None,
    }

    # Save feature importance for the winning model when available
    try:
        preprocessor = best_pipeline.named_steps["preprocess"]
        feature_names = list(preprocessor.get_feature_names_out())
        summary["feature_names_after_encoding"] = feature_names
        clf = best_pipeline.named_steps["clf"]
        if hasattr(clf, "feature_importances_"):
            importances = clf.feature_importances_.tolist()
        elif hasattr(clf, "coef_"):
            importances = np.abs(clf.coef_[0]).tolist()
        else:
            importances = None
        if importances:
            summary["global_feature_importance"] = sorted(
                zip(feature_names, importances), key=lambda t: -t[1]
            )
    except Exception as e:
        print("Could not extract feature importance:", e)

    with open(OUTPUTS_DIR / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved model to {MODELS_DIR / 'risk_model.joblib'}")
    print(f"Saved training summary to {OUTPUTS_DIR / 'training_summary.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/dataset.csv")
    parser.add_argument("--undersample-ratio", type=int, default=20)
    args = parser.parse_args()
    main(args.data, args.undersample_ratio)

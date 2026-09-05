"""
prediction.py
-------------
Turns raw model probabilities into the risk-management concepts the
dashboard shows: risk score, risk level, fraud prediction, and a
recommended action.

Risk-level thresholds are deliberately simple and configurable (not
derived from a black box): they are cut points on the model's predicted
fraud probability, chosen so that:

  LOW    (< low_threshold)             -> probability is low enough that,
                                           on this dataset, the transaction
                                           looks like the overwhelming
                                           majority of legitimate traffic.
  MEDIUM (low_threshold .. high_threshold) -> probability is elevated
                                           enough to warrant a second check.
  HIGH   (>= high_threshold)           -> probability is high enough that
                                           the transaction should be held
                                           for manual review before completion.

These are prototype thresholds for demonstration purposes, not a
regulatory or production risk policy.
"""

from __future__ import annotations

from dataclasses import dataclass

import joblib
import pandas as pd

DEFAULT_LOW_THRESHOLD = 0.10
DEFAULT_HIGH_THRESHOLD = 0.50


@dataclass
class RiskResult:
    fraud_probability: float
    risk_level: str
    prediction_label: str
    recommended_action: str


def load_model(model_path: str):
    return joblib.load(model_path)


def score_to_risk_level(
    probability: float,
    low_threshold: float = DEFAULT_LOW_THRESHOLD,
    high_threshold: float = DEFAULT_HIGH_THRESHOLD,
) -> str:
    if probability < low_threshold:
        return "LOW"
    elif probability < high_threshold:
        return "MEDIUM"
    else:
        return "HIGH"


def recommended_action_for(risk_level: str) -> str:
    return {
        "LOW": "Allow transaction — no strong risk indicators detected.",
        "MEDIUM": "Request additional verification before completing the transaction.",
        "HIGH": "Send transaction for manual risk review before it is completed.",
    }[risk_level]


def score_transaction(
    pipeline,
    feature_row: pd.DataFrame,
    low_threshold: float = DEFAULT_LOW_THRESHOLD,
    high_threshold: float = DEFAULT_HIGH_THRESHOLD,
) -> RiskResult:
    """Score a single-row DataFrame of model features (see
    data_preprocessing.build_feature_row)."""
    proba = float(pipeline.predict_proba(feature_row)[0, 1])
    risk_level = score_to_risk_level(proba, low_threshold, high_threshold)
    label = "POTENTIALLY FRAUDULENT" if proba >= high_threshold else "LEGITIMATE"
    return RiskResult(
        fraud_probability=proba,
        risk_level=risk_level,
        prediction_label=label,
        recommended_action=recommended_action_for(risk_level),
    )


def score_batch(
    pipeline,
    features_df: pd.DataFrame,
    low_threshold: float = DEFAULT_LOW_THRESHOLD,
    high_threshold: float = DEFAULT_HIGH_THRESHOLD,
) -> pd.DataFrame:
    """Score a DataFrame of many transactions (same feature columns as
    data_preprocessing.ALL_MODEL_FEATURES). Returns the input df with
    risk_score, risk_level, fraud_prediction and recommended_action columns
    appended."""
    out = features_df.copy()
    probas = pipeline.predict_proba(features_df)[:, 1]
    out["risk_score"] = probas
    out["risk_level"] = [score_to_risk_level(p, low_threshold, high_threshold) for p in probas]
    out["fraud_prediction"] = [
        "POTENTIALLY FRAUDULENT" if p >= high_threshold else "LEGITIMATE" for p in probas
    ]
    out["recommended_action"] = [recommended_action_for(r) for r in out["risk_level"]]
    return out

"""
data_preprocessing.py
----------------------
Loading, cleaning and feature engineering for RiskGuard AI.

This module is written specifically for the schema of the dataset used in
this project: the "Online Payment Fraud Detection" dataset (PaySim mobile
money simulation), which has these raw columns:

    step, type, amount, nameOrig, oldbalanceOrg, newbalanceOrig,
    nameDest, oldbalanceDest, newbalanceDest, isFraud, isFlaggedFraud

Design decisions (see README for full justification):

1. `nameOrig` / `nameDest` are dropped. They are near-unique identifier
   strings (millions of distinct values) that do not generalize to unseen
   customers/merchants and would blow up any encoding scheme. Using them
   would not improve genuine fraud detection and risks the model
   memorizing IDs instead of learning fraud patterns.

2. `isFlaggedFraud` is dropped as a *feature*. In the raw data it is set by
   a simple hard-coded business rule (essentially "amount > 200,000 on a
   TRANSFER"), it is only ever 1 for 16 out of 6.36M rows, and it is
   derived from the same event we are trying to predict. Keeping it as an
   input feature would leak rule-based knowledge of the label into the
   model. It is retained in the raw data for reference only.

3. Empirically, in this dataset fraud ONLY occurs on `TRANSFER` and
   `CASH_OUT` transactions (0 fraud cases in PAYMENT, CASH_IN, DEBIT out of
   6.35M such rows). We restrict modeling to these two transaction types.
   This is a data-driven decision confirmed by direct inspection of the
   dataset, not an assumption, and it also makes the problem
   computationally tractable on a normal laptop (~2.77M rows instead of
   6.36M) without discarding a single fraud example.

4. Two engineered "balance error" features are added, based only on the
   original numeric columns:
       errorBalanceOrig = oldbalanceOrg - amount - newbalanceOrig
       errorBalanceDest = oldbalanceDest + amount - newbalanceDest
   These capture how much a transaction's before/after balances deviate
   from what simple bookkeeping would predict, which is a strong known
   signal for these synthetic fraud patterns (fraudulent transactions
   often leave a non-zero "error" because the simulated fraud logic does
   not update balances consistently).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

RAW_DTYPES = {
    "step": "int32",
    "type": "category",
    "amount": "float64",
    "nameOrig": "string",
    "oldbalanceOrg": "float64",
    "newbalanceOrig": "float64",
    "nameDest": "string",
    "oldbalanceDest": "float64",
    "newbalanceDest": "float64",
    "isFraud": "int8",
    "isFlaggedFraud": "int8",
}

FRAUD_ELIGIBLE_TYPES = ["TRANSFER", "CASH_OUT"]

NUMERIC_FEATURES = [
    "step",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "errorBalanceOrig",
    "errorBalanceDest",
]
CATEGORICAL_FEATURES = ["type"]
TARGET = "isFraud"
ALL_MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Raw fields the user actually supplies (via the form or a batch CSV), BEFORE
# feature engineering. errorBalanceOrig / errorBalanceDest are deliberately
# excluded here -- they don't exist yet at input time, they are computed by
# add_engineered_features()/build_feature_row() using the formulas below.
RAW_NUMERIC_INPUT_FEATURES = [
    "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest",
]
RAW_INPUT_FEATURES = ["step", "type"] + RAW_NUMERIC_INPUT_FEATURES


def load_raw_dataset(path: str) -> pd.DataFrame:
    """Load the raw CSV with memory-efficient dtypes."""
    df = pd.read_csv(path, dtype=RAW_DTYPES)
    return df


def profile_dataset(df: pd.DataFrame) -> dict:
    """Return a dictionary of basic dataset-health facts, all computed live."""
    fraud_by_type = (
        df.groupby("type", observed=True)[TARGET]
        .agg(fraud_count="sum", txn_count="count")
        .assign(fraud_rate=lambda d: d["fraud_count"] / d["txn_count"])
    )
    return {
        "n_rows": int(len(df)),
        "n_cols": int(df.shape[1]),
        "missing_values": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "fraud_count": int(df[TARGET].sum()),
        "fraud_rate": float(df[TARGET].mean()),
        "flagged_fraud_count": int(df["isFlaggedFraud"].sum()),
        "n_unique_orig": int(df["nameOrig"].nunique()),
        "n_unique_dest": int(df["nameDest"].nunique()),
        "step_min": int(df["step"].min()),
        "step_max": int(df["step"].max()),
        "type_counts": df["type"].value_counts().to_dict(),
        "fraud_by_type": fraud_by_type.to_dict(orient="index"),
        "amount_describe": df["amount"].describe().to_dict(),
    }


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["errorBalanceOrig"] = (
        df["oldbalanceOrg"] - df["amount"] - df["newbalanceOrig"]
    )
    df["errorBalanceDest"] = (
        df["oldbalanceDest"] + df["amount"] - df["newbalanceDest"]
    )
    return df


def filter_to_fraud_eligible_types(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only TRANSFER and CASH_OUT rows (the only types with any fraud)."""
    return df[df["type"].isin(FRAUD_ELIGIBLE_TYPES)].copy()


def prepare_model_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Full preparation pipeline: filter -> engineer -> select model columns.

    Returns a DataFrame containing exactly ALL_MODEL_FEATURES + TARGET,
    ready to be split into train/test.
    """
    df = filter_to_fraud_eligible_types(df)
    df = add_engineered_features(df)
    keep_cols = ALL_MODEL_FEATURES + [TARGET]
    return df[keep_cols].reset_index(drop=True)


def validate_input_row(row: dict) -> list[str]:
    """Validate a single RAW transaction dict submitted via the UI form or a
    batch CSV -- i.e. before feature engineering has run. Only checks the
    fields the user actually supplies (RAW_INPUT_FEATURES); the engineered
    features (errorBalanceOrig, errorBalanceDest) are computed afterwards by
    add_engineered_features()/build_feature_row() and are intentionally not
    required here.

    Returns a list of human-readable error strings (empty list = valid).
    """
    errors = []
    for col in RAW_INPUT_FEATURES:
        if col not in row or row[col] is None or row[col] == "":
            errors.append(f"Missing value for required field '{col}'.")

    if "type" in row and row["type"] not in FRAUD_ELIGIBLE_TYPES:
        errors.append(
            f"'type' must be one of {FRAUD_ELIGIBLE_TYPES} "
            f"(the trained model only covers these transaction types, "
            f"since fraud never occurs in the other types in this dataset)."
        )

    for col in RAW_NUMERIC_INPUT_FEATURES:
        if col in row and row[col] not in (None, ""):
            try:
                val = float(row[col])
                if val < 0:
                    errors.append(f"'{col}' cannot be negative.")
            except (ValueError, TypeError):
                errors.append(f"'{col}' must be a number.")
    return errors


def build_feature_row(
    amount: float,
    oldbalanceOrg: float,
    newbalanceOrig: float,
    oldbalanceDest: float,
    newbalanceDest: float,
    type_: str,
    step: int = 1,
) -> pd.DataFrame:
    """Build a single-row DataFrame with engineered features, ready for the
    trained pipeline's `.predict_proba`."""
    row = {
        "step": step,
        "amount": amount,
        "oldbalanceOrg": oldbalanceOrg,
        "newbalanceOrig": newbalanceOrig,
        "oldbalanceDest": oldbalanceDest,
        "newbalanceDest": newbalanceDest,
        "type": type_,
    }
    frame = pd.DataFrame([row])
    frame = add_engineered_features(frame)
    return frame[ALL_MODEL_FEATURES]

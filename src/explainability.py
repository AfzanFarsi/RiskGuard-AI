"""
explainability.py
------------------
Per-transaction and global explainability using SHAP.

The final model (HistGradientBoostingClassifier) is a tree-ensemble, so we
use shap.TreeExplainer, which computes exact SHAP values efficiently for
tree-based models (no sampling/approximation needed for a single
transaction). Every explanation shown in the app is a real SHAP value
computed against the real trained model -- nothing is hard-coded.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import shap


def get_tree_explainer(pipeline):
    clf = pipeline.named_steps["clf"]
    return shap.TreeExplainer(clf)


def explain_row(pipeline, explainer, feature_row: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """Return a DataFrame of the top_n features driving this transaction's
    prediction, with columns: feature, value, shap_value, direction."""
    preprocessor = pipeline.named_steps["preprocess"]
    transformed = preprocessor.transform(feature_row)
    feature_names = list(preprocessor.get_feature_names_out())

    shap_values = explainer.shap_values(transformed)
    shap_values = np.array(shap_values)
    if shap_values.ndim == 3:
        # (n_classes, n_samples, n_features) -> take fraud class (last)
        shap_values = shap_values[-1]
    row_shap = shap_values[0]

    dense = transformed.toarray() if hasattr(transformed, "toarray") else np.asarray(transformed)
    raw_values = dense[0]

    df = pd.DataFrame({
        "feature": feature_names,
        "encoded_value": raw_values,
        "shap_value": row_shap,
    })
    df["direction"] = np.where(df["shap_value"] >= 0, "increases risk", "decreases risk")
    df["abs_shap"] = df["shap_value"].abs()
    df = df.sort_values("abs_shap", ascending=False).head(top_n)
    df["feature"] = df["feature"].str.replace("num__", "", regex=False).str.replace("cat__", "", regex=False)
    return df[["feature", "shap_value", "direction"]].reset_index(drop=True)


def clean_feature_name(name: str) -> str:
    return name.replace("num__", "").replace("cat__", "")

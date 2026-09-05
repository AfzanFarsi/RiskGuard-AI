"""
evaluation.py
-------------
Loads the real training/evaluation results produced by train_model.py
(outputs/training_summary.json) so the app can display actual metrics
instead of hard-coded numbers.
"""

from __future__ import annotations

import json
from pathlib import Path


def load_training_summary(path: str = "outputs/training_summary.json") -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"No training summary found at {path}. Run `python src/train_model.py` first."
        )
    with open(p) as f:
        return json.load(f)


def get_best_model_metrics(summary: dict) -> dict:
    return summary["results"][summary["best_model"]]


def get_model_comparison_table(summary: dict):
    rows = []
    for name, metrics in summary["results"].items():
        rows.append({
            "model": name,
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "roc_auc": metrics["roc_auc"],
            "pr_auc": metrics["pr_auc"],
            "accuracy": metrics["accuracy"],
            "train_time_seconds": metrics["train_time_seconds"],
        })
    return rows

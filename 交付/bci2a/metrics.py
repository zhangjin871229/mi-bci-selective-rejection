"""评估指标与结果保存。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]) -> dict:
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    kappa = cohen_kappa_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    report = classification_report(
        y_true, y_pred, target_names=class_names, digits=4, zero_division=0
    )
    return {
        "accuracy": float(acc),
        "f1_macro": float(f1_macro),
        "kappa": float(kappa),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }

def format_metrics_line(metrics: dict) -> str:
    return (
        f"accuracy={metrics['accuracy']:.4f}  "
        f"kappa={metrics['kappa']:.4f}  "
        f"F1={metrics['f1_macro']:.4f}"
    )

def save_metrics(metrics: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path = path.with_suffix(".txt")
    header = format_metrics_line(metrics) + "\n\n"
    report_path.write_text(header + metrics.get("classification_report", ""), encoding="utf-8")

"""Locked, validation-only classification/ranking metrics and failure tables."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_score,
    recall_score,
)

RISK_ORDER = {
    "low": 0,
    "intermediate": 1,
    "moderate": 1,
    "high": 2,
    "L": 0,
    "M": 1,
    "H": 2,
}


@dataclass(frozen=True)
class LockedMetrics:
    n: int
    labels: tuple[str, ...]
    confusion_matrix: list[list[int]]
    accuracy: float
    balanced_accuracy: float
    macro_f1: float
    macro_precision: float
    macro_recall: float
    cohen_kappa: float | None
    ordinal_mae: float
    spearman_rho: float | None
    spearman_pvalue: float | None
    kendall_tau: float | None
    kendall_pvalue: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _normalise_label(value: Any) -> str:
    return str(value).strip().lower()


def _ordinal(values: Iterable[str]) -> np.ndarray:
    out = []
    for value in values:
        key = _normalise_label(value)
        if key not in RISK_ORDER:
            raise ValueError(f"Unsupported risk label: {value!r}")
        out.append(RISK_ORDER[key])
    return np.asarray(out, dtype=float)


def locked_metrics(
    reference: Iterable[str],
    observed: Iterable[str],
    *,
    labels: tuple[str, ...] = ("low", "intermediate", "high"),
) -> LockedMetrics:
    """Compute fixed metrics; this function never fits or changes a model."""
    y_true = np.asarray([_normalise_label(x) for x in reference])
    y_pred = np.asarray([_normalise_label(x) for x in observed])
    if y_true.shape != y_pred.shape or y_true.size == 0:
        raise ValueError("reference and observed must have equal, non-zero length")

    label_values = [_normalise_label(x) for x in labels]
    if len(set(label_values)) != len(label_values):
        raise ValueError("labels must be unique")
    unknown_labels = sorted(set(label_values) - set(RISK_ORDER))
    if unknown_labels:
        raise ValueError(f"Unsupported metric labels: {unknown_labels}")

    cm = confusion_matrix(y_true, y_pred, labels=label_values)
    true_ord = _ordinal(y_true)
    pred_ord = _ordinal(y_pred)
    rho = spearmanr(true_ord, pred_ord)
    tau = kendalltau(true_ord, pred_ord)
    kappa = cohen_kappa_score(y_true, y_pred, labels=label_values)
    return LockedMetrics(
        n=int(y_true.size),
        labels=tuple(label_values),
        confusion_matrix=cm.astype(int).tolist(),
        accuracy=float(accuracy_score(y_true, y_pred)),
        balanced_accuracy=float(balanced_accuracy_score(y_true, y_pred)),
        macro_f1=float(f1_score(y_true, y_pred, labels=label_values, average="macro", zero_division=0)),
        macro_precision=float(precision_score(y_true, y_pred, labels=label_values, average="macro", zero_division=0)),
        macro_recall=float(recall_score(y_true, y_pred, labels=label_values, average="macro", zero_division=0)),
        cohen_kappa=None if pd.isna(kappa) else float(kappa),
        ordinal_mae=float(mean_absolute_error(true_ord, pred_ord)),
        spearman_rho=None if pd.isna(rho.statistic) else float(rho.statistic),
        spearman_pvalue=None if pd.isna(rho.pvalue) else float(rho.pvalue),
        kendall_tau=None if pd.isna(tau.statistic) else float(tau.statistic),
        kendall_pvalue=None if pd.isna(tau.pvalue) else float(tau.pvalue),
    )


def stratified_failures(
    frame: pd.DataFrame,
    *,
    reference_column: str = "reference_risk",
    observed_column: str = "observed_risk",
    strata: tuple[str, ...] = ("compound", "site", "concentration"),
) -> pd.DataFrame:
    """Return deterministic grouped failure summaries."""
    required = {reference_column, observed_column, *strata}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing validation columns: {missing}")
    out = frame.copy()
    out["is_failure"] = out[reference_column].map(_normalise_label) != out[observed_column].map(_normalise_label)
    out["ordinal_error"] = np.abs(_ordinal(out[reference_column]) - _ordinal(out[observed_column])).astype(int)
    return (
        out.groupby(list(strata), dropna=False, sort=True)
        .agg(
            n=("is_failure", "size"),
            failures=("is_failure", "sum"),
            failure_rate=("is_failure", "mean"),
            mean_ordinal_error=("ordinal_error", "mean"),
        )
        .reset_index()
    )

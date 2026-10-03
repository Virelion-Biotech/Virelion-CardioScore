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
    "l": 0,
    "m": 1,
    "h": 2,
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


def _label_ordinals(values: Iterable[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for value in values:
        key = _normalise_label(value)
        if key not in RISK_ORDER:
            raise ValueError(f"Unsupported risk label: {value!r}")
        out[key] = RISK_ORDER[key]
    return out


def _canonicalize_to_labels(values: Iterable[Any], labels: list[str]) -> np.ndarray:
    """Map compatible short/long risk labels onto configured label tokens."""
    label_ordinals = _label_ordinals(labels)
    ordinal_to_label: dict[int, str] = {}
    for label, ordinal in label_ordinals.items():
        if ordinal in ordinal_to_label and ordinal_to_label[ordinal] != label:
            raise ValueError(
                "Configured metric labels must contain at most one token per ordinal risk class."
            )
        ordinal_to_label[ordinal] = label

    canonical: list[str] = []
    for value in values:
        key = _normalise_label(value)
        if key in label_ordinals:
            canonical.append(key)
            continue
        if key not in RISK_ORDER:
            raise ValueError(f"Unsupported risk label: {value!r}")
        ordinal = RISK_ORDER[key]
        if ordinal not in ordinal_to_label:
            raise ValueError(
                f"Risk label {value!r} is not representable by configured metric labels {labels!r}."
            )
        canonical.append(ordinal_to_label[ordinal])
    return np.asarray(canonical)


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
    labels: tuple[str, ...] = ("low", "moderate", "high"),
) -> LockedMetrics:
    """Compute fixed metrics; this function never fits or changes a model."""
    label_values = [_normalise_label(x) for x in labels]
    if len(set(label_values)) != len(label_values):
        raise ValueError("labels must be unique")
    _label_ordinals(label_values)

    y_true = _canonicalize_to_labels(reference, label_values)
    y_pred = _canonicalize_to_labels(observed, label_values)
    if y_true.shape != y_pred.shape or y_true.size == 0:
        raise ValueError("reference and observed must have equal, non-zero length")

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


def binary_auroc_bootstrap(
    positive: Iterable[bool],
    scores: Iterable[float],
    *,
    n_bootstrap: int,
    seed: int,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """AUROC with a percentile bootstrap CI that resamples *compounds* (the independent unit).

    Resamples that contain only one class are skipped and counted; nothing is fitted or tuned.
    """
    from sklearn.metrics import roc_auc_score

    y = np.asarray(list(positive), dtype=bool)
    x = np.asarray(list(scores), dtype=float)
    if y.shape != x.shape or y.size == 0:
        raise ValueError("positive and scores must have equal, non-zero length")
    if not np.isfinite(x).all():
        raise ValueError("scores must be finite")
    if y.all() or (~y).all():
        raise ValueError("AUROC requires both positive and negative compounds")
    if not 0.0 < confidence < 1.0 or n_bootstrap < 1:
        raise ValueError("confidence must be in (0, 1) and n_bootstrap must be >= 1")
    auroc = float(roc_auc_score(y, x))
    rng = np.random.default_rng(int(seed))
    draws: list[float] = []
    for _ in range(int(n_bootstrap)):
        index = rng.integers(0, y.size, y.size)
        if y[index].all() or (~y[index]).all():
            continue
        draws.append(float(roc_auc_score(y[index], x[index])))
    if not draws:
        raise ValueError("No valid bootstrap resamples contained both classes")
    lower_q = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(draws, [lower_q, 1.0 - lower_q])
    return {
        "auroc": auroc,
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "confidence": float(confidence),
        "n": int(y.size),
        "n_positive": int(y.sum()),
        "n_negative": int((~y).sum()),
        "n_bootstrap_requested": int(n_bootstrap),
        "n_bootstrap_valid": len(draws),
        "seed": int(seed),
    }


def primary_outcome(result: dict[str, Any], rule: dict[str, Any]) -> str:
    """Apply the pre-registered three-outcome rule: 'success', 'inconclusive' or 'failure'."""
    for outcome in ("success", "inconclusive"):
        thresholds = rule[outcome]
        if (
            result["auroc"] >= float(thresholds["auroc_min"])
            and result["ci_lower"] >= float(thresholds["ci_lower_min"])
        ):
            return outcome
    return "failure"

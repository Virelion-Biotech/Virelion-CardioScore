"""Accuracy metrics for signal-level validation of beat and endpoint extraction.

Reference values come from ground truth (synthetic signals) or blinded human annotation. Agreement
between two *algorithms* is a method comparison, not accuracy, and should be labeled that way.
Correlation alone is never enough, so agreement reports bias, error and limits of agreement too.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable
from typing import Any

import numpy as np
import pandas as pd


def match_events(
    reference_s: Iterable[float], detected_s: Iterable[float], tolerance_s: float
) -> dict[str, Any]:
    """One-to-one nearest matching of detected to reference event times within a tolerance."""
    if tolerance_s <= 0:
        raise ValueError("tolerance_s must be positive")
    ref = np.sort(np.asarray(list(reference_s), dtype=float))
    det = np.sort(np.asarray(list(detected_s), dtype=float))
    if not (np.isfinite(ref).all() and np.isfinite(det).all()):
        raise ValueError("event times must be finite")
    used = np.zeros(det.size, dtype=bool)
    pairs: list[tuple[int, int]] = []
    for i, time in enumerate(ref):
        if det.size == 0:
            break
        distance = np.abs(det - time)
        distance[used] = np.inf
        j = int(np.argmin(distance))
        if distance[j] <= tolerance_s:
            used[j] = True
            pairs.append((i, j))
    errors = np.array([det[j] - ref[i] for i, j in pairs], dtype=float)
    return {
        "n_reference": int(ref.size),
        "n_detected": int(det.size),
        "tp": len(pairs),
        "fp": int(det.size - len(pairs)),
        "fn": int(ref.size - len(pairs)),
        "pairs": pairs,
        "timing_errors_s": errors,
    }


def event_detection_metrics(
    reference_s: Iterable[float], detected_s: Iterable[float], tolerance_s: float
) -> dict[str, Any]:
    matched = match_events(reference_s, detected_s, tolerance_s)
    tp, fp, fn = matched["tp"], matched["fp"], matched["fn"]
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    f1 = (
        2 * precision * recall / (precision + recall)
        if tp
        else (0.0 if (fp or fn) else float("nan"))
    )
    errors = matched["timing_errors_s"]
    return {
        "n_reference": matched["n_reference"],
        "n_detected": matched["n_detected"],
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "false_positive_rate_per_reference": fp / matched["n_reference"]
        if matched["n_reference"]
        else float("nan"),
        "false_negative_rate": fn / matched["n_reference"]
        if matched["n_reference"]
        else float("nan"),
        "timing_bias_s": float(errors.mean()) if errors.size else float("nan"),
        "timing_median_abs_error_s": float(np.median(np.abs(errors)))
        if errors.size
        else float("nan"),
        "timing_p95_abs_error_s": float(np.quantile(np.abs(errors), 0.95))
        if errors.size
        else float("nan"),
        "tolerance_s": float(tolerance_s),
    }


def lins_ccc(reference: np.ndarray, measured: np.ndarray) -> float:
    mean_r, mean_m = reference.mean(), measured.mean()
    var_r, var_m = reference.var(), measured.var()
    covariance = np.mean((reference - mean_r) * (measured - mean_m))
    denominator = var_r + var_m + (mean_r - mean_m) ** 2
    return float(2 * covariance / denominator) if denominator > 0 else float("nan")


def agreement_metrics(reference: Iterable[float], measured: Iterable[float]) -> dict[str, Any]:
    """Bias, MAE, RMSE, Bland-Altman limits of agreement, Lin's CCC and Pearson r (pairs must align)."""
    ref = np.asarray(list(reference), dtype=float)
    mea = np.asarray(list(measured), dtype=float)
    if ref.shape != mea.shape:
        raise ValueError("reference and measured must have the same length")
    keep = np.isfinite(ref) & np.isfinite(mea)
    n_missing = int((~keep).sum())
    ref, mea = ref[keep], mea[keep]
    if ref.size < 2:
        return {"n": int(ref.size), "n_missing_pairs": n_missing}
    diff = mea - ref
    sd = float(diff.std(ddof=1))
    pearson = (
        float(np.corrcoef(ref, mea)[0, 1]) if ref.std() > 0 and mea.std() > 0 else float("nan")
    )
    return {
        "n": int(ref.size),
        "n_missing_pairs": n_missing,
        "bias": float(diff.mean()),
        "mae": float(np.abs(diff).mean()),
        "rmse": float(np.sqrt(np.mean(diff**2))),
        "loa_lower": float(diff.mean() - 1.96 * sd),
        "loa_upper": float(diff.mean() + 1.96 * sd),
        "lins_ccc": lins_ccc(ref, mea),
        "pearson_r": pearson,
    }


def canonical_frame_hash(frame: pd.DataFrame) -> str:
    """SHA-256 of a DataFrame's canonical CSV bytes (column order fixed, floats to 12 significant digits)."""
    text = frame.to_csv(index=False, float_format="%.12g", lineterminator="\n")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def verify_deterministic_rebuild(
    build: Callable[[], pd.DataFrame], *, n_runs: int = 2
) -> dict[str, Any]:
    """Run ``build`` several times independently and require identical canonical output hashes."""
    if n_runs < 2:
        raise ValueError("n_runs must be at least 2")
    hashes = [canonical_frame_hash(build()) for _ in range(n_runs)]
    return {"hashes": hashes, "verified_rebuild": len(set(hashes)) == 1}
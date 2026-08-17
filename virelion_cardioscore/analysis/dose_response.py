"""Concentration-response fitting utilities for CardioScore.

This module provides an explicit four-parameter logistic (4PL) fit for
concentration-response series. Fitting is optional and should not be
interpreted as regulatory validation. Series with insufficient concentrations,
failed optimization, poor fit quality, or biologically trivial responses return
structured diagnostics instead of silently becoming scoring evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.optimize import curve_fit


_ENDPOINT_HARM_DIRECTIONS = {
    "fpd_change_pct": "absolute",
    "beat_rate_change_pct": "absolute",
    "amplitude_change_pct": "decrease",
    "stv_increase": "increase",
    "triangulation_proxy_change": "increase",
}

_ENDPOINT_EFFECT_THRESHOLDS = {
    "fpd_change_pct": 10.0,
    "beat_rate_change_pct": 15.0,
    "amplitude_change_pct": 20.0,
    "stv_increase": 0.15,
    "triangulation_proxy_change": 0.20,
}


@dataclass
class DoseResponseFit:
    endpoint: str
    success: bool
    quality_pass: bool
    n_points: int
    ec50: Optional[float] = None
    ec50_ci_low: Optional[float] = None
    ec50_ci_high: Optional[float] = None
    hill_slope: Optional[float] = None
    hill_ci_low: Optional[float] = None
    hill_ci_high: Optional[float] = None
    bottom: Optional[float] = None
    top: Optional[float] = None
    r_squared: Optional[float] = None
    rmse: Optional[float] = None
    weighted: bool = False
    monotonicity: Optional[float] = None
    monotonic_direction: Optional[str] = None
    harm_direction_compatible: Optional[bool] = None
    harmful_effect_magnitude: Optional[float] = None
    effect_threshold: Optional[float] = None
    effect_size_pass: Optional[bool] = None
    ec50_boundary_flag: bool = False
    ec50_uncertainty_fold: Optional[float] = None
    message: str = ""

    def to_dict(self) -> dict:
        return {
            "endpoint": self.endpoint,
            "success": self.success,
            "quality_pass": self.quality_pass,
            "n_points": self.n_points,
            "ec50": self.ec50,
            "ec50_ci_low": self.ec50_ci_low,
            "ec50_ci_high": self.ec50_ci_high,
            "hill_slope": self.hill_slope,
            "hill_ci_low": self.hill_ci_low,
            "hill_ci_high": self.hill_ci_high,
            "bottom": self.bottom,
            "top": self.top,
            "r_squared": self.r_squared,
            "rmse": self.rmse,
            "weighted": self.weighted,
            "monotonicity": self.monotonicity,
            "monotonic_direction": self.monotonic_direction,
            "harm_direction_compatible": self.harm_direction_compatible,
            "harmful_effect_magnitude": self.harmful_effect_magnitude,
            "effect_threshold": self.effect_threshold,
            "effect_size_pass": self.effect_size_pass,
            "ec50_boundary_flag": self.ec50_boundary_flag,
            "ec50_uncertainty_fold": self.ec50_uncertainty_fold,
            "message": self.message,
        }


def four_parameter_logistic(concentration: np.ndarray, bottom: float, top: float, ec50: float, hill_slope: float) -> np.ndarray:
    concentration = np.asarray(concentration, dtype=float)
    if np.any(concentration <= 0):
        raise ValueError("4PL fitting requires strictly positive concentrations.")
    return bottom + (top - bottom) / (1.0 + (ec50 / concentration) ** hill_slope)


def _ci95(value: float, standard_error: float) -> tuple[float, float]:
    delta = 1.96 * standard_error
    return value - delta, value + delta


def _monotonicity_score(x: np.ndarray, y: np.ndarray) -> tuple[float, str]:
    if len(x) < 2:
        return 1.0, "flat"
    deltas = np.diff(y)
    overall = float(y[-1] - y[0])
    if np.isclose(overall, 0.0):
        return 0.0, "flat"
    direction = 1.0 if overall > 0 else -1.0
    matches = np.sum((deltas * direction) >= 0.0)
    return float(matches / len(deltas)), "increasing" if direction > 0 else "decreasing"


def _harm_direction_compatible(endpoint: str, monotonic_direction: str | None, endpoint_directions: dict[str, str] | None = None) -> bool | None:
    expected = (endpoint_directions or {}).get(endpoint, _ENDPOINT_HARM_DIRECTIONS.get(endpoint))
    if expected is None or monotonic_direction in {None, "flat"}:
        return None if expected is None else False
    if expected == "absolute":
        return True
    return monotonic_direction == expected


def _harmful_effect_magnitude(y: np.ndarray, endpoint: str, endpoint_directions: dict[str, str] | None = None) -> float | None:
    direction = (endpoint_directions or {}).get(endpoint, _ENDPOINT_HARM_DIRECTIONS.get(endpoint))
    if direction is None:
        return None
    if direction == "absolute":
        return float(np.max(np.abs(y)))
    if direction == "increase":
        return float(max(0.0, np.max(y)))
    if direction == "decrease":
        return float(max(0.0, -np.min(y)))
    raise ValueError(f"Unsupported endpoint direction: {direction!r}.")


def fit_4pl(
    concentrations: np.ndarray,
    responses: np.ndarray,
    *,
    response_sem: Optional[np.ndarray] = None,
    endpoint: str = "endpoint",
    endpoint_directions: dict[str, str] | None = None,
    min_points: int = 4,
    min_r_squared: float = 0.0,
    min_monotonicity: float = 0.0,
    ec50_boundary_factor: float = 2.0,
    max_ec50_uncertainty_fold: float = 100.0,
    effect_threshold: float | None = None,
) -> DoseResponseFit:
    if effect_threshold is None:
        effect_threshold = _ENDPOINT_EFFECT_THRESHOLDS.get(endpoint)
    if effect_threshold is not None and effect_threshold < 0:
        raise ValueError("effect_threshold must be non-negative.")

    x = np.asarray(concentrations, dtype=float)
    y = np.asarray(responses, dtype=float)
    sigma = None if response_sem is None else np.asarray(response_sem, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y) & (x > 0)
    if sigma is not None:
        finite &= np.isfinite(sigma) & (sigma > 0)
    x = x[finite]
    y = y[finite]
    if sigma is not None:
        sigma = sigma[finite]

    if len(x) < min_points:
        return DoseResponseFit(endpoint=endpoint, success=False, quality_pass=False, n_points=len(x), effect_threshold=effect_threshold, message=f"Need at least {min_points} positive concentrations; got {len(x)}.")

    order = np.argsort(x)
    x = x[order]
    y = y[order]
    if sigma is not None:
        sigma = sigma[order]

    if np.unique(x).size < min_points:
        return DoseResponseFit(endpoint=endpoint, success=False, quality_pass=False, n_points=int(np.unique(x).size), effect_threshold=effect_threshold, message=f"Need at least {min_points} distinct positive concentrations.")

    span = float(np.max(y) - np.min(y))
    harmful_effect_magnitude = _harmful_effect_magnitude(y, endpoint, endpoint_directions)
    effect_size_pass = None if effect_threshold is None or harmful_effect_magnitude is None else harmful_effect_magnitude >= effect_threshold

    if span <= 1e-12:
        return DoseResponseFit(endpoint=endpoint, success=False, quality_pass=False, n_points=len(x), effect_threshold=effect_threshold, harmful_effect_magnitude=harmful_effect_magnitude, effect_size_pass=False if effect_threshold is not None else None, message="Response has negligible dynamic range; 4PL fit is not identifiable.")

    p0 = [float(np.min(y)), float(np.max(y)), float(np.median(x)), 1.0]
    y_min, y_max = float(np.min(y)), float(np.max(y))
    margin = max(1.0, span * 2.0)
    lower = [y_min - margin, y_min - margin, float(np.min(x)) * 1e-6, -10.0]
    upper = [y_max + margin, y_max + margin, float(np.max(x)) * 1e6, 10.0]

    try:
        params, covariance = curve_fit(four_parameter_logistic, x, y, p0=p0, bounds=(lower, upper), sigma=sigma, absolute_sigma=sigma is not None, maxfev=20000)
    except (RuntimeError, ValueError) as exc:
        return DoseResponseFit(endpoint=endpoint, success=False, quality_pass=False, n_points=len(x), weighted=sigma is not None, harmful_effect_magnitude=harmful_effect_magnitude, effect_threshold=effect_threshold, effect_size_pass=effect_size_pass, message=f"4PL optimization failed: {exc}")

    fitted = four_parameter_logistic(x, *params)
    residuals = y - fitted
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    rmse = float(np.sqrt(np.mean(residuals**2)))
    standard_errors = np.sqrt(np.clip(np.diag(covariance), 0.0, np.inf))
    ec50, hill_slope = float(params[2]), float(params[3])
    ec50_se, hill_se = float(standard_errors[2]), float(standard_errors[3])
    ec50_ci_low, ec50_ci_high = _ci95(ec50, ec50_se)
    hill_ci_low, hill_ci_high = _ci95(hill_slope, hill_se)
    monotonicity, monotonic_direction = _monotonicity_score(x, y)
    harm_direction_compatible = _harm_direction_compatible(endpoint, monotonic_direction, endpoint_directions)
    boundary_low = ec50 < float(np.min(x)) * ec50_boundary_factor
    boundary_high = ec50 > float(np.max(x)) / ec50_boundary_factor
    ec50_boundary_flag = bool(boundary_low or boundary_high)
    ec50_uncertainty_fold = float(ec50_ci_high / ec50_ci_low) if ec50 > 0 and ec50_ci_low > 0 else None
    finite_ci = all(np.isfinite(value) for value in [ec50_ci_low, ec50_ci_high, hill_ci_low, hill_ci_high])
    quality_pass = bool(np.isfinite(r_squared) and r_squared >= min_r_squared and finite_ci and ec50_ci_low > 0 and monotonicity >= min_monotonicity and harm_direction_compatible is not False and (effect_size_pass is not False) and not ec50_boundary_flag and ec50_uncertainty_fold is not None and ec50_uncertainty_fold <= max_ec50_uncertainty_fold)

    reasons = []
    if not np.isfinite(r_squared) or r_squared < min_r_squared:
        reasons.append(f"R-squared below {min_r_squared:.2f}")
    if monotonicity < min_monotonicity:
        reasons.append(f"monotonicity below {min_monotonicity:.2f}")
    if harm_direction_compatible is False:
        reasons.append("fitted concentration-response direction is not the configured harmful direction")
    if effect_size_pass is False:
        reasons.append(f"harmful response magnitude below configured effect threshold {float(effect_threshold):.4g}")
    if ec50_boundary_flag:
        reasons.append("EC50 lies near/outside the tested concentration range")
    if ec50_ci_low <= 0:
        reasons.append("EC50 confidence interval is not strictly positive")
    if ec50_uncertainty_fold is None or ec50_uncertainty_fold > max_ec50_uncertainty_fold:
        reasons.append("EC50 uncertainty is too wide")
    if not finite_ci:
        reasons.append("parameter confidence intervals are non-finite")

    message = "Fit passed quality criteria." if quality_pass else "Fit converged but failed quality criteria: " + "; ".join(dict.fromkeys(reasons))
    return DoseResponseFit(endpoint=endpoint, success=True, quality_pass=quality_pass, n_points=len(x), ec50=ec50, ec50_ci_low=float(ec50_ci_low), ec50_ci_high=float(ec50_ci_high), hill_slope=hill_slope, hill_ci_low=float(hill_ci_low), hill_ci_high=float(hill_ci_high), bottom=float(params[0]), top=float(params[1]), r_squared=float(r_squared), rmse=rmse, weighted=sigma is not None, monotonicity=monotonicity, monotonic_direction=monotonic_direction, harm_direction_compatible=harm_direction_compatible, harmful_effect_magnitude=harmful_effect_magnitude, effect_threshold=effect_threshold, effect_size_pass=effect_size_pass, ec50_boundary_flag=ec50_boundary_flag, ec50_uncertainty_fold=ec50_uncertainty_fold, message=message)


def fit_concentration_series(
    concentration_summary,
    *,
    endpoint_columns: Optional[list[str]] = None,
    endpoint_directions: Optional[dict[str, str]] = None,
    endpoint_thresholds: Optional[dict[str, float]] = None,
    min_points: int = 4,
    min_r_squared: float = 0.0,
    min_monotonicity: float = 0.0,
    ec50_boundary_factor: float = 2.0,
    max_ec50_uncertainty_fold: float = 100.0,
) -> list[DoseResponseFit]:
    """Fit all requested endpoint mean columns in a concentration summary."""
    if endpoint_columns is None:
        endpoint_columns = ["fpd_change_pct_mean", "beat_rate_change_pct_mean", "amplitude_change_pct_mean", "stv_increase_mean", "triangulation_proxy_change_mean"]
    results: list[DoseResponseFit] = []
    for endpoint_column in endpoint_columns:
        if endpoint_column not in concentration_summary.columns:
            continue
        endpoint = endpoint_column.removesuffix("_mean")
        sem_column = endpoint_column.removesuffix("_mean") + "_sem"
        response_sem = concentration_summary[sem_column].to_numpy(dtype=float) if sem_column in concentration_summary.columns else None
        results.append(fit_4pl(concentration_summary["concentration_uM"].to_numpy(dtype=float), concentration_summary[endpoint_column].to_numpy(dtype=float), response_sem=response_sem, endpoint=endpoint, endpoint_directions=endpoint_directions, min_points=min_points, min_r_squared=min_r_squared, min_monotonicity=min_monotonicity, ec50_boundary_factor=ec50_boundary_factor, max_ec50_uncertainty_fold=max_ec50_uncertainty_fold, effect_threshold=(endpoint_thresholds or {}).get(endpoint)))
    return results

"""Concentration-response fitting and quality gates for CardioScore."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.optimize import curve_fit

_DEFAULT_MIN_EC50_COVERAGE = 0.10


@dataclass(frozen=True)
class DoseResponseFit:
    endpoint: str
    success: bool
    quality_pass: bool
    n_points: int
    ec50: float | None = None
    ec50_ci_low: float | None = None
    ec50_ci_high: float | None = None
    hill_slope: float | None = None
    hill_ci_low: float | None = None
    hill_ci_high: float | None = None
    bottom: float | None = None
    top: float | None = None
    r_squared: float | None = None
    rmse: float | None = None
    weighted: bool = False
    monotonicity: float | None = None
    monotonic_direction: str | None = None
    harm_direction_compatible: bool | None = None
    harmful_effect_magnitude: float | None = None
    effect_threshold: float | None = None
    effect_size_pass: bool | None = None
    ec50_coverage: float | None = None
    min_ec50_coverage: float = _DEFAULT_MIN_EC50_COVERAGE
    coverage_pass: bool | None = None
    ec50_boundary_flag: bool = False
    ec50_uncertainty_fold: float | None = None
    message: str = ""

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def _four_pl(x, bottom, top, ec50, hill):
    return bottom + (top - bottom) / (1.0 + (ec50 / x) ** hill)


def _ci95(value: float, se: float) -> tuple[float, float]:
    half_width = 1.96 * se
    return float(value - half_width), float(value + half_width)


def _positive_parameter_ci95(value: float, se: float) -> tuple[float, float]:
    if not np.isfinite(value) or not np.isfinite(se) or value <= 0:
        return np.nan, np.nan
    half_width = 1.96 * se
    lower_log = np.log(value) - half_width / value
    upper_log = np.log(value) + half_width / value
    return float(np.exp(np.clip(lower_log, -745.0, 709.0))), float(np.exp(np.clip(upper_log, -745.0, 709.0)))


def _monotonicity_score(x: np.ndarray, y: np.ndarray) -> tuple[float, str]:
    order = np.argsort(x)
    diffs = np.diff(y[order])
    if len(diffs) == 0:
        return 1.0, "flat"
    increasing = float(np.mean(diffs >= 0))
    decreasing = float(np.mean(diffs <= 0))
    if increasing >= decreasing:
        return increasing, "increase"
    return decreasing, "decrease"


def _fitted_monotonic_direction(bottom: float, top: float, hill: float) -> str:
    if np.isclose(top, bottom):
        return "flat"
    return "increase" if (top - bottom) * hill > 0 else "decrease"


def _harm_direction_compatible(endpoint: str, fitted_direction: str, endpoint_directions: Optional[dict[str, str]]) -> bool | None:
    if not endpoint_directions or endpoint not in endpoint_directions or fitted_direction == "flat":
        return None
    configured = endpoint_directions[endpoint]
    if configured in {"absolute", "abs"}:
        return True
    if configured in {"increase", "inc"}:
        return fitted_direction == "increase"
    if configured in {"decrease", "dec"}:
        return fitted_direction == "decrease"
    raise ValueError(f"Unsupported endpoint direction: {configured!r} for {endpoint!r}.")


def _fitted_harmful_effect_magnitude(bottom: float, top: float, endpoint: str, endpoint_directions: Optional[dict[str, str]]) -> float | None:
    direction = (endpoint_directions or {}).get(endpoint, "absolute")
    delta = top - bottom
    if direction in {"absolute", "abs"}:
        return abs(delta)
    if direction in {"increase", "inc"}:
        return max(delta, 0.0)
    if direction in {"decrease", "dec"}:
        return max(-delta, 0.0)
    raise ValueError(f"Unsupported endpoint direction: {direction!r} for {endpoint!r}.")


def fit_4pl(
    concentrations,
    responses,
    *,
    response_sem=None,
    endpoint: str = "endpoint",
    endpoint_directions: Optional[dict[str, str]] = None,
    effect_threshold: float | None = None,
    min_points: int = 4,
    min_r_squared: float = 0.0,
    min_monotonicity: float = 0.0,
    ec50_boundary_factor: float = 2.0,
    max_ec50_uncertainty_fold: float = 100.0,
    min_ec50_coverage: float = _DEFAULT_MIN_EC50_COVERAGE,
) -> DoseResponseFit:
    x = np.asarray(concentrations, dtype=float)
    y = np.asarray(responses, dtype=float)
    sigma = None if response_sem is None else np.asarray(response_sem, dtype=float)
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("4PL fitting requires one-dimensional concentration and response arrays.")
    if x.shape != y.shape:
        raise ValueError("4PL fitting requires concentrations and responses with identical lengths.")
    if sigma is not None and (sigma.ndim != 1 or sigma.shape != x.shape):
        raise ValueError("4PL response_sem must be one-dimensional and match the response length.")
    if sigma is not None and (not np.isfinite(sigma).all() or np.any(sigma <= 0)):
        raise ValueError("4PL response_sem must contain only finite positive values.")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("4PL concentrations and responses must contain only finite values.")
    if np.any(x <= 0):
        raise ValueError("4PL concentrations must be strictly positive for log-scale fitting.")
    if np.unique(x).size != len(x):
        raise ValueError("4PL fitting requires one response per concentration; duplicate concentrations were supplied.")
    if len(x) < min_points:
        return DoseResponseFit(endpoint=endpoint, success=False, quality_pass=False, n_points=len(x), message=f"At least {min_points} concentrations are required for 4PL fitting.")
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    if sigma is not None:
        sigma = sigma[order]
    try:
        p0 = [float(np.min(y)), float(np.max(y)), float(np.median(x)), 1.0]
        bounds = ([-np.inf, -np.inf, float(np.min(x)) / 100.0, -20.0], [np.inf, np.inf, float(np.max(x)) * 100.0, 20.0])
        params, covariance = curve_fit(_four_pl, x, y, p0=p0, sigma=sigma, absolute_sigma=sigma is not None, bounds=bounds, maxfev=50000)
    except Exception as exc:  # pragma: no cover
        return DoseResponseFit(endpoint=endpoint, success=False, quality_pass=False, n_points=len(x), message=f"4PL fitting failed: {exc}")
    if covariance is None or not np.isfinite(covariance).all():
        return DoseResponseFit(endpoint=endpoint, success=False, quality_pass=False, n_points=len(x), message="4PL covariance is unavailable or non-finite.")
    fitted = _four_pl(x, *params)
    residuals = y - fitted
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((y - np.mean(y))**2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    rmse = float(np.sqrt(np.mean(residuals**2)))
    standard_errors = np.sqrt(np.clip(np.diag(covariance), 0.0, np.inf))
    ec50, hill_slope = float(params[2]), float(params[3])
    bottom, top = float(params[0]), float(params[1])
    ec50_se, hill_se = float(standard_errors[2]), float(standard_errors[3])
    ec50_ci_low, ec50_ci_high = _positive_parameter_ci95(ec50, ec50_se)
    hill_ci_low, hill_ci_high = _ci95(hill_slope, hill_se)
    monotonicity, _observed_monotonic_direction = _monotonicity_score(x, y)
    fitted_monotonic_direction = _fitted_monotonic_direction(bottom, top, hill_slope)
    harm_direction_compatible = _harm_direction_compatible(endpoint, fitted_monotonic_direction, endpoint_directions)
    fitted_harmful_effect = _fitted_harmful_effect_magnitude(bottom, top, endpoint, endpoint_directions)
    effect_size_pass = None if effect_threshold is None or fitted_harmful_effect is None else fitted_harmful_effect >= effect_threshold
    boundary_low = ec50 < float(np.min(x)) * ec50_boundary_factor
    boundary_high = ec50 > float(np.max(x)) / ec50_boundary_factor
    ec50_boundary_flag = bool(boundary_low or boundary_high)
    log_span = float(np.log10(np.max(x) / np.min(x)))
    ec50_coverage = float(np.clip(np.log10(np.max(x) / ec50) / log_span, 0.0, 1.0)) if log_span > 0 and ec50 > 0 else 0.0
    coverage_pass = ec50_coverage >= min_ec50_coverage
    ec50_uncertainty_fold = float(ec50_ci_high / ec50_ci_low) if ec50 > 0 and ec50_ci_low > 0 else None
    finite_ci = all(np.isfinite(value) for value in [ec50_ci_low, ec50_ci_high, hill_ci_low, hill_ci_high])
    quality_pass = bool(np.isfinite(r_squared) and r_squared >= min_r_squared and finite_ci and ec50_ci_low > 0 and monotonicity >= min_monotonicity and harm_direction_compatible is not False and (effect_size_pass is not False) and coverage_pass and not ec50_boundary_flag and ec50_uncertainty_fold is not None and ec50_uncertainty_fold <= max_ec50_uncertainty_fold)

    reasons = []
    if not np.isfinite(r_squared) or r_squared < min_r_squared:
        reasons.append(f"R-squared below {min_r_squared:.2f}")
    if monotonicity < min_monotonicity:
        reasons.append(f"monotonicity below {min_monotonicity:.2f}")
    if harm_direction_compatible is False:
        reasons.append("fitted concentration-response direction is not the configured harmful direction")
    if effect_size_pass is False:
        reasons.append(f"harmful fitted response magnitude below configured effect threshold {float(effect_threshold):.4g}")
    if not coverage_pass:
        reasons.append(f"EC50 coverage below minimum {min_ec50_coverage:.2f}")
    if ec50_boundary_flag:
        reasons.append("EC50 lies near/outside the tested concentration range")
    if ec50_ci_low <= 0:
        reasons.append("EC50 confidence interval is not strictly positive")
    if ec50_uncertainty_fold is None or ec50_uncertainty_fold > max_ec50_uncertainty_fold:
        reasons.append("EC50 uncertainty is too wide")
    if not finite_ci:
        reasons.append("parameter confidence intervals are non-finite")

    message = "Fit passed quality criteria." if quality_pass else "Fit converged but failed quality criteria: " + "; ".join(dict.fromkeys(reasons))
    return DoseResponseFit(endpoint=endpoint, success=True, quality_pass=quality_pass, n_points=len(x), ec50=ec50, ec50_ci_low=float(ec50_ci_low), ec50_ci_high=float(ec50_ci_high), hill_slope=hill_slope, hill_ci_low=float(hill_ci_low), hill_ci_high=float(hill_ci_high), bottom=bottom, top=top, r_squared=float(r_squared), rmse=rmse, weighted=sigma is not None, monotonicity=monotonicity, monotonic_direction=fitted_monotonic_direction, harm_direction_compatible=harm_direction_compatible, harmful_effect_magnitude=fitted_harmful_effect, effect_threshold=effect_threshold, effect_size_pass=effect_size_pass, ec50_coverage=ec50_coverage, min_ec50_coverage=min_ec50_coverage, coverage_pass=coverage_pass, ec50_boundary_flag=ec50_boundary_flag, ec50_uncertainty_fold=ec50_uncertainty_fold, message=message)


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
    min_ec50_coverage: float = _DEFAULT_MIN_EC50_COVERAGE,
) -> list[DoseResponseFit]:
    """Fit all requested endpoint mean columns in a concentration summary."""
    explicit_endpoint_columns = endpoint_columns is not None
    if endpoint_columns is None:
        endpoint_columns = ["fpd_change_pct_mean", "beat_rate_change_pct_mean", "amplitude_change_pct_mean", "stv_increase_mean", "triangulation_proxy_change_mean"]
    results: list[DoseResponseFit] = []
    for endpoint_column in endpoint_columns:
        if endpoint_column not in concentration_summary.columns:
            if explicit_endpoint_columns:
                raise ValueError(f"Requested concentration-response endpoint column {endpoint_column!r} is absent from the concentration summary.")
            continue
        endpoint = endpoint_column.removesuffix("_mean")
        sem_column = endpoint_column.removesuffix("_mean") + "_sem"
        response_sem = concentration_summary[sem_column].to_numpy(dtype=float) if sem_column in concentration_summary.columns else None
        results.append(
            fit_4pl(
                concentration_summary["concentration_uM"].to_numpy(dtype=float),
                concentration_summary[endpoint_column].to_numpy(dtype=float),
                response_sem=response_sem,
                endpoint=endpoint,
                endpoint_directions=endpoint_directions,
                min_points=min_points,
                min_r_squared=min_r_squared,
                min_monotonicity=min_monotonicity,
                ec50_boundary_factor=ec50_boundary_factor,
                max_ec50_uncertainty_fold=max_ec50_uncertainty_fold,
                effect_threshold=(endpoint_thresholds or {}).get(endpoint),
                min_ec50_coverage=min_ec50_coverage,
            )
        )
    return results

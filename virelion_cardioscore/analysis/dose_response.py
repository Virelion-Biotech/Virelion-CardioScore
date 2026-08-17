"""Concentration-response fitting utilities for CardioScore.

This module provides an explicit four-parameter logistic (4PL) fit for
concentration-response series. Fitting is optional and should not be
interpreted as regulatory validation. Series with insufficient concentrations
or failed optimization return a structured failure instead of a fabricated fit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.optimize import curve_fit


@dataclass
class DoseResponseFit:
    """Result of a four-parameter logistic concentration-response fit."""

    endpoint: str
    success: bool
    n_points: int
    ec50: Optional[float] = None
    hill_slope: Optional[float] = None
    bottom: Optional[float] = None
    top: Optional[float] = None
    r_squared: Optional[float] = None
    rmse: Optional[float] = None
    message: str = ""

    def to_dict(self) -> dict:
        return {
            "endpoint": self.endpoint,
            "success": self.success,
            "n_points": self.n_points,
            "ec50": self.ec50,
            "hill_slope": self.hill_slope,
            "bottom": self.bottom,
            "top": self.top,
            "r_squared": self.r_squared,
            "rmse": self.rmse,
            "message": self.message,
        }


def four_parameter_logistic(
    concentration: np.ndarray,
    bottom: float,
    top: float,
    ec50: float,
    hill_slope: float,
) -> np.ndarray:
    """Evaluate a 4PL model at positive concentrations."""
    concentration = np.asarray(concentration, dtype=float)
    if np.any(concentration <= 0):
        raise ValueError("4PL fitting requires strictly positive concentrations.")
    return bottom + (top - bottom) / (1.0 + (ec50 / concentration) ** hill_slope)


def fit_4pl(
    concentrations: np.ndarray,
    responses: np.ndarray,
    *,
    endpoint: str = "endpoint",
    min_points: int = 4,
) -> DoseResponseFit:
    """Fit a bounded four-parameter logistic model."""
    x = np.asarray(concentrations, dtype=float)
    y = np.asarray(responses, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y) & (x > 0)
    x = x[finite]
    y = y[finite]

    if len(x) < min_points:
        return DoseResponseFit(
            endpoint=endpoint,
            success=False,
            n_points=len(x),
            message=f"Need at least {min_points} positive concentrations; got {len(x)}.",
        )

    order = np.argsort(x)
    x = x[order]
    y = y[order]
    if np.unique(x).size < min_points:
        return DoseResponseFit(
            endpoint=endpoint,
            success=False,
            n_points=int(np.unique(x).size),
            message=f"Need at least {min_points} distinct positive concentrations.",
        )

    span = float(np.max(y) - np.min(y))
    if span <= 1e-12:
        return DoseResponseFit(
            endpoint=endpoint,
            success=False,
            n_points=len(x),
            message="Response has negligible dynamic range; 4PL fit is not identifiable.",
        )

    p0 = [float(np.min(y)), float(np.max(y)), float(np.median(x)), 1.0]
    y_min = float(np.min(y))
    y_max = float(np.max(y))
    margin = max(1.0, span * 2.0)
    lower = [y_min - margin, y_min - margin, float(np.min(x)) * 1e-6, -10.0]
    upper = [y_max + margin, y_max + margin, float(np.max(x)) * 1e6, 10.0]

    try:
        params, _ = curve_fit(
            four_parameter_logistic,
            x,
            y,
            p0=p0,
            bounds=(lower, upper),
            maxfev=20000,
        )
    except (RuntimeError, ValueError) as exc:
        return DoseResponseFit(
            endpoint=endpoint,
            success=False,
            n_points=len(x),
            message=f"4PL optimization failed: {exc}",
        )

    fitted = four_parameter_logistic(x, *params)
    residuals = y - fitted
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    rmse = float(np.sqrt(np.mean(residuals**2)))

    return DoseResponseFit(
        endpoint=endpoint,
        success=True,
        n_points=len(x),
        ec50=float(params[2]),
        hill_slope=float(params[3]),
        bottom=float(params[0]),
        top=float(params[1]),
        r_squared=float(r_squared),
        rmse=rmse,
        message="4PL fit converged.",
    )


def fit_concentration_series(
    concentration_summary,
    *,
    endpoint_columns: Optional[list[str]] = None,
    min_points: int = 4,
) -> list[DoseResponseFit]:
    """Fit all requested endpoint mean columns in a concentration summary."""
    if endpoint_columns is None:
        endpoint_columns = [
            "fpd_change_pct_mean",
            "beat_rate_change_pct_mean",
            "amplitude_change_pct_mean",
            "stv_increase_mean",
            "triangulation_proxy_change_mean",
        ]

    results: list[DoseResponseFit] = []
    for endpoint_column in endpoint_columns:
        if endpoint_column not in concentration_summary.columns:
            continue
        endpoint = endpoint_column.removesuffix("_mean")
        results.append(
            fit_4pl(
                concentration_summary["concentration_uM"].to_numpy(dtype=float),
                concentration_summary[endpoint_column].to_numpy(dtype=float),
                endpoint=endpoint,
                min_points=min_points,
            )
        )
    return results

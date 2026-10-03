"""Validation utilities for control-anchored normalization.

These utilities evaluate whether normalization reduces control drift while
preserving a treatment effect against an independently supplied reference.
They are intended for method validation and synthetic benchmarks, not for
automatic correction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from virelion_cardioscore.analysis.normalization import apply_control_anchor_correction
from virelion_cardioscore.utils.coercion import coerce_bool_series


@dataclass(frozen=True)
class NormalizationValidationResult:
    """Summary of normalization performance on a known dataset."""

    group_column: str
    endpoint: str
    control_between_sd_before: float
    control_between_sd_after: float
    control_cv_before_pct: Optional[float]
    control_cv_after_pct: Optional[float]
    treatment_effect_rmse: float
    reference_treatment_effect: Optional[float]
    n_groups: int
    n_controls: int
    n_treated: int
    passed_drift_reduction: bool
    passed_effect_preservation: bool

    def to_dict(self) -> dict:
        return {
            "group_column": self.group_column,
            "endpoint": self.endpoint,
            "control_between_sd_before": self.control_between_sd_before,
            "control_between_sd_after": self.control_between_sd_after,
            "control_cv_before_pct": self.control_cv_before_pct,
            "control_cv_after_pct": self.control_cv_after_pct,
            "treatment_effect_rmse": self.treatment_effect_rmse,
            "reference_treatment_effect": self.reference_treatment_effect,
            "n_groups": self.n_groups,
            "n_controls": self.n_controls,
            "n_treated": self.n_treated,
            "passed_drift_reduction": self.passed_drift_reduction,
            "passed_effect_preservation": self.passed_effect_preservation,
        }


def _group_control_stats(
    df: pd.DataFrame,
    group_column: str,
    endpoint: str,
) -> tuple[pd.Series, float | None]:
    """Return group-level control means and their between-group CV."""
    working = df.copy()
    working["vehicle"] = coerce_bool_series(working["vehicle"], name="vehicle")
    controls = working[working["vehicle"]].copy()
    values = pd.to_numeric(controls[endpoint], errors="coerce")
    control_frame = pd.DataFrame({"_value": values, group_column: controls[group_column]})
    control_frame = control_frame.dropna(subset=["_value"])
    group_means = control_frame.groupby(group_column, dropna=False)["_value"].mean()
    overall_mean = float(group_means.mean()) if not group_means.empty else np.nan
    if len(group_means) > 1 and not np.isclose(overall_mean, 0.0):
        cv = float(abs(group_means.std(ddof=1) / overall_mean) * 100.0)
    else:
        cv = None
    return group_means, cv


def validate_control_anchor_correction(
    df: pd.DataFrame,
    *,
    group_column: str,
    endpoint: str,
    min_controls_per_group: int = 2,
    expected_treatment_effect: float | None = None,
    max_effect_rmse: float = 1e-9,
) -> tuple[pd.DataFrame, NormalizationValidationResult]:
    """Validate control-drift reduction against an independent treatment reference.

    The normalization shift is learned from vehicle controls. Effect preservation
    is only assessed when ``expected_treatment_effect`` is supplied independently
    of the transformation. Comparing the same treatment contrast before and
    after an additive correction is intentionally not used as a validation test,
    because that identity holds by construction.
    """
    required = {"vehicle", group_column, endpoint}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing columns for normalization validation: {missing}.")
    if expected_treatment_effect is not None and not np.isfinite(expected_treatment_effect):
        raise ValueError("expected_treatment_effect must be finite when provided.")
    if max_effect_rmse < 0:
        raise ValueError("max_effect_rmse must be non-negative.")

    working = df.copy()
    working["vehicle"] = coerce_bool_series(working["vehicle"], name="vehicle")
    corrected, _ = apply_control_anchor_correction(
        working,
        group_column=group_column,
        corrected_columns=[endpoint],
        min_controls_per_group=min_controls_per_group,
    )

    before_group_means, before_cv = _group_control_stats(working, group_column, endpoint)
    after_group_means, after_cv = _group_control_stats(corrected, group_column, endpoint)

    if expected_treatment_effect is None:
        rmse = np.nan
        passed_effect_preservation = False
    else:
        treated = corrected.loc[~corrected["vehicle"]]
        controls = corrected.loc[corrected["vehicle"]]
        control_means = controls.groupby(group_column)[endpoint].mean()
        treated_means = treated.groupby(group_column)[endpoint].mean()
        observed_effects = treated_means.subtract(control_means, fill_value=np.nan).dropna()
        rmse = (
            float(np.sqrt(np.mean((observed_effects.to_numpy(dtype=float) - expected_treatment_effect) ** 2)))
            if not observed_effects.empty
            else np.nan
        )
        passed_effect_preservation = bool(np.isfinite(rmse) and rmse <= max_effect_rmse)

    result = NormalizationValidationResult(
        group_column=group_column,
        endpoint=endpoint,
        control_between_sd_before=float(before_group_means.std(ddof=1)) if len(before_group_means) > 1 else 0.0,
        control_between_sd_after=float(after_group_means.std(ddof=1)) if len(after_group_means) > 1 else 0.0,
        control_cv_before_pct=before_cv,
        control_cv_after_pct=after_cv,
        treatment_effect_rmse=rmse,
        reference_treatment_effect=expected_treatment_effect,
        n_groups=len(before_group_means),
        n_controls=int(working["vehicle"].sum()),
        n_treated=int((~working["vehicle"]).sum()),
        passed_drift_reduction=(
            float(after_group_means.std(ddof=1)) < float(before_group_means.std(ddof=1))
            if len(before_group_means) > 1
            else False
        ),
        passed_effect_preservation=passed_effect_preservation,
    )
    return corrected, result

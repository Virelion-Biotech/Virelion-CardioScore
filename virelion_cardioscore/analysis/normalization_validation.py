"""Validation utilities for control-anchored normalization.

These utilities evaluate whether normalization reduces control drift while
preserving within-group treatment-control contrasts. They are intended for
method validation and synthetic benchmarks, not for automatic correction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from virelion_cardioscore.analysis.normalization import apply_control_anchor_correction


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
    controls = df[df["vehicle"]]
    values = pd.to_numeric(controls[endpoint], errors="coerce")
    group_means = values.groupby(controls[group_column], dropna=False).mean()
    overall_mean = float(values.mean()) if values.notna().any() else np.nan
    cv = None if np.isclose(overall_mean, 0.0) else float(abs(values.std(ddof=1) / overall_mean) * 100.0)
    return group_means, cv


def validate_control_anchor_correction(
    df: pd.DataFrame,
    *,
    group_column: str,
    endpoint: str,
    min_controls_per_group: int = 2,
    max_effect_rmse: float = 1e-9,
) -> tuple[pd.DataFrame, NormalizationValidationResult]:
    """Validate additive correction on a dataset with within-group treatment effects.

    The preservation criterion compares treatment-control contrasts within each
    group before and after correction. For an additive correction this RMSE
    should be numerically zero up to floating-point tolerance.
    """
    required = {"vehicle", group_column, endpoint}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing columns for normalization validation: {missing}.")

    corrected, _ = apply_control_anchor_correction(
        df,
        group_column=group_column,
        corrected_columns=[endpoint],
        min_controls_per_group=min_controls_per_group,
    )

    before_group_means, before_cv = _group_control_stats(df, group_column, endpoint)
    after_group_means, after_cv = _group_control_stats(corrected, group_column, endpoint)

    before_effects = df[~df["vehicle"]].copy()
    after_effects = corrected[~corrected["vehicle"]].copy()

    def treatment_effects(frame: pd.DataFrame, controls: pd.DataFrame) -> pd.Series:
        control_means = controls.groupby(group_column)[endpoint].mean()
        treated_means = frame.groupby(group_column)[endpoint].mean()
        return treated_means.subtract(control_means, fill_value=np.nan).dropna()

    before_effects_by_group = treatment_effects(before_effects, df[df["vehicle"]])
    after_effects_by_group = treatment_effects(after_effects, corrected[corrected["vehicle"]])
    aligned = pd.concat(
        [before_effects_by_group.rename("before"), after_effects_by_group.rename("after")],
        axis=1,
        join="inner",
    ).dropna()
    rmse = float(np.sqrt(np.mean((aligned["after"] - aligned["before"]) ** 2))) if not aligned.empty else np.nan

    result = NormalizationValidationResult(
        group_column=group_column,
        endpoint=endpoint,
        control_between_sd_before=float(before_group_means.std(ddof=1)) if len(before_group_means) > 1 else 0.0,
        control_between_sd_after=float(after_group_means.std(ddof=1)) if len(after_group_means) > 1 else 0.0,
        control_cv_before_pct=before_cv,
        control_cv_after_pct=after_cv,
        treatment_effect_rmse=rmse,
        n_groups=len(before_group_means),
        n_controls=int(df["vehicle"].sum()),
        n_treated=int((~df["vehicle"].astype(bool)).sum()),
        passed_drift_reduction=(
            float(after_group_means.std(ddof=1)) < float(before_group_means.std(ddof=1))
            if len(before_group_means) > 1
            else False
        ),
        passed_effect_preservation=bool(np.isfinite(rmse) and rmse <= max_effect_rmse),
    )
    return corrected, result

"""Diagnostics for assumptions behind control-anchored normalization.

These checks do not alter data. They identify designs where a simple additive,
control-anchored correction may be poorly justified because treatment allocation
is imbalanced or group control shifts behave more like scale changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class NormalizationAssumptionCheck:
    """Machine-readable diagnostics for additive control correction."""

    group_column: str
    endpoint: str
    n_groups: int
    groups_with_treatment: int
    treatment_group_coverage: float
    additive_shift_cv_pct: Optional[float]
    geometric_shift_cv_pct: Optional[float]
    treatment_allocation_imbalanced: bool
    scale_like_drift_flag: bool
    usable_for_additive_correction: bool
    message: str

    def to_dict(self) -> dict:
        return {
            "group_column": self.group_column,
            "endpoint": self.endpoint,
            "n_groups": self.n_groups,
            "groups_with_treatment": self.groups_with_treatment,
            "treatment_group_coverage": self.treatment_group_coverage,
            "additive_shift_cv_pct": self.additive_shift_cv_pct,
            "geometric_shift_cv_pct": self.geometric_shift_cv_pct,
            "treatment_allocation_imbalanced": self.treatment_allocation_imbalanced,
            "scale_like_drift_flag": self.scale_like_drift_flag,
            "usable_for_additive_correction": self.usable_for_additive_correction,
            "message": self.message,
        }


def check_additive_correction_assumptions(
    df: pd.DataFrame,
    *,
    group_column: str,
    endpoint: str,
    min_treated_per_group: int = 1,
    max_shift_cv_pct: float = 50.0,
    require_treatment_in_all_groups: bool = True,
) -> NormalizationAssumptionCheck:
    """Assess treatment allocation and additive-shift plausibility.

    This is a screening diagnostic, not a formal statistical test. The
    additive-shift metric uses absolute shift magnitudes so symmetric positive
    and negative plate offsets are not falsely classified as heterogeneous.
    The geometric metric measures dispersion of positive control baselines on a
    log scale and acts only as a scale-drift warning.
    """
    required = {"vehicle", group_column, endpoint}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing columns for normalization assumption check: {missing}.")

    working = df.copy()
    working["_endpoint"] = pd.to_numeric(working[endpoint], errors="coerce")
    controls = working[working["vehicle"] == True].dropna(subset=["_endpoint"])  # noqa: E712
    treated = working[working["vehicle"] == False].dropna(subset=["_endpoint"])  # noqa: E712

    if controls.empty:
        raise ValueError("Normalization assumption check requires vehicle wells.")

    control_means = controls.groupby(group_column, dropna=False)["_endpoint"].mean()
    global_mean = float(control_means.mean())
    additive_shifts = global_mean - control_means

    positive_means = control_means[control_means > 0]
    if len(positive_means) > 1:
        log_sd = float(np.log(positive_means).std(ddof=1))
        geometric_cv = float((np.exp(log_sd) - 1.0) * 100.0)
    else:
        geometric_cv = None

    if len(additive_shifts) > 1:
        abs_shift_mean = float(additive_shifts.abs().mean())
        shift_cv = (
            float(abs(additive_shifts.abs().std(ddof=1) / abs_shift_mean) * 100.0)
            if not np.isclose(abs_shift_mean, 0.0)
            else 0.0
        )
    else:
        shift_cv = None

    total_groups = int(control_means.size)
    treated_counts = treated.groupby(group_column, dropna=False).size()
    groups_with_treatment = int(
        sum(
            group in treated_counts.index and treated_counts[group] >= min_treated_per_group
            for group in control_means.index
        )
    )
    coverage = groups_with_treatment / total_groups if total_groups else 0.0
    imbalance = groups_with_treatment < total_groups if require_treatment_in_all_groups else False
    scale_like = bool(
        len(positive_means) >= 2
        and geometric_cv is not None
        and geometric_cv > max_shift_cv_pct
    )

    usable = not imbalance and not scale_like
    if imbalance:
        message = "Treatment observations are missing or insufficient in at least one control group."
    elif scale_like:
        message = "Positive group control baselines show scale-like dispersion above the configured threshold; additive recentering may be a poor model."
    else:
        message = "No obvious treatment-allocation or scale-drift warning was detected."

    return NormalizationAssumptionCheck(
        group_column=group_column,
        endpoint=endpoint,
        n_groups=total_groups,
        groups_with_treatment=groups_with_treatment,
        treatment_group_coverage=float(coverage),
        additive_shift_cv_pct=shift_cv,
        geometric_shift_cv_pct=geometric_cv,
        treatment_allocation_imbalanced=imbalance,
        scale_like_drift_flag=scale_like,
        usable_for_additive_correction=usable,
        message=message,
    )

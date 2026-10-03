"""Optional hierarchical inference for CardioScore experiments.

This module provides a random-intercept mixed-effects model for endpoint-level
inference. It is intentionally separate from CardioScore scoring: the model
quantifies treatment effects while accounting for a declared experimental
unit, but its output does not change the risk score automatically.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MixedEffectsResult:
    """Summary of a random-intercept treatment model."""

    endpoint: str
    treatment_column: str
    group_column: str
    n_observations: int
    n_groups: int
    treatment_effect: float
    treatment_se: float
    treatment_pvalue: float
    group_variance: float
    residual_variance: float
    icc: float
    converged: bool

    def to_dict(self) -> dict:
        return {
            "endpoint": self.endpoint,
            "treatment_column": self.treatment_column,
            "group_column": self.group_column,
            "n_observations": self.n_observations,
            "n_groups": self.n_groups,
            "treatment_effect": self.treatment_effect,
            "treatment_se": self.treatment_se,
            "treatment_pvalue": self.treatment_pvalue,
            "group_variance": self.group_variance,
            "residual_variance": self.residual_variance,
            "icc": self.icc,
            "converged": self.converged,
        }


def _resolve_group_column(df: pd.DataFrame, group_column: Optional[str]) -> str:
    """Resolve grouping metadata using the same hierarchy as scoring."""
    if group_column is not None:
        if group_column not in df.columns:
            raise ValueError(f"Mixed-effects grouping column {group_column!r} is absent from the dataset.")
        return group_column

    for candidate in ("biological_replicate", "batch_id", "experiment_id", "plate_id"):
        if candidate in df.columns:
            return candidate

    raise ValueError(
        "Mixed-effects modeling requires a genuine grouping column such as "
        "'biological_replicate', 'batch_id', 'experiment_id', or 'plate_id'."
    )


def _namespace_group_identifier(df: pd.DataFrame, group_column: str) -> tuple[pd.DataFrame, str]:
    """Namespace reused group IDs by site/experiment when those fields exist."""
    namespace_columns = [
        column
        for column in ("site", "experiment_id")
        if column in df.columns and column != group_column
    ]
    for column in [*namespace_columns, group_column]:
        if df[column].isna().any() or df[column].astype(str).str.strip().eq("").any():
            raise ValueError(
                f"Mixed-effects grouping metadata column {column!r} contains missing or blank identifiers."
            )

    if not namespace_columns:
        return df, group_column

    working = df.copy()
    key_columns = [*namespace_columns, group_column]
    working["_mixed_effects_group_key"] = working[key_columns].astype(str).agg("::".join, axis=1)
    return working, "_mixed_effects_group_key"


def fit_random_intercept(
    df: pd.DataFrame,
    *,
    endpoint: str,
    treatment_column: str = "treatment",
    group_column: Optional[str] = None,
) -> MixedEffectsResult:
    """Fit ``endpoint ~ treatment + (1|group)`` using statsmodels.

    The model is an inference/diagnostic layer only. It does not modify
    CardioScore. The grouping hierarchy is biological replicate -> batch /
    experiment -> plate unless a group is explicitly configured.
    """
    try:
        import statsmodels.formula.api as smf
    except ImportError as exc:  # pragma: no cover - exercised by environment
        raise ImportError(
            "Mixed-effects modeling requires the optional 'mixed' dependency. "
            "Install with: pip install 'virelion-cardioscore[mixed]'"
        ) from exc

    if endpoint not in df.columns:
        raise ValueError(f"Endpoint column {endpoint!r} is absent from the dataset.")
    if treatment_column not in df.columns:
        raise ValueError(f"Treatment column {treatment_column!r} is absent from the dataset.")

    resolved_group = _resolve_group_column(df, group_column)
    working, model_group_column = _namespace_group_identifier(df, resolved_group)
    model_df = working[[endpoint, treatment_column, model_group_column]].copy()

    model_df[endpoint] = pd.to_numeric(model_df[endpoint], errors="coerce")
    model_df[treatment_column] = pd.to_numeric(model_df[treatment_column], errors="coerce")
    if model_df[endpoint].isna().any():
        raise ValueError(f"Endpoint column {endpoint!r} contains missing or non-numeric observations.")
    if model_df[treatment_column].isna().any():
        raise ValueError(f"Treatment column {treatment_column!r} contains missing or non-numeric observations.")
    if not np.isfinite(model_df[endpoint].to_numpy()).all():
        raise ValueError(f"Endpoint column {endpoint!r} contains non-finite observations.")
    if not np.isfinite(model_df[treatment_column].to_numpy()).all():
        raise ValueError(f"Treatment column {treatment_column!r} contains non-finite observations.")
    if not set(model_df[treatment_column].unique()).issubset({0, 1}):
        raise ValueError(f"Treatment column {treatment_column!r} must contain only 0/1 values.")

    if model_df.empty:
        raise ValueError("No complete observations remain for mixed-effects modeling.")
    n_groups = int(model_df[model_group_column].nunique())
    if n_groups < 3:
        raise ValueError("Mixed-effects modeling requires at least three independent groups.")
    if model_df[treatment_column].nunique() < 2:
        raise ValueError("Mixed-effects modeling requires both treatment levels.")

    group_counts = model_df.groupby(model_group_column, sort=False).size()
    if (group_counts < 2).any():
        raise ValueError(
            "Mixed-effects modeling requires at least two observations in every independent group."
        )

    formula = f"Q('{endpoint}') ~ Q('{treatment_column}')"
    model = smf.mixedlm(formula, model_df, groups=model_df[model_group_column])
    fit = model.fit(reml=True, method=["lbfgs", "powell", "cg"], disp=False)
    if not bool(getattr(fit, "converged", False)):
        raise ValueError("Mixed-effects model did not converge; inference is unavailable for this endpoint/grouping.")

    treatment_term = f"Q('{treatment_column}')"
    treatment_effect = float(fit.params[treatment_term])
    treatment_se = float(fit.bse[treatment_term])
    treatment_pvalue = float(fit.pvalues[treatment_term])
    group_variance = float(fit.cov_re.iloc[0, 0])
    residual_variance = float(fit.scale)
    denominator = group_variance + residual_variance
    icc = float(group_variance / denominator) if denominator > 0 else np.nan

    return MixedEffectsResult(
        endpoint=endpoint,
        treatment_column=treatment_column,
        group_column=resolved_group,
        n_observations=int(len(model_df)),
        n_groups=n_groups,
        treatment_effect=treatment_effect,
        treatment_se=treatment_se,
        treatment_pvalue=treatment_pvalue,
        group_variance=group_variance,
        residual_variance=residual_variance,
        icc=icc,
        converged=True,
    )

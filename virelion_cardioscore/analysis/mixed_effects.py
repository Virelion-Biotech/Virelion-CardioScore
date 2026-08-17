"""Optional hierarchical inference for CardioScore experiments.

This module provides a random-intercept mixed-effects model for endpoint-level
inference. It is intentionally separate from CardioScore scoring: the model
quantifies treatment effects while accounting for plate/batch clustering, but
its output does not change the risk score automatically.
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


def fit_random_intercept(
    df: pd.DataFrame,
    *,
    endpoint: str,
    treatment_column: str = "treatment",
    group_column: Optional[str] = None,
) -> MixedEffectsResult:
    """Fit ``endpoint ~ treatment + (1|group)`` using statsmodels.

    The model is an inference/diagnostic layer only. It does not modify
    CardioScore. ``group_column`` should represent a genuine higher-level
    experimental unit such as plate, batch, or biological experiment.
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

    if group_column is None:
        for candidate in ("plate_id", "batch_id", "experiment_id"):
            if candidate in df.columns:
                group_column = candidate
                break
    if group_column is None or group_column not in df.columns:
        raise ValueError(
            "Mixed-effects modeling requires a genuine grouping column such as "
            "'plate_id', 'batch_id', or 'experiment_id'."
        )

    model_df = df[[endpoint, treatment_column, group_column]].copy()
    model_df[endpoint] = pd.to_numeric(model_df[endpoint], errors="coerce")
    model_df[treatment_column] = pd.to_numeric(model_df[treatment_column], errors="coerce")
    model_df = model_df.dropna()
    if model_df.empty:
        raise ValueError("No complete observations remain for mixed-effects modeling.")
    if model_df[group_column].nunique() < 2:
        raise ValueError("Mixed-effects modeling requires at least two groups.")
    if model_df[treatment_column].nunique() < 2:
        raise ValueError("Mixed-effects modeling requires both treatment levels.")

    formula = f"Q('{endpoint}') ~ Q('{treatment_column}')"
    model = smf.mixedlm(formula, model_df, groups=model_df[group_column])
    fit = model.fit(reml=True, method=["lbfgs", "powell", "cg"], disp=False)

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
        group_column=group_column,
        n_observations=int(len(model_df)),
        n_groups=int(model_df[group_column].nunique()),
        treatment_effect=treatment_effect,
        treatment_se=treatment_se,
        treatment_pvalue=treatment_pvalue,
        group_variance=group_variance,
        residual_variance=residual_variance,
        icc=icc,
        converged=bool(getattr(fit, "converged", False)),
    )

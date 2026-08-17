"""Plate/batch variability diagnostics for CardioScore.

This module quantifies control stability across experimental groups. It is a
QC/inference layer only; it does not alter CardioScore or claim a mixed-effects
model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class VariabilityDiagnostic:
    """Summary of between-group control variability for one endpoint."""

    endpoint: str
    group_column: str
    n_groups: int
    n_controls: int
    control_mean: float
    control_sd: Optional[float]
    control_cv_pct: Optional[float]
    between_group_sd: Optional[float]
    status: str
    message: str = ""

    def to_dict(self) -> dict:
        return {
            "endpoint": self.endpoint,
            "group_column": self.group_column,
            "n_groups": self.n_groups,
            "n_controls": self.n_controls,
            "control_mean": self.control_mean,
            "control_sd": self.control_sd,
            "control_cv_pct": self.control_cv_pct,
            "between_group_sd": self.between_group_sd,
            "status": self.status,
            "message": self.message,
        }


def _resolve_group_column(df: pd.DataFrame, group_column: Optional[str]) -> str:
    if group_column is not None:
        if group_column not in df.columns:
            raise ValueError(f"Requested variability group column {group_column!r} is absent.")
        return group_column
    for candidate in ("plate_id", "batch_id", "experiment_id"):
        if candidate in df.columns:
            return candidate
    raise ValueError(
        "Plate/batch variability requires one of 'plate_id', 'batch_id', or 'experiment_id'."
    )


def control_variability(
    df: pd.DataFrame,
    *,
    endpoint_columns: Optional[list[str]] = None,
    group_column: Optional[str] = None,
    max_control_cv_pct: float = 20.0,
) -> pd.DataFrame:
    """Quantify control stability across plates/batches."""
    if df.empty:
        return pd.DataFrame()
    if "vehicle" not in df.columns:
        raise ValueError("Variability diagnostics require a 'vehicle' column.")

    group = _resolve_group_column(df, group_column)
    if endpoint_columns is None:
        endpoint_columns = [
            "fpd_ms",
            "beat_rate_bpm",
            "amplitude_uv",
            "stv",
            "triangulation_proxy",
        ]

    controls = df[df["vehicle"] == True].copy()  # noqa: E712
    rows: list[dict] = []
    for endpoint in endpoint_columns:
        if endpoint not in controls.columns:
            continue
        control = controls.copy()
        control["_value"] = pd.to_numeric(control[endpoint], errors="coerce")
        control = control.dropna(subset=["_value"])
        if control.empty:
            continue

        group_means = control.groupby(group, dropna=False)["_value"].mean()
        overall_mean = float(group_means.mean())
        n_groups = int(group_means.size)
        n_controls = int(len(control))
        control_sd = float(control["_value"].std(ddof=1)) if n_controls > 1 else None
        cv = None
        if control_sd is not None and not np.isclose(overall_mean, 0.0):
            cv = float(abs(control_sd / overall_mean) * 100.0)
        between_sd = float(group_means.std(ddof=1)) if n_groups > 1 else None

        if n_groups < 2:
            status = "insufficient_groups"
            message = "At least two plates/batches are required to estimate between-group variability."
        elif cv is not None and cv > max_control_cv_pct:
            status = "high_variability"
            message = f"Control CV {cv:.2f}% exceeds configured threshold {max_control_cv_pct:.2f}%."
        else:
            status = "stable"
            message = "Control variability is within the configured threshold."

        rows.append(
            VariabilityDiagnostic(
                endpoint=endpoint,
                group_column=group,
                n_groups=n_groups,
                n_controls=n_controls,
                control_mean=overall_mean,
                control_sd=control_sd,
                control_cv_pct=cv,
                between_group_sd=between_sd,
                status=status,
                message=message,
            ).to_dict()
        )
    return pd.DataFrame(rows)


def standardized_treatment_separation(
    effects: pd.DataFrame,
    *,
    endpoint: str,
    group_column: str,
) -> pd.DataFrame:
    """Return an exploratory treatment-vs-control separation within each group."""
    required = {endpoint, "vehicle", "compound", group_column}
    missing = sorted(required - set(effects.columns))
    if missing:
        raise ValueError(f"Missing columns for treatment separation: {missing}.")

    rows: list[dict] = []
    for keys, group in effects.groupby(["compound", group_column], dropna=False, sort=True):
        compound, group_value = keys
        controls = pd.to_numeric(group.loc[group["vehicle"] == True, endpoint], errors="coerce").dropna()  # noqa: E712
        treated = pd.to_numeric(group.loc[group["vehicle"] == False, endpoint], errors="coerce").dropna()  # noqa: E712
        if len(controls) < 2 or treated.empty:
            separation = np.nan
        else:
            control_mean = float(controls.mean())
            treated_mean = float(treated.mean())
            control_sd = float(controls.std(ddof=1))
            separation = abs(treated_mean - control_mean) / control_sd if control_sd > 0 else np.nan
        rows.append(
            {
                "compound": compound,
                group_column: group_value,
                "endpoint": endpoint,
                "n_controls": int(len(controls)),
                "n_treated": int(len(treated)),
                "standardized_separation": separation,
            }
        )
    return pd.DataFrame(rows)

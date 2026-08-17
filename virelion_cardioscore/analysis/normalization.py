"""Control-anchored batch normalization for CardioScore.

The correction is intentionally conservative: group-specific shifts are learned
from vehicle controls only and applied to all wells in that group. It is an
exploratory normalization layer, not a mixed-effects or regulatory correction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CorrectionDiagnostic:
    """Record how a control-anchored correction was applied."""

    group_column: str
    n_groups: int
    n_controls: int
    corrected_columns: tuple[str, ...]
    target_means: dict[str, float]
    group_shifts: dict[str, dict[str, float]]

    def to_dict(self) -> dict:
        return {
            "group_column": self.group_column,
            "n_groups": self.n_groups,
            "n_controls": self.n_controls,
            "corrected_columns": list(self.corrected_columns),
            "target_means": self.target_means,
            "group_shifts": self.group_shifts,
        }


def _resolve_group_column(df: pd.DataFrame, group_column: Optional[str]) -> str:
    if group_column is not None:
        if group_column not in df.columns:
            raise ValueError(f"Requested correction group column {group_column!r} is absent.")
        return group_column
    for candidate in ("plate_id", "batch_id", "experiment_id"):
        if candidate in df.columns:
            return candidate
    raise ValueError(
        "Control-anchored correction requires one of 'plate_id', 'batch_id', or 'experiment_id'."
    )


def apply_control_anchor_correction(
    df: pd.DataFrame,
    *,
    group_column: Optional[str] = None,
    corrected_columns: Optional[list[str]] = None,
    min_controls_per_group: int = 2,
    require_all_groups: bool = True,
) -> tuple[pd.DataFrame, CorrectionDiagnostic]:
    """Recenter selected endpoints using vehicle-only group control means.

    For each experimental group ``g`` and endpoint ``x``:

        corrected_x = x - mean(vehicle_g) + mean(vehicle_all)

    This preserves within-group treatment-control differences while expressing
    observations on a common control-centered scale. It is appropriate only for
    endpoints where additive recentering is scientifically defensible.
    """
    if df.empty:
        raise ValueError("Cannot correct an empty dataset.")
    if "vehicle" not in df.columns:
        raise ValueError("Control-anchored correction requires a 'vehicle' column.")

    group = _resolve_group_column(df, group_column)
    if corrected_columns is None:
        corrected_columns = [
            "fpd_ms",
            "beat_rate_bpm",
            "amplitude_uv",
            "stv",
            "triangulation_proxy",
        ]

    missing_columns = [column for column in corrected_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Correction columns are absent from the dataset: {missing_columns}.")
    if min_controls_per_group < 1:
        raise ValueError("min_controls_per_group must be at least 1.")

    working = df.copy()
    controls = working[working["vehicle"] == True]  # noqa: E712
    if controls.empty:
        raise ValueError("Control-anchored correction requires at least one vehicle well.")

    group_sizes = controls.groupby(group, dropna=False).size()
    insufficient = group_sizes[group_sizes < min_controls_per_group]
    if not insufficient.empty and require_all_groups:
        raise ValueError(
            "Control-anchored correction requires at least "
            f"{min_controls_per_group} vehicle wells per group; insufficient groups: "
            f"{list(insufficient.index)}."
        )

    usable_groups = group_sizes[group_sizes >= min_controls_per_group].index
    controls_usable = controls[controls[group].isin(usable_groups)].copy()
    if controls_usable.empty:
        raise ValueError("No experimental groups meet the minimum control requirement.")

    target_means: dict[str, float] = {}
    group_shifts: dict[str, dict[str, float]] = {}
    for column in corrected_columns:
        values = pd.to_numeric(controls_usable[column], errors="coerce").dropna()
        if values.empty:
            raise ValueError(f"No finite vehicle values are available for correction column {column!r}.")
        target_means[column] = float(values.mean())

    for group_value, indices in working.groupby(group, dropna=False).groups.items():
        if group_value not in usable_groups:
            if require_all_groups:
                raise ValueError(f"Group {group_value!r} cannot be corrected because it lacks sufficient controls.")
            continue
        group_controls = controls_usable[controls_usable[group] == group_value]
        shifts: dict[str, float] = {}
        for column in corrected_columns:
            group_mean = pd.to_numeric(group_controls[column], errors="coerce").dropna().mean()
            if not np.isfinite(group_mean):
                raise ValueError(f"Group {group_value!r} has no finite vehicle values for {column!r}.")
            shift = float(target_means[column] - group_mean)
            working.loc[indices, column] = pd.to_numeric(working.loc[indices, column], errors="coerce") + shift
            shifts[column] = shift
        group_shifts[str(group_value)] = shifts

    diagnostic = CorrectionDiagnostic(
        group_column=group,
        n_groups=int(len(usable_groups)),
        n_controls=int(len(controls_usable)),
        corrected_columns=tuple(corrected_columns),
        target_means=target_means,
        group_shifts=group_shifts,
    )
    return working, diagnostic

"""Hierarchy-aware experimental-unit summaries for CardioScore.

The pipeline historically treated wells as replicates. This module adds an
explicit hierarchy layer for datasets that contain biological or experimental
batch identifiers, without requiring those columns for existing datasets.

No mixed-effects model is claimed here. The goal is to prevent accidental
pseudoreplication by making the available experimental unit visible and by
summarizing technical wells within higher-level units before inference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

OPTIONAL_HIERARCHY_COLUMNS = (
    "biological_replicate",
    "batch_id",
    "plate_id",
    "experiment_id",
)

SUPPORTED_SCORING_UNITS = ("well", "biological_replicate", "batch", "plate")


@dataclass(frozen=True)
class HierarchySpec:
    """Describe which columns define experimental hierarchy."""

    biological_unit: Optional[str] = None
    batch_unit: Optional[str] = None
    plate_unit: Optional[str] = None


def detect_hierarchy(
    df: pd.DataFrame,
    *,
    biological_unit_column: Optional[str] = None,
    batch_unit_column: Optional[str] = None,
    plate_unit_column: Optional[str] = None,
) -> HierarchySpec:
    """Detect hierarchy columns, honoring explicit configured aliases when supplied."""
    biological = biological_unit_column or "biological_replicate"
    batch = batch_unit_column or ("batch_id" if "batch_id" in df.columns else "experiment_id")
    plate = plate_unit_column or "plate_id"
    return HierarchySpec(
        biological_unit=biological if biological in df.columns else None,
        batch_unit=batch if batch in df.columns else None,
        plate_unit=plate if plate in df.columns else None,
    )


def hierarchy_columns(
    df: pd.DataFrame,
    *,
    biological_unit_column: Optional[str] = None,
    batch_unit_column: Optional[str] = None,
    plate_unit_column: Optional[str] = None,
) -> list[str]:
    """Return detected hierarchy columns in biological-to-technical order."""
    spec = detect_hierarchy(
        df,
        biological_unit_column=biological_unit_column,
        batch_unit_column=batch_unit_column,
        plate_unit_column=plate_unit_column,
    )
    return [
        column
        for column in [spec.biological_unit, spec.batch_unit, spec.plate_unit]
        if column is not None
    ]


def _require_complete_identifier(df: pd.DataFrame, column: str) -> None:
    if df[column].isna().any():
        missing = int(df[column].isna().sum())
        raise ValueError(
            f"Experimental-unit column {column!r} contains {missing} missing value(s); "
            "missing identifiers cannot be treated as one shared unit."
        )
    if df[column].astype(str).str.strip().eq("").any():
        raise ValueError(
            f"Experimental-unit column {column!r} contains blank identifiers."
        )


def _resolve_scoring_column(
    df: pd.DataFrame,
    scoring_unit: str,
    *,
    biological_unit_column: Optional[str] = None,
    batch_unit_column: Optional[str] = None,
    plate_unit_column: Optional[str] = None,
) -> str:
    if scoring_unit not in SUPPORTED_SCORING_UNITS:
        raise ValueError(
            f"Unsupported scoring_unit: {scoring_unit!r}. "
            f"Expected one of {SUPPORTED_SCORING_UNITS}."
        )

    if scoring_unit == "well":
        column = "well"
    elif scoring_unit == "biological_replicate":
        column = biological_unit_column or "biological_replicate"
    elif scoring_unit == "batch":
        column = batch_unit_column or ("batch_id" if "batch_id" in df.columns else "experiment_id")
    else:
        column = plate_unit_column or "plate_id"

    if column not in df.columns:
        raise ValueError(
            f"scoring_unit={scoring_unit!r} requires column {column!r}, "
            "but that metadata is not present in the dataset."
        )
    _require_complete_identifier(df, column)
    return column


def aggregate_to_scoring_units(
    effects: pd.DataFrame,
    *,
    scoring_unit: str = "well",
    endpoint_columns: Optional[list[str]] = None,
    biological_unit_column: Optional[str] = None,
    batch_unit_column: Optional[str] = None,
    plate_unit_column: Optional[str] = None,
) -> pd.DataFrame:
    """Aggregate technical wells to the configured independent scoring unit."""
    if effects.empty or scoring_unit == "well":
        return effects.copy()

    if endpoint_columns is None:
        endpoint_columns = [
            "fpd_change_pct",
            "beat_rate_change_pct",
            "amplitude_change_pct",
            "stv_increase",
            "triangulation_proxy_change",
            "max_effect_pct",
        ]

    unit_column = _resolve_scoring_column(
        effects,
        scoring_unit,
        biological_unit_column=biological_unit_column,
        batch_unit_column=batch_unit_column,
        plate_unit_column=plate_unit_column,
    )
    group_columns = ["compound", "concentration_uM", unit_column]
    rows: list[dict] = []
    for keys, group in effects.groupby(group_columns, sort=True, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_columns, keys, strict=True))
        row["well"] = f"{scoring_unit}:{keys[-1]}"
        row["n_wells"] = int(group["well"].nunique())
        for endpoint in endpoint_columns:
            if endpoint not in group.columns:
                continue
            values = pd.to_numeric(group[endpoint], errors="coerce").dropna()
            row[endpoint] = float(values.mean()) if len(values) else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_experimental_units(
    effects: pd.DataFrame,
    *,
    endpoint_columns: Optional[list[str]] = None,
    biological_unit_column: Optional[str] = None,
    batch_unit_column: Optional[str] = None,
    plate_unit_column: Optional[str] = None,
) -> pd.DataFrame:
    """Summarize wells at the highest available configured experimental-unit level."""
    if effects.empty:
        return pd.DataFrame()

    if endpoint_columns is None:
        endpoint_columns = [
            "fpd_change_pct",
            "beat_rate_change_pct",
            "amplitude_change_pct",
            "stv_increase",
            "triangulation_proxy",
        ]

    metadata = hierarchy_columns(
        effects,
        biological_unit_column=biological_unit_column,
        batch_unit_column=batch_unit_column,
        plate_unit_column=plate_unit_column,
    )
    biological = biological_unit_column or "biological_replicate"
    if biological in metadata:
        unit_columns = ["compound", "concentration_uM", biological]
    else:
        batch = batch_unit_column or ("batch_id" if "batch_id" in metadata else "experiment_id")
        if batch in metadata:
            unit_columns = ["compound", "concentration_uM", batch]
        else:
            plate = plate_unit_column or "plate_id"
            if plate in metadata:
                unit_columns = ["compound", "concentration_uM", plate]
            else:
                unit_columns = ["compound", "concentration_uM", "well"]

    for column in unit_columns[2:]:
        _require_complete_identifier(effects, column)

    grouped = effects.groupby(unit_columns, sort=True, dropna=False)
    rows: list[dict] = []
    for keys, group in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(unit_columns, keys, strict=True))
        row["n_wells"] = int(group["well"].nunique()) if "well" in group else len(group)
        for endpoint in endpoint_columns:
            if endpoint not in group.columns:
                continue
            values = pd.to_numeric(group[endpoint], errors="coerce").dropna()
            row[f"{endpoint}_mean"] = float(values.mean()) if len(values) else float("nan")
            row[f"{endpoint}_sd"] = float(values.std(ddof=1)) if len(values) > 1 else float("nan")
        rows.append(row)

    return pd.DataFrame(rows)


def count_independent_units(summary: pd.DataFrame) -> pd.DataFrame:
    """Count independent units contributing to each compound/concentration."""
    if summary.empty:
        return pd.DataFrame()

    if "biological_replicate" in summary.columns:
        unit = "biological_replicate"
    elif "batch_id" in summary.columns:
        unit = "batch_id"
    elif "experiment_id" in summary.columns:
        unit = "experiment_id"
    elif "plate_id" in summary.columns:
        unit = "plate_id"
    else:
        unit = "well"

    _require_complete_identifier(summary, unit)
    return (
        summary.groupby(["compound", "concentration_uM"], sort=True, dropna=False)[unit]
        .nunique()
        .reset_index(name="n_independent_units")
    )

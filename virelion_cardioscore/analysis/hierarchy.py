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


@dataclass(frozen=True)
class HierarchySpec:
    """Describe which columns define experimental hierarchy."""

    biological_unit: Optional[str] = None
    batch_unit: Optional[str] = None
    plate_unit: Optional[str] = None


def detect_hierarchy(df: pd.DataFrame) -> HierarchySpec:
    """Detect optional hierarchy columns without inventing metadata."""
    return HierarchySpec(
        biological_unit="biological_replicate" if "biological_replicate" in df.columns else None,
        batch_unit="batch_id" if "batch_id" in df.columns else (
            "experiment_id" if "experiment_id" in df.columns else None
        ),
        plate_unit="plate_id" if "plate_id" in df.columns else None,
    )


def hierarchy_columns(df: pd.DataFrame) -> list[str]:
    """Return detected hierarchy columns in biological-to-technical order."""
    spec = detect_hierarchy(df)
    return [
        column
        for column in [spec.biological_unit, spec.batch_unit, spec.plate_unit]
        if column is not None
    ]


def summarize_experimental_units(
    effects: pd.DataFrame,
    *,
    endpoint_columns: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Summarize wells at the highest available experimental-unit level.

    The returned table has one row per compound/concentration and, when
    metadata exist, per biological replicate. Without hierarchy metadata it
    falls back to well-level summaries, preserving historical behavior.
    """
    if effects.empty:
        return pd.DataFrame()

    if endpoint_columns is None:
        endpoint_columns = [
            "fpd_change_pct",
            "beat_rate_change_pct",
            "amplitude_change_pct",
            "stv_increase",
            "triangulation_proxy_change",
        ]

    metadata = hierarchy_columns(effects)
    if "biological_replicate" in metadata:
        unit_columns = ["compound", "concentration_uM", "biological_replicate"]
    elif "batch_id" in metadata or "experiment_id" in metadata:
        batch = "batch_id" if "batch_id" in metadata else "experiment_id"
        unit_columns = ["compound", "concentration_uM", batch]
    elif "plate_id" in metadata:
        unit_columns = ["compound", "concentration_uM", "plate_id"]
    else:
        unit_columns = ["compound", "concentration_uM", "well"]

    grouped = effects.groupby(unit_columns, sort=True, dropna=False)
    rows: list[dict] = []
    for keys, group in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(unit_columns, keys))
        row["n_wells"] = int(group["well"].nunique()) if "well" in group else len(group)
        for endpoint in endpoint_columns:
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

    result = (
        summary.groupby(["compound", "concentration_uM"], sort=True, dropna=False)[unit]
        .nunique()
        .reset_index(name="n_independent_units")
    )
    return result

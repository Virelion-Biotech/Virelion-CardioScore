"""
Raw MEA trace ingestion.

Reads raw, per-electrode voltage traces from CSV and runs them through
preprocessing.filtering + features.endpoints to produce the same
well-level feature table consumed by CardioScorePipeline.

Required input columns are defined by ``REQUIRED_COLUMNS``. Optional
experimental-design metadata (plate, batch, biological replicate, and
experiment ID) are preserved through feature extraction so downstream
hierarchical analysis does not lose the experimental unit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from virelion_cardioscore.features.endpoints import extract_well_features
from virelion_cardioscore.preprocessing.beat_detection import BeatDetectionConfig
from virelion_cardioscore.preprocessing.filtering import FilterConfig

REQUIRED_COLUMNS = {
    "compound",
    "well",
    "concentration_uM",
    "vehicle",
    "electrode_id",
    "time_s",
    "voltage_uv",
}
OPTIONAL_METADATA_COLUMNS = (
    "plate_id",
    "batch_id",
    "experiment_id",
    "biological_replicate",
)


class RawTraceSchemaError(ValueError):
    """Raised when an input CSV doesn't match the expected raw-trace schema."""


@dataclass
class RawWellRecording:
    """Raw multi-electrode voltage traces and preserved experimental metadata for a well."""

    compound: str
    well: str
    concentration_uM: float
    vehicle: bool
    fs_hz: float
    electrode_traces: dict[str, np.ndarray] = field(default_factory=dict)
    metadata: dict[str, object] = field(default_factory=dict)


def validate_raw_trace_schema(df: pd.DataFrame) -> None:
    """Validate required raw-trace columns and basic numerical integrity."""
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise RawTraceSchemaError(
            f"Raw trace CSV is missing required column(s): {sorted(missing)}. "
            f"Expected columns: {sorted(REQUIRED_COLUMNS)}."
        )
    if df.empty:
        raise RawTraceSchemaError("Raw trace CSV has no rows.")

    for col in ("time_s", "voltage_uv", "concentration_uM"):
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise RawTraceSchemaError(
                f"Column '{col}' must be numeric, got dtype {df[col].dtype}. "
                "Check for stray text, units, or header rows in the CSV."
            )

    if not np.isfinite(df[["time_s", "voltage_uv", "concentration_uM"]].to_numpy(dtype=float)).all():
        raise RawTraceSchemaError("Raw trace numeric columns must contain only finite values.")
    if (df["concentration_uM"] < 0).any():
        raise RawTraceSchemaError("concentration_uM cannot be negative.")
    if (df["time_s"] < 0).any():
        raise RawTraceSchemaError("time_s cannot be negative.")

    n_nan = int(df[["time_s", "voltage_uv"]].isna().sum().sum())
    if n_nan > 0:
        raise RawTraceSchemaError(
            f"Found {n_nan} missing value(s) in time_s/voltage_uv. Raw traces must be complete."
        )

    unique_vehicle = set(pd.Series(df["vehicle"]).astype(str).str.lower().unique())
    allowed = {"true", "false", "1", "0", "1.0", "0.0", "yes", "no"}
    if not unique_vehicle.issubset(allowed):
        raise RawTraceSchemaError(
            f"Column 'vehicle' has unexpected values {sorted(unique_vehicle)}; expected True/False or 0/1."
        )

    for column in OPTIONAL_METADATA_COLUMNS:
        if column in df.columns:
            values = df[column]
            if values.isna().any() or values.astype(str).str.strip().eq("").any():
                raise RawTraceSchemaError(
                    f"Optional metadata column '{column}' contains missing or blank identifiers."
                )


def _coerce_vehicle(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin({"true", "1", "1.0", "yes"})


def _infer_sampling_rate(time_s: np.ndarray) -> float:
    """Infer fs_hz from the median gap between consecutive timestamps."""
    ordered = np.sort(np.asarray(time_s, dtype=float))
    diffs = np.diff(ordered)
    diffs = diffs[diffs > 0]
    if len(diffs) == 0:
        raise RawTraceSchemaError(
            "Could not infer sampling rate: time_s has no positive gaps between samples."
        )
    median_dt = float(np.median(diffs))
    return 1.0 / median_dt


def load_raw_traces_csv(path: str | Path) -> list[RawWellRecording]:
    """Load a long-format raw MEA trace CSV into per-well recordings."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Raw trace CSV not found: {path}")

    df = pd.read_csv(path)
    validate_raw_trace_schema(df)
    df = df.copy()
    df["vehicle"] = _coerce_vehicle(df["vehicle"])

    metadata_columns = [column for column in OPTIONAL_METADATA_COLUMNS if column in df.columns]
    group_cols = ["compound", "well", "concentration_uM", "vehicle", *metadata_columns]
    recordings: list[RawWellRecording] = []

    for keys, well_df in df.groupby(group_cols, sort=False, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        base_count = 4
        compound, well, conc, vehicle = keys[:base_count]
        metadata = dict(zip(metadata_columns, keys[base_count:]))
        electrode_traces: dict[str, np.ndarray] = {}
        fs_values = []
        for electrode_id, edf in well_df.groupby("electrode_id", sort=False):
            edf = edf.sort_values("time_s")
            times = edf["time_s"].to_numpy(dtype=float)
            if np.unique(times).size != len(times):
                raise RawTraceSchemaError(
                    f"Duplicate timestamps found for compound={compound!r}, well={well!r}, electrode={electrode_id!r}."
                )
            fs_hz = _infer_sampling_rate(times)
            fs_values.append(fs_hz)
            electrode_traces[str(electrode_id)] = edf["voltage_uv"].to_numpy(dtype=float)

        if not fs_values:
            raise RawTraceSchemaError(f"Well {well!r} contains no electrode traces.")
        fs_hz = float(np.median(fs_values))
        if max(fs_values) / min(fs_values) > 1.01:
            raise RawTraceSchemaError(
                f"Inconsistent sampling rates across electrodes in well {well!r}: {fs_values}."
            )

        recordings.append(
            RawWellRecording(
                compound=str(compound),
                well=str(well),
                concentration_uM=float(conc),
                vehicle=bool(vehicle),
                fs_hz=fs_hz,
                electrode_traces=electrode_traces,
                metadata=metadata,
            )
        )

    return recordings


def recordings_to_feature_table(
    recordings: list[RawWellRecording],
    filter_config: Optional[FilterConfig] = None,
    beat_config: Optional[BeatDetectionConfig] = None,
) -> pd.DataFrame:
    """Extract well-level features while preserving experimental metadata."""
    filter_config = filter_config or FilterConfig()
    beat_config = beat_config or BeatDetectionConfig()

    rows = []
    for rec in recordings:
        wf = extract_well_features(
            rec.electrode_traces,
            fs_hz=rec.fs_hz,
            filter_config=filter_config,
            beat_config=beat_config,
        )
        row = {
            "compound": rec.compound,
            "concentration_uM": rec.concentration_uM,
            "well": rec.well,
            "vehicle": rec.vehicle,
            **rec.metadata,
        }
        row.update(wf.to_row())
        rows.append(row)

    return pd.DataFrame(rows)


def load_raw_traces_to_feature_table(
    path: str | Path,
    filter_config: Optional[FilterConfig] = None,
    beat_config: Optional[BeatDetectionConfig] = None,
) -> pd.DataFrame:
    """Convenience wrapper: raw trace CSV -> ready-to-score feature table."""
    recordings = load_raw_traces_csv(path)
    return recordings_to_feature_table(recordings, filter_config, beat_config)

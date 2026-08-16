"""
Raw MEA trace ingestion.

Reads raw, per-electrode voltage traces from CSV (the actual output of an
MEA recording, before any beat/feature extraction) and runs them through
preprocessing.filtering + features.endpoints to produce the same
well-level feature table that virelion_cardioscore.io.synthetic produces
and analysis.pipeline.CardioScorePipeline.run() already consumes.

This is the missing link between "a lab has raw MEA files" and "the
pipeline can score them" -- previously only pre-aggregated feature CSVs
(fpd_ms, beat_rate_bpm, ...) or synthetic data could be scored.

Expected input format (long format, one row per sample per electrode)
-----------------------------------------------------------------------
Required columns:
    compound            str    compound / treatment name
    well                str    well identifier, e.g. "A01"
    concentration_uM    float  concentration in uM (0.0 for vehicle wells)
    vehicle             bool   True for vehicle/control wells
    electrode_id         str    electrode identifier within the well, e.g. "E1"
    time_s              float  sample timestamp in seconds
    voltage_uv          float  raw voltage in microvolts

Different MEA exporters (Axion, Multi Channel Systems, etc.) use different
native formats; this module defines one canonical long-format schema and
expects an upstream export/conversion step to produce it. Format-specific
readers (io/axion.py, io/mcs.py) can convert into this schema without
touching anything downstream.
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


class RawTraceSchemaError(ValueError):
    """Raised when an input CSV doesn't match the expected raw-trace schema."""


@dataclass
class RawWellRecording:
    """Raw multi-electrode voltage traces for a single well."""

    compound: str
    well: str
    concentration_uM: float
    vehicle: bool
    fs_hz: float
    electrode_traces: dict[str, np.ndarray] = field(default_factory=dict)


def validate_raw_trace_schema(df: pd.DataFrame) -> None:
    """
    Check a raw-trace DataFrame against the required schema and raise a
    RawTraceSchemaError with a specific, actionable message if it doesn't
    match -- rather than letting a cryptic KeyError surface downstream.
    """
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
                f"Check for stray text, units, or header rows in the CSV."
            )

    n_nan = df[["time_s", "voltage_uv"]].isna().sum().sum()
    if n_nan > 0:
        raise RawTraceSchemaError(
            f"Found {n_nan} missing value(s) in time_s/voltage_uv. "
            f"Raw traces must be complete -- fill, interpolate, or drop "
            f"incomplete electrodes before loading."
        )

    # vehicle should behave as a boolean even if read from CSV as
    # "True"/"False" strings or 0/1 -- coerce and warn via the specific
    # error rather than silently misclassifying wells.
    unique_vehicle = set(pd.Series(df["vehicle"]).astype(str).str.lower().unique())
    allowed = {"true", "false", "1", "0", "1.0", "0.0"}
    if not unique_vehicle.issubset(allowed):
        raise RawTraceSchemaError(
            f"Column 'vehicle' has unexpected values {sorted(unique_vehicle)}; "
            f"expected True/False (or 1/0)."
        )


def _coerce_vehicle(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin({"true", "1", "1.0"})


def _infer_sampling_rate(time_s: np.ndarray) -> float:
    """Infer fs_hz from the median gap between consecutive timestamps."""
    diffs = np.diff(np.sort(time_s))
    diffs = diffs[diffs > 0]
    if len(diffs) == 0:
        raise RawTraceSchemaError(
            "Could not infer sampling rate: time_s has no positive gaps "
            "between samples (check for duplicate timestamps)."
        )
    median_dt = float(np.median(diffs))
    return 1.0 / median_dt


def load_raw_traces_csv(path: str | Path) -> list[RawWellRecording]:
    """
    Load a long-format raw MEA trace CSV and group it into per-well,
    per-electrode voltage traces.

    Raises
    ------
    RawTraceSchemaError
        If the CSV doesn't match the expected schema (see module docstring).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Raw trace CSV not found: {path}")

    df = pd.read_csv(path)
    validate_raw_trace_schema(df)
    df = df.copy()
    df["vehicle"] = _coerce_vehicle(df["vehicle"])

    recordings: list[RawWellRecording] = []
    group_cols = ["compound", "well", "concentration_uM", "vehicle"]
    for (compound, well, conc, vehicle), well_df in df.groupby(group_cols, sort=False):
        electrode_traces: dict[str, np.ndarray] = {}
        fs_values = []
        for electrode_id, edf in well_df.groupby("electrode_id", sort=False):
            edf = edf.sort_values("time_s")
            fs_hz = _infer_sampling_rate(edf["time_s"].to_numpy())
            fs_values.append(fs_hz)
            electrode_traces[str(electrode_id)] = edf["voltage_uv"].to_numpy(dtype=float)

        # Sampling rate should be consistent across electrodes in a well;
        # use the median as a robust representative value and don't fail
        # the whole well over one electrode's clock jitter.
        fs_hz = float(np.median(fs_values))

        recordings.append(
            RawWellRecording(
                compound=str(compound),
                well=str(well),
                concentration_uM=float(conc),
                vehicle=bool(vehicle),
                fs_hz=fs_hz,
                electrode_traces=electrode_traces,
            )
        )

    return recordings


def recordings_to_feature_table(
    recordings: list[RawWellRecording],
    filter_config: Optional[FilterConfig] = None,
    beat_config: Optional[BeatDetectionConfig] = None,
) -> pd.DataFrame:
    """
    Run feature extraction on every well recording and assemble the result
    into the same feature-table schema io.synthetic.generate_synthetic_mea
    produces -- so the output of this function can be passed straight into
    CardioScorePipeline.run() with no other changes.
    """
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

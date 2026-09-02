"""Validation asset/feature manifest validation with explicit schema contracts."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ALLOWED_EVIDENCE_LEVELS = {"raw_mea_dataset", "processed_mea_summary"}
NORMALIZED_COLUMNS = {
    "compound",
    "site",
    "cell_type",
    "concentration_index",
    "concentration_uM",
    "well",
    "vehicle",
    "fpd_ms",
    "beat_rate_bpm",
    "amplitude_uv",
    "stv",
    "triangulation_proxy",
    # Runtime QC fields consumed by CardioScorePipeline.apply_qc().
    "n_electrodes",
    "noise_sd_uv",
    "beat_detection_rate",
}
REFERENCE_COLUMNS = {"compound", "reference_risk"}


@dataclass(frozen=True)
class ValidationManifest:
    source_id: str
    evidence_level: str
    asset_path: str
    asset_sha256: str
    source_url: str
    config_path: str
    features_path: str
    reference_path: str
    output_dir: str
    locked: bool = True

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ValidationManifest":
        manifest_path = Path(path)
        payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
        required = [
            "source_id", "evidence_level", "asset_path", "asset_sha256", "source_url",
            "config_path", "features_path", "reference_path", "output_dir",
        ]
        missing = [key for key in required if key not in payload]
        if missing:
            raise ValueError(f"Validation manifest missing required fields: {missing}")
        if not bool(payload.get("locked", True)):
            raise ValueError("Validation manifest must set locked: true")
        evidence = str(payload["evidence_level"]).strip()
        if evidence not in ALLOWED_EVIDENCE_LEVELS:
            raise ValueError(
                f"Unsupported evidence_level {evidence!r}; expected one of {sorted(ALLOWED_EVIDENCE_LEVELS)}"
            )
        sha = str(payload["asset_sha256"]).strip().lower()
        if len(sha) != 64 or any(ch not in "0123456789abcdef" for ch in sha):
            raise ValueError("asset_sha256 must be a 64-character hexadecimal SHA-256 digest")
        source_url = str(payload["source_url"]).strip()
        if not source_url:
            raise ValueError("source_url cannot be blank")
        return cls(
            source_id=str(payload["source_id"]),
            evidence_level=evidence,
            asset_path=str(payload["asset_path"]),
            asset_sha256=sha,
            source_url=source_url,
            config_path=str(payload["config_path"]),
            features_path=str(payload["features_path"]),
            reference_path=str(payload["reference_path"]),
            output_dir=str(payload["output_dir"]),
            locked=True,
        )


def validate_feature_schema(frame: pd.DataFrame) -> None:
    """Validate the full standardized contract required by the runtime pipeline."""
    missing = sorted(NORMALIZED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"Standardized feature table missing required columns: {missing}")

    for column in ("compound", "site", "cell_type", "well"):
        if frame[column].isna().any() or frame[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"Standardized feature table contains missing or blank {column!r} identifiers")

    vehicle = frame["vehicle"]
    vehicle_values = set(vehicle.astype(str).str.strip().str.lower())
    allowed_vehicle = {"true", "false", "1", "0", "1.0", "0.0", "yes", "no"}
    if not vehicle_values.issubset(allowed_vehicle):
        raise ValueError(f"vehicle contains unsupported values: {sorted(vehicle_values - allowed_vehicle)}")
    vehicle_bool = vehicle.astype(str).str.strip().str.lower().isin({"true", "1", "1.0", "yes"})

    index_values = pd.to_numeric(frame["concentration_index"], errors="coerce")
    if index_values.isna().any() or not np.isfinite(index_values.to_numpy()).all():
        raise ValueError("concentration_index must be numeric and finite")
    if (index_values < 0).any() or not np.isclose(index_values, np.round(index_values)).all():
        raise ValueError("concentration_index must contain non-negative integers")

    concentration_values = pd.to_numeric(frame["concentration_uM"], errors="coerce")
    if concentration_values.isna().any() or not np.isfinite(concentration_values.to_numpy()).all():
        raise ValueError("concentration_uM must be numeric and finite")
    if ((~vehicle_bool) & (concentration_values <= 0)).any():
        raise ValueError("treated wells must have strictly positive concentration_uM")
    if (vehicle_bool & (concentration_values < 0)).any():
        raise ValueError("vehicle wells may use concentration_uM=0 but cannot be negative")

    site_values = pd.to_numeric(frame["site"], errors="coerce")
    if site_values.isna().any() or not np.isfinite(site_values.to_numpy()).all():
        raise ValueError("site must be numeric and finite")

    numeric_runtime_columns = (
        "fpd_ms",
        "beat_rate_bpm",
        "amplitude_uv",
        "stv",
        "triangulation_proxy",
        "n_electrodes",
        "noise_sd_uv",
        "beat_detection_rate",
    )
    for column in numeric_runtime_columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.isna().any() or not np.isfinite(values.to_numpy()).all():
            raise ValueError(f"{column} must be numeric and finite")

    electrode_values = pd.to_numeric(frame["n_electrodes"], errors="coerce")
    if (electrode_values < 0).any() or not np.isclose(electrode_values, np.round(electrode_values)).all():
        raise ValueError("n_electrodes must contain non-negative integers")

    noise_values = pd.to_numeric(frame["noise_sd_uv"], errors="coerce")
    if (noise_values < 0).any():
        raise ValueError("noise_sd_uv cannot be negative")

    detection_values = pd.to_numeric(frame["beat_detection_rate"], errors="coerce")
    if ((detection_values < 0) | (detection_values > 1)).any():
        raise ValueError("beat_detection_rate must be between 0 and 1")


def validate_reference_schema(frame: pd.DataFrame) -> None:
    missing = sorted(REFERENCE_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"Reference table missing required columns: {missing}")
    allowed = {"low", "intermediate", "high", "L", "M", "H"}
    observed = set(frame["reference_risk"].dropna().astype(str).str.strip())
    invalid = sorted(observed - allowed)
    if invalid:
        raise ValueError(f"Unsupported reference risk labels: {invalid}")
    if frame["reference_risk"].isna().any() or frame["reference_risk"].astype(str).str.strip().eq("").any():
        raise ValueError("Reference table contains missing or blank risk labels")
    if frame["compound"].isna().any() or frame["compound"].astype(str).str.strip().eq("").any():
        raise ValueError("Reference table contains missing or blank compound identifiers")
    if frame["compound"].duplicated().any():
        duplicates = sorted(frame.loc[frame["compound"].duplicated(), "compound"].astype(str).unique())
        raise ValueError(f"Reference table contains duplicate compound labels: {duplicates}")


def validate_vehicle_structure(frame: pd.DataFrame) -> None:
    """Require a matching vehicle control for every compound under compound-level normalization."""
    if "compound" not in frame.columns or "vehicle" not in frame.columns:
        raise ValueError("Vehicle-structure validation requires compound and vehicle columns")
    vehicle_bool = frame["vehicle"].astype(str).str.strip().str.lower().isin({"true", "1", "1.0", "yes"})
    treated_compounds = set(frame.loc[~vehicle_bool, "compound"].astype(str))
    vehicle_compounds = set(frame.loc[vehicle_bool, "compound"].astype(str))
    missing_controls = sorted(treated_compounds - vehicle_compounds)
    if missing_controls:
        raise ValueError(
            "Missing matching vehicle control wells for compound(s): " + ", ".join(missing_controls)
        )


def resolve_relative(base: str | Path, value: str) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else Path(base) / candidate

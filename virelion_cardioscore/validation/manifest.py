"""Validation asset/feature manifest validation with explicit schema contracts."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

NORMALIZED_COLUMNS = {
    "compound",
    "site",
    "cell_type",
    "concentration_index",
    "concentration_uM",
    "well",
    "fpd_ms",
    "beat_rate_bpm",
    "amplitude_uv",
    "stv",
    "triangulation_proxy",
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
        return cls(**{key: payload[key] for key in required}, locked=True)


def validate_feature_schema(frame: pd.DataFrame) -> None:
    missing = sorted(NORMALIZED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"Standardized feature table missing required columns: {missing}")
    if frame["compound"].isna().any() or frame["well"].isna().any():
        raise ValueError("Standardized feature table contains missing compound or well identifiers")
    if (frame["concentration_index"] < 0).any():
        raise ValueError("concentration_index cannot be negative")


def validate_reference_schema(frame: pd.DataFrame) -> None:
    missing = sorted(REFERENCE_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"Reference table missing required columns: {missing}")
    allowed = {"low", "intermediate", "high", "L", "M", "H"}
    observed = set(frame["reference_risk"].dropna().astype(str))
    invalid = sorted(observed - allowed)
    if invalid:
        raise ValueError(f"Unsupported reference risk labels: {invalid}")
    if frame["compound"].duplicated().any():
        duplicates = sorted(frame.loc[frame["compound"].duplicated(), "compound"].astype(str).unique())
        raise ValueError(f"Reference table contains duplicate compound labels: {duplicates}")


def resolve_relative(base: str | Path, value: str) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else Path(base) / candidate

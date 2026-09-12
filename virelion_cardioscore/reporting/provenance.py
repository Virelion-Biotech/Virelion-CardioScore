"""Machine-readable provenance for CardioScore results."""

from __future__ import annotations

import hashlib
import json
import os
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Mapping

import pandas as pd
import yaml


def hash_dataframe(frame: pd.DataFrame) -> str:
    """Return a deterministic SHA-256 fingerprint for a DataFrame."""
    canonical = frame.copy()
    canonical = canonical.reindex(sorted(canonical.columns), axis=1)
    canonical = canonical.sort_values(list(canonical.columns), kind="mergesort").reset_index(drop=True)
    payload = {
        "columns": [str(column) for column in canonical.columns],
        "dtypes": [str(dtype) for dtype in canonical.dtypes],
        "data": canonical.astype(object).where(pd.notna(canonical), None).to_dict(orient="records"),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")).hexdigest()


def hash_config(config: Mapping[str, Any]) -> str:
    """Return a deterministic SHA-256 fingerprint for the effective config."""
    public_config = {str(key): value for key, value in config.items() if not str(key).startswith("_")}
    payload = yaml.safe_dump(public_config, sort_keys=True, default_flow_style=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def package_version() -> str | None:
    try:
        return version("virelion-cardioscore")
    except PackageNotFoundError:
        return None


def build_provenance(
    *,
    input_frame: pd.DataFrame,
    config: Mapping[str, Any],
    input_rows_after_qc: int,
    effect_rows: int,
    scoring_unit_rows: int,
    excluded_compounds: list[str],
    qc_log: list[str],
) -> dict[str, Any]:
    """Build an auditable provenance record for one pipeline execution."""
    return {
        "schema_version": "1.0",
        "input_sha256": hash_dataframe(input_frame),
        "configuration_sha256": hash_config(config),
        "package_version": package_version(),
        "git_commit": os.getenv("VIRELION_GIT_COMMIT") or os.getenv("GITHUB_SHA"),
        "input_rows": int(len(input_frame)),
        "rows_after_qc": int(input_rows_after_qc),
        "effect_rows": int(effect_rows),
        "scoring_unit_rows": int(scoring_unit_rows),
        "excluded_compounds": sorted(set(map(str, excluded_compounds))),
        "qc_log": list(qc_log),
        "transformation_summary": [
            "input",
            "quality_control",
            "optional_variability_correction",
            "vehicle_normalization",
            "experimental_unit_aggregation",
            "concentration_summarization",
            "concentration_coverage_gating",
            "compound_scoring",
        ],
    }

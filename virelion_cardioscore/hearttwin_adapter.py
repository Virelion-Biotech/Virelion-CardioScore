"""Versioned HeartTwin boundary for CardioScore research scoring.

The adapter intentionally emits research-score semantics and source/config
fingerprints. It does not turn preclinical MEA results into clinical risk.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from virelion_cardioscore import __version__
from virelion_cardioscore.analysis.pipeline import CardioScorePipeline

RESULT_SCHEMA = "virelion.cardioscore.hearttwin-result.v1"
DEFAULT_CONFIG = Path(__file__).resolve().parent / "config" / "default.yaml"


def _find_input(payload: dict) -> tuple[str, dict]:
    for obs in payload.get("observations", []):
        if obs.get("modality") == "safety" and "input_path" in obs.get("values", {}):
            return str(obs["values"]["input_path"]), obs["values"]
    raise ValueError("No 'safety' observation with values.input_path was provided")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    """Convert NumPy/Pandas values and all non-finite numbers to strict JSON values."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if value is None or isinstance(value, (str, int, bool)):
        return value
    try:
        missing = pd.isna(value)
        if isinstance(missing, (bool, np.bool_)) and missing:
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except (TypeError, ValueError):
            pass
    return str(value)


def score_payload(payload: dict) -> dict[str, Any]:
    """Score one HeartTwin safety observation and return the versioned output contract."""
    if not isinstance(payload, dict):
        raise ValueError("HeartTwin payload must be a JSON object.")

    input_path_raw, params = _find_input(payload)
    input_path = Path(input_path_raw).expanduser().resolve()
    if not input_path.is_file():
        raise ValueError("The safety observation input_path must identify an existing CSV file.")

    config_raw = params.get("config")
    if config_raw:
        config_path = Path(str(config_raw)).expanduser().resolve()
        if not config_path.is_file():
            raise ValueError("The safety observation config must identify an existing YAML file.")
        pipeline = CardioScorePipeline.from_config(config_path)
        config_source = "payload"
    else:
        config_path = DEFAULT_CONFIG
        pipeline = CardioScorePipeline.from_defaults()
        config_source = "package_default"

    frame = pd.read_csv(input_path)
    result = pipeline.run(frame)
    summary = _json_safe(result.summary_table.to_dict(orient="records"))
    output = {
        "schema_version": RESULT_SCHEMA,
        "producer": {
            "name": "Virelion-CardioScore",
            "version": __version__,
        },
        "entity_id": payload.get("entity_id"),
        "input": {
            "format": "csv",
            "filename": input_path.name,
            "sha256": _sha256_file(input_path),
            "rows": int(len(frame)),
        },
        "analysis": {
            "score_kind": "research_score",
            "risk_class_semantics": (
                "Heuristic research category; not an independently calibrated drug-risk "
                "probability and not an individual-patient clinical risk estimate."
            ),
            "config_source": config_source,
            "config_sha256": _sha256_file(config_path),
            "endpoint_config_sha256": _sha256_file(pipeline.engine.endpoint_config_path),
        },
        # Keep the original top-level fields so existing HeartTwin callers remain compatible.
        "summary": summary,
        "scores": [score.to_dict() for score in result.scores],
        "qc_log": result.qc_log,
        "normalization": result.normalization_diagnostic,
        "qc_rejections": _json_safe(result.qc_rejections.to_dict(orient="records")),
        "dropout_table": _json_safe(result.dropout_table.to_dict(orient="records")),
        "exclusion_table": _json_safe(result.exclusion_table.to_dict(orient="records")),
        "provenance": result.provenance,
    }
    return _json_safe(output)


def main() -> int:
    raw = (
        sys.stdin.read()
        if os.environ.get("HEARTTWIN_PAYLOAD_STDIN") == "1"
        else os.environ.get("HEARTTWIN_PAYLOAD")
    )
    if not raw:
        print("HEARTTWIN_PAYLOAD environment variable not set", file=sys.stderr)
        return 1
    try:
        payload = json.loads(raw)
        output = score_payload(payload)
        print(json.dumps(output, allow_nan=False))
        return 0
    except Exception as exc:  # noqa: BLE001 - process boundary must return a useful error
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

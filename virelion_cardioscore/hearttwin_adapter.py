"""HeartTwin local-command adapter for CardioScore."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd

from virelion_cardioscore.analysis.pipeline import CardioScorePipeline


def _find_input(payload: dict) -> tuple[str, dict]:
    for obs in payload.get("observations", []):
        if obs.get("modality") == "safety" and "input_path" in obs.get("values", {}):
            return str(obs["values"]["input_path"]), obs["values"]
    raise ValueError("No 'safety' observation with values.input_path was provided")


def main() -> int:
    raw = os.environ.get("HEARTTWIN_PAYLOAD")
    if not raw:
        print("HEARTTWIN_PAYLOAD environment variable not set", file=sys.stderr)
        return 1
    try:
        payload = json.loads(raw)
        input_path, params = _find_input(payload)
        pipeline = CardioScorePipeline.from_config(params["config"]) if params.get("config") else CardioScorePipeline.from_defaults()
        result = pipeline.run(pd.read_csv(Path(input_path)))
        output = {
            "entity_id": payload.get("entity_id"),
            "summary": result.summary_table.where(pd.notna(result.summary_table), None).to_dict(orient="records"),
            "scores": [score.to_dict() for score in result.scores],
            "qc_log": result.qc_log,
            "normalization": result.normalization_diagnostic,
        }
        print(json.dumps(output, allow_nan=False, default=str))
        return 0
    except Exception as exc:  # noqa: BLE001 - adapter boundary
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

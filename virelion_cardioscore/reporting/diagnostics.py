"""Derived report diagnostics that do not alter CardioScore."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import json

from virelion_cardioscore.analysis.concentration_drivers import concentration_drivers

if TYPE_CHECKING:
    from virelion_cardioscore.analysis.pipeline import PipelineResult


def _endpoint_directions(result: "PipelineResult") -> dict[str, str]:
    """Recover the configured endpoint directions from ScoreResult contributions."""
    for score in result.scores:
        if score.contributions:
            return {
                contribution.name: "absolute"
                for contribution in score.contributions
                if contribution.name != "dose_response_exposure_evidence"
            }
    return {}


def concentration_driver_table(result: "PipelineResult"):
    """Build concentration provenance diagnostics for the current result."""
    return concentration_drivers(
        result.concentration_table,
        endpoint_directions=_endpoint_directions(result),
        effect_threshold_pct=float(
            result.config.get("concentration_response", {}).get("effect_threshold_pct", 10.0)
        ),
    )


def enrich_json_report(result: "PipelineResult", path: str | Path) -> None:
    """Add derived diagnostics to an already-written PipelineResult JSON."""
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    drivers = concentration_driver_table(result)
    payload["concentration_drivers"] = drivers.to_dict(orient="records") if not drivers.empty else []
    payload["diagnostics"] = {
        "concentration_driver_provenance": {
            "description": "Identifies the tested concentration driving each compound/endpoint worst-case value and the fraction of tested concentrations supporting that signal.",
            "does_not_modify_score": True,
        }
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

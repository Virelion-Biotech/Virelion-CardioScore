"""Derived report diagnostics that do not alter CardioScore."""

from __future__ import annotations

import json
from html import escape
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from virelion_cardioscore.analysis.concentration_drivers import (
    ENDPOINT_DIRECTIONS,
    concentration_drivers,
)

if TYPE_CHECKING:
    from virelion_cardioscore.analysis.pipeline import PipelineResult


def _endpoint_metadata(result: "PipelineResult") -> tuple[dict[str, str], dict[str, float]]:
    """Load the same endpoint directions/thresholds used by the scoring engine."""
    scoring_cfg = result.config.get("scoring", {})
    endpoint_config = Path(scoring_cfg.get("endpoint_config", "cipa_endpoints.yaml"))
    if not endpoint_config.is_absolute():
        base_dir = Path(result.config.get("_config_dir", Path.cwd()))
        endpoint_config = base_dir / endpoint_config
        if not endpoint_config.exists():
            endpoint_config = Path(__file__).resolve().parents[1] / "config" / scoring_cfg.get("endpoint_config", "cipa_endpoints.yaml")
    if not endpoint_config.is_file():
        raise ValueError(f"Configured endpoint configuration does not exist: {endpoint_config}")
    payload = yaml.safe_load(endpoint_config.read_text(encoding="utf-8")) or {}
    endpoints = payload.get("endpoints", {})
    if not isinstance(endpoints, dict) or not endpoints:
        raise ValueError("Endpoint configuration must define a non-empty 'endpoints' mapping.")
    directions = {name: str(meta["direction"]) for name, meta in endpoints.items()}
    thresholds = {name: float(meta["effect_threshold"]) for name, meta in endpoints.items()}
    return directions, thresholds


def concentration_driver_table(result: "PipelineResult"):
    """Build concentration provenance diagnostics for the current result."""
    directions, thresholds = _endpoint_metadata(result)
    return concentration_drivers(
        result.concentration_table,
        endpoint_directions=directions or ENDPOINT_DIRECTIONS,
        endpoint_thresholds=thresholds,
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


def enrich_html_report(result: "PipelineResult", path: str | Path) -> None:
    """Insert a concentration-provenance card into the generated HTML report."""
    path = Path(path)
    html = path.read_text(encoding="utf-8")
    drivers = concentration_driver_table(result)
    if drivers.empty:
        return

    rows = []
    for _, row in drivers.iterrows():
        rows.append(
            "<tr>"
            f"<td>{escape(str(row['compound']), quote=True)}</td>"
            f"<td>{escape(str(row['endpoint']), quote=True)}</td>"
            f"<td>{float(row['driver_concentration_uM']):.4g}</td>"
            f"<td>{float(row['driver_value']):.4g}</td>"
            f"<td>{int(row['concentrations_supporting_signal'])}/{int(row['concentrations_tested'])}</td>"
            f"<td>{float(row['support_fraction']) * 100.0:.0f}%</td>"
            "</tr>"
        )

    card = (
        '<div class="card">'
        '<h2 style="margin-top:0">Concentration Provenance</h2>'
        '<p class="meta">Diagnostic only. CardioScore is unchanged; this identifies the concentration driving each endpoint and how broadly the signal is supported.</p>'
        '<table><thead><tr><th>Compound</th><th>Endpoint</th><th>Driver concentration (µM)</th>'
        '<th>Driver value</th><th>Supporting concentrations</th><th>Support</th></tr></thead>'
        f"<tbody>{''.join(rows)}</tbody></table>"
        '</div>'
    )
    marker = '<div class="card">\n    <h2 style="margin-top:0">Quality Control Log</h2>'
    if marker in html:
        html = html.replace(marker, card + "\n  " + marker, 1)
    else:
        html = html.replace("</body>", card + "</body>", 1)
    path.write_text(html, encoding="utf-8")

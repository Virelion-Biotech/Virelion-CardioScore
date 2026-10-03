"""Contract tests preventing scored STV from drifting between rhythm and repolarization semantics."""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_scored_stv_contract_is_ibi_variability():
    endpoints = yaml.safe_load(
        (ROOT / "virelion_cardioscore" / "config" / "cipa_endpoints.yaml").read_text(
            encoding="utf-8"
        )
    )
    prereg = yaml.safe_load((ROOT / "validation" / "preregistration.yaml").read_text(encoding="utf-8"))

    description = endpoints["endpoints"]["stv_increase"]["description"].lower()
    assert "inter-beat-interval" in description
    assert "repolarization" not in description
    assert prereg["scoring_commitments"]["stv_definition"] == (
        "normalized_inter_beat_interval_variability"
    )


def test_repolarization_stv_remains_separate_report_only_field():
    endpoints_source = (
        ROOT / "virelion_cardioscore" / "preprocessing" / "endpoints.py"
    ).read_text(encoding="utf-8")
    assert "fpd_stv_ms" in endpoints_source
    assert "report-only" in endpoints_source

"""Basic tests for the CardioScore engine and pipeline."""

from __future__ import annotations

import pytest

from virelion_cardioscore.analysis.cipa_scoring import CardioScoreEngine
from virelion_cardioscore.analysis.pipeline import CardioScorePipeline
from virelion_cardioscore.io.synthetic import load_synthetic_dataset


def test_engine_low_risk():
    engine = CardioScoreEngine()
    result = engine.score_compound(
        "SafeComp",
        {
            "fpd_change_pct": 3.0,
            "beat_rate_change_pct": 5.0,
            "amplitude_change_pct": -5.0,
            "stv_increase": 0.05,
            "triangulation_proxy": 0.05,
        },
    )
    assert result.risk_class == "Low"
    assert result.score < 0.30


def test_engine_high_risk():
    engine = CardioScoreEngine()
    result = engine.score_compound(
        "ToxicComp",
        {
            "fpd_change_pct": 45.0,
            "beat_rate_change_pct": 35.0,
            "amplitude_change_pct": -50.0,
            "stv_increase": 0.8,
            "triangulation_proxy": 0.7,
        },
    )
    assert result.risk_class == "High"
    assert result.score > 0.60


def test_pipeline_end_to_end():
    dataset = load_synthetic_dataset(n_compounds=3, n_concentrations=4, seed=123)
    pipeline = CardioScorePipeline.from_defaults()
    result = pipeline.run(dataset)

    assert len(result.scores) == 3
    assert not result.summary_table.empty
    assert "cardioscore" in result.summary_table.columns
    assert "risk_class" in result.summary_table.columns
    assert len(result.qc_log) > 0


def test_synthetic_dataset_structure():
    ds = load_synthetic_dataset(n_compounds=2, n_concentrations=3, seed=1)
    assert "compound" in ds.features.columns
    assert "fpd_ms" in ds.features.columns
    assert ds.features["vehicle"].dtype == bool
    assert len(ds.compounds) == 2

import pandas as pd
import pytest

from virelion_cardioscore.analysis.cipa_scoring import CardioScoreEngine
from virelion_cardioscore.analysis.score_sensitivity import (
    WeightSensitivitySpec,
    run_weight_sensitivity,
    summarize_weight_sensitivity,
)


def _values(**overrides: float) -> dict[str, float]:
    values = {
        "fpd_change_pct": 20.0,
        "beat_rate_change_pct": 10.0,
        "amplitude_change_pct": -25.0,
        "stv_increase": 0.20,
        "triangulation_proxy": 0.20,
    }
    values.update(overrides)
    return values


def test_weight_sensitivity_has_two_perturbations_per_endpoint():
    engine = CardioScoreEngine()
    results = run_weight_sensitivity(engine, {"A": _values()})

    assert len(results) == len(engine.endpoints) * 2
    assert set(results["direction"]) == {"down", "up"}
    assert results["compound"].eq("A").all()


def test_weight_sensitivity_does_not_mutate_engine():
    engine = CardioScoreEngine()
    before = {name: meta["weight"] for name, meta in engine.endpoints.items()}

    run_weight_sensitivity(engine, {"A": _values()})

    after = {name: meta["weight"] for name, meta in engine.endpoints.items()}
    assert after == before


def test_weight_sensitivity_detects_threshold_instability():
    engine = CardioScoreEngine()
    results = run_weight_sensitivity(
        engine,
        {"borderline": _values(fpd_change_pct=60.0, stv_increase=0.0, triangulation_proxy=0.0)},
        spec=WeightSensitivitySpec(relative_change=0.20),
    )

    summary = summarize_weight_sensitivity(results)
    assert isinstance(summary, pd.DataFrame)
    assert summary.loc[0, "n_perturbations"] == len(engine.endpoints) * 2
    assert 0.0 <= summary.loc[0, "risk_class_change_rate"] <= 1.0


def test_weight_sensitivity_rejects_invalid_relative_change():
    with pytest.raises(ValueError, match="less than 1.0"):
        WeightSensitivitySpec(relative_change=1.0)
    with pytest.raises(ValueError, match="non-negative"):
        WeightSensitivitySpec(relative_change=-0.1)

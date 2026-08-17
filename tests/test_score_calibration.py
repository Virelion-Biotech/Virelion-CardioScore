"""Calibration and behavioral tests for the CardioScore engine."""

from __future__ import annotations

import pytest

from virelion_cardioscore.analysis.cipa_scoring import CardioScoreEngine


@pytest.fixture()
def engine() -> CardioScoreEngine:
    return CardioScoreEngine()


def _zero_case() -> dict[str, float]:
    return {
        "fpd_change_pct": 0.0,
        "beat_rate_change_pct": 0.0,
        "amplitude_change_pct": 0.0,
        "stv_increase": 0.0,
        "triangulation_proxy": 0.0,
    }


def test_zero_signal_scores_zero(engine: CardioScoreEngine):
    result = engine.score_compound("control", _zero_case())
    assert result.score == pytest.approx(0.0)
    assert result.risk_class == "Low"


def test_single_extreme_endpoint_scores_its_weight(engine: CardioScoreEngine):
    values = _zero_case()
    values["stv_increase"] = 10.0
    result = engine.score_compound("stv_only", values)
    assert result.score == pytest.approx(0.25)


def test_large_fpd_with_normal_stv_is_not_dominated_by_stv(engine: CardioScoreEngine):
    values = _zero_case()
    values["fpd_change_pct"] = 100.0
    result = engine.score_compound("fpd_only", values)
    assert result.score == pytest.approx(0.30)


def test_large_stv_with_normal_fpd_is_limited_by_stv_weight(engine: CardioScoreEngine):
    values = _zero_case()
    values["stv_increase"] = 10.0
    result = engine.score_compound("stv_only", values)
    assert result.score == pytest.approx(0.25)


def test_moderate_signal_from_all_endpoints_exceeds_each_single_endpoint(engine: CardioScoreEngine):
    values = {
        "fpd_change_pct": 20.0,
        "beat_rate_change_pct": 30.0,
        "amplitude_change_pct": -40.0,
        "stv_increase": 0.30,
        "triangulation_proxy": 0.40,
    }
    result = engine.score_compound("multi", values)
    assert 0.30 < result.score < 1.0
    assert result.risk_class == "Moderate"


def test_monotonicity_for_harmful_fpd_signal(engine: CardioScoreEngine):
    low = _zero_case()
    high = _zero_case()
    low["fpd_change_pct"] = 10.0
    high["fpd_change_pct"] = 40.0
    low_result = engine.score_compound("low", low)
    high_result = engine.score_compound("high", high)
    assert high_result.score >= low_result.score


def test_decrease_direction_ignores_protective_amplitude_increase(engine: CardioScoreEngine):
    values = _zero_case()
    values["amplitude_change_pct"] = 40.0
    result = engine.score_compound("amplitude_up", values)
    assert result.score == pytest.approx(0.0)


def test_decrease_direction_scores_large_amplitude_loss(engine: CardioScoreEngine):
    values = _zero_case()
    values["amplitude_change_pct"] = -100.0
    result = engine.score_compound("amplitude_down", values)
    assert result.score == pytest.approx(0.15)


def test_one_extreme_endpoint_does_not_cross_moderate_threshold(engine: CardioScoreEngine):
    for endpoint in ("fpd_change_pct", "beat_rate_change_pct", "amplitude_change_pct", "stv_increase", "triangulation_proxy"):
        values = _zero_case()
        values[endpoint] = 1e6 if endpoint != "amplitude_change_pct" else -1e6
        result = engine.score_compound(endpoint, values)
        assert result.score < 0.30
        assert result.risk_class == "Low"


def test_multiple_extreme_endpoints_cross_high_threshold(engine: CardioScoreEngine):
    values = {
        "fpd_change_pct": 1e6,
        "beat_rate_change_pct": 1e6,
        "amplitude_change_pct": -1e6,
        "stv_increase": 1e6,
        "triangulation_proxy": 1e6,
    }
    result = engine.score_compound("all_extreme", values)
    assert result.score == pytest.approx(1.0)
    assert result.risk_class == "High"

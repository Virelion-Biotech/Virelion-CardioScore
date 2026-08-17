from __future__ import annotations

import numpy as np
import pytest

from virelion_cardioscore.analysis.cipa_scoring import CardioScoreEngine


@pytest.fixture
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
    result = engine.score_compound("zero", _zero_case())
    assert result.score == pytest.approx(0.0)
    assert result.risk_class == "Low"


def test_single_extreme_endpoint_scores_its_weight(engine: CardioScoreEngine):
    values = _zero_case()
    values["fpd_change_pct"] = 1e6
    result = engine.score_compound("fpd", values)
    assert result.score == pytest.approx(0.30)


def test_large_fpd_with_normal_stv_is_not_dominated_by_stv(engine: CardioScoreEngine):
    values = _zero_case()
    values["fpd_change_pct"] = 100.0
    result = engine.score_compound("fpd", values)
    assert result.score > 0.0
    assert next(item for item in result.contributions if item.name == "stv_increase").contribution == 0.0


def test_large_stv_with_normal_fpd_is_limited_by_stv_weight(engine: CardioScoreEngine):
    values = _zero_case()
    values["stv_increase"] = 1e6
    result = engine.score_compound("stv", values)
    assert result.score == pytest.approx(0.25)


def test_moderate_signal_from_all_endpoints_exceeds_each_single_endpoint(engine: CardioScoreEngine):
    values = {
        "fpd_change_pct": 25.0,
        "beat_rate_change_pct": 30.0,
        "amplitude_change_pct": -30.0,
        "stv_increase": 0.30,
        "triangulation_proxy": 0.40,
    }
    result = engine.score_compound("all", values)
    max_single = max(
        engine.score_compound(name, {**_zero_case(), name: value}).score
        for name, value in values.items()
    )
    assert result.score > max_single


def test_monotonicity_for_harmful_fpd_signal(engine: CardioScoreEngine):
    values_10 = _zero_case()
    values_30 = _zero_case()
    values_10["fpd_change_pct"] = 20.0
    values_30["fpd_change_pct"] = 40.0
    assert engine.score_compound("fpd10", values_10).score < engine.score_compound("fpd30", values_30).score


def test_decrease_direction_ignores_protective_amplitude_increase(engine: CardioScoreEngine):
    values = _zero_case()
    values["amplitude_change_pct"] = 100.0
    result = engine.score_compound("amplitude_up", values)
    assert result.score == pytest.approx(0.0)


def test_decrease_direction_scores_large_amplitude_loss(engine: CardioScoreEngine):
    values = _zero_case()
    values["amplitude_change_pct"] = -100.0
    result = engine.score_compound("amplitude_down", values)
    assert result.score == pytest.approx(0.15)


def test_one_extreme_endpoint_reaches_but_does_not_exceed_low_boundary(engine: CardioScoreEngine):
    endpoint_values = {
        "fpd_change_pct": 1e6,
        "beat_rate_change_pct": 1e6,
        "amplitude_change_pct": -1e6,
        "stv_increase": 1e6,
        "triangulation_proxy": 1e6,
    }
    expected = {
        "fpd_change_pct": (0.30, "Moderate"),
        "beat_rate_change_pct": (0.15, "Low"),
        "amplitude_change_pct": (0.15, "Low"),
        "stv_increase": (0.25, "Low"),
        "triangulation_proxy": (0.15, "Low"),
    }
    for endpoint, raw in endpoint_values.items():
        values = _zero_case()
        values[endpoint] = raw
        result = engine.score_compound(endpoint, values)
        expected_score, expected_class = expected[endpoint]
        assert result.score == pytest.approx(expected_score)
        assert result.risk_class == expected_class


def test_multiple_extreme_endpoints_cross_high_threshold(engine: CardioScoreEngine):
    values = {
        "fpd_change_pct": 1e6,
        "beat_rate_change_pct": 1e6,
        "amplitude_change_pct": -1e6,
        "stv_increase": 1e6,
        "triangulation_proxy": 1e6,
    }
    result = engine.score_compound("all_extreme", values)
    assert result.score > 0.60
    assert result.risk_class == "High"


def test_missing_endpoint_values_fail_closed(engine: CardioScoreEngine):
    values = _zero_case()
    values.pop("stv_increase")
    with pytest.raises(ValueError, match="missing"):
        engine.score_compound("incomplete", values)


def test_non_finite_endpoint_values_fail_closed(engine: CardioScoreEngine):
    values = _zero_case()
    values["stv_increase"] = np.nan
    with pytest.raises(ValueError, match="finite"):
        engine.score_compound("nan", values)

from virelion_cardioscore.analysis.cipa_scoring import CardioScoreEngine


def _endpoint_values_for_score(engine: CardioScoreEngine, target: float) -> dict[str, float]:
    """Create a pure-FPD profile whose composite score is exactly target."""
    weight = float(engine.endpoints["fpd_change_pct"]["weight"])
    threshold = float(engine.endpoints["fpd_change_pct"]["effect_threshold"])
    normalized = target / weight
    raw = threshold + normalized * 3.0 * threshold
    return {
        "fpd_change_pct": raw,
        "beat_rate_change_pct": 0.0,
        "amplitude_change_pct": 0.0,
        "stv_increase": 0.0,
        "triangulation_proxy": 0.0,
    }


def test_low_threshold_is_start_of_moderate_range():
    engine = CardioScoreEngine()

    result = engine.score_compound(
        "BoundaryLow",
        _endpoint_values_for_score(engine, engine.low_threshold),
    )

    assert result.score == engine.low_threshold
    assert result.risk_class == "Moderate"


def test_just_below_low_threshold_is_low():
    engine = CardioScoreEngine()

    result = engine.score_compound(
        "BelowBoundary",
        _endpoint_values_for_score(engine, engine.low_threshold - 1e-6),
    )

    assert result.score < engine.low_threshold
    assert result.risk_class == "Low"


def test_moderate_threshold_is_start_of_high_range():
    engine = CardioScoreEngine()

    result = engine.score_compound(
        "BoundaryModerate",
        _endpoint_values_for_score(engine, engine.moderate_threshold),
    )

    assert result.score == engine.moderate_threshold
    assert result.risk_class == "High"

from pathlib import Path


APP = Path(__file__).parents[1] / "web" / "app.js"


def test_browser_scoring_uses_concentration_level_replicate_means():
    text = APP.read_text(encoding="utf-8")
    assert "var concentrationEffects=[];" in text
    assert "fpd:mean(group.map" in text
    assert "rate:mean(group.map" in text
    assert "amp:mean(group.map" in text
    assert "stv:mean(group.map" in text
    assert "tri:mean(group.map" in text


def test_browser_scoring_uses_directional_compound_aggregation():
    text = APP.read_text(encoding="utf-8")
    assert "Math.max.apply(null,concentrationEffects.map(function(e){return Math.abs(e.fpd)}))" in text
    assert "Math.max.apply(null,concentrationEffects.map(function(e){return Math.abs(e.rate)}))" in text
    assert "Math.min.apply(null,concentrationEffects.map(function(e){return e.amp}))" in text
    assert "Math.max.apply(null,concentrationEffects.map(function(e){return e.stv}))" in text
    assert "Math.max.apply(null,concentrationEffects.map(function(e){return e.tri}))" in text


def test_browser_does_not_claim_hierarchical_or_4pl_parity():
    text = APP.read_text(encoding="utf-8")
    assert "biological-unit inference and 4PL evidence remain Python-only" in text


def test_browser_preserves_explicit_zero_weight_values():
    text = APP.read_text(encoding="utf-8")
    assert "function numericOrDefault(id, fallback)" in text
    assert "Number.isFinite(value)?value:fallback" in text
    assert "return{fpd:numericOrDefault('w_fpd',0.3)" in text

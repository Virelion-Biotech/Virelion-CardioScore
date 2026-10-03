from pathlib import Path


APP = Path(__file__).parents[1] / "web" / "app.js"
CONTRACT = Path(__file__).parents[1] / "web" / "scoring_contract.json"


def test_browser_scoring_uses_concentration_level_replicate_means():
    text = APP.read_text(encoding="utf-8")
    assert "var concentrationEffects=[];" in text
    assert "fpd:mean(group.map" in text
    assert "rate:mean(group.map" in text
    assert "amp:mean(group.map" in text
    assert "stv:mean(group.map" in text
    assert "tri:mean(group.map" in text


def test_browser_scoring_uses_contract_driven_concentration_aggregation():
    text = APP.read_text(encoding="utf-8")
    contract = CONTRACT.read_text(encoding="utf-8")
    assert "function aggregateEndpoint(values,direction,aggregation)" in text
    assert "aggregation==='mean_harmful_effect'" in text
    assert "scoringContract.concentration_aggregation" in text
    assert "aggregateEndpoint(concentrationEffects.map(function(e){return e.fpd}),scoringContract.endpoints.fpd_change_pct.direction,aggregation)" in text
    assert "aggregateEndpoint(concentrationEffects.map(function(e){return e.rate}),scoringContract.endpoints.beat_rate_change_pct.direction,aggregation)" in text
    assert "aggregateEndpoint(concentrationEffects.map(function(e){return e.amp}),scoringContract.endpoints.amplitude_change_pct.direction,aggregation)" in text
    assert "aggregateEndpoint(concentrationEffects.map(function(e){return e.stv}),scoringContract.endpoints.stv_increase.direction,aggregation)" in text
    assert "aggregateEndpoint(concentrationEffects.map(function(e){return e.tri}),scoringContract.endpoints.triangulation_proxy.direction,aggregation)" in text
    assert '"concentration_aggregation": "mean_harmful_effect"' in contract


def test_browser_uses_shared_contract_for_weights_and_thresholds():
    text = APP.read_text(encoding="utf-8")
    contract = CONTRACT.read_text(encoding="utf-8")
    assert "function weights()" in text
    assert "scoringContract.endpoints.fpd_change_pct.weight" in text
    assert "scoringContract.endpoints.beat_rate_change_pct.weight" in text
    assert "scoringContract.endpoints.amplitude_change_pct.weight" in text
    assert "scoringContract.endpoints.stv_increase.weight" in text
    assert "scoringContract.endpoints.triangulation_proxy.weight" in text
    assert "scoringContract.risk_thresholds.low" in text
    assert "scoringContract.risk_thresholds.moderate" in text
    assert '"fpd_change_pct"' in contract
    assert '"risk_thresholds"' in contract


def test_browser_discloses_python_only_features():
    text = APP.read_text(encoding="utf-8")
    assert "plate/batch normalization, biological-unit inference, and 4PL evidence remain Python-only" in text

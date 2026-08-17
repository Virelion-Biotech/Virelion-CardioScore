from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from virelion_cardioscore.analysis.cipa_scoring import CardioScoreEngine
from virelion_cardioscore.analysis.hierarchy import aggregate_to_scoring_units
from virelion_cardioscore.analysis.pipeline import CardioScorePipeline
from virelion_cardioscore.analysis.statistics import bootstrap_profile_difference
from virelion_cardioscore.io.raw_trace import load_raw_traces_csv, recordings_to_feature_table
from virelion_cardioscore.preprocessing.beat_detection import BeatDetectionConfig, detect_beats


def test_beat_detection_module_exports_real_api():
    config = BeatDetectionConfig(min_prominence_uv=5.0, min_distance_ms=200.0, refractory_ms=200.0)
    trace = np.zeros(2000, dtype=float)
    trace[500] = 20.0
    trace[1500] = 20.0
    result = detect_beats(trace, fs_hz=1000.0, config=config)
    assert result.n_beats == 2
    assert result.beat_rate_bpm == pytest.approx(60.0)


def test_decrease_direction_does_not_score_an_increase():
    engine = CardioScoreEngine()
    positive = engine.score_compound(
        "Increase",
        {
            "fpd_change_pct": 0.0,
            "beat_rate_change_pct": 0.0,
            "amplitude_change_pct": 30.0,
            "stv_increase": 0.0,
            "triangulation_proxy": 0.0,
        },
    )
    negative = engine.score_compound(
        "Decrease",
        {
            "fpd_change_pct": 0.0,
            "beat_rate_change_pct": 0.0,
            "amplitude_change_pct": -30.0,
            "stv_increase": 0.0,
            "triangulation_proxy": 0.0,
        },
    )
    assert positive.score < negative.score
    assert next(c for c in positive.contributions if c.name == "amplitude_change_pct").contribution == 0.0
    assert next(c for c in negative.contributions if c.name == "amplitude_change_pct").contribution > 0.0


def test_higher_level_unit_rejects_missing_identifier():
    effects = pd.DataFrame(
        {
            "compound": ["A", "A"],
            "concentration_uM": [1.0, 1.0],
            "well": ["W1", "W2"],
            "biological_replicate": ["B1", None],
            "fpd_change_pct": [10.0, 12.0],
            "beat_rate_change_pct": [0.0, 0.0],
            "amplitude_change_pct": [0.0, 0.0],
            "stv_increase": [0.0, 0.0],
            "triangulation_proxy_change": [0.0, 0.0],
        }
    )
    with pytest.raises(ValueError, match="missing value"):
        aggregate_to_scoring_units(effects, scoring_unit="biological_replicate")


def test_pipeline_uses_configured_endpoint_file():
    pipeline = CardioScorePipeline.from_defaults()
    assert pipeline.engine.endpoint_config_path.name == "cipa_endpoints.yaml"


def test_score_result_reports_independent_units_separately():
    engine = CardioScoreEngine()
    result = engine.score_compound(
        "A",
        {
            "fpd_change_pct": 0.0,
            "beat_rate_change_pct": 0.0,
            "amplitude_change_pct": 0.0,
            "stv_increase": 0.0,
            "triangulation_proxy": 0.0,
        },
        n_wells=8,
        n_independent_units=4,
    )
    payload = result.to_dict()
    assert payload["n_wells"] == 8
    assert payload["n_independent_units"] == 4


def test_normalization_config_uses_single_all_groups_guardrail():
    pipeline = CardioScorePipeline.from_defaults()
    assumptions = pipeline.config["variability"]["correction"]["assumptions"]
    assert "require_treatment_in_all_groups" not in assumptions
    assert pipeline.config["variability"]["correction"]["require_all_groups"] is True
    assert assumptions["fail_closed"] is True


def test_raw_trace_metadata_survive_feature_extraction(tmp_path):
    # filter_trace requires at least 10 samples per electrode, so the trace
    # needs to be long enough to actually reach feature extraction rather
    # than failing on that guard before metadata propagation is even tested.
    n_samples = 20
    time_s = [round(i * 0.001, 3) for i in range(n_samples)]
    voltage_uv = [0.0, 1.0, 0.0, -1.0] * (n_samples // 4)
    frame = pd.DataFrame(
        {
            "compound": ["A"] * n_samples,
            "well": ["W1"] * n_samples,
            "concentration_uM": [0.0] * n_samples,
            "vehicle": [True] * n_samples,
            "electrode_id": ["E1"] * n_samples,
            "time_s": time_s,
            "voltage_uv": voltage_uv,
            "plate_id": ["P1"] * n_samples,
            "batch_id": ["B1"] * n_samples,
            "biological_replicate": ["BR1"] * n_samples,
        }
    )
    source = tmp_path / "raw.csv"
    frame.to_csv(source, index=False)
    recordings = load_raw_traces_csv(source)
    features = recordings_to_feature_table(recordings)
    assert features.loc[0, "plate_id"] == "P1"
    assert features.loc[0, "batch_id"] == "B1"
    assert features.loc[0, "biological_replicate"] == "BR1"


def test_bootstrap_profile_difference_has_no_pvalue_field():
    result = bootstrap_profile_difference(
        np.array([1.0]),
        {1.0: np.array([3.0, 4.0])},
        {1.0: np.array([1.0, 2.0])},
        n_bootstrap=100,
    )
    assert not hasattr(result, "p_values")

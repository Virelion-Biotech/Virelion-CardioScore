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


def test_raw_trace_metadata_survive_feature_extraction(tmp_path):
    frame = pd.DataFrame(
        {
            "compound": ["A", "A", "A", "A"],
            "well": ["W1"] * 4,
            "concentration_uM": [0.0] * 4,
            "vehicle": [True] * 4,
            "electrode_id": ["E1"] * 4,
            "time_s": [0.000, 0.001, 0.002, 0.003],
            "voltage_uv": [0.0, 1.0, 0.0, -1.0],
            "plate_id": ["P1"] * 4,
            "batch_id": ["B1"] * 4,
            "biological_replicate": ["BR1"] * 4,
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

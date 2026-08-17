"""Regression tests for CardioScore scoring, pipeline, and I/O contracts."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from virelion_cardioscore.analysis.cipa_scoring import CardioScoreEngine
from virelion_cardioscore.analysis.pipeline import CardioScorePipeline
from virelion_cardioscore.features.endpoints import extract_well_features
from virelion_cardioscore.io.raw_trace import (
    REQUIRED_COLUMNS,
    RawTraceSchemaError,
    _infer_sampling_rate,
    validate_raw_trace_schema,
)
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


def test_public_feature_endpoint_import_path():
    traces = {
        "E1": np.zeros(2000, dtype=float),
        "E2": np.zeros(2000, dtype=float),
    }
    features = extract_well_features(traces, fs_hz=1000.0)

    assert features.n_electrodes == 2
    assert set(features.to_row()) == {
        "fpd_ms",
        "beat_rate_bpm",
        "amplitude_uv",
        "stv",
        "triangulation_proxy",
        "noise_sd_uv",
        "n_electrodes",
        "beat_detection_rate",
    }


def _valid_raw_trace_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "compound": ["TestComp", "TestComp"],
            "well": ["A01", "A01"],
            "concentration_uM": [0.0, 0.0],
            "vehicle": ["True", "True"],
            "electrode_id": ["E1", "E1"],
            "time_s": [0.000, 0.001],
            "voltage_uv": [0.0, 1.0],
        }
    )


def test_raw_trace_schema_accepts_canonical_input():
    df = _valid_raw_trace_frame()
    validate_raw_trace_schema(df)


def test_raw_trace_schema_rejects_missing_columns():
    df = _valid_raw_trace_frame().drop(columns=["voltage_uv"])

    with pytest.raises(RawTraceSchemaError, match="missing required column"):
        validate_raw_trace_schema(df)


def test_raw_trace_schema_rejects_invalid_vehicle_values():
    df = _valid_raw_trace_frame()
    df.loc[0, "vehicle"] = "maybe"

    with pytest.raises(RawTraceSchemaError, match="unexpected values"):
        validate_raw_trace_schema(df)


def test_raw_trace_sampling_rate_inference():
    time_s = np.arange(0.0, 0.01, 0.001)
    assert _infer_sampling_rate(time_s) == pytest.approx(1000.0)


def test_raw_trace_required_columns_are_explicit():
    assert REQUIRED_COLUMNS == {
        "compound",
        "well",
        "concentration_uM",
        "vehicle",
        "electrode_id",
        "time_s",
        "voltage_uv",
    }

"""Regression tests for CardioScore scoring, pipeline, I/O, and dose-response contracts."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from virelion_cardioscore.analysis.cipa_scoring import CardioScoreEngine
from virelion_cardioscore.analysis.dose_response import fit_4pl
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
    assert not result.concentration_table.empty
    assert result.dose_response_fits == {}
    assert "cardioscore" in result.summary_table.columns
    assert "risk_class" in result.summary_table.columns
    assert "concentrations_tested" in result.summary_table.columns
    assert "max_effect_pct" in result.summary_table.columns
    assert "effect_detected" in result.summary_table.columns
    assert {"n_replicates", "fpd_change_pct_mean", "fpd_change_pct_sd"}.issubset(
        result.concentration_table.columns
    )
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


def _qc_frame(stv_values: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "compound": ["TestComp"] * len(stv_values),
            "well": [f"A{i:02d}" for i in range(len(stv_values))],
            "n_electrodes": [4] * len(stv_values),
            "noise_sd_uv": [5.0] * len(stv_values),
            "beat_detection_rate": [0.95] * len(stv_values),
            "stv": stv_values,
        }
    )


def test_optional_stv_irregularity_proxy_can_reject_wells():
    pipeline = CardioScorePipeline.from_defaults()
    pipeline.config["quality_control"]["reject_wells_with_arrhythmia_proxy"] = True
    pipeline.config["quality_control"]["arrhythmia_proxy_max_stv"] = 0.5

    kept = pipeline.apply_qc(_qc_frame([0.1, 0.8]))

    assert len(kept) == 1
    assert kept.iloc[0]["well"] == "A00"
    assert "stv=0.800" in " ".join(pipeline.qc_log)


def test_irregularity_proxy_requires_explicit_threshold():
    pipeline = CardioScorePipeline.from_defaults()
    pipeline.config["quality_control"]["reject_wells_with_arrhythmia_proxy"] = True
    pipeline.config["quality_control"]["arrhythmia_proxy_max_stv"] = None

    kept = pipeline.apply_qc(_qc_frame([0.1, 0.8]))

    assert len(kept) == 2
    assert any("max_stv is not configured" in msg for msg in pipeline.qc_log)


def test_concentration_coverage_warning_is_reported_without_silent_exclusion():
    dataset = load_synthetic_dataset(n_compounds=1, n_concentrations=2, seed=9)
    pipeline = CardioScorePipeline.from_defaults()
    result = pipeline.run(dataset)

    assert len(result.scores) == 1
    assert int(result.summary_table.iloc[0]["concentrations_tested"]) == 2
    assert any("configured minimum is 3" in msg for msg in result.qc_log)


def test_replicates_are_aggregated_within_concentration():
    effects = pd.DataFrame(
        {
            "compound": ["A", "A", "A"],
            "concentration_uM": [1.0, 1.0, 2.0],
            "well": ["W1", "W2", "W3"],
            "fpd_change_pct": [10.0, 30.0, 40.0],
            "beat_rate_change_pct": [0.0, 0.0, 0.0],
            "amplitude_change_pct": [0.0, 0.0, 0.0],
            "stv_increase": [0.0, 0.0, 0.0],
            "triangulation_proxy_change": [0.0, 0.0, 0.0],
        }
    )

    concentration_summary = CardioScorePipeline.summarize_concentrations(effects)
    first = concentration_summary.loc[concentration_summary["concentration_uM"] == 1.0].iloc[0]

    assert first["n_replicates"] == 2
    assert first["fpd_change_pct_mean"] == pytest.approx(20.0)
    assert first["fpd_change_pct_sd"] == pytest.approx(np.sqrt(200.0))


def test_median_replicate_aggregation_is_supported():
    effects = pd.DataFrame(
        {
            "compound": ["A", "A", "A"],
            "concentration_uM": [1.0, 1.0, 1.0],
            "well": ["W1", "W2", "W3"],
            "fpd_change_pct": [10.0, 20.0, 100.0],
            "beat_rate_change_pct": [0.0, 0.0, 0.0],
            "amplitude_change_pct": [0.0, 0.0, 0.0],
            "stv_increase": [0.0, 0.0, 0.0],
            "triangulation_proxy_change": [0.0, 0.0, 0.0],
        }
    )
    summary = CardioScorePipeline.summarize_concentrations(effects, replicate_aggregation="median")
    assert summary.iloc[0]["fpd_change_pct_mean"] == pytest.approx(20.0)


def test_invalid_aggregation_settings_are_rejected():
    effects = pd.DataFrame(
        {
            "compound": ["A"],
            "concentration_uM": [1.0],
            "well": ["W1"],
            "fpd_change_pct": [10.0],
            "beat_rate_change_pct": [0.0],
            "amplitude_change_pct": [0.0],
            "stv_increase": [0.0],
            "triangulation_proxy_change": [0.0],
        }
    )
    with pytest.raises(ValueError, match="Unsupported replicate_aggregation"):
        CardioScorePipeline.summarize_concentrations(effects, replicate_aggregation="bogus")


def test_compound_aggregation_uses_concentration_means_not_single_wells():
    concentration_summary = pd.DataFrame(
        {
            "compound": ["A", "A"],
            "concentration_uM": [1.0, 2.0],
            "n_replicates": [2, 2],
            "fpd_change_pct_mean": [20.0, 35.0],
            "beat_rate_change_pct_mean": [0.0, 0.0],
            "amplitude_change_pct_mean": [-5.0, -15.0],
            "stv_increase_mean": [0.05, 0.10],
            "triangulation_proxy_change_mean": [0.02, 0.04],
            "max_effect_pct_mean": [20.0, 35.0],
        }
    )

    aggregate = CardioScorePipeline.aggregate_compound_effects(concentration_summary)
    row = aggregate.iloc[0]

    assert row["fpd_change_pct"] == pytest.approx(35.0)
    assert row["amplitude_change_pct"] == pytest.approx(-15.0)
    assert row["n_wells"] == 4
    assert row["concentrations_tested"] == 2


def test_4pl_fit_recovers_known_curve():
    concentrations = np.logspace(-1, 2, 7)
    expected = 80.0 / (1.0 + (10.0 / concentrations) ** 1.5)
    result = fit_4pl(concentrations, expected, endpoint="fpd_change_pct")

    assert result.success
    assert result.ec50 == pytest.approx(10.0, rel=1e-2)
    assert result.hill_slope == pytest.approx(1.5, rel=1e-2)
    assert result.r_squared > 0.999


def test_4pl_fit_rejects_insufficient_concentrations():
    result = fit_4pl(np.array([1.0, 2.0, 4.0]), np.array([1.0, 3.0, 8.0]))
    assert not result.success
    assert "at least 4" in result.message


def test_pipeline_can_optionally_fit_dose_response():
    dataset = load_synthetic_dataset(n_compounds=2, n_concentrations=6, seed=12)
    pipeline = CardioScorePipeline.from_defaults()
    pipeline.config["concentration_response"]["fit_curve"] = True
    pipeline.config["concentration_response"]["fit_min_concentrations"] = 4

    result = pipeline.run(dataset)

    assert set(result.dose_response_fits) == {"Compound_A", "Compound_B"}
    assert all(len(fits) == 5 for fits in result.dose_response_fits.values())
    assert "dose_response_fit_endpoints" in result.summary_table.columns

"""Tests for raw-trace ingestion and feature extraction."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from virelion_cardioscore.io.raw_trace import (
    RawTraceValidationError,
    infer_sampling_rate,
    load_raw_traces_csv,
    load_raw_traces_to_feature_table,
    recordings_to_feature_table,
    validate_raw_trace_schema,
)


def test_validate_raw_trace_schema_missing_column():
    frame = pd.DataFrame({"time_s": [0.0, 0.001], "voltage_uv": [1.0, 2.0]})
    with pytest.raises(RawTraceValidationError, match="missing"):
        validate_raw_trace_schema(frame)


def test_validate_raw_trace_schema_empty_dataframe():
    frame = pd.DataFrame(
        columns=["compound", "well", "concentration_uM", "vehicle", "electrode_id", "time_s", "voltage_uv"]
    )
    with pytest.raises(RawTraceValidationError, match="empty"):
        validate_raw_trace_schema(frame)


def test_validate_raw_trace_schema_non_numeric_voltage():
    frame = pd.DataFrame(
        {
            "compound": ["A", "A"],
            "well": ["W1", "W1"],
            "concentration_uM": [0.0, 0.0],
            "vehicle": [True, True],
            "electrode_id": ["E1", "E1"],
            "time_s": [0.0, 0.001],
            "voltage_uv": ["bad", 2.0],
        }
    )
    with pytest.raises(RawTraceValidationError, match="numeric"):
        validate_raw_trace_schema(frame)


def test_validate_raw_trace_schema_nan_values():
    frame = pd.DataFrame(
        {
            "compound": ["A", "A"],
            "well": ["W1", "W1"],
            "concentration_uM": [0.0, 0.0],
            "vehicle": [True, True],
            "electrode_id": ["E1", "E1"],
            "time_s": [0.0, 0.001],
            "voltage_uv": [1.0, np.nan],
        }
    )
    with pytest.raises(RawTraceValidationError, match="finite"):
        validate_raw_trace_schema(frame)


def test_validate_raw_trace_schema_bad_vehicle_values():
    frame = pd.DataFrame(
        {
            "compound": ["A", "A"],
            "well": ["W1", "W1"],
            "concentration_uM": [0.0, 0.0],
            "vehicle": ["maybe", "maybe"],
            "electrode_id": ["E1", "E1"],
            "time_s": [0.0, 0.001],
            "voltage_uv": [1.0, 2.0],
        }
    )
    with pytest.raises(RawTraceValidationError, match="vehicle"):
        validate_raw_trace_schema(frame)


def test_load_raw_traces_csv_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_raw_traces_csv(tmp_path / "missing.csv")


def test_load_raw_traces_csv_groups_by_well_and_electrode(two_compound_plate):
    frame, recordings = load_raw_traces_csv(two_compound_plate, return_recordings=True)
    assert len(frame) > 0
    assert set(frame["compound"]) == {"Compound_Safe", "Compound_Toxic"}
    assert len(recordings) == 24


def test_load_raw_traces_csv_infers_sampling_rate(two_compound_plate):
    frame, recordings = load_raw_traces_csv(two_compound_plate, return_recordings=True)
    assert infer_sampling_rate(recordings[0].time_s) == pytest.approx(1000.0)
    assert frame["time_s"].min() == pytest.approx(0.0)


def test_load_raw_traces_csv_rejects_irregular_sampling(two_compound_plate, tmp_path):
    frame = pd.read_csv(two_compound_plate)
    mask = (frame["well"] == "W01") & (frame["electrode_id"] == "E1")
    frame.loc[mask, "time_s"] = frame.loc[mask, "time_s"].to_numpy() + np.linspace(0, 0.05, mask.sum())
    path = tmp_path / "irregular.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(RawTraceValidationError, match="sampling"):
        load_raw_traces_csv(path)


def test_load_raw_traces_csv_rejects_cross_electrode_time_shift(two_compound_plate, tmp_path):
    frame = pd.read_csv(two_compound_plate)
    mask = (frame["well"] == "W01") & (frame["electrode_id"] == "E2")
    frame.loc[mask, "time_s"] = frame.loc[mask, "time_s"] + 0.001
    path = tmp_path / "shifted.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(RawTraceValidationError, match="time axis"):
        load_raw_traces_csv(path)


def test_load_raw_traces_csv_rejects_cross_electrode_length_mismatch(two_compound_plate, tmp_path):
    frame = pd.read_csv(two_compound_plate)
    mask = (frame["well"] == "W01") & (frame["electrode_id"] == "E2")
    frame = frame.loc[~(mask & (frame["time_s"] > 7.9))].copy()
    path = tmp_path / "short_electrode.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(RawTraceValidationError, match="sample count"):
        load_raw_traces_csv(path)


def test_recordings_to_feature_table_schema(two_compound_plate):
    _, recordings = load_raw_traces_csv(two_compound_plate, return_recordings=True)
    table = recordings_to_feature_table(recordings)
    expected_cols = {
        "compound", "concentration_uM", "well", "vehicle", "fpd_ms",
        "beat_rate_bpm", "amplitude_uv", "stv", "triangulation_proxy",
        "noise_sd_uv", "n_electrodes", "beat_detection_rate",
    }
    assert expected_cols.issubset(set(table.columns))
    assert len(table) == 12


def test_load_raw_traces_to_feature_table_end_to_end(two_compound_plate):
    table = load_raw_traces_to_feature_table(two_compound_plate)
    toxic_vehicle = table[(table["compound"] == "Compound_Toxic") & (table["vehicle"])]
    toxic_high_dose = table[
        (table["compound"] == "Compound_Toxic") & (table["concentration_uM"] == 10.0)
    ]
    assert toxic_high_dose["fpd_ms"].mean() > toxic_vehicle["fpd_ms"].mean() + 50


def test_feature_table_from_raw_traces_runs_through_real_pipeline(two_compound_plate):
    from virelion_cardioscore.analysis.pipeline import CardioScorePipeline

    table = load_raw_traces_to_feature_table(two_compound_plate)
    pipeline = CardioScorePipeline.from_defaults()
    pipeline.config["concentration_response"]["require_min_concentrations_for_scoring"] = False
    result = pipeline.run(table)

    assert not result.summary_table.empty
    assert set(result.summary_table["compound"]) == {"Compound_Safe", "Compound_Toxic"}

    toxic_row = result.summary_table[result.summary_table["compound"] == "Compound_Toxic"].iloc[0]
    safe_row = result.summary_table[result.summary_table["compound"] == "Compound_Safe"].iloc[0]
    assert toxic_row["cardioscore"] > safe_row["cardioscore"]

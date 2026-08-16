"""Tests for virelion_cardioscore.io.raw_trace."""

from __future__ import annotations

import pandas as pd
import pytest

from virelion_cardioscore.io.raw_trace import (
    RawTraceSchemaError,
    load_raw_traces_csv,
    load_raw_traces_to_feature_table,
    recordings_to_feature_table,
    validate_raw_trace_schema,
)


def test_validate_raw_trace_schema_missing_column():
    df = pd.DataFrame({"compound": ["A"], "well": ["W01"]})
    with pytest.raises(RawTraceSchemaError, match="missing required column"):
        validate_raw_trace_schema(df)


def test_validate_raw_trace_schema_empty_dataframe():
    df = pd.DataFrame(
        columns=["compound", "well", "concentration_uM", "vehicle", "electrode_id", "time_s", "voltage_uv"]
    )
    with pytest.raises(RawTraceSchemaError, match="no rows"):
        validate_raw_trace_schema(df)


def test_validate_raw_trace_schema_non_numeric_voltage():
    df = pd.DataFrame(
        {
            "compound": ["A"],
            "well": ["W01"],
            "concentration_uM": [1.0],
            "vehicle": [True],
            "electrode_id": ["E1"],
            "time_s": [0.0],
            "voltage_uv": ["not a number"],
        }
    )
    with pytest.raises(RawTraceSchemaError, match="must be numeric"):
        validate_raw_trace_schema(df)


def test_validate_raw_trace_schema_nan_values():
    df = pd.DataFrame(
        {
            "compound": ["A"],
            "well": ["W01"],
            "concentration_uM": [1.0],
            "vehicle": [True],
            "electrode_id": ["E1"],
            "time_s": [float("nan")],
            "voltage_uv": [1.5],
        }
    )
    with pytest.raises(RawTraceSchemaError, match="missing value"):
        validate_raw_trace_schema(df)


def test_validate_raw_trace_schema_bad_vehicle_values():
    df = pd.DataFrame(
        {
            "compound": ["A"],
            "well": ["W01"],
            "concentration_uM": [1.0],
            "vehicle": ["maybe"],
            "electrode_id": ["E1"],
            "time_s": [0.0],
            "voltage_uv": [1.5],
        }
    )
    with pytest.raises(RawTraceSchemaError, match="vehicle"):
        validate_raw_trace_schema(df)


def test_load_raw_traces_csv_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_raw_traces_csv(tmp_path / "does_not_exist.csv")


def test_load_raw_traces_csv_groups_by_well_and_electrode(two_compound_plate):
    recordings = load_raw_traces_csv(two_compound_plate)
    # 2 compounds x (1 vehicle + 2 doses) x 2 replicates = 12 wells
    assert len(recordings) == 12
    for rec in recordings:
        assert rec.fs_hz == pytest.approx(1000.0, rel=0.01)
        assert len(rec.electrode_traces) == 4


def test_load_raw_traces_csv_infers_sampling_rate(two_compound_plate):
    recordings = load_raw_traces_csv(two_compound_plate)
    for rec in recordings:
        assert 900.0 < rec.fs_hz < 1100.0


def test_recordings_to_feature_table_schema(two_compound_plate):
    recordings = load_raw_traces_csv(two_compound_plate)
    table = recordings_to_feature_table(recordings)

    expected_cols = {
        "compound",
        "concentration_uM",
        "well",
        "vehicle",
        "fpd_ms",
        "beat_rate_bpm",
        "amplitude_uv",
        "stv",
        "triangulation_proxy",
        "noise_sd_uv",
        "n_electrodes",
        "beat_detection_rate",
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
    """
    Full integration: raw voltage CSV -> feature extraction -> the actual
    CardioScorePipeline, with no schema changes needed on the pipeline side.
    """
    from virelion_cardioscore.analysis.pipeline import CardioScorePipeline

    table = load_raw_traces_to_feature_table(two_compound_plate)
    pipeline = CardioScorePipeline.from_defaults()
    result = pipeline.run(table)

    assert not result.summary_table.empty
    assert set(result.summary_table["compound"]) == {"Compound_Safe", "Compound_Toxic"}

    toxic_row = result.summary_table[result.summary_table["compound"] == "Compound_Toxic"].iloc[0]
    safe_row = result.summary_table[result.summary_table["compound"] == "Compound_Safe"].iloc[0]
    assert toxic_row["cardioscore"] > safe_row["cardioscore"]

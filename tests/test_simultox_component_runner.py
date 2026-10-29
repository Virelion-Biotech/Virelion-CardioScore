from __future__ import annotations

import hashlib

import h5py
import numpy as np
import pytest

from scripts.validation.run_simultox_component_validation import (
    EXPECTED_FILES,
    acquire_file,
    center_window,
    json_safe,
    signal_qc_flags,
)


def test_simultox_source_asset_list_is_explicit_and_limited():
    assert set(EXPECTED_FILES) == {
        "SingleWell_HiPSC_cardio_Control.h5",
        "SingleWell_HiPSC_cardio_E-4031_1uM.h5",
        "6well_HiPSC_cardio_Control.h5",
    }
    assert all(len(checksum) == 32 for checksum in EXPECTED_FILES.values())


def test_cached_source_file_is_reused_only_when_published_checksum_matches(tmp_path):
    body = b"checksum-pinned local fixture"
    target = tmp_path / "fixture.bin"
    target.write_bytes(body)
    expected_md5 = hashlib.md5(body, usedforsecurity=False).hexdigest()

    result = acquire_file("fixture.bin", expected_md5, tmp_path)

    assert result == target
    assert result.read_bytes() == body


def test_center_window_uses_mcs_tick_units_and_midpoint(tmp_path):
    path = tmp_path / "tiny_mcs.h5"
    info_dtype = np.dtype([
        ("ChannelID", "<i4"),
        ("RowIndex", "<i4"),
        ("Label", "S32"),
        ("RawDataType", "S16"),
        ("Unit", "S8"),
        ("Exponent", "<i4"),
        ("ADZero", "<i4"),
        ("Tick", "<i8"),
        ("ConversionFactor", "<i8"),
        ("ADCBits", "<i4"),
    ])
    info = np.zeros(1, dtype=info_dtype)
    info[0]["ChannelID"] = 1
    info[0]["RowIndex"] = 0
    info[0]["Label"] = b"El 1"
    info[0]["Unit"] = b"V"
    info[0]["Tick"] = 1000  # 1 ms per sample -> 1000 Hz
    with h5py.File(path, "w") as handle:
        stream = handle.require_group("Data/Recording_0/AnalogStream/Stream_0")
        stream.create_dataset("ChannelData", data=np.zeros((1, 1000), dtype=np.int32))
        stream.create_dataset("InfoChannel", data=info)

    start, duration = center_window(path, seconds=0.25)
    assert start == pytest.approx(0.375)
    assert duration == pytest.approx(0.25)


def test_json_safe_converts_nonfinite_values_to_null():
    output = json_safe({"finite": np.float64(1.5), "nan": float("nan"), "nested": [np.inf]})
    assert output == {"finite": 1.5, "nan": None, "nested": [None]}



def test_signal_qc_flags_block_single_beat_fpd_candidate():
    features = {
        "beat_detection_rate": 0.024,
        "n_beats": 1,
        "n_beats_with_fpd": 1,
    }
    flags = signal_qc_flags(features, min_beat_detection_rate=0.7)

    assert flags["passes_beat_detection_qc"] is False
    assert flags["has_minimum_detected_beats"] is False
    assert flags["has_minimum_fpd_candidates"] is False
    assert flags["fpd_estimate_eligible"] is False
    assert "beat_detection_rate_below_pipeline_threshold" in flags["qc_failures"]


def test_signal_qc_flags_allow_repeated_fpd_candidates_with_adequate_detection():
    features = {
        "beat_detection_rate": 0.96,
        "n_beats": 50,
        "n_beats_with_fpd": 21,
    }
    flags = signal_qc_flags(features, min_beat_detection_rate=0.7)

    assert flags["passes_beat_detection_qc"] is True
    assert flags["has_minimum_detected_beats"] is True
    assert flags["has_minimum_fpd_candidates"] is True
    assert flags["fpd_estimate_eligible"] is True
    assert flags["qc_failures"] == []

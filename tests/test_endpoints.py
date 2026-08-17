"""Tests for virelion_cardioscore.features.endpoints."""

from __future__ import annotations

import numpy as np
import pytest

from tests.conftest import make_electrode_trace, make_well_traces
from virelion_cardioscore.features.endpoints import (
    extract_electrode_features,
    extract_well_features,
)


def test_extract_electrode_features_fpd_accuracy():
    true_fpd_ms = 280.0
    trace = make_electrode_trace(seed=1, bpm=55.0, fpd_ms=true_fpd_ms)
    features = extract_electrode_features(trace, fs_hz=1000.0)

    assert features.fpd_ms is not None
    assert abs(features.fpd_ms - true_fpd_ms) < 5.0
    assert features.n_beats_with_fpd / features.n_beats > 0.8


@pytest.mark.parametrize("true_fpd_ms", [250.0, 280.0, 380.0, 420.0])
def test_extract_electrode_features_fpd_across_range(true_fpd_ms):
    """FPD detection should track ground truth across a range of durations,
    including prolonged (cardiotoxic-like) values."""
    trace = make_electrode_trace(seed=hash(true_fpd_ms) % 1000, bpm=55.0, fpd_ms=true_fpd_ms)
    features = extract_electrode_features(trace, fs_hz=1000.0)

    assert features.fpd_ms is not None
    assert abs(features.fpd_ms - true_fpd_ms) < 8.0


def test_extract_electrode_features_beat_rate_accuracy():
    true_bpm = 55.0
    trace = make_electrode_trace(seed=1, bpm=true_bpm, fpd_ms=280.0)
    features = extract_electrode_features(trace, fs_hz=1000.0)
    assert abs(features.beat_rate_bpm - true_bpm) < 2.0


def test_extract_well_features_averages_across_electrodes(baseline_well):
    features = extract_well_features(baseline_well, fs_hz=1000.0)
    assert features.n_electrodes == len(baseline_well)
    assert abs(features.fpd_ms - 280.0) < 5.0
    assert abs(features.beat_rate_bpm - 55.0) < 2.0
    assert features.beat_detection_rate > 0.9


def test_extract_well_features_detects_fpd_prolongation():
    baseline = make_well_traces(seed_base=10, bpm=55.0, fpd_ms=280.0)
    toxic = make_well_traces(seed_base=200, bpm=48.0, fpd_ms=420.0, amp_scale=0.9)

    baseline_features = extract_well_features(baseline, fs_hz=1000.0)
    toxic_features = extract_well_features(toxic, fs_hz=1000.0)

    assert toxic_features.fpd_ms > baseline_features.fpd_ms + 50
    assert toxic_features.beat_rate_bpm < baseline_features.beat_rate_bpm


def test_extract_well_features_empty_dict_raises():
    with pytest.raises(ValueError, match="empty"):
        extract_well_features({}, fs_hz=1000.0)


def test_extract_well_features_handles_all_noise_electrode():
    """A well with no real beats should degrade to zeros, not crash."""
    noisy = {"E1": np.random.default_rng(0).normal(0, 5, 10000)}
    features = extract_well_features(noisy, fs_hz=1000.0)

    assert features.n_electrodes == 1
    assert features.fpd_ms == 0.0
    assert features.beat_rate_bpm == 0.0


def test_extract_well_features_excludes_unreliable_electrodes():
    """
    A well with one good electrode and one all-noise electrode should
    still produce a usable well-level estimate from the reliable electrode,
    while n_electrodes still reflects the total electrode count for QC.
    """
    traces = make_well_traces(seed_base=50, n_electrodes=2, bpm=55.0, fpd_ms=280.0)
    traces["E_bad"] = np.random.default_rng(1).normal(0, 5, len(next(iter(traces.values()))))

    features = extract_well_features(traces, fs_hz=1000.0)
    assert features.n_electrodes == 3
    assert abs(features.fpd_ms - 280.0) < 10.0  # still accurate, driven by good electrodes


def test_well_features_to_row_has_expected_columns(baseline_well):
    features = extract_well_features(baseline_well, fs_hz=1000.0)
    row = features.to_row()
    expected_keys = {
        "fpd_ms",
        "beat_rate_bpm",
        "amplitude_uv",
        "stv",
        "triangulation_proxy",
        "noise_sd_uv",
        "n_electrodes",
        "beat_detection_rate",
    }
    assert set(row.keys()) == expected_keys

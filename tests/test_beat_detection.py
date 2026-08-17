"""Tests for virelion_cardioscore.preprocessing.beat_detection."""

from __future__ import annotations

import numpy as np
import pytest

from tests.conftest import make_electrode_trace
from virelion_cardioscore.preprocessing.beat_detection import (
    BeatDetectionConfig,
    detect_beats,
)
from virelion_cardioscore.preprocessing.filtering import filter_trace


@pytest.mark.parametrize("polarity", [1, -1])
def test_detect_beats_accurate_rate_both_polarities(polarity):
    fs = 1000.0
    true_bpm = 55.0
    trace = make_electrode_trace(seed=1, fs_hz=fs, bpm=true_bpm, polarity=polarity)
    filtered = filter_trace(trace, fs_hz=fs)
    result = detect_beats(filtered, fs_hz=fs)

    assert result.n_beats > 0
    assert abs(result.beat_rate_bpm - true_bpm) < 2.0
    assert result.beat_detection_rate > 0.9


def test_detect_beats_does_not_double_count_repolarization():
    """
    Regression test: an earlier version rectified the trace before peak
    detection, which caused both the depolarization spike and the
    repolarization deflection to register as separate beats (~2x true
    count). Polarity must be selected once, by peak height, not per-sample.
    """
    fs = 1000.0
    duration = 20.0
    true_bpm = 55.0
    expected_beats = duration / (60.0 / true_bpm)

    trace = make_electrode_trace(seed=2, fs_hz=fs, duration_s=duration, bpm=true_bpm)
    filtered = filter_trace(trace, fs_hz=fs)
    result = detect_beats(filtered, fs_hz=fs)

    assert result.n_beats < expected_beats * 1.3
    assert result.n_beats > expected_beats * 0.7


def test_detect_beats_empty_on_flat_trace():
    fs = 1000.0
    flat = np.random.default_rng(0).normal(0, 1, 5000)
    filtered = filter_trace(flat, fs_hz=fs)
    result = detect_beats(filtered, fs_hz=fs)

    assert result.n_beats == 0
    assert result.beat_rate_bpm == 0.0
    assert result.beat_detection_rate == 0.0
    assert result.stv == 0.0


def test_detect_beats_rejects_2d_input():
    with pytest.raises(ValueError, match="1D"):
        detect_beats(np.zeros((10, 10)), fs_hz=1000.0)


def test_detect_beats_respects_refractory_period():
    """Two peaks closer than refractory_ms apart should collapse to one."""
    fs = 1000.0
    t = np.arange(0, 2, 1 / fs)
    trace = np.zeros_like(t)
    # Two spikes only 50ms apart -- well under a 200ms refractory period.
    trace += 100 * np.exp(-((t - 0.5) ** 2) / (2 * 0.003**2))
    trace += 100 * np.exp(-((t - 0.55) ** 2) / (2 * 0.003**2))

    result = detect_beats(
        trace,
        fs_hz=fs,
        config=BeatDetectionConfig(refractory_ms=200.0, min_distance_ms=200.0),
    )
    assert result.n_beats == 1


def test_beat_detection_config_from_dict():
    cfg = BeatDetectionConfig.from_dict(
        {
            "method": "peak",
            "min_prominence_uv": 15.0,
            "min_distance_ms": 300.0,
            "refractory_ms": 250.0,
        }
    )
    assert cfg.min_prominence_uv == 15.0
    assert cfg.min_distance_ms == 300.0
    assert cfg.refractory_ms == 250.0


def test_stv_zero_for_perfectly_regular_beats():
    fs = 1000.0
    t = np.arange(0, 10, 1 / fs)
    trace = np.zeros_like(t)
    beat_time = 0.5
    while beat_time < 10:
        trace += 100 * np.exp(-((t - beat_time) ** 2) / (2 * 0.003**2))
        beat_time += 1.0

    result = detect_beats(trace, fs_hz=fs)
    assert result.stv < 0.01

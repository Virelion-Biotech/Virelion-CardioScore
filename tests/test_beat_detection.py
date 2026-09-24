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


def test_detect_beats_missed_beats_lower_detection_rate():
    """A configured prominence that misses weak beats must not report 100% detection."""
    fs = 1000.0
    t = np.arange(0, 10, 1 / fs)
    trace = np.zeros_like(t)

    # Strong beats occur every 2 s and pass the configured 20 uV threshold.
    # Weak beats occur halfway between them and are visible to the lower-
    # prominence candidate detector but should fail the configured detector.
    for beat_time in np.arange(0.5, 10.0, 2.0):
        trace += 100 * np.exp(-((t - beat_time) ** 2) / (2 * 0.003**2))
    for beat_time in np.arange(1.5, 10.0, 2.0):
        trace += 8 * np.exp(-((t - beat_time) ** 2) / (2 * 0.003**2))

    result = detect_beats(
        trace,
        fs_hz=fs,
        config=BeatDetectionConfig(min_prominence_uv=20.0, min_distance_ms=250.0),
    )

    assert 0 < result.n_beats < len(result.candidate_beat_indices)
    assert result.beat_detection_rate < 0.75


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


def test_detect_beats_rejects_nonfinite_sampling_rate():
    with pytest.raises(ValueError, match="finite and positive"):
        detect_beats(np.zeros(1000), fs_hz=np.nan)


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


# --- companion (repolarization-artifact) suppression -----------------------------------------------


def _spike_companion_trace(
    fs_hz: float,
    n_beats: int,
    period_s: float,
    companion_gap_s: float | None,
    primary_amp: float,
    companion_amp: float = 0.0,
    noise_sd: float = 0.5,
    seed: int = 0,
    lead_in_s: float = 0.5,
) -> np.ndarray:
    """A hand-built filtered-style trace with optional smaller companion deflections."""
    rng = np.random.default_rng(seed)
    duration_s = lead_in_s + n_beats * period_s + 1.0
    t = np.arange(0, duration_s, 1 / fs_hz)
    trace = np.zeros_like(t)
    for i in range(n_beats):
        beat_t = lead_in_s + i * period_s
        trace += primary_amp * np.exp(-((t - beat_t) ** 2) / (2 * 0.003**2))
        if companion_gap_s is not None and companion_amp:
            trace += companion_amp * np.exp(
                -((t - (beat_t + companion_gap_s)) ** 2) / (2 * 0.015**2)
            )
    return trace + rng.normal(0, noise_sd, size=t.shape)


def test_suppress_companions_folds_a_close_smaller_peak():
    indices = np.array([0, 270, 2050, 2320, 4100])
    amplitudes = np.array([200.0, 90.0, 205.0, 88.0, 198.0])
    kept_idx, kept_amp, has_companion = _suppress_companions(
        indices, amplitudes, fs_hz=1000.0, max_amplitude_ratio=0.6,
        max_gap_fraction_of_period=0.5, expected_period_s=2.05,
    )
    assert kept_idx.tolist() == [0, 2050, 4100]
    assert kept_amp.tolist() == [200.0, 205.0, 198.0]
    assert has_companion.tolist() == [True, True, False]


def test_suppress_companions_keeps_a_similar_amplitude_peak():
    indices = np.array([0, 270])
    amplitudes = np.array([200.0, 180.0])
    kept_idx, _, has_companion = _suppress_companions(
        indices, amplitudes, fs_hz=1000.0, max_amplitude_ratio=0.6,
        max_gap_fraction_of_period=0.5, expected_period_s=2.05,
    )
    assert kept_idx.tolist() == [0, 270] and not has_companion.any()


def test_suppress_companions_keeps_a_far_away_smaller_peak():
    indices = np.array([0, 1200])
    amplitudes = np.array([200.0, 40.0])
    kept_idx, _, _ = _suppress_companions(
        indices, amplitudes, fs_hz=1000.0, max_amplitude_ratio=0.6,
        max_gap_fraction_of_period=0.5, expected_period_s=2.05,
    )
    assert kept_idx.tolist() == [0, 1200]


def test_suppress_companions_without_a_period_estimate_is_a_no_op():
    indices = np.array([0, 270, 2050])
    amplitudes = np.array([200.0, 90.0, 205.0])
    for period in (None, 0.0, -1.0):
        kept_idx, kept_amp, has_companion = _suppress_companions(
            indices, amplitudes, fs_hz=1000.0, max_amplitude_ratio=0.6,
            max_gap_fraction_of_period=0.5, expected_period_s=period,
        )
        assert kept_idx.tolist() == indices.tolist() and not has_companion.any()


def test_suppress_companions_ratio_zero_disables_folding():
    indices = np.array([0, 270, 2050])
    amplitudes = np.array([200.0, 90.0, 205.0])
    kept_idx, _, has_companion = _suppress_companions(
        indices, amplitudes, fs_hz=1000.0, max_amplitude_ratio=0.0,
        max_gap_fraction_of_period=0.5, expected_period_s=2.05,
    )
    assert kept_idx.tolist() == indices.tolist() and not has_companion.any()


def test_detect_beats_end_to_end_folds_a_realistic_repolarization_companion():
    trace = _spike_companion_trace(
        fs_hz=1000.0, n_beats=6, period_s=2.05, companion_gap_s=0.27,
        primary_amp=200.0, companion_amp=85.0,
    )
    result = detect_beats(trace, 1000.0)
    assert result.n_beats == 6
    assert result.has_companion.tolist() == [True] * 6
    assert result.beat_detection_rate >= 0.7


def test_detect_beats_does_not_fold_two_genuinely_close_full_beats():
    trace = _spike_companion_trace(
        fs_hz=1000.0, n_beats=6, period_s=1.0, companion_gap_s=0.3,
        primary_amp=200.0, companion_amp=180.0,
    )
    result = detect_beats(trace, 1000.0)
    assert result.n_beats == 12
    assert not result.has_companion.any()


def test_detect_beats_without_companion_is_unaffected():
    trace = _spike_companion_trace(
        fs_hz=1000.0, n_beats=8, period_s=1.2, companion_gap_s=None, primary_amp=200.0
    )
    result = detect_beats(trace, 1000.0)
    assert result.n_beats == 8 and not result.has_companion.any()

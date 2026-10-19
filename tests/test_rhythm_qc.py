"""Rhythm QC must reject missed beats without mistaking slow structure for beats."""
from __future__ import annotations

import numpy as np
import pytest

from virelion_cardioscore.preprocessing.beat_detection import (
    BeatDetectionConfig,
    _estimate_expected_beat_count,
    detect_beats,
)


def pulse_train(fs, period, duration, *, alternating=False):
    t = np.arange(0, duration, 1 / fs)
    x = np.zeros_like(t)
    for i, tm in enumerate(np.arange(period / 2, duration, period)):
        amplitude = 8 if alternating and i % 2 else 100
        x += amplitude * np.exp(-((t - tm) ** 2) / (2 * .006**2))
    return t, x


@pytest.mark.parametrize("fs", [250., 1000.])
@pytest.mark.parametrize("period", [1.2, 2., 3.2])
def test_slow_half_period_structure_does_not_double_expected_count(fs, period):
    duration = 10 * period
    t, x = pulse_train(fs, period, duration)
    # A smooth, non-beat component repeats twice per depolarization cycle.
    # Its voltage autocorrelation is strong, but its slope is small compared
    # with the narrow spikes. No peak-detector output is provided to QC.
    x += 20 * np.cos(4 * np.pi * t / period)
    expected = _estimate_expected_beat_count(x, fs, duration)
    assert expected == pytest.approx(duration / period, rel=.03)


@pytest.mark.parametrize("fs", [250., 1000.])
@pytest.mark.parametrize("period", [.6, 1., 1.6])
def test_true_weak_alternating_beats_remain_in_independent_expected_count(fs, period):
    duration = 20 * period
    _, x = pulse_train(fs, period, duration, alternating=True)
    expected = _estimate_expected_beat_count(x, fs, duration)
    assert expected == pytest.approx(duration / period, rel=.03)
    result = detect_beats(x, fs, BeatDetectionConfig(min_prominence_uv=20))
    assert result.n_beats == 10
    assert result.beat_detection_rate < .75


def test_qc_does_not_become_self_validating_when_detector_threshold_is_changed():
    fs, period, duration = 1000., 1., 20.
    _, x = pulse_train(fs, period, duration, alternating=True)
    strict = detect_beats(x, fs, BeatDetectionConfig(min_prominence_uv=20))
    permissive = detect_beats(x, fs, BeatDetectionConfig(min_prominence_uv=5))
    assert strict.expected_beat_count == permissive.expected_beat_count
    assert strict.n_beats == 10 and permissive.n_beats == 20
    assert strict.beat_detection_rate < .75
    assert permissive.beat_detection_rate > .95


@pytest.mark.parametrize("seed,amplitude", [(11, 800.), (12, 600.)])
def test_jitter_does_not_turn_three_cycle_correlation_into_the_beat_period(seed, amplitude):
    from virelion_cardioscore.preprocessing.filtering import FilterConfig, filter_trace
    from virelion_cardioscore.validation.signal_truth import make_truth_recording

    truth = make_truth_recording(
        seed, duration_s=90, amplitude_uv=amplitude, noise_sd_uv=3, bpm=50,
    )
    for start in [*range(0, 81, 10), 63]:
        raw = np.round(truth.trace_uv[start*1000:(start+10)*1000], 2)
        x = filter_trace(raw, truth.fs_hz, FilterConfig())
        expected = _estimate_expected_beat_count(x, truth.fs_hz, 10.)
        assert expected == pytest.approx(10/1.2, rel=.04), start


@pytest.mark.parametrize("n", [0, 1, 10, 1000])
def test_flat_and_short_inputs_remain_safe(n):
    x = np.zeros(n)
    expected = _estimate_expected_beat_count(x, 1000., n/1000.)
    assert expected is None or np.isfinite(expected)
    if n >= 1000:
        assert expected is None


@pytest.mark.parametrize("polarity", [-1., 1.])
@pytest.mark.parametrize("offset,scale", [(0., 1.), (500., 7.)])
def test_rhythm_estimate_is_independent_of_voltage_polarity_offset_and_scale(polarity, offset, scale):
    fs, period, duration = 1000., 2., 20.
    t, x = pulse_train(fs, period, duration)
    x += 20*np.cos(4*np.pi*t/period)
    reference = _estimate_expected_beat_count(x, fs, duration)
    transformed = _estimate_expected_beat_count(offset+polarity*scale*x, fs, duration)
    assert transformed == pytest.approx(reference, rel=1e-10)

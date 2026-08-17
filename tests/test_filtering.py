"""Tests for virelion_cardioscore.preprocessing.filtering."""

from __future__ import annotations

import numpy as np
import pytest

from virelion_cardioscore.preprocessing.filtering import (
    FilterConfig,
    estimate_noise_sd,
    filter_trace,
)


def test_filter_trace_removes_dc_offset():
    fs = 1000.0
    t = np.arange(0, 5, 1 / fs)
    trace = np.full_like(t, 500.0) + np.random.default_rng(0).normal(0, 1, len(t))
    filtered = filter_trace(trace, fs_hz=fs)
    assert abs(np.mean(filtered)) < 5.0  # DC offset should be gone


def test_filter_trace_removes_mains_hum():
    fs = 1000.0
    t = np.arange(0, 5, 1 / fs)
    hum = 20 * np.sin(2 * np.pi * 50 * t)
    noise = np.random.default_rng(0).normal(0, 0.5, len(t))
    trace = hum + noise

    filtered = filter_trace(trace, fs_hz=fs, config=FilterConfig(notch_hz=50.0))
    # Power at 50Hz should drop sharply after notch filtering.
    freqs = np.fft.rfftfreq(len(t), 1 / fs)
    power_before = np.abs(np.fft.rfft(trace))
    power_after = np.abs(np.fft.rfft(filtered))
    idx_50hz = np.argmin(np.abs(freqs - 50.0))
    assert power_after[idx_50hz] < 0.2 * power_before[idx_50hz]


def test_filter_trace_preserves_signal_shape(baseline_trace):
    fs = 1000.0
    filtered = filter_trace(baseline_trace, fs_hz=fs)
    assert filtered.shape == baseline_trace.shape
    # Should still have substantial amplitude -- not filtered into flatness.
    assert np.max(np.abs(filtered)) > 20.0


def test_filter_trace_rejects_short_trace():
    with pytest.raises(ValueError, match="too short"):
        filter_trace(np.zeros(5), fs_hz=1000.0)


def test_filter_trace_rejects_2d_input():
    with pytest.raises(ValueError, match="1D"):
        filter_trace(np.zeros((10, 10)), fs_hz=1000.0)


def test_filter_trace_rejects_invalid_nyquist():
    with pytest.raises(ValueError, match="Nyquist"):
        filter_trace(
            np.random.default_rng(0).normal(0, 1, 100),
            fs_hz=1.0,
            config=FilterConfig(highpass_hz=0.5, lowpass_hz=0.4),
        )


def test_filter_trace_skips_notch_above_nyquist():
    # notch_hz above Nyquist should be silently skipped, not raise -- even
    # though highpass/lowpass are still valid at this sampling rate.
    fs = 50.0
    trace = np.random.default_rng(0).normal(0, 1, 500)
    config = FilterConfig(highpass_hz=0.5, lowpass_hz=20.0, notch_hz=50.0)
    filtered = filter_trace(trace, fs_hz=fs, config=config)
    assert filtered.shape == trace.shape


def test_estimate_noise_sd_scales_with_injected_noise():
    fs = 1000.0
    t = np.arange(0, 5, 1 / fs)
    low = np.random.default_rng(1).normal(0, 1.0, len(t))
    high = np.random.default_rng(1).normal(0, 8.0, len(t))

    noise_low = estimate_noise_sd(low, fs)
    noise_high = estimate_noise_sd(high, fs)
    assert noise_high > noise_low


def test_filter_config_from_dict_matches_yaml_keys():
    cfg = FilterConfig.from_dict(
        {"highpass_hz": 1.0, "lowpass_hz": 30.0, "notch_hz": 60.0, "notch_q": 25.0, "detrend": False}
    )
    assert cfg.highpass_hz == 1.0
    assert cfg.lowpass_hz == 30.0
    assert cfg.notch_hz == 60.0
    assert cfg.notch_q == 25.0
    assert cfg.detrend is False


def test_filter_config_from_dict_uses_defaults_for_missing_keys():
    cfg = FilterConfig.from_dict({})
    assert cfg.highpass_hz == 0.5
    assert cfg.lowpass_hz == 40.0

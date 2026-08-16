"""
Signal filtering for raw MEA field-potential traces.

Takes raw voltage traces (single electrode or well-averaged) and applies the
detrend / highpass / lowpass / notch chain defined under `preprocessing:` in
default.yaml, before beat detection or feature extraction run on them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from scipy import signal


@dataclass
class FilterConfig:
    """Mirrors the `preprocessing:` block in config/default.yaml."""

    highpass_hz: float = 0.5
    lowpass_hz: float = 40.0
    notch_hz: Optional[float] = 50.0
    notch_q: float = 30.0
    detrend: bool = True

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "FilterConfig":
        return cls(
            highpass_hz=float(cfg.get("highpass_hz", 0.5)),
            lowpass_hz=float(cfg.get("lowpass_hz", 40.0)),
            notch_hz=cfg.get("notch_hz", 50.0),
            notch_q=float(cfg.get("notch_q", 30.0)),
            detrend=bool(cfg.get("detrend", True)),
        )


def filter_trace(
    trace: np.ndarray,
    fs_hz: float,
    config: Optional[FilterConfig] = None,
) -> np.ndarray:
    """
    Apply detrend -> highpass -> lowpass -> notch to a 1D voltage trace.

    Parameters
    ----------
    trace : np.ndarray
        Raw voltage samples, shape (n_samples,), in microvolts.
    fs_hz : float
        Sampling rate in Hz.
    config : FilterConfig, optional
        Filter parameters. Defaults to the values in config/default.yaml.

    Returns
    -------
    np.ndarray
        Filtered trace, same shape as input.
    """
    if config is None:
        config = FilterConfig()

    trace = np.asarray(trace, dtype=float)
    if trace.ndim != 1:
        raise ValueError(f"filter_trace expects a 1D array, got shape {trace.shape}")
    if len(trace) < 10:
        raise ValueError("Trace too short to filter (need at least 10 samples)")

    nyquist = fs_hz / 2.0
    out = trace.copy()

    if config.detrend:
        out = signal.detrend(out, type="linear")

    # Highpass removes baseline wander / DC drift below highpass_hz.
    if config.highpass_hz and config.highpass_hz > 0:
        wn = config.highpass_hz / nyquist
        if not (0 < wn < 1):
            raise ValueError(
                f"highpass_hz={config.highpass_hz} is invalid for fs_hz={fs_hz} "
                f"(Nyquist={nyquist}); lower fs_hz or highpass_hz."
            )
        b, a = signal.butter(4, wn, btype="highpass")
        out = signal.filtfilt(b, a, out)

    # Lowpass removes high-frequency noise above lowpass_hz.
    if config.lowpass_hz and config.lowpass_hz > 0:
        wn = config.lowpass_hz / nyquist
        if not (0 < wn < 1):
            raise ValueError(
                f"lowpass_hz={config.lowpass_hz} is invalid for fs_hz={fs_hz} "
                f"(Nyquist={nyquist}); lower lowpass_hz or raise fs_hz."
            )
        b, a = signal.butter(4, wn, btype="lowpass")
        out = signal.filtfilt(b, a, out)

    # Notch removes mains hum (50/60 Hz). Skipped if it would exceed Nyquist
    # (e.g. low sampling rates) rather than raising, since it's optional.
    if config.notch_hz and config.notch_hz > 0 and config.notch_hz < nyquist:
        b, a = signal.iirnotch(config.notch_hz, config.notch_q, fs_hz)
        out = signal.filtfilt(b, a, out)

    return out


def estimate_noise_sd(trace: np.ndarray, fs_hz: float) -> float:
    """
    Estimate baseline noise standard deviation (uV) from a filtered trace.

    Uses the median absolute deviation of the high-frequency residual
    (trace minus a heavily-smoothed version of itself) as a robust noise
    estimate that isn't dominated by beat amplitude.
    """
    trace = np.asarray(trace, dtype=float)
    window = max(3, int(fs_hz * 0.02))  # ~20ms smoothing window
    if window % 2 == 0:
        window += 1
    if len(trace) <= window:
        return float(np.std(trace))

    smoothed = signal.savgol_filter(trace, window_length=window, polyorder=2)
    residual = trace - smoothed
    mad = np.median(np.abs(residual - np.median(residual)))
    return float(mad * 1.4826)  # MAD -> SD scaling for normal-ish noise

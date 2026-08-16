"""
Beat detection on filtered MEA field-potential traces.

Finds beat onsets (spike peaks) in a single-electrode trace using prominence-
based peak detection, per the `beat_detection:` block in config/default.yaml.
Downstream feature extraction (features/endpoints.py) consumes the beat
indices this module returns to compute FPD, beat rate, amplitude, STV, and
triangulation proxy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
from scipy import signal


@dataclass
class BeatDetectionConfig:
    """Mirrors the `beat_detection:` block in config/default.yaml."""

    method: str = "peak"
    min_prominence_uv: float = 20.0
    min_distance_ms: float = 250.0
    refractory_ms: float = 200.0

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "BeatDetectionConfig":
        return cls(
            method=cfg.get("method", "peak"),
            min_prominence_uv=float(cfg.get("min_prominence_uv", 20.0)),
            min_distance_ms=float(cfg.get("min_distance_ms", 250.0)),
            refractory_ms=float(cfg.get("refractory_ms", 200.0)),
        )


@dataclass
class BeatDetectionResult:
    """Beats detected in a single-electrode trace."""

    beat_indices: np.ndarray
    beat_times_s: np.ndarray
    amplitudes_uv: np.ndarray
    fs_hz: float
    duration_s: float
    n_beats: int = field(init=False)
    inter_beat_intervals_s: np.ndarray = field(init=False)
    beat_detection_rate: float = field(init=False)

    def __post_init__(self):
        self.n_beats = len(self.beat_indices)
        if self.n_beats >= 2:
            self.inter_beat_intervals_s = np.diff(self.beat_times_s)
        else:
            self.inter_beat_intervals_s = np.array([])
        self.beat_detection_rate = self._estimate_detection_rate()

    def _estimate_detection_rate(self) -> float:
        """
        Estimate the fraction of true beats actually detected.

        There's no ground truth for real recordings, so this uses the
        expected-beat-count heuristic: the median inter-beat interval (IBI)
        defines the dominant rhythm, and expected_beats = duration /
        median_IBI. This penalizes traces with dropped/skipped beats (large
        gaps relative to the dominant rhythm) without needing a reference.
        """
        if self.n_beats < 2:
            return 0.0 if self.n_beats == 0 else 1.0

        median_ibi = float(np.median(self.inter_beat_intervals_s))
        if median_ibi <= 0:
            return 1.0

        expected_beats = self.duration_s / median_ibi
        if expected_beats <= 0:
            return 1.0

        return float(np.clip(self.n_beats / expected_beats, 0.0, 1.0))

    @property
    def beat_rate_bpm(self) -> float:
        if self.n_beats < 2:
            return 0.0
        mean_ibi_s = float(np.mean(self.inter_beat_intervals_s))
        if mean_ibi_s <= 0:
            return 0.0
        return 60.0 / mean_ibi_s

    @property
    def mean_amplitude_uv(self) -> float:
        if self.n_beats == 0:
            return 0.0
        return float(np.mean(self.amplitudes_uv))

    @property
    def stv(self) -> float:
        """
        Short-term variability: mean absolute difference between consecutive
        IBIs, normalized by mean IBI (Poincare-style STV, unitless).
        """
        if len(self.inter_beat_intervals_s) < 2:
            return 0.0
        diffs = np.abs(np.diff(self.inter_beat_intervals_s))
        mean_ibi = float(np.mean(self.inter_beat_intervals_s))
        if mean_ibi <= 0:
            return 0.0
        return float(np.mean(diffs) / mean_ibi)


def detect_beats(
    trace: np.ndarray,
    fs_hz: float,
    config: Optional[BeatDetectionConfig] = None,
) -> BeatDetectionResult:
    """
    Detect beat peaks in a filtered single-electrode voltage trace.

    Parameters
    ----------
    trace : np.ndarray
        Filtered voltage trace, shape (n_samples,), in microvolts. Run
        preprocessing.filtering.filter_trace on the raw signal first.
    fs_hz : float
        Sampling rate in Hz.
    config : BeatDetectionConfig, optional
        Detection parameters. Defaults to config/default.yaml values.

    Returns
    -------
    BeatDetectionResult
    """
    if config is None:
        config = BeatDetectionConfig()

    trace = np.asarray(trace, dtype=float)
    if trace.ndim != 1:
        raise ValueError(f"detect_beats expects a 1D array, got shape {trace.shape}")

    duration_s = len(trace) / fs_hz
    min_distance_samples = max(1, int(round(config.min_distance_ms / 1000.0 * fs_hz)))

    # MEA field potentials show both a fast depolarization spike and a
    # slower, smaller repolarization deflection. Electrode polarity also
    # varies (the depolarization spike can be positive- or negative-going
    # depending on electrode placement), so we can't just threshold the raw
    # signal directly. But rectifying (abs) makes both the spike and the
    # repolarization bump register as separate positive peaks, which
    # double-counts beats. Instead: find candidate peaks in both polarities
    # separately, then pick ONE polarity for the beat markers.
    #
    # Selection uses raw peak HEIGHT, not prominence. Prominence measures
    # the drop to the nearest higher terrain, and the depol spike sits
    # immediately next to the repol deflection swinging the opposite way --
    # so both directions end up with similar prominence (the algorithm was
    # effectively measuring the same depol-to-repol swing either way,
    # regardless of which one it called the "peak"). Raw height cleanly
    # separates them, since the depolarization spike is reliably taller.
    pos_indices, pos_props = signal.find_peaks(
        trace, prominence=config.min_prominence_uv, distance=min_distance_samples
    )
    neg_indices, neg_props = signal.find_peaks(
        -trace, prominence=config.min_prominence_uv, distance=min_distance_samples
    )

    pos_mean_height = float(np.mean(np.abs(trace[pos_indices]))) if len(pos_indices) else 0.0
    neg_mean_height = float(np.mean(np.abs(trace[neg_indices]))) if len(neg_indices) else 0.0

    if pos_mean_height >= neg_mean_height:
        peak_indices = pos_indices
    else:
        peak_indices = neg_indices

    # Enforce the refractory period explicitly (belt-and-suspenders on top of
    # `distance`, since refractory_ms and min_distance_ms are configured
    # separately and may differ).
    refractory_samples = max(1, int(round(config.refractory_ms / 1000.0 * fs_hz)))
    kept: list[int] = []
    last_idx = -refractory_samples - 1
    for idx in peak_indices:
        if idx - last_idx >= refractory_samples:
            kept.append(idx)
            last_idx = idx
    beat_indices = np.array(kept, dtype=int)

    beat_times_s = beat_indices / fs_hz
    amplitudes_uv = trace[beat_indices] if len(beat_indices) else np.array([])

    return BeatDetectionResult(
        beat_indices=beat_indices,
        beat_times_s=beat_times_s,
        amplitudes_uv=amplitudes_uv,
        fs_hz=fs_hz,
        duration_s=duration_s,
    )

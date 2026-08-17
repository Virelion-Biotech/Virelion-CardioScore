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
    """Configuration for single-electrode beat detection."""

    method: str = "peak"
    min_prominence_uv: float = 20.0
    min_distance_ms: float = 250.0
    refractory_ms: float = 200.0

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "BeatDetectionConfig":
        return cls(
            method=str(cfg.get("method", "peak")),
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

    def __post_init__(self) -> None:
        self.n_beats = len(self.beat_indices)
        if self.n_beats >= 2:
            self.inter_beat_intervals_s = np.diff(self.beat_times_s)
        else:
            self.inter_beat_intervals_s = np.array([])
        self.beat_detection_rate = self._estimate_detection_rate()

    def _estimate_detection_rate(self) -> float:
        """Estimate detected beats relative to the dominant observed rhythm."""
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
        return float(np.mean(np.abs(self.amplitudes_uv)))

    @property
    def stv(self) -> float:
        """Short-term variability of consecutive inter-beat intervals."""
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
    """Detect depolarization peaks in a filtered single-electrode trace."""
    if config is None:
        config = BeatDetectionConfig()

    trace = np.asarray(trace, dtype=float)
    if trace.ndim != 1:
        raise ValueError(f"detect_beats expects a 1D array, got shape {trace.shape}")
    if fs_hz <= 0:
        raise ValueError("fs_hz must be positive")
    if not np.all(np.isfinite(trace)):
        raise ValueError("detect_beats requires finite trace values")
    if config.method != "peak":
        raise ValueError(f"Unsupported beat detection method: {config.method!r}")
    if config.min_prominence_uv < 0 or config.min_distance_ms <= 0 or config.refractory_ms <= 0:
        raise ValueError("Beat detection thresholds and distances must be positive")

    duration_s = len(trace) / fs_hz
    min_distance_samples = max(1, int(round(config.min_distance_ms / 1000.0 * fs_hz)))

    pos_indices, _ = signal.find_peaks(
        trace,
        prominence=config.min_prominence_uv,
        distance=min_distance_samples,
    )
    neg_indices, _ = signal.find_peaks(
        -trace,
        prominence=config.min_prominence_uv,
        distance=min_distance_samples,
    )

    pos_mean_height = float(np.mean(np.abs(trace[pos_indices]))) if len(pos_indices) else 0.0
    neg_mean_height = float(np.mean(np.abs(trace[neg_indices]))) if len(neg_indices) else 0.0
    peak_indices = pos_indices if pos_mean_height >= neg_mean_height else neg_indices

    refractory_samples = max(1, int(round(config.refractory_ms / 1000.0 * fs_hz)))
    kept: list[int] = []
    last_idx = -refractory_samples - 1
    for idx in peak_indices:
        idx = int(idx)
        if idx - last_idx >= refractory_samples:
            kept.append(idx)
            last_idx = idx

    beat_indices = np.asarray(kept, dtype=int)
    beat_times_s = beat_indices / fs_hz
    amplitudes_uv = trace[beat_indices] if len(beat_indices) else np.array([], dtype=float)

    return BeatDetectionResult(
        beat_indices=beat_indices,
        beat_times_s=beat_times_s,
        amplitudes_uv=amplitudes_uv,
        fs_hz=float(fs_hz),
        duration_s=float(duration_s),
    )

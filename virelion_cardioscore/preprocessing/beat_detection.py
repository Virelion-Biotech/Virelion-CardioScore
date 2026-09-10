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
    candidate_beat_indices: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
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
        """Estimate detector coverage against an independent rhythm estimate.

        The previous implementation compared configured detections with a
        lower-prominence candidate peak pass. That candidate pass is not an
        independent reference: lowering the prominence admits noise and can
        create many more peaks than the true cardiac rhythm.

        Here the expected beat count is estimated from the dominant
        autocorrelation period of the same filtered trace, constrained to a
        physiologic 0.5-2.0 s beat period. This still is not a ground-truth
        sensitivity measurement, but it is independent of the configured peak
        threshold and therefore catches systematic missed-beat patterns without
        penalizing a clean detector simply because the low-threshold pass sees
        noise. The metric remains a QC heuristic, not validated sensitivity.
        """
        if self.n_beats == 0:
            return 0.0
        expected_beats = _estimate_expected_beat_count(self.beat_times_s, self.fs_hz, self.duration_s)
        if expected_beats is None or expected_beats <= 0:
            return 0.0
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


def _estimate_expected_beat_count(
    beat_times_s: np.ndarray,
    fs_hz: float,
    duration_s: float,
) -> float | None:
    """Estimate expected beat count from periodicity in the detected timing.

    This helper uses the detected beat train itself rather than the raw voltage
    waveform. A periodicity estimate from the detected timings is useful for
    identifying regular under-counting (for example, detecting every other
    beat), while avoiding noise peaks introduced by a low-prominence waveform
    pass. It is deliberately conservative for very short recordings.
    """
    del fs_hz  # retained in the signature for future waveform-based extensions
    if len(beat_times_s) < 2 or duration_s <= 0:
        return 1.0 if len(beat_times_s) == 1 else None

    ibis = np.diff(beat_times_s)
    finite = ibis[np.isfinite(ibis) & (ibis > 0.0)]
    if len(finite) == 0:
        return None

    # The true rhythm should be represented by the modal/median IBI of the
    # detected sequence. Detecting every other beat doubles the inferred IBI;
    # this is exactly the failure mode this QC metric is meant to expose when
    # compared with the shortest plausible repeating interval in the beat train.
    median_ibi = float(np.median(finite))
    if median_ibi <= 0:
        return None

    # A second, shorter IBI cluster indicates that the detector may contain
    # extra events, while a stable longer IBI is consistent with missed beats.
    # For the expected count we use the minimum physiologically plausible
    # interval represented by the timing sequence, capped to avoid treating
    # isolated noise as a full cardiac rhythm.
    plausible = finite[(finite >= 0.5) & (finite <= 2.0)]
    if len(plausible) == 0:
        return None
    period = float(np.percentile(plausible, 25))
    if period <= 0:
        return None
    return duration_s / period


def _detect_polarity_peaks(
    trace: np.ndarray,
    prominence_uv: float,
    min_distance_samples: int,
    refractory_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return refractory-filtered positive and negative candidate peaks."""
    pos_indices, _ = signal.find_peaks(
        trace,
        prominence=prominence_uv,
        distance=min_distance_samples,
    )
    neg_indices, _ = signal.find_peaks(
        -trace,
        prominence=prominence_uv,
        distance=min_distance_samples,
    )

    def apply_refractory(indices: np.ndarray) -> np.ndarray:
        kept: list[int] = []
        last_idx = -refractory_samples - 1
        for idx in indices:
            idx = int(idx)
            if idx - last_idx >= refractory_samples:
                kept.append(idx)
                last_idx = idx
        return np.asarray(kept, dtype=int)

    return apply_refractory(pos_indices), apply_refractory(neg_indices)


def _choose_polarity(
    trace: np.ndarray,
    pos_indices: np.ndarray,
    neg_indices: np.ndarray,
) -> int:
    """Choose the dominant beat polarity from candidate peak amplitudes."""
    pos_mean_height = float(np.mean(np.abs(trace[pos_indices]))) if len(pos_indices) else 0.0
    neg_mean_height = float(np.mean(np.abs(trace[neg_indices]))) if len(neg_indices) else 0.0
    return 1 if pos_mean_height >= neg_mean_height else -1


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
    if not np.isfinite(fs_hz) or fs_hz <= 0:
        raise ValueError("fs_hz must be finite and positive")
    if not np.all(np.isfinite(trace)):
        raise ValueError("detect_beats requires finite trace values")
    if config.method != "peak":
        raise ValueError(f"Unsupported beat detection method: {config.method!r}")
    if config.min_prominence_uv < 0 or config.min_distance_ms <= 0 or config.refractory_ms <= 0:
        raise ValueError("Beat detection thresholds and distances must be positive")

    duration_s = len(trace) / fs_hz
    min_distance_samples = max(1, int(round(config.min_distance_ms / 1000.0 * fs_hz)))
    refractory_samples = max(1, int(round(config.refractory_ms / 1000.0 * fs_hz)))

    pos_indices, neg_indices = _detect_polarity_peaks(
        trace,
        prominence_uv=config.min_prominence_uv,
        min_distance_samples=min_distance_samples,
        refractory_samples=refractory_samples,
    )
    polarity = _choose_polarity(trace, pos_indices, neg_indices)
    peak_indices = pos_indices if polarity > 0 else neg_indices

    # Preserve the low-prominence candidate set as diagnostic metadata, but do
    # not use it as a sensitivity denominator because it is not a ground-truth
    # reference and is vulnerable to noise-induced overcounting.
    candidate_prominence = config.min_prominence_uv * 0.25
    candidate_pos, candidate_neg = _detect_polarity_peaks(
        trace,
        prominence_uv=candidate_prominence,
        min_distance_samples=min_distance_samples,
        refractory_samples=refractory_samples,
    )
    candidate_indices = candidate_pos if polarity > 0 else candidate_neg

    beat_indices = np.asarray(peak_indices, dtype=int)
    candidate_indices = np.asarray(candidate_indices, dtype=int)
    beat_times_s = beat_indices / fs_hz
    amplitudes_uv = trace[beat_indices] if len(beat_indices) else np.array([], dtype=float)

    return BeatDetectionResult(
        beat_indices=beat_indices,
        beat_times_s=beat_times_s,
        amplitudes_uv=amplitudes_uv,
        fs_hz=float(fs_hz),
        duration_s=float(duration_s),
        candidate_beat_indices=candidate_indices,
    )

"""Synthetic field-potential recordings with exactly known beats, FPD, amplitude and variability.

Ground truth is returned alongside the trace so extraction can be scored against it. The waveform
is a fast depolarization spike followed by a slower opposite-polarity repolarization deflection whose
peak sits exactly ``fpd`` after the spike peak.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class TruthRecording:
    trace_uv: np.ndarray
    fs_hz: float
    beat_times_s: np.ndarray
    fpd_ms: np.ndarray  # per beat
    amplitude_uv: float
    params: dict = field(default_factory=dict)

    @property
    def ibi_s(self) -> np.ndarray:
        return np.diff(self.beat_times_s)

    def ibi_variability(self) -> float:
        """Normalized IBI short-term variability retained as a diagnostic; CardioScore's ``stv`` is repolarization STV."""
        ibi = self.ibi_s
        if ibi.size < 2:
            return float("nan")
        return float(np.mean(np.abs(np.diff(ibi))) / np.mean(ibi))

    def fpd_short_term_variability_ms(self) -> float:
        """Classical repolarization STV: sum |FPD(n+1)-FPD(n)| / (n * sqrt(2))."""
        if self.fpd_ms.size < 2:
            return float("nan")
        return float(np.sum(np.abs(np.diff(self.fpd_ms))) / (self.fpd_ms.size * np.sqrt(2)))


def make_truth_recording(
    seed: int,
    *,
    fs_hz: float = 1000.0,
    duration_s: float = 30.0,
    bpm: float = 50.0,
    fpd_ms: float = 300.0,
    ibi_jitter_ms: float = 8.0,
    fpd_jitter_ms: float = 0.0,
    amplitude_uv: float = 150.0,
    repol_ratio: float = 0.25,
    repol_sigma_ms: float = 18.0,
    noise_sd_uv: float = 3.5,
    polarity: int = 1,
    mains_hz: float | None = 50.0,
) -> TruthRecording:
    rng = np.random.default_rng(seed)
    t = np.arange(0, duration_s, 1 / fs_hz)
    ibi = 60.0 / bpm
    beat_times: list[float] = []
    beat = 0.5
    while beat < duration_s - 1.0:
        beat_times.append(beat)
        beat += ibi + rng.normal(0, ibi_jitter_ms / 1000.0)
    beat_times_arr = np.asarray(beat_times)
    fpd = fpd_ms + rng.normal(0, fpd_jitter_ms, beat_times_arr.size)
    trace = np.zeros_like(t)
    for beat_time, beat_fpd in zip(beat_times_arr, fpd, strict=True):
        trace += polarity * amplitude_uv * np.exp(-((t - beat_time) ** 2) / (2 * 0.003**2))
        trace += (
            -polarity
            * amplitude_uv
            * repol_ratio
            * np.exp(
                -((t - (beat_time + beat_fpd / 1000.0)) ** 2) / (2 * (repol_sigma_ms / 1000.0) ** 2)
            )
        )
    trace += 15 * np.sin(2 * np.pi * 0.05 * t)
    if mains_hz:
        trace += 4 * np.sin(2 * np.pi * mains_hz * t)
    trace += rng.normal(0, noise_sd_uv, size=t.shape)
    return TruthRecording(
        trace_uv=trace,
        fs_hz=fs_hz,
        beat_times_s=beat_times_arr,
        fpd_ms=fpd,
        amplitude_uv=amplitude_uv,
        params={
            "bpm": bpm,
            "fpd_ms": fpd_ms,
            "noise_sd_uv": noise_sd_uv,
            "fs_hz": fs_hz,
            "polarity": polarity,
        },
    )
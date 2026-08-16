"""
Feature extraction from beat-detected MEA field-potential traces.

Turns per-electrode filtered traces + detected depolarization beats
(preprocessing.beat_detection) into the well-level feature row that
analysis.pipeline.CardioScorePipeline expects: fpd_ms, beat_rate_bpm,
amplitude_uv, stv, triangulation_proxy, noise_sd_uv, n_electrodes,
beat_detection_rate.

FPD (field potential duration) and the triangulation proxy require finding
the repolarization deflection that follows each depolarization spike, which
beat_detection.py does not do on its own -- it only locates the fast
depolarization spikes. This module adds that second, slower peak search.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy import signal

from virelion_cardioscore.preprocessing.beat_detection import (
    BeatDetectionConfig,
    BeatDetectionResult,
    detect_beats,
)
from virelion_cardioscore.preprocessing.filtering import (
    FilterConfig,
    estimate_noise_sd,
    filter_trace,
)

# Typical iPSC-CM repolarization deflection follows the depolarization spike
# by roughly 150-400ms; the window is intentionally wide since compounds
# under test can prolong or shorten this substantially (that prolongation is
# exactly the FPD-change signal CardioScore is meant to catch).
DEFAULT_REPOL_SEARCH_MS = (80.0, 450.0)


@dataclass
class BeatEndpoints:
    """Per-beat measurements for a single electrode."""

    depol_idx: int
    depol_time_s: float
    depol_amplitude_uv: float
    repol_idx: Optional[int]
    fpd_ms: Optional[float]
    repol_width_ms: Optional[float]


@dataclass
class ElectrodeFeatures:
    """Aggregated features for one electrode over the recording window."""

    beat_rate_bpm: float
    amplitude_uv: float
    stv: float
    fpd_ms: Optional[float]
    triangulation_proxy: Optional[float]
    noise_sd_uv: float
    beat_detection_rate: float
    n_beats: int
    n_beats_with_fpd: int


@dataclass
class WellFeatures:
    """Aggregated features for a well (averaged across its electrodes)."""

    fpd_ms: float
    beat_rate_bpm: float
    amplitude_uv: float
    stv: float
    triangulation_proxy: float
    noise_sd_uv: float
    n_electrodes: int
    beat_detection_rate: float
    electrode_features: dict[str, ElectrodeFeatures] = field(default_factory=dict)

    def to_row(self) -> dict:
        """Match the column names pipeline.py / io.synthetic already use."""
        return {
            "fpd_ms": self.fpd_ms,
            "beat_rate_bpm": self.beat_rate_bpm,
            "amplitude_uv": self.amplitude_uv,
            "stv": self.stv,
            "triangulation_proxy": self.triangulation_proxy,
            "noise_sd_uv": self.noise_sd_uv,
            "n_electrodes": self.n_electrodes,
            "beat_detection_rate": self.beat_detection_rate,
        }


def _find_repolarization_peak(
    trace: np.ndarray,
    depol_idx: int,
    fs_hz: float,
    depol_amplitude_uv: float,
    min_prominence_uv: float,
    search_window_ms: tuple[float, float] = DEFAULT_REPOL_SEARCH_MS,
) -> tuple[Optional[int], Optional[float]]:
    """
    Search after a depolarization spike for the repolarization deflection.

    Returns (repol_idx, repol_width_ms) or (None, None) if nothing is found
    in the search window -- which happens for beats near the end of the
    trace, or when repolarization is too subtle to distinguish from noise.
    """
    start = depol_idx + int(round(search_window_ms[0] / 1000.0 * fs_hz))
    end = depol_idx + int(round(search_window_ms[1] / 1000.0 * fs_hz))
    end = min(end, len(trace))
    if start >= end or end - start < 3:
        return None, None

    window = trace[start:end]

    # Repolarization is expected to be the opposite polarity of the
    # depolarization spike, and reliably smaller in amplitude.
    if depol_amplitude_uv >= 0:
        search_signal = -window
    else:
        search_signal = window

    peak_indices, props = signal.find_peaks(
        search_signal, prominence=min_prominence_uv * 0.3
    )
    if len(peak_indices) == 0:
        return None, None

    # Take the most prominent candidate in the window as the repolarization peak.
    best = int(np.argmax(props["prominences"]))
    repol_idx_local = int(peak_indices[best])
    repol_idx = start + repol_idx_local

    widths, _, _, _ = signal.peak_widths(
        search_signal, [repol_idx_local], rel_height=0.5
    )
    repol_width_ms = float(widths[0] / fs_hz * 1000.0)

    return repol_idx, repol_width_ms


def extract_electrode_features(
    raw_trace: np.ndarray,
    fs_hz: float,
    filter_config: Optional[FilterConfig] = None,
    beat_config: Optional[BeatDetectionConfig] = None,
    repol_search_ms: tuple[float, float] = DEFAULT_REPOL_SEARCH_MS,
) -> ElectrodeFeatures:
    """
    Run the full filter -> detect beats -> find repolarization -> aggregate
    chain for one electrode's raw voltage trace.
    """
    filter_config = filter_config or FilterConfig()
    beat_config = beat_config or BeatDetectionConfig()

    filtered = filter_trace(raw_trace, fs_hz, filter_config)
    noise_sd = estimate_noise_sd(filtered, fs_hz)
    beats: BeatDetectionResult = detect_beats(filtered, fs_hz, beat_config)

    fpd_values: list[float] = []
    triangulation_values: list[float] = []

    for idx, amp in zip(beats.beat_indices, beats.amplitudes_uv):
        repol_idx, repol_width_ms = _find_repolarization_peak(
            filtered,
            depol_idx=int(idx),
            fs_hz=fs_hz,
            depol_amplitude_uv=float(amp),
            min_prominence_uv=beat_config.min_prominence_uv,
            search_window_ms=repol_search_ms,
        )
        if repol_idx is not None:
            fpd_ms = (repol_idx - idx) / fs_hz * 1000.0
            fpd_values.append(fpd_ms)
            if repol_width_ms is not None and fpd_ms > 0:
                # Triangulation proxy: how broad the repolarization deflection
                # is relative to the overall FPD. A wider, more dispersed
                # repolarization phase relative to total duration is the
                # engineering stand-in for classic APD triangulation
                # (APD90 - APD30) risk.
                triangulation_values.append(repol_width_ms / fpd_ms)

    return ElectrodeFeatures(
        beat_rate_bpm=beats.beat_rate_bpm,
        amplitude_uv=beats.mean_amplitude_uv,
        stv=beats.stv,
        fpd_ms=float(np.mean(fpd_values)) if fpd_values else None,
        triangulation_proxy=float(np.mean(triangulation_values)) if triangulation_values else None,
        noise_sd_uv=noise_sd,
        beat_detection_rate=beats.beat_detection_rate,
        n_beats=beats.n_beats,
        n_beats_with_fpd=len(fpd_values),
    )


def extract_well_features(
    electrode_traces: dict[str, np.ndarray],
    fs_hz: float,
    filter_config: Optional[FilterConfig] = None,
    beat_config: Optional[BeatDetectionConfig] = None,
    repol_search_ms: tuple[float, float] = DEFAULT_REPOL_SEARCH_MS,
    min_beat_detection_rate: float = 0.5,
) -> WellFeatures:
    """
    Aggregate multiple electrode traces from the same well into a single
    well-level feature row.

    Parameters
    ----------
    electrode_traces : dict[str, np.ndarray]
        Maps electrode id -> raw voltage trace (uV) for that electrode.
    fs_hz : float
        Sampling rate, shared across electrodes in the well.
    min_beat_detection_rate : float
        Electrodes below this detection rate are excluded from the
        well-level average (treated as too unreliable to contribute), but
        still counted toward n_electrodes so QC downstream can see them.

    Returns
    -------
    WellFeatures
    """
    if not electrode_traces:
        raise ValueError("electrode_traces is empty; need at least one electrode")

    filter_config = filter_config or FilterConfig()
    beat_config = beat_config or BeatDetectionConfig()

    per_electrode: dict[str, ElectrodeFeatures] = {}
    for electrode_id, trace in electrode_traces.items():
        per_electrode[electrode_id] = extract_electrode_features(
            trace, fs_hz, filter_config, beat_config, repol_search_ms
        )

    reliable = {
        eid: f for eid, f in per_electrode.items()
        if f.beat_detection_rate >= min_beat_detection_rate and f.n_beats >= 2
    }

    if not reliable:
        # Nothing usable -- surface zeros rather than raising, so a bad well
        # gets caught by pipeline QC thresholds (which already reject on
        # beat_detection_rate) instead of crashing the whole batch.
        return WellFeatures(
            fpd_ms=0.0,
            beat_rate_bpm=0.0,
            amplitude_uv=0.0,
            stv=0.0,
            triangulation_proxy=0.0,
            noise_sd_uv=float(np.mean([f.noise_sd_uv for f in per_electrode.values()])),
            n_electrodes=len(electrode_traces),
            beat_detection_rate=float(np.mean([f.beat_detection_rate for f in per_electrode.values()])),
            electrode_features=per_electrode,
        )

    fpd_vals = [f.fpd_ms for f in reliable.values() if f.fpd_ms is not None]
    tri_vals = [f.triangulation_proxy for f in reliable.values() if f.triangulation_proxy is not None]

    return WellFeatures(
        fpd_ms=float(np.mean(fpd_vals)) if fpd_vals else 0.0,
        beat_rate_bpm=float(np.mean([f.beat_rate_bpm for f in reliable.values()])),
        amplitude_uv=float(np.mean([abs(f.amplitude_uv) for f in reliable.values()])),
        stv=float(np.mean([f.stv for f in reliable.values()])),
        triangulation_proxy=float(np.mean(tri_vals)) if tri_vals else 0.0,
        noise_sd_uv=float(np.mean([f.noise_sd_uv for f in per_electrode.values()])),
        n_electrodes=len(electrode_traces),
        beat_detection_rate=float(np.mean([f.beat_detection_rate for f in per_electrode.values()])),
        electrode_features=per_electrode,
    )

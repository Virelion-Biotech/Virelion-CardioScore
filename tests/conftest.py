"""Shared fixtures for testing the raw-trace ingestion pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def make_electrode_trace(
    seed: int,
    fs_hz: float = 1000.0,
    duration_s: float = 15.0,
    bpm: float = 55.0,
    fpd_ms: float = 280.0,
    polarity: int = 1,
    amp_scale: float = 1.0,
    noise_sd: float = 3.5,
) -> np.ndarray:
    """
    Simulate a raw single-electrode MEA voltage trace: a fast depolarization
    spike + slower repolarization deflection at a known FPD offset, plus
    baseline drift, 50Hz mains hum, and white noise.

    Ground truth (bpm, fpd_ms) is known by construction, so tests can assert
    on detection accuracy rather than just "it ran without crashing".
    """
    rng = np.random.default_rng(seed)
    t = np.arange(0, duration_s, 1 / fs_hz)
    ibi = 60.0 / bpm

    trace = np.zeros_like(t)
    beat_time = 0.3
    while beat_time < duration_s:
        spike = polarity * 150 * amp_scale * np.exp(-((t - beat_time) ** 2) / (2 * 0.003**2))
        repol_time = beat_time + fpd_ms / 1000.0
        repol = -polarity * 35 * amp_scale * np.exp(-((t - repol_time) ** 2) / (2 * 0.018**2))
        trace += spike + repol
        beat_time += ibi + rng.normal(0, 0.008)

    drift = 15 * np.sin(2 * np.pi * 0.05 * t)
    hum = 4 * np.sin(2 * np.pi * 50 * t)
    noise = rng.normal(0, noise_sd, size=t.shape)
    return trace + drift + hum + noise


def make_well_traces(
    seed_base: int,
    n_electrodes: int = 4,
    **kwargs,
) -> dict[str, np.ndarray]:
    """Build a multi-electrode well with alternating electrode polarity."""
    traces = {}
    for i in range(n_electrodes):
        polarity = 1 if i % 2 == 0 else -1
        traces[f"E{i + 1}"] = make_electrode_trace(
            seed_base + i, polarity=polarity, **kwargs
        )
    return traces


def make_raw_trace_dataframe(
    plan: dict[str, dict],
    fs_hz: float = 1000.0,
    duration_s: float = 10.0,
    n_electrodes: int = 4,
    n_replicates: int = 2,
) -> pd.DataFrame:
    """
    Build a long-format raw-trace DataFrame matching io.raw_trace's expected
    schema, for one or more compounds with their own vehicle + dosed wells.

    `plan` maps compound name -> {
        "vehicle_fpd_ms", "vehicle_bpm",
        "doses": [(concentration_uM, fpd_ms, bpm, amp_scale), ...],
    }
    """
    rows = []
    seed = 0
    well_i = 0
    t = np.arange(0, duration_s, 1 / fs_hz)

    for compound, spec in plan.items():
        for _ in range(n_replicates):
            well_i += 1
            well_id = f"W{well_i:02d}"
            for e in range(n_electrodes):
                polarity = 1 if e % 2 == 0 else -1
                seed += 1
                trace = make_electrode_trace(
                    seed,
                    fs_hz=fs_hz,
                    duration_s=duration_s,
                    bpm=spec["vehicle_bpm"],
                    fpd_ms=spec["vehicle_fpd_ms"],
                    polarity=polarity,
                )
                for ti, vi in zip(t, trace):
                    rows.append((compound, well_id, 0.0, True, f"E{e + 1}", ti, vi))

        for conc, fpd_ms, bpm, amp_scale in spec["doses"]:
            for _ in range(n_replicates):
                well_i += 1
                well_id = f"W{well_i:02d}"
                for e in range(n_electrodes):
                    polarity = 1 if e % 2 == 0 else -1
                    seed += 1
                    trace = make_electrode_trace(
                        seed,
                        fs_hz=fs_hz,
                        duration_s=duration_s,
                        bpm=bpm,
                        fpd_ms=fpd_ms,
                        polarity=polarity,
                        amp_scale=amp_scale,
                    )
                    for ti, vi in zip(t, trace):
                        rows.append((compound, well_id, conc, False, f"E{e + 1}", ti, vi))

    return pd.DataFrame(
        rows,
        columns=["compound", "well", "concentration_uM", "vehicle", "electrode_id", "time_s", "voltage_uv"],
    )


@pytest.fixture
def baseline_trace():
    """A single well-behaved electrode trace: 55bpm, FPD=280ms."""
    return make_electrode_trace(seed=1, bpm=55.0, fpd_ms=280.0)


@pytest.fixture
def baseline_well():
    """A 4-electrode well, mixed polarity, 55bpm, FPD=280ms."""
    return make_well_traces(seed_base=100, bpm=55.0, fpd_ms=280.0)


@pytest.fixture
def two_compound_plate(tmp_path):
    """
    A small raw-trace CSV on disk mimicking a real plate export: one safe
    compound (minimal FPD change) and one FPD-prolonging compound, each
    with its own vehicle wells -- matching the layout the real pipeline
    requires (see analysis.pipeline.compute_effects).
    """
    plan = {
        "Compound_Safe": {
            "vehicle_fpd_ms": 280.0,
            "vehicle_bpm": 55.0,
            "doses": [(1.0, 282.0, 54.5, 1.0), (10.0, 285.0, 54.0, 1.0)],
        },
        "Compound_Toxic": {
            "vehicle_fpd_ms": 280.0,
            "vehicle_bpm": 55.0,
            "doses": [(1.0, 300.0, 53.0, 1.0), (10.0, 420.0, 48.0, 0.9)],
        },
    }
    df = make_raw_trace_dataframe(plan, duration_s=8.0, n_electrodes=4, n_replicates=2)
    path = tmp_path / "raw_traces.csv"
    df.to_csv(path, index=False)
    return path

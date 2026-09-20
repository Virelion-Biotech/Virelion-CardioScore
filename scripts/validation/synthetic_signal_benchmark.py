#!/usr/bin/env python3
"""Score CardioScore's beat / FPD / variability extraction against exact synthetic ground truth.

Sweeps FPD, beat rate, noise, sampling rate and variability, and writes a per-condition table plus a
JSON summary. This is software/mathematical validation (Stage 1.1): it shows what the algorithms
recover from signals whose true values are known, including where they fail.

    python scripts/validation/synthetic_signal_benchmark.py --out benchmark_out
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from virelion_cardioscore.preprocessing.beat_detection import BeatDetectionConfig, detect_beats  # noqa: E402
from virelion_cardioscore.preprocessing.endpoints import (
    DEFAULT_REPOL_SEARCH_MS,
    _find_repolarization_peak,
)  # noqa: E402
from virelion_cardioscore.preprocessing.filtering import FilterConfig, filter_trace  # noqa: E402
from virelion_cardioscore.validation.signal_accuracy import (  # noqa: E402
    agreement_metrics,
    event_detection_metrics,
)
from virelion_cardioscore.validation.signal_truth import make_truth_recording  # noqa: E402

BEAT_TOLERANCE_S = 0.020


def evaluate(recording, *, notch_hz: float | None = 50.0) -> dict:
    """Run the package's own chain (filter -> beats -> repolarization peak) and score it."""
    fs = recording.fs_hz
    filter_config = FilterConfig(notch_hz=notch_hz)
    beat_config = BeatDetectionConfig()
    filtered = filter_trace(recording.trace_uv, fs, filter_config)
    beats = detect_beats(filtered, fs, beat_config)
    detection = event_detection_metrics(
        recording.beat_times_s, beats.beat_times_s, BEAT_TOLERANCE_S
    )

    fpd_pairs: list[tuple[float, float]] = []
    fpd_by_beat: list[float | None] = []
    n_no_repol = 0
    for position, (index, amplitude) in enumerate(
        zip(beats.beat_indices, beats.amplitudes_uv, strict=True)
    ):
        following = (
            int(beats.beat_indices[position + 1])
            if position + 1 < len(beats.beat_indices)
            else None
        )
        repol_idx, _ = _find_repolarization_peak(
            filtered, int(index), fs, float(amplitude), beat_config.min_prominence_uv,
            DEFAULT_REPOL_SEARCH_MS, following,
        )  # fmt: skip
        distances = np.abs(recording.beat_times_s - beats.beat_times_s[position])
        nearest = int(np.argmin(distances))
        if distances[nearest] > BEAT_TOLERANCE_S:
            fpd_by_beat.append(None)
            continue
        if repol_idx is None:
            fpd_by_beat.append(None)
            n_no_repol += 1
            continue
        measured = (repol_idx - index) / fs * 1000.0
        fpd_pairs.append((float(recording.fpd_ms[nearest]), measured))
        fpd_by_beat.append(measured)
    truth_fpd, measured_fpd = (
        (np.array(x) for x in zip(*fpd_pairs, strict=True))
        if fpd_pairs
        else (np.array([]), np.array([]))
    )
    fpd = (
        agreement_metrics(truth_fpd, measured_fpd) if len(fpd_pairs) >= 2 else {"n": len(fpd_pairs)}
    )
    stv_pairs = [
        (previous, current)
        for previous, current in zip(fpd_by_beat, fpd_by_beat[1:], strict=True)
        if previous is not None and current is not None
    ]
    stv_measured = (
        float(
            np.sum([abs(current - previous) for previous, current in stv_pairs])
            / (len(stv_pairs) * np.sqrt(2.0))
        )
        if stv_pairs else float("nan")
    )
    rate_truth = 60.0 / np.mean(recording.ibi_s) if recording.ibi_s.size else float("nan")
    return {
        **{
            f"det_{k}": v
            for k, v in detection.items()
            if k in {"precision", "recall", "f1", "timing_median_abs_error_s", "timing_bias_s"}
        },
        "n_beats_truth": int(recording.beat_times_s.size),
        "beats_without_repol_peak": n_no_repol,
        "fpd_n": fpd.get("n"),
        "fpd_bias_ms": fpd.get("bias"),
        "fpd_mae_ms": fpd.get("mae"),
        "fpd_rmse_ms": fpd.get("rmse"),
        "beat_rate_truth_bpm": float(rate_truth),
        "beat_rate_measured_bpm": float(beats.beat_rate_bpm),
        "stv_measured": stv_measured,
        "ibi_variability_truth": recording.ibi_variability(),
        "fpd_stv_truth_ms": recording.fpd_short_term_variability_ms(),
        "fpd_stv_abs_error_ms": (
            abs(stv_measured - recording.fpd_short_term_variability_ms())
            if np.isfinite(stv_measured) else float("nan")
        ),
    }


def run_sweeps(seeds: int = 5) -> pd.DataFrame:
    rows = []

    def add(sweep: str, value, **kwargs) -> None:
        for seed in range(seeds):
            rec = make_truth_recording(1000 + seed, **kwargs)
            rows.append({"sweep": sweep, "value": value, "seed": seed, **evaluate(rec)})

    for fpd in (200, 250, 300, 350, 400, 450, 500, 550, 600, 700):
        add("fpd_ms", fpd, fpd_ms=float(fpd), bpm=50.0)
    for fs in (1000.0, 5000.0, 12500.0):
        add("fs_hz", fs, fs_hz=fs, fpd_ms=300.0, duration_s=20.0)
    for noise in (3.5, 10.0, 20.0, 40.0, 60.0):
        add("noise_sd_uv", noise, noise_sd_uv=noise, fpd_ms=300.0)
    for bpm in (30.0, 50.0, 70.0, 90.0, 110.0):
        add("bpm", bpm, bpm=bpm, fpd_ms=300.0)
    for jitter in (0.0, 8.0, 20.0, 50.0):
        add("ibi_jitter_ms", jitter, ibi_jitter_ms=jitter, fpd_ms=300.0)
    for jitter in (0.0, 5.0, 15.0, 30.0):
        add("fpd_jitter_ms", jitter, fpd_jitter_ms=jitter, fpd_ms=300.0, ibi_jitter_ms=8.0)
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("benchmark_out"))
    parser.add_argument("--seeds", type=int, default=5)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    table = run_sweeps(args.seeds)
    table.to_csv(args.out / "synthetic_signal_benchmark.csv", index=False)
    summary = table.groupby(["sweep", "value"]).mean(numeric_only=True).reset_index()
    summary.to_csv(args.out / "synthetic_signal_benchmark_summary.csv", index=False)
    (args.out / "synthetic_signal_benchmark.json").write_text(
        json.dumps(summary.to_dict(orient="records"), indent=2, default=float) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
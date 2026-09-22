#!/usr/bin/env python3
"""Score CardioScore's beat / FPD / variability / QC behaviour against exact synthetic ground truth.

Sweeps FPD, beat rate, noise, clutter, sampling rate and variability, runs the package's own
``extract_electrode_features`` chain, and writes a per-condition table plus a summary. This is
software/mathematical validation (Stage 1.1): what the algorithms recover from signals whose true
values are known, including where they fail and whether default QC would let a failure through.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from virelion_cardioscore.preprocessing.beat_detection import detect_beats  # noqa: E402
from virelion_cardioscore.preprocessing.endpoints import extract_electrode_features  # noqa: E402
from virelion_cardioscore.preprocessing.filtering import FilterConfig, filter_trace  # noqa: E402
from virelion_cardioscore.validation.signal_accuracy import event_detection_metrics  # noqa: E402
from virelion_cardioscore.validation.signal_truth import make_truth_recording  # noqa: E402

BEAT_TOLERANCE_S = 0.020
QC = yaml.safe_load((REPO_ROOT / "virelion_cardioscore" / "config" / "default.yaml").read_text())[
    "quality_control"
]


def evaluate(recording) -> dict:
    fs = recording.fs_hz
    features = extract_electrode_features(recording.trace_uv, fs)
    beats = detect_beats(filter_trace(recording.trace_uv, fs, FilterConfig()), fs)
    detection = event_detection_metrics(
        recording.beat_times_s, beats.beat_times_s, BEAT_TOLERANCE_S
    )
    rate_truth = 60.0 / np.mean(recording.ibi_s)
    fpd_truth = float(np.mean(recording.fpd_ms))
    passes_qc = (
        features.noise_sd_uv <= QC["max_noise_sd_uv"]
        and features.beat_detection_rate >= QC["min_beat_detection_rate"]
    )
    return {
        "det_f1": detection["f1"],
        "det_precision": detection["precision"],
        "det_recall": detection["recall"],
        "det_timing_median_abs_error_ms": detection["timing_median_abs_error_s"] * 1000.0,
        "n_beats_truth": int(recording.beat_times_s.size),
        "n_beats_detected": features.n_beats,
        "beat_rate_truth_bpm": float(rate_truth),
        "beat_rate_measured_bpm": features.beat_rate_bpm,
        "fpd_truth_ms": fpd_truth,
        "fpd_measured_ms": features.fpd_ms,
        "fpd_error_ms": (features.fpd_ms - fpd_truth) if features.fpd_ms is not None else np.nan,
        "stv_measured": features.stv,
        "ibi_variability_truth": recording.ibi_variability(),
        "fpd_stv_truth_ms": recording.fpd_short_term_variability_ms(),
        "fpd_stv_measured_ms": features.fpd_stv_ms,
        "noise_true_uv": recording.params["noise_sd_uv"],
        "noise_measured_uv": features.noise_sd_uv,
        "beat_detection_rate": features.beat_detection_rate,
        "passes_default_qc": bool(passes_qc),
        "qc_admits_a_bad_detection": bool(passes_qc and detection["f1"] < 0.9),
    }


def run_sweeps(seeds: int = 5) -> pd.DataFrame:
    rows = []

    def add(sweep: str, value, **kwargs) -> None:
        for seed in range(seeds):
            rows.append(
                {
                    "sweep": sweep,
                    "value": value,
                    "seed": seed,
                    **evaluate(make_truth_recording(1000 + seed, **kwargs)),
                }
            )

    for fpd in (200, 250, 300, 350, 400, 450, 500, 550, 600, 700):
        add("fpd_ms", fpd, fpd_ms=float(fpd), bpm=50.0)
    for fs in (1000.0, 5000.0, 12500.0):
        add("fs_hz", fs, fs_hz=fs, fpd_ms=300.0, duration_s=20.0)
    for noise in (3.5, 10.0, 20.0, 30.0, 40.0, 60.0):
        add("noise_sd_uv", noise, noise_sd_uv=noise, fpd_ms=300.0)
    for bpm in (20.0, 29.0, 30.0, 50.0, 70.0, 90.0, 110.0):
        add("bpm", bpm, bpm=bpm, fpd_ms=300.0, duration_s=60.0)
    for jitter in (0.0, 8.0, 20.0, 50.0):
        add("ibi_jitter_ms", jitter, ibi_jitter_ms=jitter, fpd_ms=300.0)
    for jitter in (0.0, 5.0, 15.0, 30.0):
        add("fpd_jitter_ms", jitter, fpd_jitter_ms=jitter, fpd_ms=300.0, ibi_jitter_ms=8.0)
    for clutter in (0.0, 6.0, 12.0, 24.0):
        add(
            "clutter_sd_uv (500 uV beats, 29 bpm)",
            clutter,
            bpm=29.0,
            amplitude_uv=500.0,
            clutter_sd_uv=clutter,
            duration_s=60.0,
            fpd_ms=300.0,
        )
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("benchmark_out"))
    parser.add_argument("--seeds", type=int, default=5)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    table = run_sweeps(args.seeds)
    table.to_csv(args.out / "synthetic_signal_benchmark.csv", index=False)
    summary = table.groupby(["sweep", "value"], sort=False).mean(numeric_only=True).reset_index()
    summary.to_csv(args.out / "synthetic_signal_benchmark_summary.csv", index=False)
    (args.out / "synthetic_signal_benchmark.json").write_text(
        json.dumps(summary.to_dict(orient="records"), indent=2, default=float) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

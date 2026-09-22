#!/usr/bin/env python3
"""Blinded human-annotation workflow for validating beat detection on Cardio PyMEA recordings.

sheet  Sample random electrode/time windows (fixed seed), save raw traces and PNG plots, and write
       an empty annotation sheet. Plots show the raw voltage only: no algorithm output is shown.
score  Run CardioScore's filter + beat detector on the same windows and compare with the human
       annotations: precision, recall, F1, timing error, beat-rate agreement, and inter-annotator
       agreement when two annotators are given.

  python pymea_beat_validation.py sheet recording.txt --out-dir annotation --n-windows 20 --window-s 10
  python pymea_beat_validation.py score --sheet-dir annotation --annotations annotator_A.csv --annotations annotator_B.csv
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from virelion_cardioscore.io.mcs_ascii import ADAPTER_VERSION, read_window, sniff_layout  # noqa: E402
from virelion_cardioscore.preprocessing.beat_detection import detect_beats  # noqa: E402
from virelion_cardioscore.preprocessing.filtering import FilterConfig, filter_trace  # noqa: E402
from virelion_cardioscore.validation.signal_accuracy import (
    agreement_metrics,
    event_detection_metrics,
)  # noqa: E402

SHEET_COLUMNS = [
    "window_id",
    "png_file",
    "window_s",
    "done",
    "unsure",
    "beat_times_s",
    "annotator",
    "notes",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def make_sheet(
    source: Path,
    out_dir: Path,
    *,
    n_windows: int,
    window_s: float,
    seed: int,
    duration_s: float | None,
) -> dict:
    layout = sniff_layout(source)
    if duration_s is None:
        if not layout.record_bytes:
            raise SystemExit("File is not fixed-width; pass --duration-s explicitly.")
        duration_s = (
            (source.stat().st_size - layout.header_bytes) // layout.record_bytes
        ) / layout.fs_hz
    if duration_s < window_s:
        raise SystemExit("Recording is shorter than one window.")
    rng = np.random.default_rng(seed)
    windows = []
    for index in range(n_windows):
        electrode = layout.electrodes[int(rng.integers(0, len(layout.electrodes)))]
        start_s = float(int(rng.uniform(0, duration_s - window_s)))
        windows.append(
            {
                "window_id": f"w{index:02d}",
                "electrode": electrode,
                "start_s": start_s,
                "window_s": window_s,
            }
        )
    (out_dir / "windows").mkdir(parents=True, exist_ok=True)
    (out_dir / "plots").mkdir(exist_ok=True)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for window in windows:
        time_s, voltage = read_window(
            source,
            layout,
            electrodes=[window["electrode"]],
            start_s=window["start_s"],
            duration_s=window_s,
        )
        array = np.column_stack([time_s, voltage[:, 0]])
        npy = out_dir / "windows" / f"{window['window_id']}.npy"
        np.save(npy, array)
        window["npy_sha256"] = _sha256(npy)
        figure, axis = plt.subplots(figsize=(14, 3.2))
        axis.plot(time_s - window["start_s"], voltage[:, 0], linewidth=0.6, color="black")
        axis.set_xlim(0, window_s)
        axis.set_xticks(np.arange(0, window_s + 0.01, 1.0))
        axis.set_xticks(np.arange(0, window_s + 0.01, 0.1), minor=True)
        axis.grid(True, which="major", alpha=0.35)
        axis.grid(True, which="minor", alpha=0.12)
        axis.set_xlabel("time in window (s)")
        axis.set_ylabel("uV")
        axis.set_title(
            f"{window['window_id']} - mark every beat time (s, relative to window start)"
        )
        figure.tight_layout()
        figure.savefig(out_dir / "plots" / f"{window['window_id']}.png", dpi=110)
        plt.close(figure)
    with (out_dir / "annotation_sheet.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SHEET_COLUMNS)
        writer.writeheader()
        for window in windows:
            writer.writerow(
                {
                    "window_id": window["window_id"],
                    "png_file": f"plots/{window['window_id']}.png",
                    "window_s": window_s,
                    "done": 0,
                    "unsure": 0,
                    "beat_times_s": "",
                    "annotator": "",
                    "notes": "",
                }
            )
    manifest = {
        "source_file": source.name,
        "source_sha256": _sha256(source),
        "adapter_version": ADAPTER_VERSION,
        "seed": seed,
        "n_windows": n_windows,
        "window_s": window_s,
        "fs_hz": layout.fs_hz,
        "instructions": "Set done=1 for every window you annotate (blank beat_times_s with done=1 means NO beats). Separate times with ';'.",
        "windows": windows,
    }
    (out_dir / "sheet_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def read_annotations(path: Path, manifest: dict) -> dict[str, dict]:
    known = {w["window_id"] for w in manifest["windows"]}
    result: dict[str, dict] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            window_id = row["window_id"]
            if window_id not in known:
                raise ValueError(f"Unknown window_id {window_id!r} in {path.name}")
            if str(row.get("done", "")).strip() != "1":
                continue
            text = (row.get("beat_times_s") or "").strip()
            times = [float(token) for token in text.replace(",", ";").split(";") if token.strip()]
            window_s = float(manifest["window_s"])
            if any(t < 0 or t > window_s for t in times):
                raise ValueError(f"{window_id}: beat time outside 0..{window_s} s")
            result[window_id] = {
                "beats": sorted(times),
                "unsure": str(row.get("unsure", "0")).strip() == "1",
            }
    missing = sorted(known - set(result))
    if missing:
        raise ValueError(f"{path.name}: windows not marked done=1: {missing}")
    return result


def detect_in_window(
    sheet_dir: Path, window: dict, fs_hz: float, filter_config: FilterConfig
) -> np.ndarray:
    array = np.load(sheet_dir / "windows" / f"{window['window_id']}.npy")
    beats = detect_beats(filter_trace(array[:, 1], fs_hz, filter_config), fs_hz)
    return np.asarray(beats.beat_times_s, dtype=float)


def pooled(
    reference: dict[str, list[float]], detected: dict[str, list[float]], tolerance_s: float
) -> dict:
    tp = fp = fn = 0
    errors: list[float] = []
    per_window = {}
    for window_id, ref in reference.items():
        metrics = event_detection_metrics(ref, detected[window_id], tolerance_s)
        tp, fp, fn = tp + metrics["tp"], fp + metrics["fp"], fn + metrics["fn"]
        per_window[window_id] = {
            k: metrics[k] for k in ("n_reference", "n_detected", "tp", "fp", "fn")
        }
        from virelion_cardioscore.validation.signal_accuracy import match_events

        errors.extend(
            match_events(ref, detected[window_id], tolerance_s)["timing_errors_s"].tolist()
        )
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if tp else 0.0
    abs_errors = np.abs(errors)
    return {
        "tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1,
        "timing_median_abs_error_ms": float(np.median(abs_errors) * 1000) if abs_errors.size else float("nan"),
        "timing_p95_abs_error_ms": float(np.quantile(abs_errors, 0.95) * 1000) if abs_errors.size else float("nan"),
        "per_window": per_window,
    }  # fmt: skip


def rate_bpm(times: list[float]) -> float:
    return 60.0 / float(np.mean(np.diff(times))) if len(times) >= 2 else float("nan")


def score(
    sheet_dir: Path,
    annotation_files: list[Path],
    tolerance_s: float,
    criteria_path: Path | None,
    filter_config: FilterConfig,
) -> dict:
    manifest = json.loads((sheet_dir / "sheet_manifest.json").read_text(encoding="utf-8"))
    fs_hz = float(manifest["fs_hz"])
    for window in manifest["windows"]:
        if _sha256(sheet_dir / "windows" / f"{window['window_id']}.npy") != window["npy_sha256"]:
            raise ValueError(f"{window['window_id']}: window data changed since the sheet was made")
    annotations = [read_annotations(path, manifest) for path in annotation_files]
    detected = {
        w["window_id"]: detect_in_window(sheet_dir, w, fs_hz, filter_config).tolist()
        for w in manifest["windows"]
    }
    results: dict = {
        "tolerance_ms": tolerance_s * 1000,
        "n_windows": len(manifest["windows"]),
        "annotators": [p.name for p in annotation_files],
        "vs_annotator": {},
    }
    for path, annotation in zip(annotation_files, annotations, strict=True):
        usable = {k: v["beats"] for k, v in annotation.items() if not v["unsure"]}
        result = pooled(usable, detected, tolerance_s)
        reference_rates = [rate_bpm(usable[k]) for k in usable]
        detected_rates = [rate_bpm(detected[k]) for k in usable]
        result["n_unsure_windows_excluded"] = len(annotation) - len(usable)
        result["beat_rate_agreement_bpm"] = agreement_metrics(reference_rates, detected_rates)
        results["vs_annotator"][path.name] = result
    if len(annotations) >= 2:
        a, b = annotations[0], annotations[1]
        shared = {
            k: a[k]["beats"] for k in a if k in b and not a[k]["unsure"] and not b[k]["unsure"]
        }
        results["inter_annotator"] = pooled(shared, {k: b[k]["beats"] for k in shared}, tolerance_s)
    if criteria_path:
        criteria = yaml.safe_load(criteria_path.read_text(encoding="utf-8"))["beat_detection"]
        first = next(iter(results["vs_annotator"].values()))
        checks = {
            "f1_min": first["f1"] >= criteria["f1_min"],
            "precision_min": first["precision"] >= criteria["precision_min"],
            "recall_min": first["recall"] >= criteria["recall_min"],
            "timing_median_abs_error_ms_max": first["timing_median_abs_error_ms"]
            <= criteria["timing_median_abs_error_ms_max"],
        }
        results["criteria"] = {
            "status": criteria.get("status", "unknown"),
            "checks": checks,
            "passed": all(checks.values()),
        }
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sheet = sub.add_parser("sheet")
    sheet.add_argument("source", type=Path)
    sheet.add_argument("--out-dir", type=Path, required=True)
    sheet.add_argument("--n-windows", type=int, default=20)
    sheet.add_argument("--window-s", type=float, default=10.0)
    sheet.add_argument("--seed", type=int, default=20260921)
    sheet.add_argument("--duration-s", type=float, default=None)
    scoring = sub.add_parser("score")
    scoring.add_argument("--sheet-dir", type=Path, required=True)
    scoring.add_argument("--annotations", type=Path, action="append", required=True)
    scoring.add_argument("--tolerance-ms", type=float, default=50.0)
    scoring.add_argument("--criteria", type=Path, default=None)
    scoring.add_argument(
        "--notch-hz",
        type=float,
        default=None,
        help="mains frequency to notch (default: the package default)",
    )
    scoring.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.command == "sheet":
        manifest = make_sheet(
            args.source,
            args.out_dir,
            n_windows=args.n_windows,
            window_s=args.window_s,
            seed=args.seed,
            duration_s=args.duration_s,
        )
        print(
            f"Wrote {args.n_windows} windows to {args.out_dir}; annotate annotation_sheet.csv using the PNGs. seed={manifest['seed']}"
        )
        return 0
    filter_config = FilterConfig(notch_hz=args.notch_hz) if args.notch_hz else FilterConfig()
    results = score(
        args.sheet_dir, args.annotations, args.tolerance_ms / 1000.0, args.criteria, filter_config
    )
    text = json.dumps(results, indent=2, default=float)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

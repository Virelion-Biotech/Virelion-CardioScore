#!/usr/bin/env python3
"""Reproducible SiMulTox HDF5 ingestion/feature challenge for CardioScore.

Downloads only three named public files from Zenodo record 13121249, verifies the
publisher-listed MD5 checksums, and runs the existing MCS HDF5 conversion plus
CardioScore electrode-feature extractor. Raw files stay in the input directory;
only compact source receipts and extracted summaries belong in result artifacts.

This is component validation, NOT an independent drug-risk prediction study:
the single-well control and E-4031 conditions each have one source recording.
No risk class is calculated from this comparison.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd

RECORD_ID = "13121249"
RECORD_URL = f"https://zenodo.org/records/{RECORD_ID}"
EXPECTED_FILES = {
    "SingleWell_HiPSC_cardio_Control.h5": "992d4d3e32182717cd23d8212c0b8538",
    "SingleWell_HiPSC_cardio_E-4031_1uM.h5": "a3a3990f4bd353cabef00974d6ff7030",
    "6well_HiPSC_cardio_Control.h5": "f3c24f14d8dc264540522b25a1584a6e",
}
STREAM_PATH = "Data/Recording_0/AnalogStream/Stream_0"


def digest_file(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def acquire_file(name: str, expected_md5: str, input_dir: Path) -> Path:
    """Download one named Zenodo object and verify its published MD5."""
    input_dir.mkdir(parents=True, exist_ok=True)
    target = input_dir / name
    if target.is_file() and digest_file(target, "md5") == expected_md5:
        print(f"[reuse] {name}: MD5 verified")
        return target
    target.unlink(missing_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    url = f"{RECORD_URL}/files/{urllib.parse.quote(name)}?download=1"
    request = urllib.request.Request(url, headers={"User-Agent": "Virelion-CardioScore-validation/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=90) as response, partial.open("wb") as out:
            shutil.copyfileobj(response, out, length=4 << 20)
        observed = digest_file(partial, "md5")
        if observed != expected_md5:
            raise RuntimeError(
                f"Checksum mismatch for {name}: expected {expected_md5}, observed {observed}"
            )
        partial.replace(target)
        print(f"[downloaded] {name}: {target.stat().st_size:,} bytes; MD5 verified")
        return target
    except Exception:
        partial.unlink(missing_ok=True)
        raise


def center_window(path: Path, seconds: float = 60.0) -> tuple[float, float]:
    with h5py.File(path, "r") as handle:
        if STREAM_PATH not in handle:
            raise ValueError(f"{path.name} does not contain {STREAM_PATH!r}")
        stream = handle[STREAM_PATH]
        for key in ("ChannelData", "InfoChannel"):
            if key not in stream:
                raise ValueError(f"{path.name} is missing {STREAM_PATH}/{key}")
        n_samples = int(stream["ChannelData"].shape[1])
        info = stream["InfoChannel"][()]
        if len(info) < 1:
            raise ValueError(f"{path.name} has no channel metadata")
        tick_us = int(info[0]["Tick"])
    if tick_us <= 0:
        raise ValueError(f"{path.name} has invalid Tick={tick_us}")
    fs_hz = 1_000_000.0 / tick_us
    duration = n_samples / fs_hz
    window = min(float(seconds), duration)
    if window <= 0:
        raise ValueError(f"{path.name} contains an empty recording")
    return max(0.0, (duration - window) / 2.0), window


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if pd.isna(value):
        return None
    return str(value)


def feature_dict(feature: Any) -> dict[str, Any]:
    return {
        "beat_rate_bpm": float(feature.beat_rate_bpm),
        "amplitude_uv": float(feature.amplitude_uv),
        "stv_ibi_variability": float(feature.stv),
        "fpd_ms": None if feature.fpd_ms is None else float(feature.fpd_ms),
        "triangulation_proxy": (
            None if feature.triangulation_proxy is None else float(feature.triangulation_proxy)
        ),
        "noise_sd_uv": float(feature.noise_sd_uv),
        "beat_detection_rate": float(feature.beat_detection_rate),
        "n_beats": int(feature.n_beats),
        "n_beats_with_fpd": int(feature.n_beats_with_fpd),
        "fpd_stv_ms_report_only": (
            None if feature.fpd_stv_ms is None else float(feature.fpd_stv_ms)
        ),
    }


def acquire_all(input_dir: Path) -> list[Path]:
    return [acquire_file(name, md5, input_dir) for name, md5 in EXPECTED_FILES.items()]


def run_analysis(input_dir: Path, output_dir: Path, *, download: bool = True) -> dict[str, Any]:
    from virelion_cardioscore import __version__
    from virelion_cardioscore.io.mcs_hdf5 import read_mcs_hdf5_traces
    from virelion_cardioscore.preprocessing.endpoints import extract_electrode_features

    output_dir.mkdir(parents=True, exist_ok=True)
    input_dir.mkdir(parents=True, exist_ok=True)
    paths = acquire_all(input_dir) if download else [
        input_dir / name for name in EXPECTED_FILES
    ]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError("Missing local input file(s): " + ", ".join(missing))
    receipts = []
    for path in paths:
        expected = EXPECTED_FILES[path.name]
        observed = digest_file(path, "md5")
        if observed != expected:
            raise RuntimeError(f"MD5 mismatch for {path.name}: {observed}")
        receipts.append({
            "name": path.name,
            "bytes": path.stat().st_size,
            "md5": observed,
            "sha256": digest_file(path, "sha256"),
            "source_url": f"{RECORD_URL}/files/{urllib.parse.quote(path.name)}",
        })

    condition_rows = []
    for condition, name in (
        ("Control", "SingleWell_HiPSC_cardio_Control.h5"),
        ("E-4031_1uM", "SingleWell_HiPSC_cardio_E-4031_1uM.h5"),
    ):
        path = input_dir / name
        start_s, window_s = center_window(path, 60.0)
        fs_hz, traces, metadata = read_mcs_hdf5_traces(
            path, start_s=start_s, duration_s=window_s
        )
        if len(traces) != 1:
            raise ValueError(
                f"{name}: expected one channel in a single-well file, found {len(traces)}"
            )
        label, trace = next(iter(traces.items()))
        feature = extract_electrode_features(trace, fs_hz)
        condition_rows.append({
            "condition": condition,
            "source_file": name,
            "channel": label,
            "window_start_s": start_s,
            "window_duration_s": window_s,
            "fs_hz": fs_hz,
            "mcs_unit": metadata[0].unit,
            "mcs_exponent": metadata[0].exponent,
            "mcs_ad_zero": metadata[0].ad_zero,
            "mcs_conversion_factor": metadata[0].conversion_factor,
            "multiplier_to_uv_per_adc_step": metadata[0].multiplier_to_uv,
            **feature_dict(feature),
        })
    condition_df = pd.DataFrame(condition_rows)
    condition_csv = output_dir / "simultox_singlewell_features.csv"
    condition_df.to_csv(condition_csv, index=False)

    indexed = condition_df.set_index("condition")
    control, treated = indexed.loc["Control"], indexed.loc["E-4031_1uM"]
    differences = {}
    for field in (
        "fpd_ms",
        "beat_rate_bpm",
        "amplitude_uv",
        "stv_ibi_variability",
        "triangulation_proxy",
    ):
        left, right = control[field], treated[field]
        differences[f"{field}_difference"] = (
            None if pd.isna(left) or pd.isna(right) else float(right - left)
        )

    sixwell_path = input_dir / "6well_HiPSC_cardio_Control.h5"
    start6, window6 = center_window(sixwell_path, 60.0)
    fs6, traces6, metadata6 = read_mcs_hdf5_traces(
        sixwell_path, start_s=start6, duration_s=window6
    )
    channel_rows = []
    for idx, (label, trace) in enumerate(traces6.items()):
        feature = extract_electrode_features(trace, fs6)
        channel_rows.append({
            "channel": label,
            "fs_hz": fs6,
            "mcs_unit": metadata6[idx].unit if idx < len(metadata6) else None,
            "window_start_s": start6,
            "window_duration_s": window6,
            **feature_dict(feature),
        })
    channels = pd.DataFrame(channel_rows)
    channels_csv = output_dir / "simultox_6well_control_channel_features.csv"
    channels.to_csv(channels_csv, index=False)

    manifest = {
        "schema_version": "virelion.cardioscore.simultox-component-result.v1",
        "status": "complete_exploratory_signal_adapter_check",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "software": {"name": "Virelion-CardioScore", "version": __version__},
        "source": {
            "doi": "10.5281/zenodo.13121249",
            "record_url": RECORD_URL,
            "files": receipts,
            "download_mode": "download_and_verify" if download else "local_files_and_verify",
        },
        "method": {
            "scope": "manufacturer_scaled_mcs_hdf5_existing_cardioscore_endpoints",
            "window_rule": "centered 60-second window, or full recording when shorter",
            "manufacturer_formula": (
                "(ChannelData - ADZero) * ConversionFactor * 10**Exponent in InfoChannel.Unit"
            ),
            "sampling_rule": "fs_hz = 1_000_000 / Tick_us",
        },
        "singlewell_conditions": condition_rows,
        "descriptive_e4031_minus_control": differences,
        "sixwell_control": {
            "n_channels": int(len(channels)),
            "median_beat_detection_rate": (
                None if channels.empty else float(channels["beat_detection_rate"].median())
            ),
            "channels_with_at_least_two_beats": int((channels["n_beats"] >= 2).sum()),
            "channels_with_fpd": int((channels["n_beats_with_fpd"] > 0).sum()),
        },
        "outputs": {
            "condition_features": condition_csv.name,
            "sixwell_channel_features": channels_csv.name,
        },
        "interpretation_guardrail": (
            "This evaluates public MCS HDF5 decoding, physical unit conversion and existing "
            "signal-feature extraction. There is only one source recording per single-well "
            "condition; the comparison is descriptive, is not independent replicated risk "
            "validation, and does not assign a CardioScore risk class."
        ),
    }
    summary_path = output_dir / "simultox_component_summary.json"
    summary_path.write_text(
        json.dumps(json_safe(manifest), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("data/external/simultox_2024"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/simultox_component_validation"))
    parser.add_argument("--local-only", action="store_true", help="Use already downloaded source files; do not download.")
    args = parser.parse_args()
    run_analysis(args.input_dir, args.output_dir, download=not args.local_only)
    print(f"Results written to {args.output_dir.resolve()}")
    print("Scope: component validation only; no drug-risk class or predictive metric produced.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

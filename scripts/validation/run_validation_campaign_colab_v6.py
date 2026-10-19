#!/usr/bin/env python3
"""CardioScore public-validation campaign v6: source-adapter execution.

This run reuses the established Drive cache. It does not change CardioScore weights,
thresholds, detector settings, endpoint definitions, or clinical-risk labels.

It adds two explicitly scoped analyses:
  * Ballouz/Vandenberg: exploratory FPD candidate extraction from the published averaged-beat
    MATLAB arrays, paired by plate/well across baseline/cis/nif conditions. This is directional
    source-adapter checking, not locked validation evidence.
  * SiMulTox: strict manufacturer-defined MCS HDF5 scaling into microvolts followed by the existing
    CardioScore signal endpoint extractor. The single-well Control/E-4031 comparison remains
    descriptive because it is one recording per condition, not an independent replicated study.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np

BASE_RUNNER_COMMIT = "b1def2d24302827f6b391e1c5054606dbd9f04e1"
BASE_RUNNER_URL = (
    "https://raw.githubusercontent.com/Virelion-Biotech/Virelion-CardioScore/"
    f"{BASE_RUNNER_COMMIT}/scripts/validation/run_validation_campaign_colab.py"
)
PACKAGE_PIN_V6 = "0efe6355d5eb4a43ccf2d4b6d923510f95a3d60c"
FORCE_V6 = {
    "_campaign:preflight",
    "ballouz_vandenberg_2021:analysis",
    "simultox_2024:analysis",
}


def _numeric_leaves(obj: Any) -> list[np.ndarray]:
    out: list[np.ndarray] = []

    def walk(value: Any) -> None:
        arr = np.asarray(value)
        if arr.dtype == object:
            for item in arr.flat:
                walk(item)
            return
        if np.issubdtype(arr.dtype, np.number) and arr.size:
            out.append(np.asarray(arr, dtype=float).squeeze())

    walk(obj)
    return out


def _well_fs_hz(time_cell: Any) -> float | None:
    leaves = [x.reshape(-1) for x in _numeric_leaves(time_cell) if x.size >= 3]
    if not leaves:
        return None
    time = max(leaves, key=lambda x: x.size)
    diffs = np.diff(time)
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if diffs.size < 2:
        return None
    dt = float(np.median(diffs))
    if dt <= 0 or not math.isfinite(dt):
        return None
    jitter = float(np.max(np.abs(diffs - dt)) / dt)
    if jitter > 0.01:
        return None
    return 1.0 / dt


def _fpd_candidate(waveform: np.ndarray, fs_hz: float) -> dict[str, Any]:
    """Exploratory depolarization-to-repolarization candidate on one averaged beat.

    The source publication defines FPD as depolarization peak to repolarization-wave peak.
    This helper intentionally reports QC diagnostics and is not wired into the frozen
    CardioScore endpoint implementation.
    """
    from scipy import signal

    x = np.asarray(waveform, dtype=float).reshape(-1)
    finite = np.isfinite(x)
    if x.size < max(20, int(round(0.15 * fs_hz))) or not finite.all():
        return {"valid": False, "reason": "short_or_nonfinite_waveform"}
    centered = x - float(np.median(x))
    search_start_n = max(1, int(round(0.080 * fs_hz)))
    search_end_n = max(search_start_n + 3, int(round(0.800 * fs_hz)))
    latest_depol = max(1, x.size - search_start_n - 3)
    depol_idx = int(np.argmax(np.abs(centered[:latest_depol])))
    depol_amp = float(centered[depol_idx])
    if not math.isfinite(depol_amp) or abs(depol_amp) <= 0:
        return {"valid": False, "reason": "no_depolarization_peak"}

    start = depol_idx + search_start_n
    end = min(x.size, depol_idx + search_end_n)
    if end - start < 3:
        return {"valid": False, "reason": "insufficient_repolarization_window"}
    segment = centered[start:end]
    opposite = -segment if depol_amp > 0 else segment

    pre = centered[max(0, depol_idx - int(round(0.10 * fs_hz))):depol_idx]
    if pre.size >= 5:
        med = float(np.median(pre))
        mad = float(np.median(np.abs(pre - med))) * 1.4826
    else:
        mad = 0.0
    min_prom = max(3.0 * mad, abs(depol_amp) * 0.01, 1e-12)
    peaks, props = signal.find_peaks(opposite, prominence=min_prom)
    if len(peaks) == 0:
        return {
            "valid": False,
            "reason": "no_opposite_polarity_repolarization_peak",
            "depol_idx": depol_idx,
            "depol_amplitude_native": depol_amp,
            "minimum_prominence_native": min_prom,
        }
    best = int(np.argmax(props["prominences"]))
    repol_local = int(peaks[best])
    repol_idx = start + repol_local
    prominence = float(props["prominences"][best])
    fpd_ms = float((repol_idx - depol_idx) / fs_hz * 1000.0)
    ratio = prominence / abs(depol_amp)
    valid = 80.0 <= fpd_ms <= 800.0 and ratio >= 0.01
    return {
        "valid": bool(valid),
        "reason": None if valid else "candidate_failed_qc",
        "depol_idx": depol_idx,
        "repol_idx": repol_idx,
        "fpd_ms": fpd_ms,
        "depol_amplitude_native": depol_amp,
        "repolarization_prominence_native": prominence,
        "repolarization_to_depolarization_prominence_ratio": float(ratio),
        "minimum_prominence_native": float(min_prom),
    }


def analysis_ballouz_v6(root: Path, atomic_json) -> dict:
    import pandas as pd
    import scipy.io

    raw = root / "raw" / "ballouz_vandenberg_2021" / "files"
    result = root / "results" / "ballouz_vandenberg_2021"
    result.mkdir(parents=True, exist_ok=True)
    expected = [
        f"plate{plate}_{condition}.mat"
        for plate in (1, 2)
        for condition in ("baseline", "cis", "nif")
    ]
    missing = [name for name in expected if not (raw / name).exists()]
    if missing:
        marker = result / "analysis_blocked.json"
        atomic_json(marker, {"status": "blocked", "reason": "missing_mat_files", "files": missing})
        return {"status": "blocked", "outputs": [{"path": str(marker)}]}

    rows: list[dict[str, Any]] = []
    electrode_rows: list[dict[str, Any]] = []
    for plate in (1, 2):
        for condition in ("baseline", "cis", "nif"):
            path = raw / f"plate{plate}_{condition}.mat"
            payload = scipy.io.loadmat(
                path,
                variable_names=["AllWells_AvgBeats", "AllWells_UnaveragedTime"],
                squeeze_me=False,
                struct_as_record=False,
            )
            if "AllWells_AvgBeats" not in payload or "AllWells_UnaveragedTime" not in payload:
                marker = result / "analysis_blocked.json"
                atomic_json(
                    marker,
                    {"status": "blocked", "reason": "required_ballouz_variables_missing", "file": path.name},
                )
                return {"status": "blocked", "outputs": [{"path": str(marker)}]}
            avg = np.asarray(payload["AllWells_AvgBeats"], dtype=object)
            times = np.asarray(payload["AllWells_UnaveragedTime"], dtype=object)
            if avg.shape != (6, 8) or times.shape != (6, 8):
                marker = result / "analysis_blocked.json"
                atomic_json(
                    marker,
                    {
                        "status": "blocked",
                        "reason": "unexpected_ballouz_well_grid",
                        "file": path.name,
                        "avg_shape": list(avg.shape),
                        "time_shape": list(times.shape),
                    },
                )
                return {"status": "blocked", "outputs": [{"path": str(marker)}]}

            for row_idx in range(6):
                for col_idx in range(8):
                    fs_hz = _well_fs_hz(times[row_idx, col_idx])
                    waveforms = [
                        x.reshape(-1)
                        for x in _numeric_leaves(avg[row_idx, col_idx])
                        if x.size >= 100
                    ]
                    if fs_hz is None or not waveforms:
                        rows.append(
                            {
                                "plate": plate,
                                "condition": condition,
                                "well_row_zero_based": row_idx,
                                "well_col_zero_based": col_idx,
                                "fs_hz_from_exported_timestamps": fs_hz,
                                "n_electrode_waveforms": len(waveforms),
                                "n_valid_fpd_electrodes": 0,
                                "fpd_median_ms": np.nan,
                            }
                        )
                        continue
                    candidates = []
                    for electrode_idx, waveform in enumerate(waveforms):
                        candidate = _fpd_candidate(waveform, fs_hz)
                        electrode_rows.append(
                            {
                                "plate": plate,
                                "condition": condition,
                                "well_row_zero_based": row_idx,
                                "well_col_zero_based": col_idx,
                                "electrode_index": electrode_idx,
                                "fs_hz_from_exported_timestamps": fs_hz,
                                **candidate,
                            }
                        )
                        if candidate.get("valid") and candidate.get("fpd_ms") is not None:
                            candidates.append(float(candidate["fpd_ms"]))
                    rows.append(
                        {
                            "plate": plate,
                            "condition": condition,
                            "well_row_zero_based": row_idx,
                            "well_col_zero_based": col_idx,
                            "fs_hz_from_exported_timestamps": fs_hz,
                            "n_electrode_waveforms": len(waveforms),
                            "n_valid_fpd_electrodes": len(candidates),
                            "fpd_median_ms": float(np.median(candidates)) if len(candidates) >= 4 else np.nan,
                        }
                    )

    wells = pd.DataFrame(rows)
    electrodes = pd.DataFrame(electrode_rows)
    wells_csv = result / "ballouz_exploratory_fpd_by_well.csv"
    electrodes_csv = result / "ballouz_exploratory_fpd_by_electrode.csv"
    wells.to_csv(wells_csv, index=False)
    electrodes.to_csv(electrodes_csv, index=False)

    wide = wells.pivot_table(
        index=["plate", "well_row_zero_based", "well_col_zero_based"],
        columns="condition",
        values="fpd_median_ms",
        aggfunc="first",
    ).reset_index()
    required = {"baseline", "cis", "nif"}
    if not required.issubset(wide.columns):
        marker = result / "analysis_blocked.json"
        atomic_json(marker, {"status": "blocked", "reason": "paired_conditions_not_recoverable"})
        return {"status": "blocked", "outputs": [{"path": str(marker)}]}
    paired = wide.dropna(subset=["baseline", "cis", "nif"]).copy()
    paired["cis_minus_baseline_ms"] = paired["cis"] - paired["baseline"]
    paired["nif_minus_cis_ms"] = paired["nif"] - paired["cis"]
    paired["nif_minus_baseline_ms"] = paired["nif"] - paired["baseline"]
    paired["cis_prolonged"] = paired["cis_minus_baseline_ms"] > 0
    paired["nif_reversed_from_cis"] = paired["nif_minus_cis_ms"] < 0
    paired["nif_closer_to_baseline"] = (
        paired["nif_minus_baseline_ms"].abs() < paired["cis_minus_baseline_ms"].abs()
    )
    paired_csv = result / "ballouz_exploratory_paired_effects.csv"
    paired.to_csv(paired_csv, index=False)

    summary = {
        "status": "complete_exploratory_adapter_check",
        "scope": "paired_directional_fpd_candidate_extraction",
        "n_wells_total": int(len(wells)),
        "n_paired_wells_with_valid_fpd_all_conditions": int(len(paired)),
        "median_exported_sampling_rate_hz": (
            float(wells["fs_hz_from_exported_timestamps"].dropna().median())
            if wells["fs_hz_from_exported_timestamps"].notna().any()
            else None
        ),
        "median_cis_minus_baseline_ms": (
            float(paired["cis_minus_baseline_ms"].median()) if len(paired) else None
        ),
        "median_nif_minus_cis_ms": (
            float(paired["nif_minus_cis_ms"].median()) if len(paired) else None
        ),
        "fraction_cis_prolonged": float(paired["cis_prolonged"].mean()) if len(paired) else None,
        "fraction_nif_reversed_from_cis": (
            float(paired["nif_reversed_from_cis"].mean()) if len(paired) else None
        ),
        "fraction_nif_closer_to_baseline": (
            float(paired["nif_closer_to_baseline"].mean()) if len(paired) else None
        ),
        "source_concentration_conflict": {
            "MEA_guide_txt": "states 200 micromolar cisapride and 50 micromolar nifedipine",
            "published_methods_and_Dryad_methods": "state 200 nM cisapride and 50 nM nifedipine",
            "handling": "condition identities baseline/cis/nif are used; numeric concentrations are not encoded in this analysis",
        },
        "interpretation_guardrail": (
            "Exploratory source-adapter check only. FPD candidates are derived from published averaged-beat arrays using an explicit peak rule that has not been frozen or independently human-validated for this dataset. No CardioScore risk score or clinical claim is produced."
        ),
    }
    summary_json = result / "ballouz_exploratory_summary.json"
    atomic_json(summary_json, summary)
    (result / "analysis_blocked.json").unlink(missing_ok=True)
    return {
        "outputs": [
            {"path": str(wells_csv)},
            {"path": str(electrodes_csv)},
            {"path": str(paired_csv)},
            {"path": str(summary_json)},
        ],
        "analysis_scope": "exploratory_ballouz_paired_fpd_challenge",
        "adapter_required": True,
        "not_locked_validation_evidence": True,
    }


def _feature_dict(feature: Any) -> dict[str, Any]:
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


def _center_window(path: Path, seconds: float = 60.0) -> tuple[float, float]:
    import h5py

    stream_path = "Data/Recording_0/AnalogStream/Stream_0"
    with h5py.File(path, "r") as h:
        stream = h[stream_path]
        n = int(stream["ChannelData"].shape[1])
        info = stream["InfoChannel"][()]
        tick_us = int(info[0]["Tick"])
    fs = 1_000_000.0 / tick_us
    duration = n / fs
    window = min(float(seconds), duration)
    start = max(0.0, (duration - window) / 2.0)
    return start, window


def analysis_simultox_v6(root: Path, atomic_json) -> dict:
    import pandas as pd

    from virelion_cardioscore.io.mcs_hdf5 import read_mcs_hdf5_traces
    from virelion_cardioscore.preprocessing.endpoints import extract_electrode_features

    raw = root / "raw" / "simultox_2024"
    result = root / "results" / "simultox_2024"
    result.mkdir(parents=True, exist_ok=True)
    control = raw / "SingleWell_HiPSC_cardio_Control.h5"
    treated = raw / "SingleWell_HiPSC_cardio_E-4031_1uM.h5"
    sixwell = raw / "6well_HiPSC_cardio_Control.h5"
    for path in (control, treated, sixwell):
        if not path.exists():
            marker = result / "analysis_blocked.json"
            atomic_json(marker, {"status": "blocked", "reason": "expected_hdf5_file_missing", "file": path.name})
            return {"status": "blocked", "outputs": [{"path": str(marker)}]}

    condition_rows = []
    for condition, path in (("Control", control), ("E-4031_1uM", treated)):
        start, window = _center_window(path, 60.0)
        fs_hz, traces, metadata = read_mcs_hdf5_traces(
            path, start_s=start, duration_s=window
        )
        if len(traces) != 1:
            marker = result / "analysis_blocked.json"
            atomic_json(
                marker,
                {"status": "blocked", "reason": "singlewell_file_did_not_contain_one_channel", "file": path.name, "n_channels": len(traces)},
            )
            return {"status": "blocked", "outputs": [{"path": str(marker)}]}
        label, trace = next(iter(traces.items()))
        feature = extract_electrode_features(trace, fs_hz)
        condition_rows.append(
            {
                "condition": condition,
                "source_file": path.name,
                "channel": label,
                "window_start_s": start,
                "window_duration_s": window,
                "fs_hz": fs_hz,
                "mcs_unit": metadata[0].unit,
                "mcs_exponent": metadata[0].exponent,
                "mcs_ad_zero": metadata[0].ad_zero,
                "mcs_conversion_factor": metadata[0].conversion_factor,
                "multiplier_to_uv_per_adc_step": metadata[0].multiplier_to_uv,
                **_feature_dict(feature),
            }
        )

    conditions = pd.DataFrame(condition_rows)
    condition_csv = result / "simultox_singlewell_cardioscore_features.csv"
    conditions.to_csv(condition_csv, index=False)

    c = conditions.set_index("condition").loc["Control"]
    t = conditions.set_index("condition").loc["E-4031_1uM"]
    effect = {
        "fpd_ms_difference": (
            None if pd.isna(c["fpd_ms"]) or pd.isna(t["fpd_ms"])
            else float(t["fpd_ms"] - c["fpd_ms"])
        ),
        "beat_rate_bpm_difference": float(t["beat_rate_bpm"] - c["beat_rate_bpm"]),
        "amplitude_uv_difference": float(t["amplitude_uv"] - c["amplitude_uv"]),
        "ibi_variability_difference": float(t["stv_ibi_variability"] - c["stv_ibi_variability"]),
        "triangulation_proxy_difference": (
            None if pd.isna(c["triangulation_proxy"]) or pd.isna(t["triangulation_proxy"])
            else float(t["triangulation_proxy"] - c["triangulation_proxy"])
        ),
    }

    start6, window6 = _center_window(sixwell, 60.0)
    fs6, traces6, _ = read_mcs_hdf5_traces(sixwell, start_s=start6, duration_s=window6)
    channel_rows = []
    for label, trace in traces6.items():
        feature = extract_electrode_features(trace, fs6)
        channel_rows.append({"channel": label, "fs_hz": fs6, **_feature_dict(feature)})
    channels = pd.DataFrame(channel_rows)
    channel_csv = result / "simultox_6well_control_channel_features.csv"
    channels.to_csv(channel_csv, index=False)

    summary = {
        "status": "complete_exploratory_signal_adapter_check",
        "scope": "manufacturer_scaled_mcs_hdf5_existing_cardioscore_endpoints",
        "manufacturer_formula": "(ChannelData - ADZero) * ConversionFactor * 10**Exponent in Unit",
        "manufacturer_tick_rule": "fs_hz = 1_000_000 / Tick_us",
        "singlewell_conditions": condition_rows,
        "descriptive_e4031_minus_control": effect,
        "sixwell_control": {
            "n_channels": int(len(channels)),
            "median_beat_detection_rate": float(channels["beat_detection_rate"].median()),
            "channels_with_at_least_two_beats": int((channels["n_beats"] >= 2).sum()),
            "channels_with_fpd": int((channels["n_beats_with_fpd"] > 0).sum()),
        },
        "interpretation_guardrail": (
            "The MCS physical conversion is specification-backed, but the Control/E-4031 comparison contains one source recording per condition and is not treated as an independent replicated risk-validation experiment. No CardioScore risk class is assigned from this comparison."
        ),
    }
    summary_json = result / "simultox_signal_adapter_summary.json"
    atomic_json(summary_json, summary)
    (result / "analysis_blocked.json").unlink(missing_ok=True)
    return {
        "outputs": [
            {"path": str(condition_csv)},
            {"path": str(channel_csv)},
            {"path": str(summary_json)},
        ],
        "analysis_scope": "exploratory_simultox_signal_adapter_check",
        "adapter_required": False,
        "not_independent_risk_validation": True,
    }


def main() -> int:
    base_source = urllib.request.urlopen(BASE_RUNNER_URL).read()
    print("Base campaign runner commit:", BASE_RUNNER_COMMIT)
    print("Base runner SHA-256:", hashlib.sha256(base_source).hexdigest())
    print("Package pin with strict MCS HDF5 adapter:", PACKAGE_PIN_V6)
    print("Source-adapter execution patch: active")

    ns = {"__name__": "cardioscore_campaign_base_v2"}
    exec(compile(base_source, "run_validation_campaign_colab.py", "exec"), ns)
    ns["PACKAGE_PIN"] = PACKAGE_PIN_V6
    ns["FORCE"].update(FORCE_V6)

    base_preflight = ns["preflight"]
    atomic_json = ns["atomic_json"]

    def preflight_v6(root: Path) -> dict:
        base = base_preflight(root)
        import subprocess
        import sys

        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "tests/test_mcs_hdf5.py"],
            cwd=ns["LOCAL_REPO"],
            text=True,
            capture_output=True,
        )
        log = root / "results" / "_preflight" / "mcs_hdf5_pytest.txt"
        log.write_text(
            "$ python -m pytest -q tests/test_mcs_hdf5.py\n\nSTDOUT\n"
            + proc.stdout
            + "\nSTDERR\n"
            + proc.stderr,
            encoding="utf-8",
        )
        if proc.returncode:
            raise RuntimeError("MCS HDF5 adapter tests failed; refusing source-adapter execution")
        outputs = list(base.get("outputs", [])) + [{"path": str(log)}]
        return {**base, "outputs": outputs, "mcs_hdf5_adapter_test": "passed"}

    ns["preflight"] = preflight_v6
    ns["analysis_ballouz"] = lambda root: analysis_ballouz_v6(root, atomic_json)
    ns["analysis_simultox"] = lambda root: analysis_simultox_v6(root, atomic_json)

    return int(ns["main"]())


if __name__ == "__main__":
    raise SystemExit(main())

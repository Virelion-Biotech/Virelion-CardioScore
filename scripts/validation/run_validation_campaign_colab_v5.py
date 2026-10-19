#!/usr/bin/env python3
"""CardioScore public-validation campaign v5: deep adapter introspection.

This wrapper reuses the established Drive cache/checkpoints and unchanged CardioScore scoring code.
It reruns only the analysis stages that still need source-specific adapters:
  * Ballouz/Vandenberg MATLAB raw MEA: robust MAT-format inspection + small waveform samples.
  * SiMulTox MCS HDF5: extract self-describing acquisition/scaling metadata + small raw samples.
  * Daley optical microtissue: add explicit directional dose-response corroboration summary.

No CardioScore weights, thresholds, detector settings, endpoint definitions, or clinical-risk labels
are changed. The output remains evidence-scoped and fail-closed.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import shutil
import sys
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np

BASE_RUNNER_COMMIT = "b1def2d24302827f6b391e1c5054606dbd9f04e1"
BASE_RUNNER_URL = (
    "https://raw.githubusercontent.com/Virelion-Biotech/Virelion-CardioScore/"
    f"{BASE_RUNNER_COMMIT}/scripts/validation/run_validation_campaign_colab.py"
)
V4_COMMIT = "a46f27162a70bd100d39faed1c1c59e4b217c884"
V4_URL = (
    "https://raw.githubusercontent.com/Virelion-Biotech/Virelion-CardioScore/"
    f"{V4_COMMIT}/scripts/validation/run_validation_campaign_colab_v4.py"
)

FORCE_ANALYSIS = {
    "ballouz_vandenberg_2021:analysis",
    "simultox_2024:analysis",
    "daley_2024_microtissue:analysis",
}

BALL_VAR_NAMES = (
    "FileData",
    "AllWells_AvgBeats",
    "AllWells_UnaveragedTime",
    "AllWells_UnaveragedData",
)


def safe_scalar(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and not math.isfinite(value):
            return str(value)
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        return safe_scalar(value.item())
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return safe_scalar(value.item())
    return str(value)


def numeric_sample_summary(arr: np.ndarray, max_sample: int = 100000) -> dict:
    a = np.asarray(arr)
    info = {
        "shape": list(a.shape),
        "dtype": str(a.dtype),
        "size": int(a.size),
    }
    if not np.issubdtype(a.dtype, np.number) or a.size == 0:
        return info
    flat = a.reshape(-1)
    if flat.size > max_sample:
        step = max(1, flat.size // max_sample)
        flat = flat[::step][:max_sample]
    try:
        vals = flat.astype(float, copy=False)
    except Exception:
        return info
    finite = vals[np.isfinite(vals)]
    info["finite_fraction_sample"] = float(finite.size / vals.size) if vals.size else None
    if finite.size:
        info.update({
            "min_sample": float(np.min(finite)),
            "max_sample": float(np.max(finite)),
            "mean_sample": float(np.mean(finite)),
            "median_sample": float(np.median(finite)),
        })
    return info


def summarize_mat_object(obj: Any, *, depth: int = 0, max_depth: int = 3) -> Any:
    if depth > max_depth:
        return {"type": type(obj).__name__, "truncated_depth": True}

    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool, bytes, np.generic)):
        return safe_scalar(obj)

    if isinstance(obj, np.ndarray):
        base = {
            "type": "ndarray",
            "shape": list(obj.shape),
            "dtype": str(obj.dtype),
            "size": int(obj.size),
        }
        if obj.dtype != object:
            base.update(numeric_sample_summary(obj))
            return base

        nonempty = []
        for idx, cell in np.ndenumerate(obj):
            try:
                arr = np.asarray(cell)
                is_empty = arr.size == 0
            except Exception:
                is_empty = cell is None
            if not is_empty:
                nonempty.append(idx)
        base["nonempty_cell_count"] = len(nonempty)
        base["sample_cells"] = []
        for idx in nonempty[:8]:
            base["sample_cells"].append({
                "index": list(idx),
                "summary": summarize_mat_object(obj[idx], depth=depth + 1, max_depth=max_depth),
            })
        return base

    fieldnames = getattr(obj, "_fieldnames", None)
    if fieldnames:
        fields = {}
        for name in list(fieldnames)[:40]:
            try:
                fields[str(name)] = summarize_mat_object(
                    getattr(obj, name), depth=depth + 1, max_depth=max_depth
                )
            except Exception as exc:
                fields[str(name)] = {"error": f"{type(exc).__name__}: {exc}"}
        return {
            "type": type(obj).__name__,
            "mat_struct": True,
            "fields": fields,
            "field_count": len(fieldnames),
        }

    if isinstance(obj, (list, tuple)):
        return {
            "type": type(obj).__name__,
            "length": len(obj),
            "items": [
                summarize_mat_object(x, depth=depth + 1, max_depth=max_depth)
                for x in list(obj)[:8]
            ],
        }

    return {"type": type(obj).__name__, "repr": repr(obj)[:500]}


def detect_mat_format(path: Path) -> dict:
    with path.open("rb") as handle:
        head = handle.read(128)
    text = head.decode("latin-1", errors="replace").replace("\x00", "")
    if head.startswith(b"\x89HDF\r\n\x1a\n") or "MATLAB 7.3 MAT-file" in text:
        fmt = "matlab_v7.3_hdf5"
    elif "MATLAB 5.0 MAT-file" in text:
        fmt = "matlab_level5"
    else:
        fmt = "unknown_mat"
    return {
        "format": fmt,
        "header_text": text[:120],
        "first16_hex": head[:16].hex(),
    }


def load_level5_variable(path: Path, name: str) -> Any:
    import scipy.io
    payload = scipy.io.loadmat(
        path,
        variable_names=[name],
        squeeze_me=False,
        struct_as_record=False,
        chars_as_strings=True,
    )
    if name not in payload:
        raise KeyError(f"{name} absent")
    return payload[name]


def h5_mat_tree(path: Path, limit: int = 600) -> dict:
    import h5py
    items = []
    with h5py.File(path, "r") as h:
        def visit(name: str, node: Any) -> None:
            if len(items) >= limit:
                return
            if isinstance(node, h5py.Dataset):
                item = {
                    "path": name,
                    "kind": "dataset",
                    "shape": list(node.shape),
                    "dtype": str(node.dtype),
                }
                if node.size and node.size <= 256 and node.dtype.kind not in {"O", "V"}:
                    try:
                        item["values"] = np.asarray(node[()]).tolist()
                    except Exception:
                        pass
                items.append(item)
            else:
                items.append({"path": name, "kind": "group"})
        h.visititems(visit)
        top = list(h.keys())
    return {"top_level_keys": top, "items": items, "truncated": len(items) >= limit}


def object_cell_nonempty_coords(obj: Any) -> list[tuple[int, ...]]:
    arr = np.asarray(obj)
    coords = []
    if arr.dtype != object:
        return coords
    for idx, cell in np.ndenumerate(arr):
        try:
            if np.asarray(cell).size > 0:
                coords.append(idx)
        except Exception:
            if cell is not None:
                coords.append(idx)
    return coords


def save_ballouz_small_samples(
    result: Path,
    stem: str,
    loaded: dict[str, Any],
    *,
    max_wells: int = 3,
    max_elements_per_array: int = 500000,
) -> dict:
    avg = loaded.get("AllWells_AvgBeats")
    tim = loaded.get("AllWells_UnaveragedTime")
    raw = loaded.get("AllWells_UnaveragedData")
    if avg is None:
        return {"saved": False, "reason": "AllWells_AvgBeats_not_loaded"}

    avg_arr = np.asarray(avg)
    if avg_arr.dtype != object:
        return {"saved": False, "reason": "AllWells_AvgBeats_not_object_cell_array"}

    candidates = object_cell_nonempty_coords(avg_arr)
    payload = {}
    manifest = []
    saved = 0
    for idx in candidates:
        if saved >= max_wells:
            break
        a = np.asarray(avg_arr[idx])
        if a.size == 0 or a.size > max_elements_per_array or a.dtype == object:
            continue
        keybase = "r" + "_".join(str(i) for i in idx)
        payload[f"{keybase}_avg"] = a

        entry = {
            "index_zero_based": list(idx),
            "avg_shape": list(a.shape),
            "avg_dtype": str(a.dtype),
        }

        if tim is not None:
            try:
                t = np.asarray(np.asarray(tim)[idx])
                entry["time_shape"] = list(t.shape)
                entry["time_dtype"] = str(t.dtype)
                if t.size <= max_elements_per_array and t.dtype != object:
                    payload[f"{keybase}_time"] = t
            except Exception as exc:
                entry["time_error"] = f"{type(exc).__name__}: {exc}"

        if raw is not None:
            try:
                r = np.asarray(np.asarray(raw)[idx])
                entry["unaveraged_shape"] = list(r.shape)
                entry["unaveraged_dtype"] = str(r.dtype)
            except Exception as exc:
                entry["unaveraged_error"] = f"{type(exc).__name__}: {exc}"

        manifest.append(entry)
        saved += 1

    if not payload:
        return {"saved": False, "reason": "no_small_numeric_avgbeat_cells"}

    out = result / f"{stem}_sample_avgbeats.npz"
    np.savez_compressed(out, **payload)
    return {"saved": True, "path": str(out), "wells": manifest}


def analysis_ballouz_v5(root: Path, atomic_json) -> dict:
    raw = root / "raw" / "ballouz_vandenberg_2021" / "files"
    result = root / "results" / "ballouz_vandenberg_2021"
    result.mkdir(parents=True, exist_ok=True)
    mats = sorted(raw.glob("*.mat"))
    if len(mats) != 6:
        marker = result / "analysis_blocked.json"
        atomic_json(marker, {
            "status": "blocked",
            "reason": "expected_six_mat_files_not_present",
            "found": [p.name for p in mats],
        })
        return {"status": "blocked", "outputs": [{"path": str(marker)}]}

    guide = next(iter(raw.glob("*guide*.txt")), None)
    files = []
    sample_outputs = []
    any_target_loaded = False

    for path in mats:
        info = {
            "file": path.name,
            "bytes": path.stat().st_size,
            **detect_mat_format(path),
            "variables": {},
        }
        if info["format"] == "matlab_v7.3_hdf5":
            try:
                info["hdf5_tree"] = h5_mat_tree(path)
                if any(name in info["hdf5_tree"]["top_level_keys"] for name in BALL_VAR_NAMES):
                    any_target_loaded = True
            except Exception as exc:
                info["hdf5_error"] = f"{type(exc).__name__}: {exc}"
        else:
            loaded = {}
            for name in BALL_VAR_NAMES:
                try:
                    obj = load_level5_variable(path, name)
                    loaded[name] = obj
                    info["variables"][name] = {
                        "loaded": True,
                        "summary": summarize_mat_object(obj, max_depth=3),
                    }
                    any_target_loaded = True
                except Exception as exc:
                    info["variables"][name] = {
                        "loaded": False,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
            sample = save_ballouz_small_samples(result, path.stem, loaded)
            info["sample_export"] = sample
            if sample.get("saved"):
                sample_outputs.append(Path(sample["path"]))
        files.append(info)

    out = result / "matlab_deep_intake_summary.json"
    state = (
        "intake_complete_adapter_design_ready"
        if any_target_loaded
        else "blocked_matlab_target_variables_unreadable"
    )
    atomic_json(out, {
        "status": state,
        "scope": "raw_mea_perturbation_rescue_structure_and_small_samples",
        "source_semantics": {
            "baseline": "absence of drug",
            "cis": "10 min after addition of 200 micromolar cisapride",
            "nif": "10 min after addition of 50 micromolar nifedipine",
            "plate1_cell_lines": ["142C1", "273C1", "274C1"],
            "plate2_cell_lines": ["8C1", "15C1", "479C1"],
            "source": "MEA_guide.txt",
        },
        "files": files,
        "guide_file": str(guide) if guide else None,
        "note": (
            "This stage exposes MATLAB structure and small averaged-waveform samples only. "
            "It does not infer FPD or treatment effects until axes/units/well mapping are verified."
        ),
    })
    if guide:
        shutil.copy2(guide, result / guide.name)

    if not any_target_loaded:
        return {
            "status": "blocked",
            "outputs": [{"path": str(out)}],
            "analysis_scope": "deep_mat_structure_intake_failed",
            "adapter_required": True,
        }
    outputs = [{"path": str(out)}] + [{"path": str(p)} for p in sample_outputs]
    return {
        "outputs": outputs,
        "analysis_scope": "deep_mat_structure_intake",
        "adapter_required": True,
    }


def decode_h5_value(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.bytes_):
        return bytes(value).decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        return decode_h5_value(value.item())
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return decode_h5_value(value.item())
        return [decode_h5_value(x) for x in value.tolist()]
    if isinstance(value, tuple):
        return [decode_h5_value(x) for x in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def read_mcs_h5_metadata(path: Path, result: Path) -> dict:
    import h5py

    stream_path = "Data/Recording_0/AnalogStream/Stream_0"
    with h5py.File(path, "r") as h:
        stream = h[stream_path]
        data = stream["ChannelData"]
        stamps = np.asarray(stream["ChannelDataTimeStamps"][()])
        info_ds = stream["InfoChannel"]
        info_arr = info_ds[()]

        rows = []
        if info_arr.dtype.names:
            for row in info_arr:
                item = {}
                for field in info_arr.dtype.names:
                    item[field] = decode_h5_value(row[field])
                rows.append(item)

        distinct = {}
        fields_of_interest = [
            "Unit", "Exponent", "ADZero", "Tick", "ConversionFactor",
            "ADCBits", "RawDataType", "Label", "ChannelID", "RowIndex",
        ]
        for field in fields_of_interest:
            vals = [r.get(field) for r in rows if field in r]
            uniq = []
            seen = set()
            for v in vals:
                marker = json.dumps(v, sort_keys=True, default=str)
                if marker not in seen:
                    seen.add(marker)
                    uniq.append(v)
            distinct[field] = uniq[:100]

        data_shape = list(data.shape)
        data_dtype = str(data.dtype)
        sample_n = min(int(data.shape[1]), 20000)
        raw_sample = np.asarray(data[:, :sample_n])
        sample_path = result / f"{path.stem}_raw_sample.npz"
        np.savez_compressed(
            sample_path,
            channel_data_raw=raw_sample,
            channel_data_timestamps=stamps,
        )

    return {
        "file": path.name,
        "bytes": path.stat().st_size,
        "channel_data_shape": data_shape,
        "channel_data_dtype": data_dtype,
        "timestamps": stamps.tolist(),
        "info_channel_rows": rows,
        "distinct_metadata": distinct,
        "raw_sample_export": str(sample_path),
        "metadata_complete_for_scaling_review": all(
            distinct.get(k) for k in ("Unit", "Exponent", "ADZero", "Tick", "ConversionFactor")
        ),
        "note": (
            "Raw integer samples and manufacturer metadata are exported. Physical-unit conversion "
            "is intentionally not asserted in this stage; the conversion formula will be locked only "
            "after metadata values are reviewed against the MCS format semantics."
        ),
    }


def analysis_simultox_v5(root: Path, atomic_json, simultox_files: dict) -> dict:
    raw = root / "raw" / "simultox_2024"
    result = root / "results" / "simultox_2024"
    result.mkdir(parents=True, exist_ok=True)
    files = []
    sample_outputs = []
    for name in simultox_files:
        path = raw / name
        if not path.exists():
            marker = result / "analysis_blocked.json"
            atomic_json(marker, {
                "status": "blocked",
                "reason": "expected_hdf5_file_missing",
                "file": name,
            })
            return {"status": "blocked", "outputs": [{"path": str(marker)}]}
        meta = read_mcs_h5_metadata(path, result)
        files.append(meta)
        sample_outputs.append(Path(meta["raw_sample_export"]))

    out = result / "hdf5_deep_metadata_summary.json"
    atomic_json(out, {
        "status": "intake_complete_scaling_review_ready",
        "scope": "control_vs_e4031_mcs_hdf5_metadata_and_raw_samples",
        "files": files,
        "note": (
            "The two single-well files are condition-labeled Control and E-4031 1uM by source filename. "
            "The 6-well file is Control. No unlabeled timestamped recordings are introduced."
        ),
    })
    return {
        "outputs": [{"path": str(out)}] + [{"path": str(p)} for p in sample_outputs],
        "analysis_scope": "mcs_hdf5_scaling_metadata_intake",
        "adapter_required": True,
    }


def analysis_daley_v5(root: Path, atomic_json, base_analysis_daley) -> dict:
    import pandas as pd

    base = base_analysis_daley(root)
    if base.get("status") == "blocked":
        return base

    result = root / "results" / "daley_2024_microtissue"
    summary_csv = result / "apd_concentration_summary.csv"
    frame = pd.read_csv(summary_csv)
    norm = frame[frame["normalization"].astype(str).str.lower() == "norm"].copy()

    rows = []
    for (metric, drug), group in norm.groupby(["metric", "drug"]):
        group = group.sort_values("concentration_uM")
        if group.empty:
            continue
        baseline = group.iloc[0]
        highest = group.iloc[-1]
        x = group["concentration_uM"].to_numpy(float)
        y = group["mean"].to_numpy(float)
        try:
            from scipy.stats import spearmanr
            rho, p = spearmanr(x, y)
            rho = None if not np.isfinite(rho) else float(rho)
            p = None if not np.isfinite(p) else float(p)
        except Exception:
            rho = p = None
        rows.append({
            "metric": metric,
            "drug": drug,
            "n_concentrations": int(len(group)),
            "baseline_concentration_uM": float(baseline["concentration_uM"]),
            "baseline_normalized_mean": float(baseline["mean"]),
            "highest_concentration_uM": float(highest["concentration_uM"]),
            "highest_n": int(highest["n"]),
            "highest_normalized_mean": float(highest["mean"]),
            "highest_normalized_sd": None if pd.isna(highest["std"]) else float(highest["std"]),
            "highest_percent_change_vs_1": float((highest["mean"] - 1.0) * 100.0),
            "direction_at_highest": (
                "increase" if highest["mean"] > 1.0
                else "decrease" if highest["mean"] < 1.0
                else "no_change"
            ),
            "spearman_concentration_vs_mean": rho,
            "spearman_p_value_descriptive": p,
        })

    directional = pd.DataFrame(rows)
    directional_csv = result / "daley_directional_corroboration.csv"
    directional.to_csv(directional_csv, index=False)
    meta = result / "daley_directional_corroboration.json"
    atomic_json(meta, {
        "status": "complete",
        "scope": "cross_modal_directional_corroboration",
        "n_rows": int(len(directional)),
        "interpretation_guardrail": (
            "These optical APD/APDtri dose-response summaries are cross-modal corroboration only. "
            "They are not MEA truth labels, not independent CardioScore risk classifications, and "
            "technical observations are not treated as independent compounds."
        ),
        "rows": directional.to_dict(orient="records"),
    })
    outputs = list(base.get("outputs", [])) + [
        {"path": str(directional_csv)},
        {"path": str(meta)},
    ]
    return {
        "outputs": outputs,
        "analysis_scope": "cross_modal_apd_triangulation_directional",
    }


def main() -> int:
    base_source = urllib.request.urlopen(BASE_RUNNER_URL).read()
    v4_source = urllib.request.urlopen(V4_URL).read()
    print("Base campaign runner commit:", BASE_RUNNER_COMMIT)
    print("Base runner SHA-256:", hashlib.sha256(base_source).hexdigest())
    print("Authenticated Dryad helper commit:", V4_COMMIT)
    print("Adapter deep-introspection patch: active")

    ns = {"__name__": "cardioscore_campaign_base_v2"}
    exec(compile(base_source, "run_validation_campaign_colab.py", "exec"), ns)

    v4ns = {"__name__": "cardioscore_campaign_v4_helpers"}
    exec(compile(v4_source, "run_validation_campaign_colab_v4.py", "exec"), v4ns)

    ns["dryad_full_dataset"] = v4ns["make_authenticated_dryad"](ns)

    base_analysis_daley = ns["analysis_daley"]
    atomic_json = ns["atomic_json"]
    simultox_files = ns["SIMULTOX_FILES"]

    ns["analysis_ballouz"] = lambda root: analysis_ballouz_v5(root, atomic_json)
    ns["analysis_simultox"] = lambda root: analysis_simultox_v5(root, atomic_json, simultox_files)
    ns["analysis_daley"] = lambda root: analysis_daley_v5(root, atomic_json, base_analysis_daley)

    ns["FORCE"].update(FORCE_ANALYSIS)
    return int(ns["main"]())


if __name__ == "__main__":
    raise SystemExit(main())

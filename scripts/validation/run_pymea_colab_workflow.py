#!/usr/bin/env python3
"""Colab workflow for blinded Cardio PyMEA beat-detection validation.

PREPARE downloads and verifies the full Cardio PyMEA release, caches it in Google Drive,
creates blinded held-out windows, and downloads an annotation package ZIP.
SCORE accepts that package plus one or two completed annotation CSVs, runs the frozen
validation protocol, and downloads a complete result ZIP.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

VALIDATION_CODE_SHA = "39767da5144946f73717b34c2ba7f1e63766afef"
PROTOCOL_COMMIT = "7c09d907f5c4949ef25209465de56df44a855d99"
FROZEN_CRITERIA_PATH = "validation/signal_validation_criteria_v1.yaml"
DATASET_DOI = "10.5068/D14H5C"
ZENODO_RECORD_ID = "6524838"
ZENODO_API = f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}"
EXPECTED_FILES = {
    "Day9__Nov1_130pm.txt": "03383581f3ec977563477f5b9b311a2f",
    "File_Parameter_List.xlsx": "84ea2ecb4730685d4bd5786bf5bc3e4a",
    "README_cardio_pymea.txt": "80f28001227b50db740e351a8151c729",
    "untreated_MEA120_only.xlsx": "6d75438174c3c7210da967ed21aa43a7",
}
WORK = Path("/content/cardioscore_pymea_validation")
REPO = WORK / "repo"
CACHE_REL = Path("CardioScore_validation_data/CardioPyMEA_D14H5C")
DEV_SEED = 20260921
DEV_N = 20
DEV_WINDOW_S = 10.0
HELDOUT_SEED = int(os.environ.get("CARDIOSCORE_PYMEA_HELDOUT_SEED", "20261019"))
HELDOUT_N = int(os.environ.get("CARDIOSCORE_PYMEA_N_WINDOWS", "40"))
HELDOUT_WINDOW_S = float(os.environ.get("CARDIOSCORE_PYMEA_WINDOW_S", "10"))


def run(cmd: list[str], *, cwd: Path | None = None) -> None:
    print("+", " ".join(map(str, cmd)))
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True, text=True)


def digest(path: Path, algorithm: str, chunk: int = 8 << 20) -> str:
    h = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def setup_repo() -> tuple[dict, Path, Path]:
    import yaml

    if WORK.exists():
        shutil.rmtree(WORK)
    WORK.mkdir(parents=True)
    run(["git", "clone", "-q", "https://github.com/Virelion-Biotech/Virelion-CardioScore.git", str(REPO)])
    run(["git", "checkout", "-q", PROTOCOL_COMMIT], cwd=REPO)
    diff = subprocess.run(
        [
            "git", "diff", "--exit-code", VALIDATION_CODE_SHA, PROTOCOL_COMMIT, "--",
            "virelion_cardioscore",
            "scripts/validation/pymea_beat_validation.py",
            "scripts/validation/synthetic_signal_benchmark.py",
            "scripts/validation/inspect_raw_source.py",
        ],
        cwd=REPO,
        text=True,
        capture_output=True,
    )
    if diff.returncode:
        raise RuntimeError("Frozen protocol changes executable validation code relative to VALIDATION_CODE_SHA.")
    run([sys.executable, "-m", "pip", "install", "-q", "-e", ".[dev,mixed]"], cwd=REPO)
    criteria_path = REPO / FROZEN_CRITERIA_PATH
    criteria = yaml.safe_load(criteria_path.read_text(encoding="utf-8"))
    assert criteria["status"] == "frozen"
    assert criteria["beat_detection"]["status"] == "frozen"
    assert criteria["validation_code_sha"] == VALIDATION_CODE_SHA
    setup = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "validation_code_sha": VALIDATION_CODE_SHA,
        "protocol_commit": PROTOCOL_COMMIT,
        "criteria_sha256": digest(criteria_path, "sha256"),
    }
    return setup, criteria_path, criteria


def run_preflight() -> Path:
    evidence = WORK / "environment_evidence"
    evidence.mkdir(exist_ok=True)
    tests = [
        "tests/test_mcs_ascii.py",
        "tests/test_signal_accuracy.py",
        "tests/test_pymea_beat_validation.py",
        "tests/test_stv_semantics.py",
    ]
    cmd = [sys.executable, "-m", "pytest", "-q", *tests]
    proc = subprocess.run(cmd, cwd=REPO, text=True, capture_output=True)
    (evidence / "focused_pytest.txt").write_text(
        "$ " + " ".join(cmd) + "\n\nSTDOUT\n" + proc.stdout + "\nSTDERR\n" + proc.stderr,
        encoding="utf-8",
    )
    print(proc.stdout)
    if proc.returncode:
        print(proc.stderr)
        raise RuntimeError("Focused validation tests failed.")
    bench = evidence / "synthetic_benchmark"
    run([
        sys.executable,
        str(REPO / "scripts/validation/synthetic_signal_benchmark.py"),
        "--out", str(bench), "--seeds", "5",
    ], cwd=REPO)
    return evidence


def acquire_dataset() -> tuple[dict[str, Path], dict]:
    from google.colab import drive
    import requests

    drive.mount("/content/drive", force_remount=False)
    cache = Path("/content/drive/MyDrive") / CACHE_REL
    cache.mkdir(parents=True, exist_ok=True)
    record = requests.get(ZENODO_API, timeout=60).json()
    entries = {item["key"]: item for item in record.get("files", [])}
    missing = sorted(set(EXPECTED_FILES) - set(entries))
    if missing:
        raise RuntimeError(f"Zenodo record missing expected file(s): {missing}")
    rows = []
    for name, expected_md5 in EXPECTED_FILES.items():
        entry = entries[name]
        api_checksum = str(entry.get("checksum", ""))
        api_md5 = api_checksum.split(":", 1)[-1] if api_checksum else ""
        if api_md5 and api_md5.lower() != expected_md5:
            raise RuntimeError(f"Published checksum changed for {name}: {api_md5}")
        target = cache / name
        if target.exists() and digest(target, "md5") == expected_md5:
            print("✓ cached + verified:", name)
            rows.append({"file": name, "bytes": target.stat().st_size, "md5": expected_md5, "source": "drive_cache"})
            continue
        target.unlink(missing_ok=True)
        part = cache / (name + ".part")
        url = entry.get("links", {}).get("content") or entry.get("links", {}).get("self")
        if not url:
            raise RuntimeError(f"No Zenodo download URL for {name}")
        print(f"Downloading {name} ({entry.get('size', 'unknown')} bytes)")
        run(["curl", "-L", "--fail", "--retry", "6", "--retry-delay", "10", "--connect-timeout", "30", "-C", "-", "-o", str(part), url])
        actual = digest(part, "md5")
        if actual != expected_md5:
            part.unlink(missing_ok=True)
            raise RuntimeError(f"MD5 mismatch for {name}: {actual} != {expected_md5}")
        os.replace(part, target)
        rows.append({"file": name, "bytes": target.stat().st_size, "md5": actual, "source": "zenodo"})
        print("✓ downloaded + verified:", name)
    files = {name: cache / name for name in EXPECTED_FILES}
    manifest = {
        "dataset_doi": DATASET_DOI,
        "zenodo_record_id": ZENODO_RECORD_ID,
        "zenodo_api": ZENODO_API,
        "record_title": record.get("metadata", {}).get("title"),
        "acquired_at_utc": datetime.now(timezone.utc).isoformat(),
        "files": rows,
    }
    (WORK / "dataset_acquisition.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return files, manifest


def load_pymea_module():
    path = REPO / "scripts/validation/pymea_beat_validation.py"
    spec = importlib.util.spec_from_file_location("pymea_validation_locked", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def find_historical_manifest(raw_sha: str) -> tuple[Path | None, dict | None]:
    matches: list[tuple[Path, dict]] = []
    for path in Path("/content/drive/MyDrive").rglob("sheet_manifest.json"):
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if (
            obj.get("source_sha256") == raw_sha
            and int(obj.get("seed", -1)) == DEV_SEED
            and int(obj.get("n_windows", -1)) == DEV_N
            and float(obj.get("window_s", -1)) == DEV_WINDOW_S
        ):
            matches.append((path, obj))
    if len(matches) > 1:
        raise RuntimeError("Multiple matching historical development manifests found:\n" + "\n".join(str(p) for p, _ in matches))
    return matches[0] if matches else (None, None)


def prepare(setup: dict, criteria_path: Path, criteria: dict, evidence: Path) -> None:
    from google.colab import files as colab_files
    from virelion_cardioscore.io.mcs_ascii import sniff_layout

    data, _ = acquire_dataset()
    prep = WORK / "prepare"
    prep.mkdir(exist_ok=True)
    inventory = prep / "source_inventory.json"
    run([
        sys.executable,
        str(REPO / "scripts/validation/inspect_raw_source.py"),
        *[str(data[name]) for name in EXPECTED_FILES],
        "--out", str(inventory),
    ], cwd=REPO)
    raw = data["Day9__Nov1_130pm.txt"]
    layout = sniff_layout(raw)
    raw_sha = digest(raw, "sha256")
    historical_path, historical = find_historical_manifest(raw_sha)
    pymea = load_pymea_module()
    exact = historical is not None
    if exact:
        dev = historical
        print("✓ exact historical manifest:", historical_path)
    else:
        print("No exact historical manifest found; reconstructing defaults and downgrading evidence state.")
        dev = pymea.make_sheet(
            raw,
            prep / "reconstructed_historical_default_development_batch",
            n_windows=DEV_N,
            window_s=DEV_WINDOW_S,
            seed=DEV_SEED,
            duration_s=None,
            exclude=[],
        )
    exclusions = [
        (str(w["electrode"]), float(w["start_s"]), float(w["start_s"]) + float(w["window_s"]))
        for w in dev["windows"]
    ]
    heldout_dir = prep / "heldout_annotation"
    heldout = pymea.make_sheet(
        raw,
        heldout_dir,
        n_windows=HELDOUT_N,
        window_s=HELDOUT_WINDOW_S,
        seed=HELDOUT_SEED,
        duration_s=None,
        exclude=exclusions,
    )
    overlaps = []
    for w in heldout["windows"]:
        start, end = float(w["start_s"]), float(w["start_s"]) + float(w["window_s"])
        for electrode, ex_start, ex_end in exclusions:
            if w["electrode"] == electrode and start < ex_end and end > ex_start:
                overlaps.append(w["window_id"])
    if overlaps:
        raise RuntimeError(f"Held-out overlap detected: {overlaps}")
    state = "confirmatory_heldout_candidate" if exact else "provisional_exploratory_reconstructed_development_exclusions"
    root = prep / "annotation_package"
    shutil.copytree(heldout_dir, root / "heldout_annotation")
    shutil.copy2(criteria_path, root / "signal_validation_criteria_v1.yaml")
    shutil.copy2(inventory, root / "source_inventory.json")
    shutil.copy2(WORK / "dataset_acquisition.json", root / "dataset_acquisition.json")
    shutil.copytree(evidence, root / "environment_evidence")
    exclusions_doc = {
        "source_sha256": raw_sha,
        "exact_development_manifest_supplied": exact,
        "historical_manifest_path": str(historical_path) if historical_path else None,
        "historical_default_reconstruction_used": not exact,
        "development_manifest": dev,
        "excluded_regions": [list(x) for x in exclusions],
    }
    (root / "development_exclusions.json").write_text(json.dumps(exclusions_doc, indent=2) + "\n", encoding="utf-8")
    protocol = {
        **setup,
        "dataset_doi": DATASET_DOI,
        "zenodo_record_id": ZENODO_RECORD_ID,
        "evidence_state": state,
        "confirmatory_eligible_if_annotations_complete": exact,
        "source_filename": raw.name,
        "source_sha256": raw_sha,
        "source_bytes": raw.stat().st_size,
        "fs_hz": float(layout.fs_hz),
        "n_source_electrodes": len(layout.electrodes),
        "heldout_seed": HELDOUT_SEED,
        "heldout_n_windows": HELDOUT_N,
        "heldout_window_s": HELDOUT_WINDOW_S,
        "frozen_acceptance": criteria["beat_detection"],
    }
    (root / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    (root / "ANNOTATE_ME.txt").write_text(
        "Open heldout_annotation/plots/*.png and fill annotation_sheet.csv.\n"
        "Set done=1 for every row. Enter semicolon-separated beat times relative to each 10-s window.\n"
        "Do not inspect CardioScore detections before finishing annotation. Save as annotator_A.csv; a second independent annotator is preferred.\n",
        encoding="utf-8",
    )
    out = Path(f"/content/cardioscore_pymea_annotation_package_{HELDOUT_SEED}.zip")
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                archive.write(path, arcname=str(path.relative_to(root)))
    print("Evidence state:", state)
    print("Annotation package:", out)
    colab_files.download(str(out))


def score(setup: dict, criteria_path: Path, criteria: dict, evidence: Path) -> None:
    import pandas as pd
    from google.colab import files as colab_files

    uploaded = colab_files.upload()
    upload_dir = WORK / "score_uploads"
    upload_dir.mkdir(exist_ok=True)
    paths = []
    for name, data in uploaded.items():
        path = upload_dir / Path(name).name
        path.write_bytes(data)
        paths.append(path)
    zips = [p for p in paths if p.suffix.lower() == ".zip"]
    csvs = sorted(p for p in paths if p.suffix.lower() == ".csv")
    if len(zips) != 1 or not (1 <= len(csvs) <= 2):
        raise RuntimeError("Upload exactly one annotation-package ZIP plus one or two completed CSVs.")
    score_root = WORK / "score"
    package_dir = score_root / "package"
    results_dir = score_root / "results"
    package_dir.mkdir(parents=True)
    results_dir.mkdir(parents=True)
    with zipfile.ZipFile(zips[0]) as archive:
        bad = [n for n in archive.namelist() if n.startswith(("/", "\\")) or ".." in Path(n).parts]
        if bad:
            raise RuntimeError(f"Unsafe ZIP members: {bad}")
        archive.extractall(package_dir)
    protocol = json.loads((package_dir / "protocol.json").read_text(encoding="utf-8"))
    if protocol["validation_code_sha"] != VALIDATION_CODE_SHA or protocol["protocol_commit"] != PROTOCOL_COMMIT:
        raise RuntimeError("Annotation package does not match frozen validation code/protocol.")
    packaged_criteria = package_dir / "signal_validation_criteria_v1.yaml"
    if digest(packaged_criteria, "sha256") != digest(criteria_path, "sha256"):
        raise RuntimeError("Frozen criteria hash mismatch.")
    annotations = []
    for src in csvs:
        dst = results_dir / src.name
        shutil.copy2(src, dst)
        annotations.append(dst)
    pymea = load_pymea_module()
    results = pymea.score(
        package_dir / "heldout_annotation",
        annotations,
        float(criteria["beat_detection"]["tolerance_ms"]) / 1000.0,
        packaged_criteria,
        pymea.FilterConfig(),
    )
    exact = bool(protocol["confirmatory_eligible_if_annotations_complete"])
    passed = results.get("criteria", {}).get("passed")
    interpretation = {
        "protocol_evidence_state": protocol["evidence_state"],
        "exact_development_exclusions": exact,
        "criteria_confirmatory": results.get("criteria", {}).get("confirmatory"),
        "criteria_passed": passed,
        "confirmatory_interpretation_allowed": bool(exact and passed is not None),
        "confirmatory_passed": passed if exact else None,
        "reason": None if exact else "Exact historical development manifest unavailable; provisional/exploratory only.",
    }
    results["evidence_interpretation"] = interpretation
    (results_dir / "pymea_heldout_validation_results.json").write_text(json.dumps(results, indent=2, default=float) + "\n", encoding="utf-8")
    rows = []
    summary = []
    for annotator, metrics in results["vs_annotator"].items():
        summary.append({k: metrics[k] for k in ["tp", "fp", "fn", "precision", "recall", "f1", "timing_median_abs_error_ms", "timing_p95_abs_error_ms"]} | {"annotator": annotator})
        for window_id, item in metrics["per_window"].items():
            rows.append({"annotator": annotator, "window_id": window_id, **item})
    pd.DataFrame(summary).to_csv(results_dir / "metric_summary.csv", index=False)
    pd.DataFrame(rows).to_csv(results_dir / "per_window_detection_counts.csv", index=False)
    pd.DataFrame(results["qc"]["windows"]).to_csv(results_dir / "qc_windows.csv", index=False)
    for name in ["protocol.json", "signal_validation_criteria_v1.yaml", "source_inventory.json", "dataset_acquisition.json", "development_exclusions.json"]:
        src = package_dir / name
        if src.exists():
            shutil.copy2(src, results_dir / name)
    shutil.copy2(package_dir / "heldout_annotation/sheet_manifest.json", results_dir / "heldout_sheet_manifest.json")
    if (package_dir / "environment_evidence").exists():
        shutil.copytree(package_dir / "environment_evidence", results_dir / "environment_evidence", dirs_exist_ok=True)
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "validation_code_sha": VALIDATION_CODE_SHA,
        "protocol_commit": PROTOCOL_COMMIT,
        "evidence_interpretation": interpretation,
        "artifacts": {},
    }
    for path in sorted(results_dir.rglob("*")):
        if path.is_file():
            manifest["artifacts"][str(path.relative_to(results_dir))] = {"bytes": path.stat().st_size, "sha256": digest(path, "sha256")}
    (results_dir / "RESULTS_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    out = Path("/content/cardioscore_pymea_heldout_validation_results.zip")
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
        for path in sorted(results_dir.rglob("*")):
            if path.is_file():
                archive.write(path, arcname=str(path.relative_to(results_dir)))
    print(json.dumps(interpretation, indent=2))
    for row in summary:
        print(row)
    colab_files.download(str(out))


def main() -> None:
    mode = os.environ.get("CARDIOSCORE_PYMEA_MODE", "PREPARE").strip().upper()
    if mode not in {"PREPARE", "SCORE"}:
        raise ValueError("CARDIOSCORE_PYMEA_MODE must be PREPARE or SCORE")
    setup, criteria_path, criteria = setup_repo()
    evidence = run_preflight()
    if mode == "PREPARE":
        prepare(setup, criteria_path, criteria, evidence)
    else:
        score(setup, criteria_path, criteria, evidence)


if __name__ == "__main__":
    main()

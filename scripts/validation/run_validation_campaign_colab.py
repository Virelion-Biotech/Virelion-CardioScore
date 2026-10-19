#!/usr/bin/env python3
"""Resumable all-public-dataset CardioScore validation campaign for Google Colab.

The campaign downloads the agreed public datasets, caches raw assets and completed stages in
Google Drive, resumes safely after runtime loss, runs source-appropriate analyses without inventing
schema/mappings, and exports one compact results ZIP. Large raw files remain in Drive and are never
placed in the download bundle.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import traceback
import urllib.parse
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

PACKAGE_PIN = "ba58932542c80327c4993579758027e596c551ec"
REPO_URL = "https://github.com/Virelion-Biotech/Virelion-CardioScore.git"
CAMPAIGN_ID = "cardioscore-public-validation-v1"
RUNNER_SCHEMA_VERSION = 2
DRIVE_REL = Path("CardioScore_validation_campaign_v1")
LOCAL_WORK = Path("/content/cardioscore_validation_campaign_runtime")
LOCAL_REPO = LOCAL_WORK / "repo"
AUTO_DOWNLOAD = os.environ.get("CARDIOSCORE_CAMPAIGN_AUTO_DOWNLOAD", "1").lower() in {"1", "true", "yes", "y"}
FORCE = {x.strip() for x in os.environ.get("CARDIOSCORE_CAMPAIGN_FORCE", "").split(",") if x.strip()}

PYMEA_DEV_SEED = 20260921
PYMEA_DEV_N = 20
PYMEA_DEV_WINDOW_S = 10.0
PYMEA_HELDOUT_SEED = int(os.environ.get("CARDIOSCORE_CAMPAIGN_PYMEA_SEED", "20261019"))
PYMEA_HELDOUT_N = int(os.environ.get("CARDIOSCORE_CAMPAIGN_PYMEA_WINDOWS", "40"))
PYMEA_WINDOW_S = float(os.environ.get("CARDIOSCORE_CAMPAIGN_PYMEA_WINDOW_S", "10"))

PYMEA_ZENODO_RECORD = "6524838"
PYMEA_FILES = {
    "Day9__Nov1_130pm.txt": "03383581f3ec977563477f5b9b311a2f",
    "File_Parameter_List.xlsx": "84ea2ecb4730685d4bd5786bf5bc3e4a",
    "README_cardio_pymea.txt": "80f28001227b50db740e351a8151c729",
    "untreated_MEA120_only.xlsx": "6d75438174c3c7210da967ed21aa43a7",
}
SIMULTOX_FILES = {
    "SingleWell_HiPSC_cardio_Control.h5": "992d4d3e32182717cd23d8212c0b8538",
    "SingleWell_HiPSC_cardio_E-4031_1uM.h5": "a3a3990f4bd353cabef00974d6ff7030",
    "6well_HiPSC_cardio_Control.h5": "f3c24f14d8dc264540522b25a1584a6e",
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run(cmd: list[str], *, cwd: Path | None = None, capture: bool = False) -> subprocess.CompletedProcess:
    print("+", " ".join(map(str, cmd)))
    return subprocess.run(
        list(map(str, cmd)),
        cwd=str(cwd) if cwd else None,
        check=True,
        text=True,
        capture_output=capture,
    )


def atomic_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def digest(path: Path, algorithm: str = "sha256", chunk: int = 8 << 20) -> str:
    h = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def safe_extract(zip_path: Path, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.infolist():
            name = member.filename
            if name.startswith(("/", "\\")) or ".." in Path(name).parts:
                raise RuntimeError(f"Unsafe ZIP member: {name}")
        zf.extractall(out_dir)
    return sorted(p for p in out_dir.rglob("*") if p.is_file())


def curl_download(url: str, target: Path, *, expected_md5: str | None = None) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if expected_md5 is None or digest(target, "md5") == expected_md5.lower():
            print("✓ cached:", target.name)
            return target
        print("Cached checksum mismatch; replacing", target.name)
        target.unlink()
    part = target.with_suffix(target.suffix + ".part")
    cmd = [
        "curl", "-L", "--fail", "--retry", "8", "--retry-delay", "10",
        "--connect-timeout", "30", "-C", "-", "-o", str(part), url,
    ]
    try:
        run(cmd)
    except subprocess.CalledProcessError:
        part.unlink(missing_ok=True)
        run([
            "curl", "-L", "--fail", "--retry", "8", "--retry-delay", "10",
            "--connect-timeout", "30", "-o", str(part), url,
        ])
    if expected_md5 is not None:
        actual = digest(part, "md5")
        if actual.lower() != expected_md5.lower():
            part.unlink(missing_ok=True)
            raise RuntimeError(f"MD5 mismatch for {target.name}: {actual} != {expected_md5}")
    os.replace(part, target)
    return target


def checkpoint_path(root: Path, dataset: str, stage: str) -> Path:
    return root / "checkpoints" / dataset / f"{stage}.json"


def outputs_exist(record: dict) -> bool:
    outputs = record.get("outputs") or []
    return all(Path(item["path"]).exists() for item in outputs if item.get("required", True))


def stage(
    root: Path,
    dataset: str,
    name: str,
    fn: Callable[[], dict],
    *,
    continue_on_error: bool = True,
) -> dict:
    key = f"{dataset}:{name}"
    cp = checkpoint_path(root, dataset, name)
    if key not in FORCE and cp.exists():
        try:
            old = json.loads(cp.read_text(encoding="utf-8"))
        except Exception:
            old = None
        if (
            old
            and old.get("runner_schema_version") == RUNNER_SCHEMA_VERSION
            and old.get("status") in {"complete", "blocked", "reference_only"}
            and outputs_exist(old)
        ):
            print(f"↪ resume: {key} ({old['status']})")
            return old
    record = {
        "dataset": dataset,
        "stage": name,
        "started_at_utc": now(),
        "status": "running",
        "runner_schema_version": RUNNER_SCHEMA_VERSION,
    }
    atomic_json(cp, record)
    try:
        payload = fn() or {}
        status = payload.pop("status", "complete")
        record.update(payload)
        record["status"] = status
        record["finished_at_utc"] = now()
        atomic_json(cp, record)
        return record
    except Exception as exc:
        record.update({
            "status": "failed",
            "finished_at_utc": now(),
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        })
        atomic_json(cp, record)
        print(f"✗ {key}: {record['error']}")
        if not continue_on_error:
            raise
        return record


def setup_drive_and_repo() -> tuple[Path, dict]:
    from google.colab import drive
    import yaml

    drive.mount("/content/drive", force_remount=False)
    root = Path("/content/drive/MyDrive") / DRIVE_REL
    for name in ("raw", "results", "checkpoints", "exports", "logs"):
        (root / name).mkdir(parents=True, exist_ok=True)

    if LOCAL_WORK.exists():
        shutil.rmtree(LOCAL_WORK)
    LOCAL_WORK.mkdir(parents=True)
    run(["git", "clone", "-q", REPO_URL, str(LOCAL_REPO)])
    run(["git", "checkout", "-q", PACKAGE_PIN], cwd=LOCAL_REPO)
    run([sys.executable, "-m", "pip", "install", "-q", "-e", ".[dev,mixed]", "h5py", "requests", "openpyxl", "beautifulsoup4"], cwd=LOCAL_REPO)
    repo_str = str(LOCAL_REPO)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)
    importlib.invalidate_caches()
    registry = yaml.safe_load((LOCAL_REPO / "validation/campaign_datasets.yaml").read_text(encoding="utf-8"))
    if registry.get("campaign_id") != CAMPAIGN_ID:
        raise RuntimeError("Campaign registry ID does not match runner.")
    (root / "campaign_registry.yaml").write_text(
        (LOCAL_REPO / "validation/campaign_datasets.yaml").read_text(encoding="utf-8"), encoding="utf-8"
    )
    return root, registry


def preflight(root: Path) -> dict:
    out = root / "results" / "_preflight"
    out.mkdir(parents=True, exist_ok=True)
    tests = [
        "tests/test_mcs_ascii.py",
        "tests/test_signal_accuracy.py",
        "tests/test_pymea_beat_validation.py",
        "tests/test_stv_semantics.py",
        "tests/test_colab_runner_blinova.py",
    ]
    cmd = [sys.executable, "-m", "pytest", "-q", *tests]
    proc = subprocess.run(cmd, cwd=LOCAL_REPO, text=True, capture_output=True)
    log = out / "focused_pytest.txt"
    log.write_text("$ " + " ".join(cmd) + "\n\nSTDOUT\n" + proc.stdout + "\nSTDERR\n" + proc.stderr, encoding="utf-8")
    if proc.returncode:
        raise RuntimeError("Focused repository tests failed; refusing real-data execution. See focused_pytest.txt")
    bench = out / "synthetic_benchmark"
    if not bench.exists():
        run([
            sys.executable,
            str(LOCAL_REPO / "scripts/validation/synthetic_signal_benchmark.py"),
            "--out", str(bench), "--seeds", "5",
        ], cwd=LOCAL_REPO)
    return {"outputs": [{"path": str(log)}, {"path": str(bench)}], "tests": tests}


def _absolute_dryad_href(href: str) -> str:
    if href.startswith("http://") or href.startswith("https://"):
        return href
    return "https://datadryad.org" + href


def _dryad_api_files(doi: str) -> tuple[list[dict], dict]:
    """Resolve published Dryad files via dataset -> version -> files REST links."""
    import requests

    encoded = urllib.parse.quote("doi:" + doi, safe="")
    dataset_url = f"https://datadryad.org/api/v2/datasets/{encoded}"
    r = requests.get(dataset_url, timeout=60, headers={"Accept": "application/json"})
    r.raise_for_status()
    dataset = r.json()

    version_href = (dataset.get("_links", {}).get("stash:version") or {}).get("href")
    if not version_href:
        versions_url = dataset_url + "/versions"
        vr = requests.get(versions_url, timeout=60, headers={"Accept": "application/json"})
        vr.raise_for_status()
        payload = vr.json()
        versions = (payload.get("_embedded") or {}).get("stash:versions") or []
        if not versions:
            raise RuntimeError("Dryad dataset exposes no published version.")
        chosen = max(versions, key=lambda x: int(x.get("id", -1)))
        version_href = (chosen.get("_links", {}).get("self") or {}).get("href") or f"/api/v2/versions/{chosen['id']}"

    version_url = _absolute_dryad_href(version_href)
    v = requests.get(version_url, timeout=60, headers={"Accept": "application/json"})
    v.raise_for_status()
    version = v.json()
    files_href = (version.get("_links", {}).get("stash:files") or {}).get("href")
    if not files_href:
        files_href = version_href.rstrip("/") + "/files"

    files_url = _absolute_dryad_href(files_href)
    entries: list[dict] = []
    seen: set[str] = set()
    while files_url and files_url not in seen:
        seen.add(files_url)
        fr = requests.get(files_url, timeout=60, headers={"Accept": "application/json"})
        fr.raise_for_status()
        payload = fr.json()
        entries.extend((payload.get("_embedded") or {}).get("stash:files") or [])
        next_href = (payload.get("_links", {}).get("next") or {}).get("href")
        files_url = _absolute_dryad_href(next_href) if next_href else ""

    if not entries:
        raise RuntimeError("Dryad REST API returned zero files.")
    return entries, {"dataset_url": dataset_url, "version_url": version_url}


def _dryad_landing_files(doi: str) -> list[dict]:
    """Legacy fallback for records whose REST API path is unavailable."""
    import requests
    from bs4 import BeautifulSoup

    url = f"https://datadryad.org/dataset/doi:{doi}"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    entries = []
    for a in soup.find_all("a", href=True):
        href = str(a.get("href", ""))
        if "/downloads/file_stream/" not in href:
            continue
        name = a.get_text(" ", strip=True)
        if not name or name.lower() == "download":
            continue
        entries.append({
            "path": name,
            "_links": {"stash:download": {"href": href}},
            "digest": None,
            "digestType": None,
        })
    dedup = {}
    for item in entries:
        dedup[(item["path"], item["_links"]["stash:download"]["href"])] = item
    return list(dedup.values())


def dryad_full_dataset(root: Path, dataset: str, doi: str) -> dict:
    """Download each published Dryad file, preserving paths and provenance."""
    raw = root / "raw" / dataset
    extracted = raw / "files"
    raw.mkdir(parents=True, exist_ok=True)
    extracted.mkdir(parents=True, exist_ok=True)
    marker = raw / "SOURCE_UNAVAILABLE.json"

    method = "dryad_rest_api"
    try:
        entries, provenance = _dryad_api_files(doi)
    except Exception as api_exc:
        try:
            entries = _dryad_landing_files(doi)
            if not entries:
                raise RuntimeError("Landing page exposed no downloadable file links.")
            provenance = {
                "landing_url": f"https://datadryad.org/dataset/doi:{doi}",
                "api_error": str(api_exc),
            }
            method = "dryad_landing_file_stream_fallback"
        except Exception as landing_exc:
            atomic_json(marker, {
                "doi": doi,
                "checked_at_utc": now(),
                "api_error": str(api_exc),
                "landing_fallback_error": str(landing_exc),
            })
            return {
                "status": "blocked",
                "reason": "published_files_unavailable",
                "doi": doi,
                "outputs": [{"path": str(marker)}],
            }

    downloaded = []
    receipts = []
    for entry in entries:
        rel = Path(str(entry.get("path") or ""))
        if not rel.name or rel.is_absolute() or ".." in rel.parts:
            raise RuntimeError(f"Unsafe/invalid Dryad path: {rel}")
        href = ((entry.get("_links") or {}).get("stash:download") or {}).get("href")
        if not href:
            raise RuntimeError(f"Dryad file has no download link: {rel}")
        url = _absolute_dryad_href(str(href))
        target = extracted / rel
        target.parent.mkdir(parents=True, exist_ok=True)

        digest_value = str(entry.get("digest") or "").lower() or None
        digest_type = str(entry.get("digestType") or "").lower()
        expected_sha256 = digest_value if digest_type in {"sha-256", "sha256"} else None

        if target.exists() and expected_sha256 and digest(target) == expected_sha256:
            print("✓ cached + Dryad SHA-256 verified:", rel)
        elif target.exists() and expected_sha256 is None:
            print("✓ cached:", rel)
        else:
            if target.exists():
                target.unlink()
            curl_download(url, target)
            if expected_sha256 and digest(target) != expected_sha256:
                target.unlink(missing_ok=True)
                raise RuntimeError(f"Dryad SHA-256 mismatch for {rel}")

        downloaded.append(target)
        receipts.append({
            "path": str(rel),
            "bytes": target.stat().st_size,
            "sha256": digest(target),
            "dryad_digest": digest_value,
            "dryad_digest_type": entry.get("digestType"),
            "download_url": url,
        })

    marker.unlink(missing_ok=True)
    receipt = raw / "acquisition.json"
    atomic_json(receipt, {
        "doi": doi,
        "source": "Dryad",
        "method": method,
        "acquired_at_utc": now(),
        **provenance,
        "files": receipts,
    })
    return {
        "outputs": [{"path": str(receipt)}, {"path": str(extracted)}],
        "n_files": len(downloaded),
        "raw_dir": str(extracted),
        "download_method": method,
    }


def acquire_blinova(root: Path) -> dict:
    raw = root / "raw" / "blinova_2018_cipa"
    file = curl_download(
        "https://cipaproject.org/wp-content/uploads/2018/09/Blinova_etal_2018_data.xlsx",
        raw / "Blinova_etal_2018_data.xlsx",
    )
    receipt = raw / "acquisition.json"
    atomic_json(receipt, {"source": "CiPA Project", "url": "https://cipaproject.org/wp-content/uploads/2018/09/Blinova_etal_2018_data.xlsx", "acquired_at_utc": now(), "sha256": digest(file), "bytes": file.stat().st_size})
    return {"outputs": [{"path": str(file)}, {"path": str(receipt)}], "raw_file": str(file)}


def acquire_pymea(root: Path) -> dict:
    import requests
    raw = root / "raw" / "cardio_pymea_2022"
    raw.mkdir(parents=True, exist_ok=True)
    meta = requests.get(f"https://zenodo.org/api/records/{PYMEA_ZENODO_RECORD}", timeout=60)
    meta.raise_for_status()
    record = meta.json()
    entries = {x["key"]: x for x in record.get("files", [])}
    if sorted(set(PYMEA_FILES) - set(entries)):
        raise RuntimeError("Zenodo Cardio PyMEA mirror missing expected files.")
    downloaded = []
    for name, md5 in PYMEA_FILES.items():
        entry = entries[name]
        api_md5 = str(entry.get("checksum", "")).split(":", 1)[-1]
        if api_md5 and api_md5.lower() != md5.lower():
            raise RuntimeError(f"Zenodo checksum changed for {name}")
        url = entry.get("links", {}).get("content") or entry.get("links", {}).get("self")
        file = curl_download(url, raw / name, expected_md5=md5)
        downloaded.append(file)
    receipt = raw / "acquisition.json"
    atomic_json(receipt, {"doi": "10.5068/D14H5C", "zenodo_record": PYMEA_ZENODO_RECORD, "acquired_at_utc": now(), "files": [{"name": p.name, "bytes": p.stat().st_size, "md5": PYMEA_FILES[p.name]} for p in downloaded]})
    return {"outputs": [{"path": str(receipt)}] + [{"path": str(p)} for p in downloaded], "raw_dir": str(raw)}


def acquire_simultox(root: Path) -> dict:
    raw = root / "raw" / "simultox_2024"
    raw.mkdir(parents=True, exist_ok=True)
    files = []
    for name, md5 in SIMULTOX_FILES.items():
        url = f"https://zenodo.org/records/13121249/files/{urllib.parse.quote(name)}?download=1"
        files.append(curl_download(url, raw / name, expected_md5=md5))
    receipt = raw / "acquisition.json"
    atomic_json(receipt, {"doi": "10.5281/zenodo.13121249", "acquired_at_utc": now(), "selection_rule": "three agreed self-describing files only", "files": [{"name": p.name, "bytes": p.stat().st_size, "md5": SIMULTOX_FILES[p.name]} for p in files]})
    return {"outputs": [{"path": str(receipt)}] + [{"path": str(p)} for p in files], "raw_dir": str(raw)}


def acquire_geiger(root: Path) -> dict:
    import requests
    raw = root / "raw" / "geiger_2025_nam"
    raw.mkdir(parents=True, exist_ok=True)
    meta = requests.get("https://api.figshare.com/v2/articles/30466295", timeout=60)
    meta.raise_for_status()
    article = meta.json()
    downloaded = []
    for entry in article.get("files", []):
        name = entry["name"]
        url = entry.get("download_url")
        if not url:
            continue
        downloaded.append(curl_download(url, raw / name, expected_md5=entry.get("computed_md5")))
    if not downloaded:
        raise RuntimeError("Figshare record returned no downloadable files.")
    receipt = raw / "acquisition.json"
    atomic_json(receipt, {"doi": "10.6084/m9.figshare.30466295", "figshare_id": 30466295, "title": article.get("title"), "acquired_at_utc": now(), "files": [{"name": p.name, "bytes": p.stat().st_size, "sha256": digest(p)} for p in downloaded]})
    return {"outputs": [{"path": str(receipt)}] + [{"path": str(p)} for p in downloaded], "raw_dir": str(raw)}


def acquire_altrocchi(root: Path) -> dict:
    raw = root / "raw" / "altrocchi_2023_chronic"
    raw.mkdir(parents=True, exist_ok=True)
    article = curl_download(
        "https://public-pages-files-2025.frontiersin.org/journals/pharmacology/articles/10.3389/fphar.2023.1229960/pdf",
        raw / "Altrocchi_2023_article.pdf",
    )
    supplement = raw / "DataSheet1_s001"
    try:
        curl_download("https://doi.org/10.3389/fphar.2023.1229960.s001", supplement)
        outputs = [article, supplement]
    except Exception as exc:
        marker = raw / "SUPPLEMENT_DOWNLOAD_NOTE.json"
        atomic_json(marker, {"error": str(exc), "doi": "10.3389/fphar.2023.1229960.s001"})
        outputs = [article, marker]
    receipt = raw / "acquisition.json"
    atomic_json(receipt, {"doi": "10.3389/fphar.2023.1229960", "role": "qualitative_reference_only", "acquired_at_utc": now(), "files": [{"name": p.name, "bytes": p.stat().st_size} for p in outputs]})
    return {"status": "reference_only", "outputs": [{"path": str(receipt)}] + [{"path": str(p)} for p in outputs], "raw_dir": str(raw)}


def files_under(path: Path) -> list[Path]:
    return sorted(p for p in path.rglob("*") if p.is_file() and not p.name.endswith(".part"))


def inventory(root: Path, dataset: str) -> dict:
    raw = root / "raw" / dataset
    result = root / "results" / dataset
    result.mkdir(parents=True, exist_ok=True)
    source_files = [p for p in files_under(raw) if p.name not in {"acquisition.json", "SOURCE_UNAVAILABLE.json", "SUPPLEMENT_DOWNLOAD_NOTE.json"}]
    if not source_files:
        marker = result / "inventory_blocked.json"
        atomic_json(marker, {"status": "blocked", "reason": "no_source_files", "dataset": dataset})
        return {"status": "blocked", "outputs": [{"path": str(marker)}]}
    out = result / "source_inventory.json"
    run([sys.executable, str(LOCAL_REPO / "scripts/validation/inspect_raw_source.py"), *[str(p) for p in source_files], "--out", str(out)], cwd=LOCAL_REPO)
    (result / "inventory_blocked.json").unlink(missing_ok=True)
    return {"outputs": [{"path": str(out)}], "n_source_files": len(source_files)}


def analysis_blinova(root: Path) -> dict:
    raw = root / "raw" / "blinova_2018_cipa" / "Blinova_etal_2018_data.xlsx"
    result = root / "results" / "blinova_2018_cipa"
    result.mkdir(parents=True, exist_ok=True)
    audit = result / "workbook_audit.json"
    run([sys.executable, str(LOCAL_REPO / "scripts/audit_blinova_workbook.py"), str(raw), "--json", str(audit)], cwd=LOCAL_REPO)
    import pandas as pd
    runner_path = LOCAL_REPO / "scripts/validation/run_colab_validation.py"
    spec = importlib.util.spec_from_file_location("cardioscore_colab_runner_campaign", runner_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    semantic = module.audit_blinova_frame(pd.read_excel(raw), result)
    semantic_path = result / "semantic_audit.json"
    atomic_json(semantic_path, semantic)
    return {"outputs": [{"path": str(audit)}, {"path": str(semantic_path)}], "analysis_scope": "semantic_component_multisite"}


def load_pymea_module():
    path = LOCAL_REPO / "scripts/validation/pymea_beat_validation.py"
    spec = importlib.util.spec_from_file_location("pymea_campaign", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def analysis_pymea(root: Path) -> dict:
    from virelion_cardioscore.io.mcs_ascii import sniff_layout
    raw = root / "raw" / "cardio_pymea_2022" / "Day9__Nov1_130pm.txt"
    result = root / "results" / "cardio_pymea_2022"
    result.mkdir(parents=True, exist_ok=True)
    layout = sniff_layout(raw)
    raw_sha = digest(raw)
    matches = []
    drive_root = Path("/content/drive/MyDrive")
    for path in drive_root.rglob("sheet_manifest.json"):
        if any(part in {"reconstructed_development_windows", "reconstructed_historical_default_development_batch"} for part in path.parts):
            continue
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if obj.get("source_sha256") == raw_sha and int(obj.get("seed", -1)) == PYMEA_DEV_SEED and int(obj.get("n_windows", -1)) == PYMEA_DEV_N and float(obj.get("window_s", -1)) == PYMEA_DEV_WINDOW_S:
            matches.append((path, obj))
    if len(matches) > 1:
        raise RuntimeError("Multiple matching historical PyMEA development manifests found; refusing to guess.")
    pymea = load_pymea_module()
    if matches:
        historical_path, dev = matches[0]
        exact = True
    else:
        historical_path = None
        dev_dir = result / "reconstructed_development_windows"
        if dev_dir.exists():
            shutil.rmtree(dev_dir)
        dev = pymea.make_sheet(raw, dev_dir, n_windows=PYMEA_DEV_N, window_s=PYMEA_DEV_WINDOW_S, seed=PYMEA_DEV_SEED, duration_s=None, exclude=[])
        exact = False
    exclusions = [(str(w["electrode"]), float(w["start_s"]), float(w["start_s"]) + float(w["window_s"])) for w in dev["windows"]]
    heldout = result / "heldout_annotation"
    if heldout.exists():
        shutil.rmtree(heldout)
    manifest = pymea.make_sheet(raw, heldout, n_windows=PYMEA_HELDOUT_N, window_s=PYMEA_WINDOW_S, seed=PYMEA_HELDOUT_SEED, duration_s=None, exclude=exclusions)
    evidence_state = "confirmatory_heldout_candidate" if exact else "provisional_exploratory_reconstructed_development_exclusions"
    state = {
        "status": "awaiting_blinded_human_annotation",
        "evidence_state": evidence_state,
        "source_sha256": raw_sha,
        "fs_hz": float(layout.fs_hz),
        "n_electrodes": len(layout.electrodes),
        "historical_manifest_path": str(historical_path) if historical_path else None,
        "heldout_manifest": manifest,
    }
    state_path = result / "pymea_signal_validation_state.json"
    atomic_json(state_path, state)
    package = result / f"cardioscore_pymea_annotation_package_{PYMEA_HELDOUT_SEED}.zip"
    with zipfile.ZipFile(package, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
        for p in sorted(heldout.rglob("*")):
            if p.is_file():
                zf.write(p, arcname=str(Path("heldout_annotation") / p.relative_to(heldout)))
        zf.writestr("EVIDENCE_STATE.txt", evidence_state + "\n")
    return {"outputs": [{"path": str(state_path)}, {"path": str(package)}, {"path": str(heldout / "sheet_manifest.json")}], "analysis_scope": "raw_signal_blinded_annotation_preparation", "evidence_state": evidence_state}


def analysis_ballouz(root: Path) -> dict:
    import scipy.io
    raw = root / "raw" / "ballouz_vandenberg_2021" / "files"
    result = root / "results" / "ballouz_vandenberg_2021"
    result.mkdir(parents=True, exist_ok=True)
    mats = sorted(raw.rglob("*.mat")) if raw.exists() else []
    if not mats:
        marker = result / "analysis_blocked.json"
        atomic_json(marker, {"status": "blocked", "reason": "no_matlab_source_files_after_acquisition"})
        return {"status": "blocked", "outputs": [{"path": str(marker)}]}
    (result / "analysis_blocked.json").unlink(missing_ok=True)
    summary = []
    for p in mats:
        try:
            variables = [{"name": n, "shape": list(shape), "class": cls} for n, shape, cls in scipy.io.whosmat(p)]
        except Exception as exc:
            variables = [{"error": f"{type(exc).__name__}: {exc}"}]
        summary.append({"file": p.name, "bytes": p.stat().st_size, "variables": variables})
    guide = next(iter(raw.rglob("*guide*.txt")), None)
    out = result / "matlab_intake_summary.json"
    atomic_json(out, {
        "status": "intake_complete_adapter_required",
        "scope": "raw_mea_perturbation_rescue_challenge",
        "n_mat_files": len(mats),
        "files": summary,
        "guide_file": str(guide) if guide else None,
        "note": "No treatment/well/channel mapping is guessed from MATLAB structures; source-specific adapter remains required for endpoint scoring.",
    })
    if guide:
        shutil.copy2(guide, result / guide.name)
    return {"outputs": [{"path": str(out)}], "analysis_scope": "structural_raw_mea_intake", "adapter_required": True}


def h5_tree(path: Path, limit: int = 800) -> dict:
    import h5py
    items = []
    with h5py.File(path, "r") as handle:
        def visit(name, node):
            if len(items) >= limit:
                return
            if isinstance(node, h5py.Dataset):
                items.append({"path": name, "shape": list(node.shape), "dtype": str(node.dtype)})
            else:
                items.append({"path": name, "group": True})
        handle.visititems(visit)
    return {"file": path.name, "items": items, "truncated": len(items) >= limit}


def analysis_simultox(root: Path) -> dict:
    raw = root / "raw" / "simultox_2024"
    result = root / "results" / "simultox_2024"
    result.mkdir(parents=True, exist_ok=True)
    trees = [h5_tree(raw / name) for name in SIMULTOX_FILES]
    out = result / "hdf5_intake_summary.json"
    atomic_json(out, {
        "status": "intake_complete_adapter_required",
        "scope": "control_vs_e4031_format_and_pharmacology_challenge",
        "files": trees,
        "note": "Only self-describing agreed files are used; timestamped recordings remain excluded until condition metadata are authoritative.",
    })
    return {"outputs": [{"path": str(out)}], "analysis_scope": "structural_hdf5_intake", "adapter_required": True}


def profile_workbook(path: Path, out_dir: Path) -> dict:
    import pandas as pd
    book = pd.ExcelFile(path)
    result = {"file": path.name, "sheets": {}}
    for sheet in book.sheet_names:
        frame = pd.read_excel(path, sheet_name=sheet)
        sheet_info = {"rows": int(len(frame)), "columns": [str(x) for x in frame.columns]}
        numeric = frame.select_dtypes(include="number")
        if not numeric.empty:
            desc = numeric.describe().transpose().reset_index().rename(columns={"index": "column"})
            csv = out_dir / ("sheet_" + re.sub(r"[^A-Za-z0-9_.-]+", "_", sheet) + "_numeric_summary.csv")
            desc.to_csv(csv, index=False)
            sheet_info["numeric_summary_csv"] = str(csv)
        result["sheets"][sheet] = sheet_info
    return result


def analysis_geiger(root: Path) -> dict:
    raw = root / "raw" / "geiger_2025_nam"
    result = root / "results" / "geiger_2025_nam"
    result.mkdir(parents=True, exist_ok=True)
    books = sorted(raw.rglob("*.xlsx"))
    preferred = [
        p for p in books
        if all(token in p.name.lower() for token in ("raw", "data", "summary"))
    ]
    if not preferred and len(books) == 1:
        preferred = books
    if len(preferred) != 1:
        marker = result / "analysis_blocked.json"
        atomic_json(marker, {
            "status": "blocked",
            "reason": "raw_data_summary_workbook_not_unambiguously_identified",
            "xlsx_files": [p.name for p in books],
            "candidates": [p.name for p in preferred],
        })
        return {"status": "blocked", "outputs": [{"path": str(marker)}]}
    profile = profile_workbook(preferred[0], result)
    out = result / "processed_workbook_profile.json"
    atomic_json(out, {
        "status": "complete",
        "scope": "directional_combination_processed_summary",
        "selected_workbook": preferred[0].name,
        **profile,
    })
    (result / "analysis_blocked.json").unlink(missing_ok=True)
    return {"outputs": [{"path": str(out)}], "analysis_scope": "processed_directional_summary"}


def read_daley_file(path: Path):
    import pandas as pd
    for sep in ("\t", ",", None):
        try:
            frame = pd.read_csv(path, sep=sep, engine="python" if sep is None else "c")
            if len(frame.columns) >= 3:
                return frame
        except Exception:
            pass
    raise ValueError(f"Could not parse {path.name}")


def analysis_daley(root: Path) -> dict:
    import pandas as pd
    raw = root / "raw" / "daley_2024_microtissue" / "files"
    result = root / "results" / "daley_2024_microtissue"
    result.mkdir(parents=True, exist_ok=True)
    rows = []
    parsed_files = 0
    for path in sorted(raw.rglob("APD*.txt")):
        m = re.match(r"(APDmxr|APDtri)_(Norm|Raw)_(.+)\.txt$", path.name)
        if not m:
            continue
        metric, normalization, drug = m.groups()
        frame = read_daley_file(path)
        cols = {str(c).strip().lower(): c for c in frame.columns}
        conc = cols.get("concentration")
        value = cols.get("value")
        if conc is None or value is None:
            continue
        frame[conc] = pd.to_numeric(frame[conc], errors="coerce")
        frame[value] = pd.to_numeric(frame[value], errors="coerce")
        clean = frame.dropna(subset=[conc, value])
        grouped = clean.groupby(conc)[value].agg(["count", "mean", "std", "median"]).reset_index()
        for _, r in grouped.iterrows():
            rows.append({"metric": metric, "normalization": normalization, "drug": drug, "concentration_uM": float(r[conc]), "n": int(r["count"]), "mean": float(r["mean"]), "std": None if pd.isna(r["std"]) else float(r["std"]), "median": float(r["median"])})
        parsed_files += 1
    if not rows:
        marker = result / "analysis_blocked.json"
        atomic_json(marker, {"status": "blocked", "reason": "expected_APD_files_not_parseable"})
        return {"status": "blocked", "outputs": [{"path": str(marker)}]}
    (result / "analysis_blocked.json").unlink(missing_ok=True)
    csv = result / "apd_concentration_summary.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    meta = result / "analysis_summary.json"
    atomic_json(meta, {"status": "complete", "scope": "cross_modal_apd_and_triangulation", "parsed_files": parsed_files, "rows": len(rows), "note": "Optical APD/APDtri corroboration only; not treated as MEA ground truth."})
    return {"outputs": [{"path": str(csv)}, {"path": str(meta)}], "analysis_scope": "cross_modal_apd_triangulation"}


def analysis_patellike(root: Path, dataset: str) -> dict:
    raw = root / "raw" / dataset
    result = root / "results" / dataset
    result.mkdir(parents=True, exist_ok=True)
    files = [p for p in files_under(raw) if p.name not in {"dataset.zip", "acquisition.json", "SOURCE_UNAVAILABLE.json"}]
    if not files:
        marker = result / "analysis_blocked.json"
        atomic_json(marker, {"status": "blocked", "reason": "source_archive_not_available_or_empty", "dataset": dataset})
        return {"status": "blocked", "outputs": [{"path": str(marker)}]}
    table_profiles = []
    for p in files:
        if p.suffix.lower() in {".xlsx", ".xls"}:
            try:
                table_profiles.append(profile_workbook(p, result))
            except Exception as exc:
                table_profiles.append({"file": p.name, "error": str(exc)})
    out = result / "intake_analysis.json"
    atomic_json(out, {
        "status": "intake_complete_adapter_required",
        "dataset": dataset,
        "n_files": len(files),
        "spreadsheet_profiles": table_profiles,
        "note": "No endpoint or treatment mapping is inferred without an authoritative source-specific adapter.",
    })
    return {"outputs": [{"path": str(out)}], "adapter_required": True, "analysis_scope": "external_archive_intake"}


def analysis_altrocchi(root: Path) -> dict:
    result = root / "results" / "altrocchi_2023_chronic"
    result.mkdir(parents=True, exist_ok=True)
    out = result / "reference_scope.json"
    atomic_json(out, {"status": "reference_only", "scope": "qualitative_chronic_corroboration", "doi": "10.3389/fphar.2023.1229960", "note": "Not scored as a machine-readable raw MEA benchmark."})
    return {"status": "reference_only", "outputs": [{"path": str(out)}]}


def export_campaign(root: Path, registry: dict, records: dict[str, dict]) -> Path:
    import pandas as pd
    export_dir = root / "exports"
    export_dir.mkdir(parents=True, exist_ok=True)
    status_rows = []
    for key, rec in sorted(records.items()):
        status_rows.append({"key": key, "dataset": rec.get("dataset"), "stage": rec.get("stage"), "status": rec.get("status"), "error": rec.get("error"), "reason": rec.get("reason")})
    pd.DataFrame(status_rows).to_csv(export_dir / "campaign_status.csv", index=False)
    atomic_json(export_dir / "campaign_summary.json", {
        "campaign_id": CAMPAIGN_ID,
        "package_pin": PACKAGE_PIN,
        "runner_schema_version": RUNNER_SCHEMA_VERSION,
        "generated_at_utc": now(),
        "environment": {"python": sys.version, "platform": platform.platform()},
        "records": records,
    })
    bundle = export_dir / "CardioScore_public_validation_campaign_results.zip"
    tmp = bundle.with_suffix(".zip.tmp")
    if tmp.exists():
        tmp.unlink()
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
        for base, label in ((root / "results", "results"), (root / "checkpoints", "checkpoints"), (export_dir, "exports")):
            for p in sorted(base.rglob("*")):
                if not p.is_file() or p == bundle or p == tmp:
                    continue
                if p.stat().st_size > 80 * 1024 * 1024:
                    continue
                arc = Path(label) / p.relative_to(base)
                zf.write(p, arcname=str(arc))
        zf.writestr("campaign_registry.yaml", (root / "campaign_registry.yaml").read_text(encoding="utf-8"))
    os.replace(tmp, bundle)
    return bundle


def main() -> int:
    root, registry = setup_drive_and_repo()
    print("Drive campaign root:", root)
    records: dict[str, dict] = {}

    records["_campaign:preflight"] = stage(root, "_campaign", "preflight", lambda: preflight(root), continue_on_error=False)

    acquisition: dict[str, Callable[[], dict]] = {
        "blinova_2018_cipa": lambda: acquire_blinova(root),
        "patel_2019_optopacing": lambda: dryad_full_dataset(root, "patel_2019_optopacing", "10.5061/dryad.mg2565q"),
        "cardio_pymea_2022": lambda: acquire_pymea(root),
        "ballouz_vandenberg_2021": lambda: dryad_full_dataset(root, "ballouz_vandenberg_2021", "10.5061/dryad.612jm640k"),
        "simultox_2024": lambda: acquire_simultox(root),
        "geiger_2025_nam": lambda: acquire_geiger(root),
        "daley_2024_microtissue": lambda: dryad_full_dataset(root, "daley_2024_microtissue", "10.5061/dryad.sqv9s4nbn"),
        "altrocchi_2023_chronic": lambda: acquire_altrocchi(root),
    }
    analyses: dict[str, Callable[[], dict]] = {
        "blinova_2018_cipa": lambda: analysis_blinova(root),
        "patel_2019_optopacing": lambda: analysis_patellike(root, "patel_2019_optopacing"),
        "cardio_pymea_2022": lambda: analysis_pymea(root),
        "ballouz_vandenberg_2021": lambda: analysis_ballouz(root),
        "simultox_2024": lambda: analysis_simultox(root),
        "geiger_2025_nam": lambda: analysis_geiger(root),
        "daley_2024_microtissue": lambda: analysis_daley(root),
        "altrocchi_2023_chronic": lambda: analysis_altrocchi(root),
    }

    for item in registry["datasets"]:
        dataset = item["id"]
        print("\n" + "=" * 90)
        print("DATASET:", dataset, "|", item["role"])
        print("=" * 90)
        acq = stage(root, dataset, "acquire", acquisition[dataset])
        records[f"{dataset}:acquire"] = acq
        inv = stage(root, dataset, "inventory", lambda d=dataset: inventory(root, d))
        records[f"{dataset}:inventory"] = inv
        ana = stage(root, dataset, "analysis", analyses[dataset])
        records[f"{dataset}:analysis"] = ana
        atomic_json(root / "campaign_live_state.json", {"updated_at_utc": now(), "records": records})

    bundle = export_campaign(root, registry, records)
    print("\nCampaign complete enough to export.")
    print("Persistent Drive root:", root)
    print("Result bundle:", bundle)
    print("If Colab crashes on a later rerun, completed stages are resumed from Drive checkpoints.")
    if AUTO_DOWNLOAD:
        from google.colab import files as colab_files
        colab_files.download(str(bundle))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

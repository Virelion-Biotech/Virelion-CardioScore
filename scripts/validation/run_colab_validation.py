#!/usr/bin/env python3
"""Single-runtime Colab validation runner for CardioScore.

This script is designed to be fetched from GitHub and executed directly inside
one Google Colab runtime. It replaces notebook-to-notebook handoffs with a
single run workspace and a machine-readable run manifest.

Fail-closed principles:
- never invent source metadata, endpoint values, reference labels, or mappings;
- never treat technical wells as independent compounds;
- never silently map unexpected CiPA platform/event labels;
- never publish raw source assets;
- never report headline external metrics when reference compounds are excluded;
- external validation requires an explicit source->feature verified rebuild.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

REPO = "Virelion-Biotech/Virelion-CardioScore"
PIN = "869150cd5fb5ccf155fb066258404bd4df163ade"
BRANCH = "main"
ROOT = Path("/content/cardioscore_validation")
RUN_LABEL = os.environ.get("CARDIOSCORE_RUN_LABEL", "").strip()
PUBLISH_RESULTS = os.environ.get("CARDIOSCORE_PUBLISH", "1").strip().lower() in {"1", "true", "yes", "y"}
ALLOWED_PUBLISH_NAMES = {
    "locked_external_validation.json",
    "locked_compound_validation.csv",
    "locked_failures_by_compound.csv",
    "secondary_sensitivity.csv",
    "qc_log.csv",
    "qc_summary.json",
    "raw_mea_lineage.json",
    "source_receipt.json",
    "blinova_reference.csv",
    "blinova_semantic_summary.csv",
}


def log(message: str) -> None:
    print(f"[CardioScore] {message}")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def file_inventory(path: Path) -> list[dict]:
    import tarfile
    import zipfile

    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as zf:
            return [
                {"name": info.filename, "bytes": info.file_size, "crc": info.CRC}
                for info in zf.infolist()
            ]
    if tarfile.is_tarfile(path):
        with tarfile.open(path, "r:*") as tf:
            return [
                {"name": info.name, "bytes": info.size, "type": str(info.type)}
                for info in tf.getmembers()
            ]
    return [{"name": path.name, "bytes": path.stat().st_size}]


def install_pinned_package() -> str:
    log(f"Installing CardioScore at pinned revision {PIN}")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            f"git+https://github.com/{REPO}.git@{PIN}",
        ],
        check=True,
    )
    return PIN


def choose_run_label() -> str:
    global RUN_LABEL
    if RUN_LABEL:
        return RUN_LABEL
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    RUN_LABEL = input(f"Run label [{stamp}]: ").strip() or stamp
    if not all(ch.isalnum() or ch in "-_" for ch in RUN_LABEL):
        raise ValueError("Run label may contain only letters, numbers, '-' and '_'.")
    return RUN_LABEL


def prepare_workspace() -> dict[str, Path]:
    label = choose_run_label()
    run_dir = ROOT / "runs" / label
    if run_dir.exists():
        raise FileExistsError(
            f"Run directory already exists: {run_dir}. Use a new CARDIOSCORE_RUN_LABEL."
        )
    inputs = run_dir / "input"
    derived = run_dir / "derived"
    results = run_dir / "results"
    work = run_dir / "work"
    for path in (inputs, derived, results, work):
        path.mkdir(parents=True, exist_ok=True)
    return {
        "run": run_dir,
        "input": inputs,
        "derived": derived,
        "results": results,
        "work": work,
    }


def upload_inputs(input_dir: Path) -> list[Path]:
    from google.colab import files

    log("Upload all source/validation assets needed for this run in one selection.")
    log("Raw source datasets remain local; they are never copied to the GitHub publish area.")
    uploaded = files.upload()
    paths = []
    for name, data in uploaded.items():
        target = input_dir / Path(name).name
        target.write_bytes(data)
        paths.append(target)
    if not paths:
        raise RuntimeError("No files were uploaded.")
    log("Uploaded: " + ", ".join(p.name for p in paths))
    return paths


def csv_columns(path: Path) -> set[str]:
    import pandas as pd

    return set(pd.read_csv(path, nrows=5).columns)


def xlsx_columns(path: Path) -> set[str]:
    import pandas as pd

    excel = pd.ExcelFile(path)
    if not excel.sheet_names:
        return set()
    frame = pd.read_excel(path, sheet_name=excel.sheet_names[0], nrows=5)
    return set(frame.columns)


def classify_inputs(paths: list[Path]) -> dict[str, list[Path] | Path | None]:
    raw_required = {
        "compound",
        "well",
        "concentration_uM",
        "vehicle",
        "electrode_id",
        "time_s",
        "voltage_uv",
    }
    feature_required = {
        "compound",
        "site",
        "cell_type",
        "concentration_index",
        "concentration_uM",
        "well",
        "vehicle",
        "fpd_ms",
        "beat_rate_bpm",
        "amplitude_uv",
        "stv",
        "triangulation_proxy",
        "n_electrodes",
        "noise_sd_uv",
        "beat_detection_rate",
    }
    blinova_required = {
        "Drug_Name",
        "Cell_type",
        "risk",
        "Platform",
        "Type_of_EADs",
        "conc",
        "EAD",
        "ddFPDc",
        "site",
    }
    classified: dict[str, list[Path] | Path | None] = {
        "blinova": None,
        "raw_csv": None,
        "feature_csv": None,
        "source_candidates": [],
        "reference": None,
        "source_receipt": None,
        "derivation_manifest": None,
    }
    for path in paths:
        lname = path.name.lower()
        if lname == "reference.csv":
            classified["reference"] = path
            continue
        if lname == "source_receipt.json":
            classified["source_receipt"] = path
            continue
        if lname == "derivation_manifest.json":
            classified["derivation_manifest"] = path
            continue
        if path.suffix.lower() in {".xlsx", ".xls"}:
            try:
                cols = xlsx_columns(path)
            except Exception as exc:
                log(f"Workbook inspection failed for {path.name}: {exc}")
                classified["source_candidates"].append(path)
                continue
            if blinova_required.issubset(cols):
                classified["blinova"] = path
            else:
                classified["source_candidates"].append(path)
            continue
        if path.suffix.lower() == ".csv":
            try:
                cols = csv_columns(path)
            except Exception as exc:
                log(f"CSV inspection failed for {path.name}: {exc}")
                classified["source_candidates"].append(path)
                continue
            if raw_required.issubset(cols):
                classified["raw_csv"] = path
            elif feature_required.issubset(cols):
                classified["feature_csv"] = path
            else:
                classified["source_candidates"].append(path)
            continue
        classified["source_candidates"].append(path)
    return classified


def make_source_receipt(source: Path, derived_dir: Path, source_id: str | None = None) -> Path:
    receipt = {
        "source_id": source_id or source.stem,
        "source_url": "user_uploaded_to_colab",
        "source_filename": source.name,
        "source_path": str(source),
        "byte_size": source.stat().st_size,
        "sha256": sha256_file(source),
        "inventory": file_inventory(source),
        "acquired_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    receipt_dir = derived_dir / "source_receipts"
    receipt_dir.mkdir(parents=True, exist_ok=True)
    safe_name = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in source.name)
    path = receipt_dir / f"{safe_name}.json"
    write_json(path, receipt)
    return path


def run_blinova(path: Path, derived_dir: Path) -> dict:
    import pandas as pd

    required = {
        "Drug_Name",
        "Cell_type",
        "risk",
        "Platform",
        "Type_of_EADs",
        "conc",
        "EAD",
        "ddFPDc",
        "site",
    }
    df = pd.read_excel(path)
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Blinova source is missing required columns: {missing}")

    # Canonicalize the known sotalol naming variants before counting compounds.
    # Preserve the raw Drug_Name column for provenance; never broadly fuzzy-match names.
    raw_names = df["Drug_Name"].astype(str).str.strip()
    canonical_names = (
        raw_names.str.lower()
        .str.replace(r"d[\s,.-]*l[\s,.-]*sotalol", "sotalol", regex=True)
        .str.replace(r"dl[\s,.-]*sotalol", "sotalol", regex=True)
    )
    raw_to_canonical = (
        pd.DataFrame({"raw": raw_names, "canonical": canonical_names})
        .drop_duplicates()
        .sort_values(["canonical", "raw"])
    )
    canonical_n = canonical_names.nunique()
    if canonical_n != 28:
        raise ValueError(
            f"Expected the released 28-drug panel after documented name normalization; "
            f"found {canonical_n} canonical compounds. Raw aliases: "
            f"{raw_to_canonical.to_dict(orient='records')}"
        )
    if df["site"].nunique() != 10:
        raise ValueError(f"Expected 10 sites, found {df['site'].nunique()}.")
    if not pd.to_numeric(df["conc"], errors="coerce").notna().all():
        raise ValueError("Blinova conc contains non-numeric values.")
    if not pd.to_numeric(df["ddFPDc"], errors="coerce").notna().all():
        raise ValueError("Blinova ddFPDc contains non-numeric values.")

    risk_map = {"l": "low", "m": "intermediate", "h": "high"}
    frame = df.copy()
    frame["compound"] = (
        frame["Drug_Name"]
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"d[\s,.-]*l[\s,.-]*sotalol", "sotalol", regex=True)
        .str.replace(r"dl[\s,.-]*sotalol", "sotalol", regex=True)
    )
    frame["raw_Drug_Name"] = frame["Drug_Name"]
    frame["reference_risk"] = (
        frame["risk"].astype(str).str.strip().str.lower().map(risk_map)
    )
    if frame["reference_risk"].isna().any():
        raise ValueError("Unexpected Blinova risk label; no mapping was invented.")

    platforms = set(frame["Platform"].astype(str).str.strip())
    allowed = {"AXN", "CLY", "ECR", "AMD", "MCS"}
    unexpected = sorted(platforms - allowed)
    if unexpected:
        raise ValueError(
            f"Unexpected Blinova platform code(s): {unexpected}. "
            "Do not silently relabel them."
        )

    event = frame["Type_of_EADs"].astype(str).str.strip().str.upper()
    broad = pd.to_numeric(frame["EAD"], errors="coerce")
    if broad.isna().any():
        raise ValueError("Blinova EAD contains non-numeric values.")
    frame["is_abcd_arrhythmia"] = event.isin({"A", "B", "C", "D"})
    frame["is_Q_quiescence"] = event.eq("Q")
    if ((frame["is_abcd_arrhythmia"] | frame["is_Q_quiescence"]) & ~broad.eq(1)).any():
        raise ValueError("Blinova event semantics are internally inconsistent.")

    for drug in ("verapamil", "terfenadine"):
        mask = frame["compound"].eq(drug)
        if mask.any() and int(frame.loc[mask, "is_abcd_arrhythmia"].sum()) != 0:
            raise ValueError(f"{drug} contains an unexpected A-D event; inspect the source.")

    reference = frame[["compound", "reference_risk"]].drop_duplicates()
    if reference.groupby("compound").size().max() != 1:
        raise ValueError("Blinova reference mapping is not one-to-one by compound.")

    reference_path = derived_dir / "blinova_reference.csv"
    semantic_path = derived_dir / "blinova_semantic_summary.csv"
    reference.to_csv(reference_path, index=False)
    frame[
        [
            "compound",
            "Drug_Name",
            "risk",
            "Platform",
            "site",
            "Cell_type",
            "conc",
            "ddFPDc",
            "EAD",
            "Type_of_EADs",
            "is_abcd_arrhythmia",
            "is_Q_quiescence",
        ]
    ].to_csv(semantic_path, index=False)

    return {
        "status": "complete",
        "reference_path": str(reference_path),
        "semantic_summary_path": str(semantic_path),
        "n_rows": int(len(frame)),
        "n_compounds": int(reference["compound"].nunique()),
        "n_sites": int(frame["site"].nunique()),
        "n_abcd_events": int(frame["is_abcd_arrhythmia"].sum()),
        "n_Q_events": int(frame["is_Q_quiescence"].sum()),
    }


def dataframe_sha256(df) -> str:
    payload = df.to_csv(index=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def run_raw_mea(path: Path, derived_dir: Path, results_dir: Path) -> dict:
    from virelion_cardioscore.io.raw_trace import load_raw_traces_to_feature_table

    first = load_raw_traces_to_feature_table(path)
    second = load_raw_traces_to_feature_table(path)
    first_hash = dataframe_sha256(first)
    second_hash = dataframe_sha256(second)
    verified = first_hash == second_hash
    if not verified:
        raise RuntimeError("Raw source -> feature extraction was not deterministic across two rebuilds.")

    feature_path = derived_dir / "raw_mea_features.csv"
    first.to_csv(feature_path, index=False)

    lineage = {
        "package_commit": PIN,
        "source_filename": path.name,
        "source_sha256": sha256_file(path),
        "canonical_features": str(feature_path),
        "canonical_features_sha256": sha256_file(feature_path),
        "rebuild_1_sha256": first_hash,
        "rebuild_2_sha256": second_hash,
        "verified_rebuild": True,
        "adapter": "virelion_cardioscore.io.raw_trace.load_raw_traces_to_feature_table",
        "manual_annotation_required_for_accuracy": True,
    }
    lineage_path = results_dir / "raw_mea_lineage.json"
    write_json(lineage_path, lineage)

    log(f"Raw MEA feature extraction: {len(first)} wells -> {feature_path.name}")
    return {
        "status": "complete",
        "feature_path": str(feature_path),
        "lineage_path": str(lineage_path),
        "n_feature_rows": int(len(first)),
        "verified_rebuild": True,
    }


def find_locked_assets(classified: dict, derived_dir: Path) -> dict[str, Path] | None:
    features = classified["feature_csv"]
    reference = classified["reference"]
    receipt = classified["source_receipt"]
    manifest = classified["derivation_manifest"]

    if features is None and (derived_dir / "raw_mea_features.csv").exists():
        features = derived_dir / "raw_mea_features.csv"
    if reference is None and (derived_dir / "blinova_reference.csv").exists():
        reference = derived_dir / "blinova_reference.csv"
    if receipt is None and (derived_dir / "source_receipt.json").exists():
        receipt = derived_dir / "source_receipt.json"

    if all(x is not None for x in (features, reference, receipt, manifest)):
        return {
            "features": features,
            "reference": reference,
            "receipt": receipt,
            "manifest": manifest,
        }
    return None


def config_sha256() -> str:
    import virelion_cardioscore

    cfg = Path(virelion_cardioscore.__file__).resolve().parent / "config" / "default.yaml"
    return sha256_file(cfg)


def run_locked_external(assets: dict[str, Path], results_dir: Path, input_dir: Path) -> dict:
    import pandas as pd
    import yaml

    from virelion_cardioscore import __file__ as package_file
    from virelion_cardioscore.analysis.pipeline import CardioScorePipeline
    from virelion_cardioscore.validation.manifest import (
        validate_feature_schema,
        validate_reference_schema,
        validate_vehicle_structure,
    )
    from virelion_cardioscore.validation.metrics import locked_metrics, stratified_failures

    features = pd.read_csv(assets["features"])
    reference = pd.read_csv(assets["reference"])
    receipt = json.loads(assets["receipt"].read_text(encoding="utf-8"))
    derivation = json.loads(assets["manifest"].read_text(encoding="utf-8"))

    assert receipt.get("sha256"), "Source receipt must contain sha256."
    assert receipt.get("source_filename"), "Source receipt must name the exact source file."
    source_path = input_dir / Path(str(receipt["source_filename"])).name
    assert source_path.is_file(), f"Source file named by receipt is not uploaded: {source_path.name}"
    assert sha256_file(source_path) == receipt["sha256"], "Uploaded source SHA-256 does not match source receipt."
    assert receipt.get("source_url") not in {None, "", "REPLACE_ME", "user_uploaded_to_colab"}, (
        "Locked validation requires an explicit authoritative source URL/identifier."
    )
    assert derivation.get("verified_rebuild") is True, "verified_rebuild=true is required."
    assert derivation.get("source_sha256") == receipt["sha256"], "Source hash mismatch."
    assert derivation.get("canonical_features_sha256") == sha256_file(assets["features"])
    assert derivation.get("reference_sha256") == sha256_file(assets["reference"])
    assert derivation.get("config_sha256") == config_sha256(), (
        "Locked feature/reference files were not derived under the current pinned default config."
    )
    reference_provenance = derivation.get("reference_provenance")
    assert isinstance(reference_provenance, dict), (
        "Locked validation requires reference_provenance metadata in derivation_manifest.json."
    )
    assert reference_provenance.get("source_url"), (
        "reference_provenance.source_url is required so the reference labels have documented provenance."
    )
    assert reference_provenance.get("source_id"), (
        "reference_provenance.source_id is required so the reference labels have documented provenance."
    )

    validate_feature_schema(features)
    validate_reference_schema(reference)

    package_dir = Path(package_file).resolve().parent
    config_path = package_dir / "config" / "default.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if bool(config.get("scoring", {}).get("normalize_by_vehicle", True)):
        validate_vehicle_structure(features)

    leakage = [
        col
        for col in features.columns
        if "reference_risk" in col.lower() or "risk_class" in col.lower()
    ]
    if leakage:
        raise ValueError(f"Leakage-prone label column(s) in feature table: {leakage}")

    pipeline = CardioScorePipeline.from_defaults()
    result = pipeline.run(features)
    observed = result.summary_table[
        ["compound", "risk_class", "cardioscore"]
    ].rename(columns={"risk_class": "observed_risk"})
    joined = reference.merge(
        observed, on="compound", how="inner", validate="one_to_one"
    )

    excluded = sorted(set(reference["compound"]) - set(joined["compound"]))
    payload = {
        "package_commit": PIN,
        "locked": True,
        "status": "complete" if not excluded else "blocked_by_exclusions",
        "source_id": receipt.get("source_id"),
        "source_sha256": receipt.get("sha256"),
        "derivation_manifest": derivation,
        "excluded_or_unscoreable_compounds": excluded,
        "qc_log": result.qc_log,
    }

    failures = pd.DataFrame()
    if not excluded:
        metrics = locked_metrics(
            joined["reference_risk"], joined["observed_risk"]
        )
        failures = stratified_failures(joined, strata=("compound",))
        payload["metrics"] = metrics.to_dict()
        payload["compound_scores"] = joined.to_dict(orient="records")
        payload["failures"] = failures.to_dict(orient="records")
        joined.to_csv(results_dir / "locked_compound_validation.csv", index=False)
        failures.to_csv(results_dir / "locked_failures_by_compound.csv", index=False)
        log(json.dumps(metrics.to_dict(), indent=2))
    else:
        payload["metrics"] = None
        payload["compound_scores"] = joined.to_dict(orient="records")
        payload["failures"] = []
        joined.to_csv(results_dir / "locked_compound_validation.csv", index=False)
        log("Headline external metrics BLOCKED because reference compounds were not scoreable.")
        log("Excluded compounds: " + ", ".join(excluded))

    write_json(results_dir / "locked_external_validation.json", payload)
    return {
        "status": payload["status"],
        "result_path": str(results_dir / "locked_external_validation.json"),
        "n_reference": int(len(reference)),
        "n_scored": int(len(joined)),
        "n_excluded": int(len(excluded)),
    }


def load_endpoint_config() -> dict:
    from virelion_cardioscore import __file__ as package_file
    import yaml

    endpoint_cfg = Path(package_file).resolve().parent / "config" / "cipa_endpoints.yaml"
    return yaml.safe_load(endpoint_cfg.read_text(encoding="utf-8"))


def score_with_endpoint_config(features, base, endpoint_cfg: dict, work_dir: Path):
    import yaml

    cfg = copy.deepcopy(base.config)
    key = hashlib.sha256(
        json.dumps(endpoint_cfg, sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]
    endpoint_path = work_dir / f"endpoints_{key}.yaml"
    endpoint_path.write_text(yaml.safe_dump(endpoint_cfg, sort_keys=False), encoding="utf-8")
    cfg["scoring"]["endpoint_config"] = str(endpoint_path)
    pipeline = __import__(
        "virelion_cardioscore.analysis.pipeline",
        fromlist=["CardioScorePipeline"],
    ).CardioScorePipeline(cfg)
    return pipeline.run(features).summary_table[
        ["compound", "cardioscore", "risk_class"]
    ].copy()


def run_robustness(assets: dict[str, Path], results_dir: Path, work_dir: Path) -> dict:
    import pandas as pd

    locked_path = results_dir / "locked_external_validation.json"
    if not locked_path.exists():
        return {"status": "blocked", "reason": "locked_external_validation.json is missing."}
    locked = json.loads(locked_path.read_text(encoding="utf-8"))
    if locked.get("status") != "complete":
        return {
            "status": "blocked",
            "reason": "Primary locked external result is not complete.",
        }

    features = pd.read_csv(assets["features"])
    from virelion_cardioscore.analysis.pipeline import CardioScorePipeline

    base = CardioScorePipeline.from_defaults()
    endpoint_cfg = load_endpoint_config()
    baseline = base.run(features).summary_table[
        ["compound", "cardioscore", "risk_class"]
    ].copy()

    scenarios = {
        "baseline": {k: float(v["weight"]) for k, v in endpoint_cfg["endpoints"].items()},
        "fpd_only": {
            "fpd_change_pct": 1,
            "beat_rate_change_pct": 0,
            "amplitude_change_pct": 0,
            "stv_increase": 0,
            "triangulation_proxy": 0,
        },
        "no_fpd": {
            "fpd_change_pct": 0,
            "beat_rate_change_pct": 0.30,
            "amplitude_change_pct": 0.20,
            "stv_increase": 0.30,
            "triangulation_proxy": 0.20,
        },
    }

    tables = []
    for name, weights in scenarios.items():
        if name == "baseline":
            current = baseline
        else:
            cfg_copy = copy.deepcopy(endpoint_cfg)
            for endpoint, weight in weights.items():
                cfg_copy["endpoints"][endpoint]["weight"] = float(weight)
            total = sum(float(v["weight"]) for v in cfg_copy["endpoints"].values())
            for meta in cfg_copy["endpoints"].values():
                meta["weight"] = float(meta["weight"]) / total
            current = score_with_endpoint_config(features, base, cfg_copy, work_dir)
        current = current.rename(
            columns={
                "cardioscore": f"score_{name}",
                "risk_class": f"risk_{name}",
            }
        )
        tables.append(current)

    sensitivity = tables[0]
    for table in tables[1:]:
        sensitivity = sensitivity.merge(
            table, on="compound", how="inner", validate="one_to_one"
        )

    for name, factor in (("thresholds_minus10", 0.90), ("thresholds_plus10", 1.10)):
        cfg_copy = copy.deepcopy(endpoint_cfg)
        for meta in cfg_copy["endpoints"].values():
            meta["effect_threshold"] = float(meta["effect_threshold"]) * factor
        current = score_with_endpoint_config(features, base, cfg_copy, work_dir)
        current = current.rename(
            columns={
                "cardioscore": f"{name}_score",
                "risk_class": f"{name}_risk",
            }
        )
        sensitivity = sensitivity.merge(
            current, on="compound", how="inner", validate="one_to_one"
        )

    sensitivity.to_csv(results_dir / "secondary_sensitivity.csv", index=False)
    return {
        "status": "complete",
        "sensitivity_path": str(results_dir / "secondary_sensitivity.csv"),
        "n_compounds": int(len(sensitivity)),
    }


def run_qc_summary(
    assets: dict[str, Path] | None,
    results_dir: Path,
) -> dict:
    import pandas as pd

    features_path = assets["features"] if assets else None
    reference_path = assets["reference"] if assets else None
    if features_path is None or reference_path is None:
        return {"status": "blocked", "reason": "Feature/reference assets are unavailable."}

    from virelion_cardioscore.analysis.pipeline import CardioScorePipeline

    features = pd.read_csv(features_path)
    reference = pd.read_csv(reference_path)
    pipeline = CardioScorePipeline.from_defaults()
    result = pipeline.run(features)
    summary = {
        "n_input_rows": int(len(features)),
        "n_scored_compounds": int(len(result.summary_table)),
        "n_reference_compounds": int(len(reference)),
        "n_excluded_reference_compounds": int(
            len(set(reference["compound"]) - set(result.summary_table["compound"]))
        ),
        "qc_log_entries": int(len(result.qc_log)),
    }
    pd.DataFrame({"qc_log": result.qc_log}).to_csv(
        results_dir / "qc_log.csv", index=False
    )
    write_json(results_dir / "qc_summary.json", summary)
    return {"status": "complete", **summary}


def publish_results(results_dir: Path, run_label: str) -> dict:
    try:
        from google.colab import userdata

        token = userdata.get("GITHUB_TOKEN")
    except Exception:
        token = None
    if not token:
        return {
            "status": "blocked",
            "reason": "Colab Secret GITHUB_TOKEN is missing.",
        }

    artifact_files = [p for p in results_dir.iterdir() if p.is_file()]
    unexpected = [
        p.name for p in artifact_files if p.name not in ALLOWED_PUBLISH_NAMES
    ]
    if unexpected:
        raise ValueError(f"Refusing to publish unapproved result file(s): {unexpected}")
    if not artifact_files:
        return {"status": "blocked", "reason": "No publishable result artifacts exist."}

    publish_work = Path(tempfile.mkdtemp(prefix="cardioscore-github-publish-"))
    askpass = publish_work / "askpass.sh"
    askpass.write_text(
        "#!/bin/sh\n"
        "case \"$1\" in\n"
        "  *Username*) printf '%s\\n' \"x-access-token\" ;;\n"
        "  *Password*) printf '%s\\n' \"$GITHUB_TOKEN\" ;;\n"
        "  *) exit 1 ;;\n"
        "esac\n",
        encoding="utf-8",
    )
    askpass.chmod(0o700)

    env = os.environ.copy()
    env["GITHUB_TOKEN"] = token
    env["GIT_ASKPASS"] = str(askpass)
    env["GIT_TERMINAL_PROMPT"] = "0"

    repo_dir = publish_work / "repo"
    subprocess.run(
        [
            "git",
            "clone",
            "--depth",
            "1",
            "--branch",
            BRANCH,
            f"https://github.com/{REPO}.git",
            str(repo_dir),
        ],
        env=env,
        check=True,
    )
    before = subprocess.check_output(
        ["git", "-C", str(repo_dir), "rev-parse", "HEAD"], text=True
    ).strip()

    destination = repo_dir / "validation_results" / run_label
    destination.mkdir(parents=True, exist_ok=False)
    for path in artifact_files:
        shutil.copy2(path, destination / path.name)

    manifest = {
        "run_label": run_label,
        "cardioscore_package_pin": PIN,
        "main_commit_at_publish_clone": before,
        "artifacts": {},
    }
    for path in sorted(destination.iterdir()):
        if path.is_file():
            manifest["artifacts"][path.name] = {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
    write_json(destination / "PUBLISH_MANIFEST.json", manifest)

    status = subprocess.check_output(
        ["git", "-C", str(repo_dir), "status", "--short", "--untracked-files=all"],
        text=True,
    )
    changed = [line[3:] for line in status.splitlines() if line.strip()]
    expected_prefix = f"validation_results/{run_label}/"
    bad = [path for path in changed if not path.startswith(expected_prefix)]
    if bad:
        raise RuntimeError(f"Safety stop: unexpected Git changes: {bad}")

    subprocess.run(
        ["git", "-C", str(repo_dir), "config", "user.name", "Syed Umer Hannan"],
        check=True,
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(repo_dir),
            "config",
            "user.email",
            "syedumerhannan@icloud.com",
        ],
        check=True,
    )
    commit_message = f"Publish CardioScore validation results: {run_label}"
    subprocess.run(
        ["git", "-C", str(repo_dir), "add", f"validation_results/{run_label}"],
        env=env,
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repo_dir), "commit", "-m", commit_message],
        env=env,
        check=True,
    )
    local_commit = subprocess.check_output(
        ["git", "-C", str(repo_dir), "rev-parse", "HEAD"], text=True
    ).strip()
    subprocess.run(
        ["git", "-C", str(repo_dir), "push", "origin", BRANCH],
        env=env,
        check=True,
    )
    remote_commit = subprocess.check_output(
        ["git", "-C", str(repo_dir), "rev-parse", "origin/main"], text=True
    ).strip()
    if remote_commit != local_commit:
        raise RuntimeError(
            f"Remote verification failed: origin/main={remote_commit}, local={local_commit}"
        )
    return {
        "status": "complete",
        "commit": local_commit,
        "destination": f"validation_results/{run_label}",
    }


def main() -> None:
    install_pinned_package()
    paths = prepare_workspace()
    uploaded = upload_inputs(paths["input"])
    classified = classify_inputs(uploaded)

    run_manifest = {
        "run_label": RUN_LABEL,
        "package_pin": PIN,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_files": {
            p.name: {
                "bytes": p.stat().st_size,
                "sha256": sha256_file(p),
            }
            for p in uploaded
        },
        "stages": {},
    }
    write_json(paths["run"] / "run_manifest.json", run_manifest)

    # Intake: receipt for every plausible authoritative source.
    source_candidates = list(classified["source_candidates"])
    if classified["blinova"] is not None:
        source_candidates.append(classified["blinova"])
    if classified["raw_csv"] is not None:
        source_candidates.append(classified["raw_csv"])
    receipt_paths = []
    for source in source_candidates:
        receipt = make_source_receipt(source, paths["derived"])
        receipt_paths.append(receipt)
        log(f"Source receipt: {receipt}")
    if len(receipt_paths) == 1:
        shutil.copy2(receipt_paths[0], paths["derived"] / "source_receipt.json")

    # Stage 01: Blinova/CiPA semantic/component validation.
    if classified["blinova"] is not None:
        try:
            run_manifest["stages"]["blinova_summary"] = run_blinova(
                classified["blinova"], paths["derived"]
            )
        except Exception as exc:
            run_manifest["stages"]["blinova_summary"] = {
                "status": "failed",
                "error": str(exc),
            }
            log(f"Blinova stage FAILED: {exc}")

    # Stage 02: raw MEA -> features. Unsupported source formats stop here.
    raw_path = classified["raw_csv"]
    if raw_path is not None:
        try:
            run_manifest["stages"]["raw_mea"] = run_raw_mea(
                raw_path, paths["derived"], paths["results"]
            )
        except Exception as exc:
            run_manifest["stages"]["raw_mea"] = {
                "status": "failed",
                "error": str(exc),
            }
            log(f"Raw MEA stage FAILED: {exc}")
    elif classified["source_candidates"]:
        run_manifest["stages"]["raw_mea"] = {
            "status": "blocked",
            "reason": "Source format not supported by the current generic raw-trace runner. Build/review a source-specific adapter instead of guessing."
        }

    # Refresh receipt/manifest locations from derived outputs.
    locked_assets = find_locked_assets(classified, paths["derived"])

    # Stage 03: locked external test only when explicit lineage is supplied.
    if locked_assets is not None:
        try:
            run_manifest["stages"]["locked_external"] = run_locked_external(
                locked_assets, paths["results"]
            )
        except Exception as exc:
            run_manifest["stages"]["locked_external"] = {
                "status": "failed",
                "error": str(exc),
            }
            log(f"Locked external stage FAILED: {exc}")
    else:
        run_manifest["stages"]["locked_external"] = {
            "status": "blocked",
            "reason": "Need canonical_features.csv, reference.csv, source_receipt.json, and derivation_manifest.json with verified_rebuild=true."
        }

    # Stage 04: secondary robustness only after a complete primary result.
    if locked_assets is not None:
        try:
            run_manifest["stages"]["robustness"] = run_robustness(
                locked_assets, paths["results"], paths["work"]
            )
        except Exception as exc:
            run_manifest["stages"]["robustness"] = {
                "status": "failed",
                "error": str(exc),
            }
            log(f"Robustness stage FAILED: {exc}")
        try:
            run_manifest["stages"]["qc"] = run_qc_summary(
                locked_assets, paths["results"]
            )
        except Exception as exc:
            run_manifest["stages"]["qc"] = {
                "status": "failed",
                "error": str(exc),
            }
            log(f"QC stage FAILED: {exc}")
    else:
        run_manifest["stages"]["robustness"] = {
            "status": "blocked",
            "reason": "Primary locked external result is unavailable.",
        }

    run_manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(paths["run"] / "run_manifest.json", run_manifest)

    # Surface a compact state report.
    print("\n=== CardioScore validation state ===")
    for stage, info in run_manifest["stages"].items():
        print(f"{stage:20s} {info.get('status', 'unknown')}")
    print(f"Run workspace: {paths['run']}")
    print(f"Results:        {paths['results']}")

    if PUBLISH_RESULTS:
        try:
            publish = publish_results(paths["results"], RUN_LABEL)
        except Exception as exc:
            publish = {"status": "failed", "error": str(exc)}
        run_manifest["stages"]["publish"] = publish
        write_json(paths["run"] / "run_manifest.json", run_manifest)
        print(f"publish              {publish.get('status')}")
        if publish.get("commit"):
            print(f"GitHub commit        {publish['commit']}")
    else:
        run_manifest["stages"]["publish"] = {
            "status": "skipped",
            "reason": "CARDIOSCORE_PUBLISH is disabled.",
        }
        write_json(paths["run"] / "run_manifest.json", run_manifest)


if __name__ == "__main__":
    main()

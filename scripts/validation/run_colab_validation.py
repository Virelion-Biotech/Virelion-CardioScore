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

def resolve_runner_revision(environ=None) -> tuple[str, str | None]:
    """Return (runner commit SHA or 'unpinned-main', runner-source SHA-256)."""
    environ = os.environ if environ is None else environ
    sha = str(environ.get("CARDIOSCORE_RUNNER_SHA", "")).strip().lower()
    source_sha = str(environ.get("CARDIOSCORE_RUNNER_SOURCE_SHA256", "")).strip().lower() or None
    hex_digits = set("0123456789abcdef")
    if sha and not (len(sha) == 40 and set(sha) <= hex_digits):
        raise ValueError("CARDIOSCORE_RUNNER_SHA must be a full 40-character commit SHA.")
    if source_sha and not (len(source_sha) == 64 and set(source_sha) <= hex_digits):
        raise ValueError("CARDIOSCORE_RUNNER_SOURCE_SHA256 must be a 64-character hex digest.")
    return (sha or "unpinned-main", source_sha)


RUNNER_REVISION, RUNNER_SOURCE_SHA256 = resolve_runner_revision()
PIN = "fdada43a8e07199f4eedccf7f8b941cd6cb351e8"
BRANCH = "main"
ROOT = Path("/content/cardioscore_validation")
RUN_LABEL = os.environ.get("CARDIOSCORE_RUN_LABEL", "").strip()
PUBLISH_RESULTS = os.environ.get("CARDIOSCORE_PUBLISH", "1").strip().lower() in {"1", "true", "yes", "y"}
ALLOWED_PUBLISH_NAMES = {
    "locked_external_validation.json",
    "locked_compound_validation.csv",
    "locked_failures_by_compound.csv",
    "locked_dropout_by_concentration.csv",
    "locked_exclusions.csv",
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
    log(f"Runner revision {RUNNER_REVISION}")
    if RUNNER_REVISION == "unpinned-main":
        log(
            "WARNING: runner was fetched from a moving branch, so this run cannot identify "
            "the runner code. Use the SHA-pinned launcher in VALIDATION_EXECUTION.md."
        )
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


# ---------------------------------------------------------------------------
# Blinova 2018 workbook contract
#
# Structural problems stop the stage. Documented source quirks are retained as
# explicit audit flags and never rewritten or relabeled.
# ---------------------------------------------------------------------------
BLINOVA_REQUIRED_COLUMNS = (
    "Drug_Name", "Cell_type", "risk", "Platform", "Type_of_EADs",
    "conc", "EAD", "ddFPDc", "site",
)
BLINOVA_PANEL_SIZE = 28
BLINOVA_N_SITES = 10
BLINOVA_NAME_ALIASES = {
    "d,l sotalol": "sotalol",
    "d,l,sotalol": "sotalol",
}
BLINOVA_RISK_MAP = {"l": "low", "m": "intermediate", "h": "high"}
BLINOVA_DOCUMENTED_PLATFORMS = ("AXN", "CLY", "ECR", "AMD", "MCS")
BLINOVA_ARRHYTHMIA_EVENT_TYPES = ("A", "B", "C", "D")
BLINOVA_QUIESCENCE_EVENT_TYPE = "Q"
# Type_of_EADs may combine letters (e.g. AC, ABC). A-D are the paper's four arrhythmia-like event
# types and Q is quiescence; any other letter or non-letter character is preserved and flagged,
# never interpreted.
BLINOVA_KNOWN_EVENT_LETTERS = frozenset("ABCDQ")
BLINOVA_NO_EVENT_CLAIMS = ("terfenadine", "verapamil")
BLINOVA_MAX_LISTED_ROWS = 25


def _blinova_flag(code: str, severity: str, message: str, **detail) -> dict:
    return {"code": code, "severity": severity, "message": message, **detail}


def _excel_rows(index) -> list[int]:
    return [int(i) + 2 for i in list(index)[:BLINOVA_MAX_LISTED_ROWS]]


def _count_by(series) -> dict[str, int]:
    return {str(k): int(v) for k, v in series.value_counts().sort_index().items()}


def audit_blinova_frame(df, derived_dir: Path) -> dict:
    import pandas as pd

    missing_cols = sorted(set(BLINOVA_REQUIRED_COLUMNS) - set(df.columns))
    if missing_cols:
        raise ValueError(f"Blinova source is missing required columns: {missing_cols}")
    if df.empty:
        raise ValueError("Blinova source has no rows.")

    frame = df.reset_index(drop=True).copy()
    flags: list[dict] = []

    # Compound identity
    if frame["Drug_Name"].isna().any():
        raise ValueError(
            f"Blinova Drug_Name is missing in spreadsheet row(s) "
            f"{_excel_rows(frame.index[frame['Drug_Name'].isna()])}."
        )
    raw_names = frame["Drug_Name"].astype(str).str.strip()
    frame["compound"] = raw_names.str.lower().replace(BLINOVA_NAME_ALIASES)
    n_canonical = int(frame["compound"].nunique())
    if n_canonical != BLINOVA_PANEL_SIZE:
        mapping = (
            pd.DataFrame({"raw": raw_names, "canonical": frame["compound"]})
            .drop_duplicates()
            .sort_values(["canonical", "raw"])
        )
        raise ValueError(
            f"Expected the released {BLINOVA_PANEL_SIZE}-drug panel after documented name "
            f"normalization; found {n_canonical} canonical compounds. "
            f"Raw aliases: {mapping.to_dict(orient='records')}"
        )

    if frame["site"].isna().any():
        raise ValueError(
            f"Blinova site is missing in spreadsheet row(s) "
            f"{_excel_rows(frame.index[frame['site'].isna()])}."
        )
    site = pd.to_numeric(frame["site"], errors="coerce")
    if site.isna().any():
        raise ValueError(
            f"Blinova site contains missing or non-numeric values in spreadsheet row(s) "
            f"{_excel_rows(frame.index[site.isna()])}."
        )
    if int(site.nunique()) != BLINOVA_N_SITES:
        raise ValueError(f"Expected {BLINOVA_N_SITES} sites, found {int(site.nunique())}.")

    # Numeric fields: conc/EAD are complete; ddFPDc may legitimately be missing.
    conc = pd.to_numeric(frame["conc"], errors="coerce")
    if conc.isna().any():
        raise ValueError(
            "Blinova conc contains missing or non-numeric values in spreadsheet row(s) "
            f"{_excel_rows(frame.index[conc.isna()])}."
        )
    ead = pd.to_numeric(frame["EAD"], errors="coerce")
    if ead.isna().any():
        raise ValueError(
            "Blinova EAD contains missing or non-numeric values in spreadsheet row(s) "
            f"{_excel_rows(frame.index[ead.isna()])}."
        )
    dd = pd.to_numeric(frame["ddFPDc"], errors="coerce")
    dd_malformed = frame["ddFPDc"].notna() & dd.isna()
    if dd_malformed.any():
        examples = sorted({str(v) for v in frame.loc[dd_malformed, "ddFPDc"]})[:10]
        raise ValueError(f"Blinova ddFPDc contains malformed non-numeric value(s): {examples}")
    n_dd_missing = int(dd.isna().sum())

    # Risk: blanks are audit flags; labeled rows alone define the compound reference.
    def _clean(value):
        if pd.isna(value) or str(value).strip() == "":
            return None
        return str(value).strip().lower()

    risk_text = frame["risk"].map(_clean)
    risk_missing = risk_text.isna()
    unknown_risk = risk_text.notna() & ~risk_text.isin(list(BLINOVA_RISK_MAP))
    if unknown_risk.any():
        raise ValueError(
            f"Unexpected Blinova risk label(s) {sorted(set(risk_text[unknown_risk]))}; "
            "no mapping was invented."
        )
    frame["reference_risk"] = risk_text.map(BLINOVA_RISK_MAP)
    if risk_missing.any():
        flags.append(
            _blinova_flag(
                "MISSING_RISK_ROWS",
                "warning",
                "Source rows have no risk label. Labels were NOT injected into these rows; "
                "the compound reference uses only labeled rows of the same compound.",
                n_rows=int(risk_missing.sum()),
                spreadsheet_rows=_excel_rows(frame.index[risk_missing]),
                rows_by_compound=_count_by(frame.loc[risk_missing, "compound"]),
            )
        )

    # Platform/cell type: preserve unexpected values, never silently relabel.
    def _label(value):
        if pd.isna(value) or str(value).strip() == "":
            return "<missing>"
        return str(value).strip()

    platform = frame["Platform"].map(_label)
    cell_type = frame["Cell_type"].map(_label)
    dataset_key = frame["site"].astype(str) + "|" + cell_type  # site x cell-type dataset
    documented_platform = platform.isin(list(BLINOVA_DOCUMENTED_PLATFORMS))
    if (~documented_platform).any():
        odd = platform[~documented_platform]
        flags.append(
            _blinova_flag(
                "UNDOCUMENTED_PLATFORM_CODE",
                "warning",
                "Platform code(s) outside the paper's documented set. Codes were NOT relabeled.",
                n_rows=int(len(odd)),
                rows_by_code=_count_by(odd),
                rows_by_dataset=_count_by(dataset_key[odd.index]),
                site_celltype_datasets=sorted(set(dataset_key[odd.index])),
                compounds_affected=sorted({str(c) for c in frame.loc[odd.index, "compound"]}),
            )
        )
    if (cell_type == "<missing>").any():
        flags.append(
            _blinova_flag(
                "MISSING_CELL_TYPE_ROWS",
                "warning",
                "Rows without Cell_type; preserved as '<missing>'.",
                n_rows=int((cell_type == "<missing>").sum()),
                spreadsheet_rows=_excel_rows(frame.index[cell_type == "<missing>"]),
            )
        )

    # Event semantics. Labels are parsed as letter sets: A-D = arrhythmia-like types (combinations
    # such as AC or ABC are several types in one well), Q = quiescence. Anything else is kept
    # verbatim and flagged as unresolved.
    known = BLINOVA_KNOWN_EVENT_LETTERS
    abcd = frozenset(BLINOVA_ARRHYTHMIA_EVENT_TYPES)
    event = frame["Type_of_EADs"].map(
        lambda v: "" if pd.isna(v) else str(v).strip().upper()
    )
    has_label = event.ne("")
    letters = event.map(lambda v: frozenset(v) if v.isalpha() else frozenset())
    is_abcd = letters.map(lambda ls: bool(ls & abcd))
    is_q = letters.map(lambda ls: BLINOVA_QUIESCENCE_EVENT_TYPE in ls)
    unresolved = has_label & (~event.map(str.isalpha) | letters.map(lambda ls: bool(ls - known)))
    inconsistent = has_label & ~ead.eq(1)
    if inconsistent.any():
        raise ValueError(
            "Blinova event semantics are internally inconsistent: "
            f"{int(inconsistent.sum())} row(s) carry a Type_of_EADs label but EAD != 1 "
            f"(spreadsheet rows {_excel_rows(frame.index[inconsistent])})."
        )
    if unresolved.any():
        flags.append(
            _blinova_flag(
                "EVENT_LABEL_UNRESOLVED",
                "warning",
                "Type_of_EADs labels with letters other than A-D/Q or non-letter characters "
                "are kept verbatim and NOT interpreted. Confirm their meaning in the paper's "
                "Data S1 legend before using them.",
                n_rows=int(unresolved.sum()),
                counts=_count_by(event[unresolved]),
                rows_by_compound=_count_by(frame.loc[unresolved, "compound"]),
            )
        )
    ead1_unlabeled = ead.eq(1) & ~has_label
    if ead1_unlabeled.any():
        flags.append(
            _blinova_flag(
                "EAD_FLAG_WITHOUT_EVENT_TYPE",
                "warning",
                "EAD == 1 rows with no Type_of_EADs label; EAD semantics remain unresolved.",
                n_rows=int(ead1_unlabeled.sum()),
                rows_by_compound=_count_by(frame.loc[ead1_unlabeled, "compound"]),
            )
        )
    elif has_label.any() and ead.eq(1).equals(has_label):
        flags.append(
            _blinova_flag(
                "EAD_EQUALS_ANY_EVENT_LABEL",
                "info",
                "EAD == 1 exactly when Type_of_EADs is non-blank (Q quiescence included), so EAD "
                "means 'any event label', not 'arrhythmia'. Use the A-D letters for "
                "arrhythmia-like events.",
                n_rows=int(has_label.sum()),
            )
        )

    # Published-claim cross-check: warning only.
    dataset_key = frame["site"].astype(str) + "|" + cell_type  # site x cell-type dataset
    frame["_ead1"] = ead.eq(1)
    q_only = is_q & ~is_abcd & ~unresolved
    for compound in BLINOVA_NO_EVENT_CLAIMS:
        mask = frame["compound"].eq(compound)
        if not mask.any():
            continue
        stats = {
            "n_rows": int(mask.sum()),
            "n_site_celltype_datasets": int(dataset_key[mask].nunique()),
            "rows_abcd_events": int((mask & is_abcd).sum()),
            "datasets_with_abcd_events": int(dataset_key[mask & is_abcd].nunique()),
            "rows_Q_events": int((mask & is_q).sum()),
            "rows_Q_only": int((mask & q_only).sum()),
            "rows_EAD_eq_1": int((mask & frame["_ead1"]).sum()),
            "datasets_with_EAD_eq_1": int(dataset_key[mask & frame["_ead1"]].nunique()),
        }
        if stats["rows_abcd_events"] > 0:
            flags.append(
                _blinova_flag(
                    "PUBLISHED_CLAIM_MISMATCH",
                    "warning",
                    f"Paper reports no arrhythmia-like events for {compound}; workbook has A-D labels "
                    "(including combinations).",
                    compound=compound,
                    **stats,
                )
            )
        elif stats["rows_EAD_eq_1"] > 0 and stats["rows_EAD_eq_1"] == stats["rows_Q_only"]:
            flags.append(
                _blinova_flag(
                    "EAD_EXPLAINED_BY_QUIESCENCE",
                    "info",
                    f"Paper reports no arrhythmia-like events for {compound}; every EAD == 1 "
                    "row is a Q (quiescence) label, so the workbook is consistent with that claim.",
                    compound=compound,
                    **stats,
                )
            )
        elif stats["rows_EAD_eq_1"] > 0:
            flags.append(
                _blinova_flag(
                    "EAD_FLAG_WITHOUT_ARRHYTHMIA_TYPE",
                    "warning",
                    f"Paper reports no arrhythmia-like events for {compound}; EAD == 1 rows "
                    "exist that are neither A-D nor solely Q labels.",
                    compound=compound,
                    **stats,
                )
            )
    frame = frame.drop(columns="_ead1")

    # Compound-level reference: only labeled rows, no imputation.
    labeled = frame.loc[frame["reference_risk"].notna(), ["compound", "reference_risk"]].drop_duplicates()
    conflicts = labeled.groupby("compound")["reference_risk"].nunique()
    conflicts = conflicts[conflicts > 1]
    if len(conflicts):
        raise ValueError(
            f"Blinova reference mapping is not one-to-one by compound: {sorted(conflicts.index)}"
        )
    unlabeled = sorted(set(frame["compound"]) - set(labeled["compound"]))
    if unlabeled:
        raise ValueError(
            f"Compound(s) with no labeled risk row; no label was invented: {unlabeled}"
        )
    reference = labeled.sort_values("compound").reset_index(drop=True)

    frame["platform_documented"] = documented_platform
    frame["risk_missing"] = risk_missing
    frame["is_abcd_arrhythmia"] = is_abcd
    frame["is_Q_quiescence"] = is_q
    frame["event_label_unresolved"] = unresolved

    reference_path = derived_dir / "blinova_reference.csv"
    semantic_path = derived_dir / "blinova_semantic_summary.csv"
    flags_path = derived_dir / "blinova_audit_flags.json"
    reference.to_csv(reference_path, index=False)
    frame[
        [
            "compound", "Drug_Name", "risk", "Platform", "site", "Cell_type",
            "conc", "ddFPDc", "EAD", "Type_of_EADs",
            "is_abcd_arrhythmia", "is_Q_quiescence", "event_label_unresolved",
            "platform_documented", "risk_missing",
        ]
    ].to_csv(semantic_path, index=False)
    write_json(
        flags_path,
        {
            "n_rows": int(len(frame)),
            "n_compounds": int(reference["compound"].nunique()),
            "flags": flags,
        },
    )

    return {
        "status": "complete",
        "reference_path": str(reference_path),
        "semantic_summary_path": str(semantic_path),
        "flags_path": str(flags_path),
        "n_rows": int(len(frame)),
        "n_compounds": int(reference["compound"].nunique()),
        "n_sites": int(site.nunique()),
        # Rows carrying any A-D letter (combinations such as AC/ABC included) / any Q label.
        "n_abcd_events": int(is_abcd.sum()),
        "n_Q_events": int(is_q.sum()),
        "n_unresolved_event_label_rows": int(unresolved.sum()),
        "event_label_counts": _count_by(event[has_label]),
        "n_ddFPDc_missing": n_dd_missing,
        "ddFPDc_missing_fraction": float(n_dd_missing / len(frame)),
        "n_flags": len(flags),
        "flags": flags,
    }


def run_blinova(path: Path, derived_dir: Path) -> dict:
    import pandas as pd
    return audit_blinova_frame(pd.read_excel(path), derived_dir)


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


FREEZE_REPO_PATHS = ("validation/preregistration.yaml", "validation/freeze_manifest.json")


def _http_get(url: str) -> bytes:
    import urllib.request

    with urllib.request.urlopen(url, timeout=60) as response:  # noqa: S310 - fixed https GitHub URL
        return response.read()


def load_freeze_files(
    environ=None, fetch=None, runner_revision: str | None = None
) -> tuple[bytes, dict, str]:
    """Return (pre-registration bytes, freeze manifest, source description).

    Files come from the immutable GitHub commit the runner itself was fetched from, so the commit
    history shows when the plan was frozen. ``CARDIOSCORE_FREEZE_DIR`` overrides this for offline
    or test use and is recorded as a local override.
    """
    from virelion_cardioscore.validation.freeze import load_manifest

    environ = os.environ if environ is None else environ
    override = str(environ.get("CARDIOSCORE_FREEZE_DIR", "")).strip()
    if override:
        base = Path(override)
        prereg_bytes = (base / "preregistration.yaml").read_bytes()
        manifest = load_manifest((base / "freeze_manifest.json").read_bytes())
        return prereg_bytes, manifest, f"local_override:{base}"
    revision = RUNNER_REVISION if runner_revision is None else runner_revision
    if revision == "unpinned-main":
        raise RuntimeError(
            "The runner is not SHA-pinned, so the freeze cannot be read from an immutable commit."
        )
    fetch = fetch or _http_get
    base_url = f"https://raw.githubusercontent.com/{REPO}/{revision}/"
    prereg_bytes = fetch(base_url + FREEZE_REPO_PATHS[0])
    manifest = load_manifest(fetch(base_url + FREEZE_REPO_PATHS[1]))
    return prereg_bytes, manifest, f"github:{REPO}@{revision}"


def verify_locked_freeze(
    environ=None, fetch=None, runner_revision: str | None = None, package_sha: str | None = None
) -> tuple[dict, dict | None]:
    """Verify the pre-registration/freeze; returns (audit info, parsed pre-registration or None)."""
    import virelion_cardioscore

    try:
        from virelion_cardioscore.validation.freeze import parse_preregistration, verify_freeze

        prereg_bytes, manifest, source = load_freeze_files(environ, fetch, runner_revision)
    except ImportError as exc:
        return {
            "verified": False,
            "problems": [
                f"The installed CardioScore package (PIN {PIN}) has no freeze module ({exc}). "
                "Set PIN to a commit that contains virelion_cardioscore/validation/freeze.py."
            ],
        }, None
    except Exception as exc:
        return {"verified": False, "problems": [f"Freeze files unavailable: {exc}"]}, None
    config_dir = Path(virelion_cardioscore.__file__).resolve().parent / "config"
    problems = verify_freeze(
        manifest,
        prereg_bytes=prereg_bytes,
        config_dir=config_dir,
        package_sha=PIN if package_sha is None else package_sha,
    )
    info = {
        "verified": not problems,
        "problems": problems,
        "source": source,
        "validation_version": manifest.get("validation_version"),
        "frozen_at_utc": manifest.get("frozen_at_utc"),
        "package_sha": manifest.get("package_sha"),
        "preregistration_sha256": manifest.get("preregistration_sha256"),
    }
    return info, (parse_preregistration(prereg_bytes) if not problems else None)


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


def evaluate_primary_analysis(joined, prereg: dict, *, score_override=None) -> dict:
    """Pre-registered primary analysis (AUROC + compound bootstrap CI + three-outcome rule)."""
    from virelion_cardioscore.validation.metrics import (
        RISK_ORDER,
        binary_auroc_bootstrap,
        primary_outcome,
    )

    spec = prereg["primary_analysis"]
    positive_ordinals = {RISK_ORDER[str(name).strip().lower()] for name in spec["positive_class"]}
    positive = [
        RISK_ORDER[str(value).strip().lower()] in positive_ordinals for value in joined["reference_risk"]
    ]
    scores = joined["cardioscore"] if score_override is None else score_override
    ci = spec["confidence_interval"]
    result = binary_auroc_bootstrap(
        positive,
        scores,
        n_bootstrap=int(ci["n_bootstrap"]),
        seed=int(ci["seed"]),
        confidence=float(ci["confidence"]),
    )
    result["outcome"] = primary_outcome(result, spec["outcome_rule"])
    return result


def run_locked_external(
    assets: dict[str, Path],
    results_dir: Path,
    input_dir: Path,
    prereg: dict | None = None,
    freeze_info: dict | None = None,
) -> dict:
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
    exclusion_reasons = {
        str(row["compound"]): {"reason": str(row["reason"]), "detail": str(row["detail"])}
        for _, row in result.exclusion_table.iterrows()
    }
    for compound in excluded:
        exclusion_reasons.setdefault(
            str(compound), {"reason": "absent_from_feature_table", "detail": ""}
        )
    summary = result.summary_table
    informative = (
        sorted(
            set(summary.loc[summary["informative_dropout"].astype(bool), "compound"].astype(str))
            & set(joined["compound"].astype(str))
        )
        if "informative_dropout" in summary.columns
        else []
    )
    result.dropout_table.to_csv(results_dir / "locked_dropout_by_concentration.csv", index=False)
    result.exclusion_table.to_csv(results_dir / "locked_exclusions.csv", index=False)
    payload = {
        "package_commit": PIN,
        "locked": True,
        "status": "complete" if not excluded else "blocked_by_exclusions",
        "source_id": receipt.get("source_id"),
        "source_sha256": receipt.get("sha256"),
        "derivation_manifest": derivation,
        "freeze": freeze_info,
        "excluded_or_unscoreable_compounds": excluded,
        "exclusion_reasons": exclusion_reasons,
        "informative_dropout_compounds": informative,
        "primary_analysis": None,
        "dropout_sensitivity": None,
        "qc_log": result.qc_log,
    }

    failures = pd.DataFrame()
    if not excluded:
        metrics = locked_metrics(
            joined["reference_risk"], joined["observed_risk"]
        )
        failures = stratified_failures(joined, strata=("compound",))
        payload["metrics"] = metrics.to_dict()
        if prereg is not None:
            payload["primary_analysis"] = evaluate_primary_analysis(joined, prereg)
            log("Primary analysis: " + json.dumps(payload["primary_analysis"], indent=2))
            if informative:
                assert (
                    prereg["secondary_analyses"]["dropout_sensitivity"][
                        "reclassify_informative_dropout_as"
                    ]
                    == "high"
                ), "Only reclassification of informative-dropout compounds as 'high' is implemented."
                sensitivity = joined.copy()
                flagged = sensitivity["compound"].astype(str).isin(informative)
                sensitivity.loc[flagged, "cardioscore"] = 1.0
                sensitivity.loc[flagged, "observed_risk"] = "High"
                payload["dropout_sensitivity"] = {
                    "compounds_reclassified_as_high": informative,
                    "primary_analysis": evaluate_primary_analysis(sensitivity, prereg),
                    "three_class_metrics": locked_metrics(
                        sensitivity["reference_risk"], sensitivity["observed_risk"]
                    ).to_dict(),
                }
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
        "n_informative_dropout_compounds": int(len(informative)),
        "primary_auroc": (payload["primary_analysis"] or {}).get("auroc"),
        "primary_outcome": (payload["primary_analysis"] or {}).get("outcome"),
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
        "runner_revision": RUNNER_REVISION,
        "runner_source_sha256": RUNNER_SOURCE_SHA256,
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
            blinova = run_blinova(classified["blinova"], paths["derived"])
            run_manifest["stages"]["blinova_summary"] = blinova
            for flag in blinova["flags"]:
                log(
                    f"Blinova {flag['severity']} [{flag['code']}]: "
                    f"{flag['message']}"
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
                locked_assets, paths["results"], paths["input"]
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
        note = f"  ({info['n_flags']} flag(s), see run_manifest.json)" if info.get("n_flags") else ""
        print(f"{stage:20s} {info.get('status', 'unknown')}{note}")
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

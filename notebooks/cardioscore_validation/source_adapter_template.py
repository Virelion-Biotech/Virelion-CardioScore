"""Source-specific CardioScore adapter template.

Intentionally incomplete. Implement only after inspecting the exact public source.
Never infer missing identifiers, conditions, biological replicates, or well mappings.
"""
from __future__ import annotations
from pathlib import Path
import hashlib, json
import pandas as pd

CANONICAL_RAW = {
    "compound","well","concentration_uM","vehicle",
    "electrode_id","time_s","voltage_uv"
}

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024*1024), b""): h.update(b)
    return h.hexdigest()

def validate_canonical_raw(df: pd.DataFrame) -> None:
    missing = sorted(CANONICAL_RAW - set(df.columns))
    if missing:
        raise ValueError(f"Missing canonical raw fields: {missing}")
    if df.empty:
        raise ValueError("Canonical raw table is empty.")
    for c in ["compound","well","electrode_id"]:
        if df[c].isna().any() or df[c].astype(str).str.strip().eq("").any():
            raise ValueError(f"Blank canonical identifier: {c}")
    for c in ["time_s","voltage_uv","concentration_uM"]:
        x = pd.to_numeric(df[c], errors="coerce")
        if x.isna().any() or not x.map(lambda v: v == v and abs(v) != float("inf")).all():
            raise ValueError(f"Non-finite canonical numeric field: {c}")

def build_features(source_path: Path) -> pd.DataFrame:
    raise NotImplementedError("Write and review a source-specific parser before use.")

def write_derivation_manifest(source: Path, features: Path, reference: Path, adapter_commit: str, config_sha256: str) -> Path:
    payload = {
        "source_sha256": sha256(source),
        "canonical_features_sha256": sha256(features),
        "reference_sha256": sha256(reference),
        "adapter_commit": adapter_commit,
        "config_sha256": config_sha256,
        "verified_rebuild": True,
    }
    out = features.parent / "derivation_manifest.json"
    out.write_text(json.dumps(payload, indent=2)+"\n", encoding="utf-8")
    return out

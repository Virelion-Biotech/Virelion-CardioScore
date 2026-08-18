#!/usr/bin/env python3
"""Run a locked external validation from a provenance manifest.

This command never tunes or mutates the CardioScore configuration. It verifies the
immutable asset, standardized feature/reference schemas, runs the existing pipeline,
and writes deterministic validation artifacts.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from virelion_cardioscore.analysis.pipeline import CardioScorePipeline
from virelion_cardioscore.validation.integrity import verify_sha256
from virelion_cardioscore.validation.manifest import (
    ValidationManifest,
    resolve_relative,
    validate_feature_schema,
    validate_reference_schema,
)
from virelion_cardioscore.validation.metrics import locked_metrics, stratified_failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    manifest = ValidationManifest.from_yaml(manifest_path)
    base = manifest_path.parent
    asset = resolve_relative(base, manifest.asset_path)
    config = resolve_relative(base, manifest.config_path)
    features_path = resolve_relative(base, manifest.features_path)
    reference_path = resolve_relative(base, manifest.reference_path)
    output_dir = resolve_relative(base, manifest.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not asset.is_file():
        raise FileNotFoundError(f"Validation asset does not exist: {asset}")
    if not verify_sha256(asset, manifest.asset_sha256):
        raise ValueError(f"SHA-256 mismatch for validation asset: {asset}")

    features = pd.read_csv(features_path)
    reference = pd.read_csv(reference_path)
    validate_feature_schema(features)
    validate_reference_schema(reference)

    result = CardioScorePipeline.from_config(config).run(features)
    observed = result.summary_table[["compound", "risk_class", "cardioscore"]].rename(
        columns={"risk_class": "observed_risk"}
    )
    joined = reference.merge(observed, on="compound", how="inner", validate="one_to_one")
    if len(joined) != len(reference):
        missing = sorted(set(reference["compound"]) - set(observed["compound"]))
        raise ValueError(f"Reference compounds missing from observed CardioScore output: {missing}")

    metrics = locked_metrics(joined["reference_risk"], joined["observed_risk"])
    failures = stratified_failures(joined, strata=("compound",))

    payload = {
        "source_id": manifest.source_id,
        "evidence_level": manifest.evidence_level,
        "asset_sha256": manifest.asset_sha256,
        "locked": True,
        "metrics": metrics.to_dict(),
        "compound_scores": joined.to_dict(orient="records"),
        "failures": failures.to_dict(orient="records"),
    }
    (output_dir / "validation.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    failures.to_csv(output_dir / "failures_by_compound.csv", index=False)
    joined.to_csv(output_dir / "compound_validation.csv", index=False)
    print(json.dumps({"metrics": metrics.to_dict(), "output_dir": str(output_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

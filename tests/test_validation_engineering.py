from __future__ import annotations

from pathlib import Path
import zipfile

import pandas as pd
import pytest

from virelion_cardioscore.validation.integrity import build_asset_manifest, inventory_archive, sha256_file, verify_sha256
from virelion_cardioscore.validation.manifest import validate_feature_schema, validate_reference_schema
from virelion_cardioscore.validation.metrics import locked_metrics, stratified_failures


def test_locked_metrics_are_deterministic() -> None:
    result = locked_metrics(["low", "intermediate", "high"], ["low", "high", "high"])
    assert result.n == 3
    assert result.accuracy == pytest.approx(2 / 3)
    assert result.ordinal_mae == pytest.approx(1 / 3)
    assert result.confusion_matrix == [[1, 0, 0], [0, 0, 1], [0, 0, 1]]


def test_stratified_failures() -> None:
    frame = pd.DataFrame(
        {
            "compound": ["A", "A", "B"],
            "reference_risk": ["low", "low", "high"],
            "observed_risk": ["low", "high", "high"],
        }
    )
    failures = stratified_failures(frame, strata=("compound",))
    row_a = failures.loc[failures["compound"] == "A"].iloc[0]
    assert int(row_a["n"]) == 2
    assert int(row_a["failures"]) == 1
    assert float(row_a["failure_rate"]) == pytest.approx(0.5)


def test_archive_inventory_is_path_safe(tmp_path: Path) -> None:
    archive = tmp_path / "asset.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("data/example.csv", "a,b\n1,2\n")
    entries = inventory_archive(archive)
    assert entries[0].path == "data/example.csv"
    assert entries[0].sha256 == sha256_file(tmp_path / "asset.zip") or entries[0].sha256 is not None
    manifest = build_asset_manifest(
        archive,
        source_id="test",
        source_url="https://example.invalid",
        acquisition_date="2026-01-01",
    )
    assert manifest.byte_size == archive.stat().st_size
    assert verify_sha256(archive, manifest.sha256)


def test_feature_and_reference_schema_validation() -> None:
    features = pd.DataFrame(
        {
            "compound": ["A"],
            "site": [1],
            "cell_type": ["CDI"],
            "concentration_index": [1],
            "concentration_uM": [1.0],
            "well": ["A01"],
            "fpd_ms": [300.0],
            "beat_rate_bpm": [60.0],
            "amplitude_uv": [100.0],
            "stv": [1.0],
            "triangulation_proxy": [0.2],
        }
    )
    reference = pd.DataFrame({"compound": ["A"], "reference_risk": ["high"]})
    validate_feature_schema(features)
    validate_reference_schema(reference)


def test_reference_rejects_duplicates() -> None:
    reference = pd.DataFrame({"compound": ["A", "A"], "reference_risk": ["high", "low"]})
    with pytest.raises(ValueError, match="duplicate"):
        validate_reference_schema(reference)

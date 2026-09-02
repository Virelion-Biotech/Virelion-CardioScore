from __future__ import annotations

from pathlib import Path
import zipfile

import pandas as pd
import pytest

from virelion_cardioscore.validation.integrity import (
    build_asset_manifest,
    inventory_archive,
    sha256_file,
    verify_sha256,
)
from virelion_cardioscore.validation.manifest import (
    validate_feature_schema,
    validate_reference_schema,
    validate_vehicle_structure,
)
from virelion_cardioscore.validation.metrics import locked_metrics, stratified_failures


def _valid_features(**overrides: object) -> pd.DataFrame:
    row = {
        "compound": "A",
        "site": 1,
        "cell_type": "CDI",
        "concentration_index": 1,
        "concentration_uM": 1.0,
        "well": "A01",
        "vehicle": False,
        "fpd_ms": 300.0,
        "beat_rate_bpm": 60.0,
        "amplitude_uv": 100.0,
        "stv": 1.0,
        "triangulation_proxy": 0.2,
        "n_electrodes": 4,
        "noise_sd_uv": 5.0,
        "beat_detection_rate": 0.9,
    }
    row.update(overrides)
    return pd.DataFrame([row])


def test_locked_metrics_are_deterministic() -> None:
    result = locked_metrics(["low", "intermediate", "high"], ["low", "high", "high"])
    assert result.n == 3
    assert result.accuracy == pytest.approx(2 / 3)
    assert result.ordinal_mae == pytest.approx(1 / 3)
    assert result.confusion_matrix == [[1, 0, 0], [0, 0, 1], [0, 0, 1]]


def test_locked_metrics_normalizes_published_short_labels() -> None:
    result = locked_metrics(["L", "M", "H"], ["low", "intermediate", "high"], labels=("L", "M", "H"))
    assert result.labels == ("l", "m", "h")
    assert result.accuracy == pytest.approx(1.0)
    assert result.ordinal_mae == pytest.approx(0.0)


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
    assert entries[0].sha256 == sha256_file_from_member(entries[0].path, archive)
    manifest = build_asset_manifest(
        archive,
        source_id="test",
        source_url="https://example.invalid",
        acquisition_date="2026-01-01",
    )
    assert manifest.byte_size == archive.stat().st_size
    assert verify_sha256(archive, manifest.sha256)


def sha256_file_from_member(member_path: str, archive: Path) -> str:
    digest = __import__("hashlib").sha256()
    with zipfile.ZipFile(archive) as zf, zf.open(member_path) as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_archive_inventory_rejects_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("../escape.txt", "unsafe")
    with pytest.raises(ValueError, match="Unsafe archive member path"):
        inventory_archive(archive)


def test_feature_and_reference_schema_validation() -> None:
    features = _valid_features()
    reference = pd.DataFrame({"compound": ["A"], "reference_risk": ["high"]})
    validate_feature_schema(features)
    validate_reference_schema(reference)


def test_feature_schema_requires_runtime_qc_fields() -> None:
    features = _valid_features().drop(columns=["noise_sd_uv"])
    with pytest.raises(ValueError, match="noise_sd_uv"):
        validate_feature_schema(features)


def test_vehicle_zero_concentration_is_allowed_for_controls() -> None:
    features = _valid_features(
        well="V01",
        vehicle=True,
        concentration_uM=0.0,
        concentration_index=0,
    )
    validate_feature_schema(features)


def test_treated_zero_concentration_is_rejected() -> None:
    features = _valid_features(concentration_uM=0.0, vehicle=False)
    with pytest.raises(ValueError, match="strictly positive"):
        validate_feature_schema(features)


def test_negative_vehicle_concentration_is_rejected() -> None:
    features = _valid_features(vehicle=True, concentration_uM=-1.0)
    with pytest.raises(ValueError, match="cannot be negative"):
        validate_feature_schema(features)


def test_invalid_beat_detection_rate_is_rejected() -> None:
    features = _valid_features(beat_detection_rate=1.2)
    with pytest.raises(ValueError, match="between 0 and 1"):
        validate_feature_schema(features)


def test_invalid_electrode_count_is_rejected() -> None:
    features = _valid_features(n_electrodes=2.5)
    with pytest.raises(ValueError, match="non-negative integers"):
        validate_feature_schema(features)


def test_vehicle_structure_requires_control_for_each_treated_compound() -> None:
    features = pd.DataFrame(
        [
            _valid_features(compound="A").iloc[0].to_dict(),
            _valid_features(compound="B").iloc[0].to_dict(),
        ]
    )
    with pytest.raises(ValueError, match="Missing matching vehicle control"):
        validate_vehicle_structure(features)


def test_vehicle_structure_accepts_matching_control() -> None:
    treated = _valid_features(compound="A", well="A01")
    control = _valid_features(
        compound="A",
        well="V01",
        vehicle=True,
        concentration_uM=0.0,
        concentration_index=0,
    )
    validate_vehicle_structure(pd.concat([treated, control], ignore_index=True))


def test_reference_rejects_duplicates() -> None:
    reference = pd.DataFrame({"compound": ["A", "A"], "reference_risk": ["high", "low"]})
    with pytest.raises(ValueError, match="duplicate"):
        validate_reference_schema(reference)

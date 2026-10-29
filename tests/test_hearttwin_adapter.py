from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from virelion_cardioscore.hearttwin_adapter import RESULT_SCHEMA, score_payload


EXAMPLE = Path(__file__).parents[1] / "examples" / "data" / "example_features.csv"


def _payload(path: Path, values: dict | None = None) -> dict:
    return {
        "entity_id": "synthetic-validation-entity",
        "observations": [
            {
                "modality": "safety",
                "values": {"input_path": str(path), **(values or {})},
            }
        ],
    }


def test_hearttwin_adapter_emits_versioned_research_result_with_provenance():
    output = score_payload(_payload(EXAMPLE))

    assert output["schema_version"] == RESULT_SCHEMA
    assert output["producer"]["name"] == "Virelion-CardioScore"
    assert output["entity_id"] == "synthetic-validation-entity"
    assert output["input"]["filename"] == EXAMPLE.name
    assert output["input"]["sha256"] == hashlib.sha256(EXAMPLE.read_bytes()).hexdigest()
    assert output["input"]["rows"] > 0
    assert output["analysis"]["score_kind"] == "research_score"
    assert output["analysis"]["config_sha256"]
    assert output["analysis"]["endpoint_config_sha256"]
    assert output["scores"]
    assert output["summary"]
    assert "qc_log" in output
    assert "qc_rejections" in output
    assert "dropout_table" in output
    assert "exclusion_table" in output
    assert "provenance" in output
    assert "not an independently calibrated" in output["analysis"]["risk_class_semantics"]

    # The whole adapter response must be strict JSON: NaN/Infinity are not allowed.
    encoded = json.dumps(output, allow_nan=False)
    assert "synthetic-validation-entity" in encoded
    assert str(EXAMPLE.resolve()) not in encoded


def test_hearttwin_adapter_rejects_missing_safety_input():
    with pytest.raises(ValueError, match="No 'safety' observation"):
        score_payload({"entity_id": "x", "observations": [{"modality": "ecg", "values": {}}]})


def test_hearttwin_adapter_rejects_missing_input_file(tmp_path):
    missing = tmp_path / "does-not-exist.csv"
    with pytest.raises(ValueError, match="existing CSV file"):
        score_payload(_payload(missing))


def test_hearttwin_adapter_rejects_missing_explicit_config(tmp_path):
    missing = tmp_path / "does-not-exist.yaml"
    with pytest.raises(ValueError, match="existing YAML file"):
        score_payload(_payload(EXAMPLE, {"config": str(missing)}))

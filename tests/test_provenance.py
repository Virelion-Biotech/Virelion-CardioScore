from __future__ import annotations

import json

import pandas as pd

from virelion_cardioscore.analysis.pipeline import CardioScorePipeline
from virelion_cardioscore.io.synthetic import load_synthetic_dataset


def test_pipeline_result_exposes_and_serializes_provenance(tmp_path):
    dataset = load_synthetic_dataset(n_compounds=2, n_concentrations=4, seed=21)
    result = CardioScorePipeline.from_defaults().run(dataset)

    assert result.provenance["schema_version"] == "1.0"
    assert len(result.provenance["input_sha256"]) == 64
    assert len(result.provenance["configuration_sha256"]) == 64
    assert result.provenance["input_rows"] == len(dataset.features)
    assert 0 < result.provenance["rows_after_qc"] <= result.provenance["input_rows"]
    assert result.provenance["effect_rows"] == len(result.feature_table)
    assert result.provenance["scoring_unit_rows"] >= 0
    assert "compound_scoring" in result.provenance["transformation_summary"]

    output = tmp_path / "result.json"
    result.to_json(output)
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["provenance"] == result.provenance


def test_provenance_input_hash_is_order_invariant():
    pipeline = CardioScorePipeline.from_defaults()
    dataset = load_synthetic_dataset(n_compounds=1, n_concentrations=4, seed=33)
    shuffled = dataset.features.sample(frac=1.0, random_state=7).reset_index(drop=True)

    first = pipeline.run(dataset)
    second = pipeline.run(pd.DataFrame(shuffled))

    assert first.provenance["input_sha256"] == second.provenance["input_sha256"]
    assert first.provenance["configuration_sha256"] == second.provenance["configuration_sha256"]

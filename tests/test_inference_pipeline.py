from __future__ import annotations

from virelion_cardioscore.analysis.pipeline import CardioScorePipeline
from virelion_cardioscore.io.synthetic import load_synthetic_dataset


def test_inference_is_disabled_by_default():
    dataset = load_synthetic_dataset(n_compounds=1, n_concentrations=3, seed=21)
    result = CardioScorePipeline.from_defaults().run(dataset)
    assert result.inference_table.empty


def test_opt_in_bootstrap_inference_is_exposed():
    dataset = load_synthetic_dataset(n_compounds=1, n_concentrations=3, seed=21)
    # Synthetic fixture wells are treated as explicitly independent units for
    # this infrastructure test; production users should provide biological,
    # batch, or plate identifiers instead.
    dataset.features["biological_replicate"] = dataset.features["well"]
    pipeline = CardioScorePipeline.from_defaults()
    pipeline.config["inference"]["enabled"] = True
    pipeline.config["inference"]["n_bootstrap"] = 200

    result = pipeline.run(dataset)

    assert not result.inference_table.empty
    assert {"compound", "concentration_uM", "n_replicates"}.issubset(
        result.inference_table.columns
    )
    assert set(result.inference_table["cluster_column"]) == {"biological_replicate"}
    assert any(column.endswith("_ci_low") for column in result.inference_table.columns)
    assert any(column.endswith("_ci_high") for column in result.inference_table.columns)

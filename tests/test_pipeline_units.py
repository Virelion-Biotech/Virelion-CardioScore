from __future__ import annotations

import pandas as pd

from virelion_cardioscore.analysis.pipeline import CardioScorePipeline


def _dataset() -> pd.DataFrame:
    rows = []
    for well, bio, fpd in [
        ("W1", "B1", 100.0),
        ("W2", "B1", 120.0),
        ("W3", "B2", 140.0),
        ("W4", "B2", 160.0),
    ]:
        rows.append(
            {
                "compound": "A",
                "well": well,
                "concentration_uM": 1.0,
                "vehicle": False,
                "fpd_ms": fpd,
                "beat_rate_bpm": 60.0,
                "amplitude_uv": 100.0,
                "stv": 0.1,
                "triangulation_proxy": 0.1,
                "n_electrodes": 4,
                "noise_sd_uv": 5.0,
                "beat_detection_rate": 0.95,
                "biological_replicate": bio,
            }
        )

    rows.append(
        {
            "compound": "A",
            "well": "V1",
            "concentration_uM": 0.0,
            "vehicle": True,
            "fpd_ms": 100.0,
            "beat_rate_bpm": 60.0,
            "amplitude_uv": 100.0,
            "stv": 0.1,
            "triangulation_proxy": 0.1,
            "n_electrodes": 4,
            "noise_sd_uv": 5.0,
            "beat_detection_rate": 0.95,
            "biological_replicate": "BV",
        }
    )
    return pd.DataFrame(rows)


def _run(scoring_unit: str):
    pipeline = CardioScorePipeline.from_defaults()
    pipeline.config["experimental_units"]["scoring_unit"] = scoring_unit
    return pipeline.run(_dataset())


def test_well_scoring_preserves_four_independent_wells():
    result = _run("well")

    assert result.summary_table.iloc[0]["n_wells"] == 4
    assert result.summary_table.iloc[0]["n_independent_units"] == 4


def test_biological_replicate_scoring_collapses_technical_wells():
    result = _run("biological_replicate")

    assert result.summary_table.iloc[0]["n_wells"] == 4
    assert result.summary_table.iloc[0]["n_independent_units"] == 2
    assert any("collapsed 4 technical wells into 2 independent units" in msg for msg in result.qc_log)
    assert result.concentration_table.iloc[0]["n_replicates"] == 2


def test_missing_biological_replicate_metadata_fails_pipeline():
    pipeline = CardioScorePipeline.from_defaults()
    pipeline.config["experimental_units"]["scoring_unit"] = "biological_replicate"

    dataset = _dataset().drop(columns=["biological_replicate"])

    try:
        pipeline.run(dataset)
    except ValueError as exc:
        assert "biological_replicate" in str(exc)
    else:
        raise AssertionError("Expected missing experimental-unit metadata to fail")

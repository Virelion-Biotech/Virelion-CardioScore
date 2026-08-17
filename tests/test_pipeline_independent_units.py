from __future__ import annotations

import pandas as pd

from virelion_cardioscore.analysis.pipeline import CardioScorePipeline


def _nested_feature_table() -> pd.DataFrame:
    rows = []
    for compound in ["A"]:
        for replicate in ["B1", "B2"]:
            for concentration, fpd in [(1.0, 110.0), (10.0, 130.0)]:
                for technical in [1, 2]:
                    rows.append(
                        {
                            "compound": compound,
                            "concentration_uM": concentration,
                            "well": f"{replicate}_{concentration}_{technical}",
                            "biological_replicate": replicate,
                            "vehicle": False,
                            "fpd_ms": fpd,
                            "beat_rate_bpm": 55.0,
                            "amplitude_uv": 180.0,
                            "stv": 0.05,
                            "triangulation_proxy": 0.10,
                            "noise_sd_uv": 5.0,
                            "n_electrodes": 4,
                            "beat_detection_rate": 0.95,
                        }
                    )
    return pd.DataFrame(rows)


def test_compound_independent_units_are_unique_across_concentrations():
    pipeline = CardioScorePipeline.from_defaults()
    pipeline.config["experimental_units"]["scoring_unit"] = "biological_replicate"

    # Matching vehicle rows are required by the default normalization path.
    frame = _nested_feature_table()
    vehicles = frame.copy()
    vehicles["vehicle"] = True
    vehicles["well"] = vehicles["well"] + "_V"
    vehicles["fpd_ms"] = 100.0
    combined = pd.concat([frame, vehicles], ignore_index=True)

    result = pipeline.run(combined)
    score = result.scores[0]

    # Two biological replicates are reused across two concentrations; they
    # must not be counted as four independent biological units.
    assert score.n_independent_units == 2
    assert score.n_wells == 8
    assert result.summary_table.iloc[0]["n_independent_units"] == 2

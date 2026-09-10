from __future__ import annotations

import pandas as pd
import pytest

from virelion_cardioscore.analysis.pipeline import CardioScorePipeline


def _effects_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "compound": ["A", "A", "A", "A"],
            "concentration_uM": [0.0, 0.0, 1.0, 1.0],
            "well": ["V1", "V2", "T1", "T2"],
            "vehicle": ["True", "False", "False", "false"],
            "fpd_ms": [100.0, 101.0, 120.0, 121.0],
            "beat_rate_bpm": [60.0, 60.0, 55.0, 55.0],
            "amplitude_uv": [100.0, 100.0, 90.0, 90.0],
            "stv": [0.04, 0.04, 0.08, 0.08],
            "triangulation_proxy": [0.18, 0.18, 0.30, 0.30],
            "n_electrodes": [8, 8, 8, 8],
            "noise_sd_uv": [5.0, 5.0, 5.0, 5.0],
            "beat_detection_rate": [0.95, 0.95, 0.95, 0.95],
        }
    )


def test_compute_effects_does_not_treat_string_false_as_vehicle():
    pipeline = CardioScorePipeline.from_defaults()
    effects = pipeline.compute_effects(_effects_frame())

    assert len(effects) == 2
    assert set(effects["well"]) == {"T1", "T2"}
    assert effects["fpd_change_pct"].tolist() == pytest.approx([20.0, 21.0])


def test_pipeline_run_accepts_string_boolean_encoding():
    pipeline = CardioScorePipeline.from_defaults()
    result = pipeline.run(_effects_frame())

    assert len(result.scores) == 1
    assert result.scores[0].compound == "A"
    assert result.summary_table.iloc[0]["n_wells"] == 2

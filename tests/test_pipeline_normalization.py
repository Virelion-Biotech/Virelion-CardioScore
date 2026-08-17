from __future__ import annotations

import pandas as pd
import pytest

from virelion_cardioscore.analysis.pipeline import CardioScorePipeline


def _dataset() -> pd.DataFrame:
    rows = []
    common = {
        "beat_rate_bpm": 60.0,
        "amplitude_uv": 100.0,
        "stv": 0.1,
        "triangulation_proxy": 0.1,
        "n_electrodes": 4,
        "noise_sd_uv": 5.0,
        "beat_detection_rate": 0.95,
    }
    for plate, values in {
        "P1": [100.0, 102.0, 130.0, 132.0],
        "P2": [120.0, 122.0, 150.0, 152.0],
    }.items():
        for index, fpd in enumerate(values):
            rows.append(
                {
                    **common,
                    "plate_id": plate,
                    "compound": "A",
                    "well": f"{plate}_{index}",
                    "concentration_uM": 1.0,
                    "vehicle": index < 2,
                    "fpd_ms": fpd,
                }
            )
    return pd.DataFrame(rows)


def test_pipeline_control_anchor_correction_reduces_between_plate_drift():
    pipeline = CardioScorePipeline.from_defaults()
    pipeline.config["variability"]["enabled"] = True
    pipeline.config["variability"]["group_column"] = "plate_id"
    pipeline.config["variability"]["correction"]["enabled"] = True
    pipeline.config["variability"]["correction"]["corrected_columns"] = ["fpd_ms"]

    result = pipeline.run(_dataset())

    before = result.variability_before_correction
    after = result.variability_table
    before_sd = float(before.loc[before["endpoint"] == "fpd_ms", "between_group_sd"].iloc[0])
    after_sd = float(after.loc[after["endpoint"] == "fpd_ms", "between_group_sd"].iloc[0])

    assert result.normalization_diagnostic["n_groups"] == 2
    assert before_sd > 0
    assert after_sd == pytest.approx(0.0)
    assert after_sd < before_sd
    assert any("control-anchored recentering" in msg for msg in result.qc_log)


def test_pipeline_normalization_requires_variability_diagnostics():
    pipeline = CardioScorePipeline.from_defaults()
    pipeline.config["variability"]["enabled"] = False
    pipeline.config["variability"]["correction"]["enabled"] = True

    with pytest.raises(ValueError, match="requires variability.enabled=true"):
        pipeline.run(_dataset())


def test_pipeline_default_normalization_is_disabled():
    pipeline = CardioScorePipeline.from_defaults()
    result = pipeline.run(_dataset())

    assert result.normalization_diagnostic == {}
    assert result.variability_before_correction.empty

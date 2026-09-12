from __future__ import annotations

import pandas as pd
import pytest

from virelion_cardioscore.analysis.pipeline import CardioScorePipeline
from virelion_cardioscore.analysis.robustness_classification import (
    evaluate_classification_stability,
    perturb_feature_table,
)


def _frame() -> pd.DataFrame:
    rows = []
    for concentration in [1.0, 3.0, 10.0]:
        rows.extend([
            {
                "compound": "A",
                "well": f"V{int(concentration)}",
                "concentration_uM": 0.0 if concentration == 1.0 else concentration,
                "vehicle": concentration == 1.0,
                "fpd_ms": 100.0 if concentration == 1.0 else 135.0,
                "beat_rate_bpm": 60.0,
                "amplitude_uv": 100.0 if concentration == 1.0 else 80.0,
                "stv": 0.04 if concentration == 1.0 else 0.12,
                "triangulation_proxy": 0.18 if concentration == 1.0 else 0.35,
                "noise_sd_uv": 5.0,
                "n_electrodes": 8,
                "beat_detection_rate": 0.95,
            }
        ])
    return pd.DataFrame(rows)


def test_perturb_feature_table_is_deterministic():
    frame = _frame()
    first = perturb_feature_table(frame, relative_noise_sd=0.01, seed=7)
    second = perturb_feature_table(frame, relative_noise_sd=0.01, seed=7)
    pd.testing.assert_frame_equal(first, second)


def test_perturb_feature_table_rejects_negative_noise():
    with pytest.raises(ValueError, match="relative_noise_sd"):
        perturb_feature_table(_frame(), relative_noise_sd=-0.1)


def test_classification_stability_reports_repeatability():
    pipeline = CardioScorePipeline.from_defaults()
    pipeline.config["concentration_response"]["require_min_concentrations_for_scoring"] = False
    result = evaluate_classification_stability(
        _frame(),
        pipeline=pipeline,
        relative_noise_sd=0.001,
        n_scenarios=5,
        seed=11,
    )
    assert len(result) == 1
    assert result.iloc[0]["compound"] == "A"
    assert result.iloc[0]["n_scenarios"] == 5
    assert 0.0 <= result.iloc[0]["class_stability_rate"] <= 1.0
    assert result.iloc[0]["max_absolute_score_delta"] >= result.iloc[0]["mean_absolute_score_delta"] >= 0.0

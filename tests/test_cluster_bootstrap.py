"""Tests for independent-unit bootstrap integration in CardioScorePipeline."""

from __future__ import annotations

import pandas as pd
import pytest

from virelion_cardioscore.analysis.pipeline import CardioScorePipeline


def test_pipeline_bootstrap_uses_independent_clusters():
    pipeline = CardioScorePipeline.from_defaults()

    effects = pd.DataFrame(
        {
            "compound": ["A"] * 6,
            "concentration_uM": [1.0] * 6,
            "well": ["W1", "W2", "W3", "W4", "W5", "W6"],
            "biological_replicate": ["BR1", "BR1", "BR2", "BR2", "BR3", "BR3"],
            "fpd_change_pct": [10.0, 12.0, 20.0, 21.0, 30.0, 29.0],
            "beat_rate_change_pct": [1.0] * 6,
            "amplitude_change_pct": [-2.0] * 6,
            "stv_increase": [0.01] * 6,
            "triangulation_proxy_change": [0.02] * 6,
        }
    )

    result = pipeline.bootstrap_concentration_inference(
        effects,
        n_bootstrap=200,
        confidence=0.95,
        seed=42,
    )

    assert len(result) == 1
    assert result.iloc[0]["cluster_column"] == "biological_replicate"
    assert int(result.iloc[0]["n_replicates"]) == 6
    assert int(result.iloc[0]["n_clusters"]) == 3


def test_pipeline_bootstrap_fails_without_independent_unit_metadata():
    pipeline = CardioScorePipeline.from_defaults()

    effects = pd.DataFrame(
        {
            "compound": ["A", "A"],
            "concentration_uM": [1.0, 1.0],
            "well": ["W1", "W2"],
            "fpd_change_pct": [10.0, 20.0],
            "beat_rate_change_pct": [1.0, 2.0],
            "amplitude_change_pct": [-2.0, -3.0],
            "stv_increase": [0.01, 0.02],
            "triangulation_proxy_change": [0.02, 0.03],
        }
    )

    with pytest.raises(
        ValueError,
        match="Cluster-aware bootstrap requires independent-unit metadata",
    ):
        pipeline.bootstrap_concentration_inference(effects)


def test_pipeline_bootstrap_honors_explicit_cluster_column():
    pipeline = CardioScorePipeline.from_defaults()

    effects = pd.DataFrame(
        {
            "compound": ["A"] * 4,
            "concentration_uM": [1.0] * 4,
            "well": ["W1", "W2", "W3", "W4"],
            "site": ["S1", "S1", "S2", "S2"],
            "fpd_change_pct": [10.0, 11.0, 20.0, 21.0],
            "beat_rate_change_pct": [1.0] * 4,
            "amplitude_change_pct": [-2.0] * 4,
            "stv_increase": [0.01] * 4,
            "triangulation_proxy_change": [0.02] * 4,
        }
    )

    result = pipeline.bootstrap_concentration_inference(
        effects,
        n_bootstrap=100,
        confidence=0.95,
        seed=42,
        cluster_column="site",
    )

    assert result.iloc[0]["cluster_column"] == "site"
    assert int(result.iloc[0]["n_clusters"]) == 2


def test_pipeline_bootstrap_rejects_missing_cluster_identifier():
    pipeline = CardioScorePipeline.from_defaults()

    effects = pd.DataFrame(
        {
            "compound": ["A", "A"],
            "concentration_uM": [1.0, 1.0],
            "well": ["W1", "W2"],
            "biological_replicate": ["BR1", None],
            "fpd_change_pct": [10.0, 20.0],
            "beat_rate_change_pct": [1.0, 2.0],
            "amplitude_change_pct": [-2.0, -3.0],
            "stv_increase": [0.01, 0.02],
            "triangulation_proxy_change": [0.02, 0.03],
        }
    )

    with pytest.raises(ValueError, match="contains missing or blank identifiers"):
        pipeline.bootstrap_concentration_inference(effects)

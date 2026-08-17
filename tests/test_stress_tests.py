"""Stress tests for clustering, plate drift, and pseudoreplication."""

import pytest

from virelion_cardioscore.analysis.stress_tests import (
    StressTestSpec,
    conventional_treatment_effect,
    make_known_effect_dataset,
    true_treatment_effect,
)


def test_balanced_clean_design_recovers_known_effect():
    spec = StressTestSpec(treatment_effect=12.0, seed=1)
    data = make_known_effect_dataset(spec)

    estimate = conventional_treatment_effect(data)

    assert abs(estimate - true_treatment_effect(spec)) < 2.0


def test_balanced_additive_plate_drift_preserves_pooled_effect():
    spec = StressTestSpec(
        treatment_effect=15.0,
        group_offsets=(-30.0, -10.0, 10.0, 30.0),
        seed=2,
    )
    data = make_known_effect_dataset(spec)

    estimate = conventional_treatment_effect(data)

    assert abs(estimate - true_treatment_effect(spec)) < 2.0


def test_treatment_allocated_plate_imbalance_creates_conventional_bias():
    spec = StressTestSpec(
        treatment_effect=10.0,
        group_offsets=(-40.0, -20.0, 20.0, 40.0),
        groups_with_treatment=(2, 3),
        seed=3,
    )
    data = make_known_effect_dataset(spec)

    estimate = conventional_treatment_effect(data)

    assert estimate - true_treatment_effect(spec) > 20.0


def test_mixed_effects_recovers_treatment_effect_in_balanced_drift_case():
    statsmodels = pytest.importorskip("statsmodels")
    del statsmodels

    from virelion_cardioscore.analysis.mixed_effects import fit_random_intercept

    spec = StressTestSpec(
        treatment_effect=10.0,
        group_offsets=(-40.0, -20.0, 20.0, 40.0),
        seed=4,
    )
    data = make_known_effect_dataset(spec)
    data["_treatment"] = (~data["vehicle"]).astype(int)

    result = fit_random_intercept(
        data,
        endpoint="fpd_ms",
        treatment_column="_treatment",
        group_column="plate_id",
    )

    assert result.converged
    assert abs(result.treatment_effect - true_treatment_effect(spec)) < 2.0
    assert 0.0 <= result.icc <= 1.0


def test_multiplicative_drift_with_treatment_imbalance_is_not_additive():
    spec = StressTestSpec(
        treatment_effect=10.0,
        group_offsets=(0.0, 0.0, 0.0, 0.0),
        group_scales=(0.8, 0.9, 1.1, 1.2),
        groups_with_treatment=(2, 3),
        seed=5,
    )
    data = make_known_effect_dataset(spec)

    estimate = conventional_treatment_effect(data)

    assert abs(estimate - true_treatment_effect(spec)) > 0.5

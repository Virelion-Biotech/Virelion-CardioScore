"""Regression tests for robustness sweeps."""

import pandas as pd

from virelion_cardioscore.analysis.robustness import (
    RobustnessGrid,
    run_robustness_matrix,
    summarize_robustness_matrix,
)


def test_robustness_matrix_is_reproducible():
    grid = RobustnessGrid(
        offset_magnitudes=(0.0, 20.0),
        treatment_group_counts=(4, 2),
        replicate_counts=(2,),
        noise_sds=(1.0,),
        n_repeats=2,
        seed=7,
    )
    first = run_robustness_matrix(grid)
    second = run_robustness_matrix(grid)

    pd.testing.assert_frame_equal(first, second)


def test_robustness_matrix_has_expected_number_of_scenarios():
    grid = RobustnessGrid(
        offset_magnitudes=(0.0, 10.0),
        treatment_group_counts=(4, 2, 1),
        replicate_counts=(2, 4),
        noise_sds=(1.0,),
        n_repeats=3,
    )
    results = run_robustness_matrix(grid)

    assert len(results) == 2 * 3 * 2 * 1 * 3
    assert set(results["offset_magnitude"]) == {0.0, 10.0}
    assert set(results["treated_groups"]) == {1, 2, 4}


def test_zero_drift_is_robust_to_treatment_allocation():
    grid = RobustnessGrid(
        offset_magnitudes=(0.0,),
        treatment_group_counts=(1, 2, 4),
        replicate_counts=(8,),
        noise_sds=(0.5,),
        n_repeats=5,
        bias_tolerance=2.0,
    )
    results = run_robustness_matrix(grid)

    assert results["within_tolerance"].mean() == 1.0


def test_high_drift_and_treatment_imbalance_reduce_recovery():
    grid = RobustnessGrid(
        offset_magnitudes=(0.0, 40.0),
        treatment_group_counts=(1, 4),
        replicate_counts=(8,),
        noise_sds=(0.5,),
        n_repeats=8,
        bias_tolerance=2.0,
    )
    results = run_robustness_matrix(grid)
    summary = summarize_robustness_matrix(results, bias_tolerance=2.0)

    low_drift_imbalanced = summary[
        (summary["offset_magnitude"] == 0.0) & (summary["treated_groups"] == 1)
    ]["recovery_rate"].iloc[0]
    high_drift_imbalanced = summary[
        (summary["offset_magnitude"] == 40.0) & (summary["treated_groups"] == 1)
    ]["recovery_rate"].iloc[0]

    assert low_drift_imbalanced > high_drift_imbalanced

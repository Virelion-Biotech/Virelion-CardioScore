from __future__ import annotations

import numpy as np

from virelion_cardioscore.analysis.dose_response import fit_4pl
from virelion_cardioscore.analysis.pipeline import CardioScorePipeline
from virelion_cardioscore.io.synthetic import load_synthetic_dataset


def test_4pl_good_fit_passes_monotonicity_and_ec50_range_gates():
    concentrations = np.logspace(-1, 2, 7)
    responses = 80.0 / (1.0 + (10.0 / concentrations) ** 1.5)

    result = fit_4pl(
        concentrations,
        responses,
        endpoint="fpd_change_pct",
        min_r_squared=0.99,
        min_monotonicity=0.80,
        ec50_boundary_factor=2.0,
        max_ec50_uncertainty_fold=100.0,
    )

    assert result.success
    assert result.quality_pass
    assert result.monotonicity == 1.0
    assert result.monotonic_direction == "increasing"
    assert not result.ec50_boundary_flag
    assert result.ec50_uncertainty_fold is not None


def test_non_monotonic_series_is_flagged():
    concentrations = np.logspace(0, 2, 6)
    responses = np.array([1.0, 12.0, 4.0, 18.0, 8.0, 22.0])

    result = fit_4pl(
        concentrations,
        responses,
        endpoint="endpoint",
        min_r_squared=0.0,
        min_monotonicity=0.80,
    )

    assert result.success
    assert result.monotonicity is not None
    assert result.monotonicity < 0.80
    assert not result.quality_pass
    assert "monotonicity" in result.message


def test_ec50_near_unobserved_boundary_is_flagged():
    concentrations = np.logspace(0, np.log10(32), 6)
    responses = 90.0 / (1.0 + (100.0 / concentrations) ** 1.3)

    result = fit_4pl(
        concentrations,
        responses,
        endpoint="endpoint",
        min_r_squared=0.0,
        min_monotonicity=0.80,
        ec50_boundary_factor=2.0,
    )

    assert result.success
    assert result.ec50_boundary_flag
    assert not result.quality_pass
    assert "EC50" in result.message


def test_pipeline_reports_quality_diagnostics():
    dataset = load_synthetic_dataset(n_compounds=2, n_concentrations=6, seed=17)
    pipeline = CardioScorePipeline.from_defaults()
    pipeline.config["concentration_response"]["fit_curve"] = True

    result = pipeline.run(dataset)

    for column in [
        "dose_response_boundary_flags",
        "dose_response_high_uncertainty",
        "dose_response_mean_monotonicity",
    ]:
        assert column in result.summary_table.columns

    assert set(result.dose_response_fits) == {"Compound_A", "Compound_B"}


def test_dose_response_result_serialization_includes_diagnostics():
    concentrations = np.logspace(-1, 2, 7)
    responses = 80.0 / (1.0 + (10.0 / concentrations) ** 1.5)
    result = fit_4pl(concentrations, responses, endpoint="fpd_change_pct")
    payload = result.to_dict()

    for key in [
        "quality_pass",
        "monotonicity",
        "monotonic_direction",
        "ec50_boundary_flag",
        "ec50_uncertainty_fold",
    ]:
        assert key in payload

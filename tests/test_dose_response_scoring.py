from __future__ import annotations

import numpy as np
import pytest

from virelion_cardioscore.analysis.cipa_scoring import CardioScoreEngine
from virelion_cardioscore.analysis.dose_response import DoseResponseFit, fit_4pl, four_parameter_logistic
from virelion_cardioscore.analysis.pipeline import CardioScorePipeline


def test_dose_response_evidence_contribution_is_optional():
    engine = CardioScoreEngine()
    endpoints = {
        "fpd_change_pct": 45.0,
        "beat_rate_change_pct": 0.0,
        "amplitude_change_pct": 0.0,
        "stv_increase": 0.0,
        "triangulation_proxy": 0.0,
    }

    baseline = engine.score_compound("A", endpoints)
    evidence_score = engine.score_compound(
        "A",
        endpoints,
        dose_response_evidence=1.0,
        dose_response_weight=0.20,
    )

    assert evidence_score.score > baseline.score
    assert any(c.name == "dose_response_exposure_evidence" for c in evidence_score.contributions)
    assert evidence_score.metadata["dose_response_evidence"] == pytest.approx(1.0)
    assert evidence_score.to_dict()["metadata"]["dose_response_weight"] == pytest.approx(0.20)


def test_dose_response_weight_requires_valid_evidence():
    engine = CardioScoreEngine()
    endpoints = {name: 0.0 for name in engine.endpoints}

    with pytest.raises(ValueError, match="dose_response_evidence is required"):
        engine.score_compound("A", endpoints, dose_response_weight=0.10)

    with pytest.raises(ValueError, match="between 0 and 1"):
        engine.score_compound("A", endpoints, dose_response_evidence=1.5)


def test_exposure_evidence_is_weighted_by_endpoint_importance():
    fits = [
        DoseResponseFit(
            endpoint="fpd_change_pct",
            success=True,
            quality_pass=True,
            n_points=6,
            ec50=1.0,
        ),
        DoseResponseFit(
            endpoint="beat_rate_change_pct",
            success=True,
            quality_pass=True,
            n_points=6,
            ec50=100.0,
        ),
        DoseResponseFit(
            endpoint="amplitude_change_pct",
            success=True,
            quality_pass=False,
            n_points=6,
            ec50=0.1,
        ),
    ]
    evidence = CardioScorePipeline.dose_response_exposure_evidence(
        fits,
        min_concentration_uM=0.1,
        max_concentration_uM=100.0,
        endpoint_weights={
            "fpd_change_pct": 0.30,
            "beat_rate_change_pct": 0.15,
            "amplitude_change_pct": 0.15,
        },
    )

    assert evidence is not None
    assert 0.0 < evidence < 1.0


def test_exposure_evidence_returns_none_without_quality_passing_fits():
    fit = DoseResponseFit(
        endpoint="fpd_change_pct",
        success=True,
        quality_pass=False,
        n_points=6,
        ec50=1.0,
    )
    evidence = CardioScorePipeline.dose_response_exposure_evidence(
        [fit],
        min_concentration_uM=0.1,
        max_concentration_uM=100.0,
        endpoint_weights={"fpd_change_pct": 0.30},
    )

    assert evidence is None


def test_4pl_rejects_protective_amplitude_direction_for_scoring():
    concentrations = np.asarray([0.1, 0.3, 1.0, 3.0, 10.0, 30.0])
    responses = four_parameter_logistic(concentrations, 0.0, 60.0, 2.0, 1.2)
    fit = fit_4pl(
        concentrations,
        responses,
        endpoint="amplitude_change_pct",
        min_points=6,
        min_r_squared=0.95,
        min_monotonicity=0.95,
    )

    assert fit.success
    assert fit.monotonic_direction == "increasing"
    assert fit.harm_direction_compatible is False
    assert not fit.quality_pass
    assert "harmful direction" in fit.message


def test_4pl_accepts_harmful_amplitude_direction_for_scoring():
    concentrations = np.asarray([0.1, 0.3, 1.0, 3.0, 10.0, 30.0])
    responses = four_parameter_logistic(concentrations, 0.0, -60.0, 2.0, 1.2)
    fit = fit_4pl(
        concentrations,
        responses,
        endpoint="amplitude_change_pct",
        min_points=6,
        min_r_squared=0.95,
        min_monotonicity=0.95,
    )

    assert fit.success
    assert fit.monotonic_direction == "decreasing"
    assert fit.harm_direction_compatible is True
    assert fit.quality_pass

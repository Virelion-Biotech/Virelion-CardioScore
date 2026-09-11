from __future__ import annotations

import numpy as np
import pytest

from virelion_cardioscore.analysis.dose_response import (
    fit_4pl,
    four_parameter_logistic,
)
from virelion_cardioscore.analysis.pipeline import CardioScorePipeline


AMPLITUDE_METADATA = {"amplitude_change_pct": "decrease"}
AMPLITUDE_THRESHOLD = 20.0


def test_4pl_rejects_protective_amplitude_direction_for_scoring():
    concentrations = np.asarray([0.1, 0.3, 1.0, 3.0, 10.0, 30.0])
    responses = four_parameter_logistic(concentrations, 0.0, 60.0, 2.0, 1.2)
    fit = fit_4pl(
        concentrations,
        responses,
        endpoint="amplitude_change_pct",
        endpoint_directions=AMPLITUDE_METADATA,
        effect_threshold=AMPLITUDE_THRESHOLD,
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
        endpoint_directions=AMPLITUDE_METADATA,
        effect_threshold=AMPLITUDE_THRESHOLD,
        min_points=6,
        min_r_squared=0.95,
        min_monotonicity=0.95,
    )

    assert fit.success
    assert fit.monotonic_direction == "decreasing"
    assert fit.harm_direction_compatible is True
    assert fit.quality_pass

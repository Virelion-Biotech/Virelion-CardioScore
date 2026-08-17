from __future__ import annotations

import numpy as np
import pytest

from virelion_cardioscore.analysis.statistics import (
    bootstrap_ci,
    bootstrap_profile_difference,
)


def test_bootstrap_ci_is_reproducible():
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    first = bootstrap_ci(values, n_bootstrap=500, seed=7)
    second = bootstrap_ci(values, n_bootstrap=500, seed=7)

    assert first.to_dict() == second.to_dict()
    assert first.estimate == pytest.approx(3.0)
    assert first.ci_low <= first.estimate <= first.ci_high


def test_bootstrap_ci_rejects_tiny_sample():
    with pytest.raises(ValueError, match="At least two finite observations"):
        bootstrap_ci(np.array([1.0]), n_bootstrap=100)


def test_profile_difference_detects_separated_groups():
    result = bootstrap_profile_difference(
        np.array([1.0, 10.0]),
        {1.0: np.array([10.0, 11.0, 9.0]), 10.0: np.array([20.0, 21.0, 19.0])},
        {1.0: np.array([1.0, 2.0, 0.0]), 10.0: np.array([8.0, 9.0, 7.0])},
        n_bootstrap=500,
        seed=11,
    )

    assert result.concentrations == (1.0, 10.0)
    assert all(diff > 0 for diff in result.differences)
    assert all(low > 0 for low in result.ci_low)
    assert all(p < 0.10 for p in result.p_values)


def test_profile_difference_requires_matched_replicates():
    with pytest.raises(ValueError, match="At least two finite replicates"):
        bootstrap_profile_difference(
            np.array([1.0]),
            {1.0: np.array([1.0])},
            {1.0: np.array([0.0, 1.0])},
            n_bootstrap=100,
        )

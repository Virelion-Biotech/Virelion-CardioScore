from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from virelion_cardioscore.analysis.mixed_effects import fit_random_intercept

statsmodels = pytest.importorskip("statsmodels")


def _dataset() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    rows = []
    for plate, plate_shift in [("P1", 0.0), ("P2", 8.0), ("P3", -6.0), ("P4", 4.0)]:
        for treatment in [0, 1]:
            for _replicate in range(3):
                rows.append(
                    {
                        "plate_id": plate,
                        "treatment": treatment,
                        "fpd_ms": 100.0 + plate_shift + 15.0 * treatment + rng.normal(0, 1.0),
                    }
                )
    return pd.DataFrame(rows)


def test_random_intercept_recovers_treatment_effect_and_group_variance():
    result = fit_random_intercept(_dataset(), endpoint="fpd_ms", group_column="plate_id")

    assert result.converged
    assert result.n_groups == 4
    assert result.n_observations == 24
    assert result.treatment_effect == pytest.approx(15.0, abs=2.0)
    assert result.treatment_se > 0
    assert result.group_variance >= 0
    assert 0 <= result.icc <= 1


def test_random_intercept_requires_multiple_groups():
    frame = _dataset().query("plate_id == 'P1'")
    with pytest.raises(ValueError, match="at least two groups"):
        fit_random_intercept(frame, endpoint="fpd_ms", group_column="plate_id")


def test_random_intercept_requires_both_treatment_levels():
    frame = _dataset().query("treatment == 1")
    with pytest.raises(ValueError, match="both treatment levels"):
        fit_random_intercept(frame, endpoint="fpd_ms", group_column="plate_id")

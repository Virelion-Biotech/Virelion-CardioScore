from __future__ import annotations

import pandas as pd
import pytest

from virelion_cardioscore.analysis.mixed_effects import fit_random_intercept

statsmodels = pytest.importorskip("statsmodels")


def _base_frame() -> pd.DataFrame:
    rows = []
    values = {
        "G1": [99.0, 101.0, 109.0, 111.0],
        "G2": [105.0, 107.0, 115.0, 117.0],
        "G3": [95.0, 97.0, 105.0, 107.0],
    }
    for group, group_values in values.items():
        rows.extend(
            [
                {"biological_replicate": group, "treatment": 0, "fpd_ms": group_values[0]},
                {"biological_replicate": group, "treatment": 0, "fpd_ms": group_values[1]},
                {"biological_replicate": group, "treatment": 1, "fpd_ms": group_values[2]},
                {"biological_replicate": group, "treatment": 1, "fpd_ms": group_values[3]},
            ]
        )
    return pd.DataFrame(rows)


def test_mixed_effects_requires_three_independent_groups():
    frame = _base_frame().query("biological_replicate != 'G3'")
    with pytest.raises(ValueError, match="at least three independent groups"):
        fit_random_intercept(frame, endpoint="fpd_ms")


def test_mixed_effects_requires_two_observations_per_group():
    frame = _base_frame().iloc[[0, 1, 2, 3, 4]].copy()
    with pytest.raises(ValueError, match="at least two observations in every independent group"):
        fit_random_intercept(frame, endpoint="fpd_ms")


def test_mixed_effects_prefers_biological_replicate_over_plate():
    frame = _base_frame().assign(
        plate_id=["P1", "P1", "P2", "P2", "P3", "P3", "P4", "P4", "P5", "P5", "P6", "P6"]
    )
    result = fit_random_intercept(frame, endpoint="fpd_ms")
    assert result.group_column == "biological_replicate"
    assert result.n_groups == 3

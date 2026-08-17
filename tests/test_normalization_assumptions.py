from __future__ import annotations

import pandas as pd
import pytest

from virelion_cardioscore.analysis.normalization_assumptions import check_additive_correction_assumptions


def _balanced_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "plate_id": ["P1", "P1", "P1", "P2", "P2", "P2"],
            "vehicle": [True, True, False, True, True, False],
            "fpd_ms": [100.0, 102.0, 130.0, 120.0, 122.0, 150.0],
        }
    )


def test_balanced_design_is_usable():
    result = check_additive_correction_assumptions(
        _balanced_frame(), group_column="plate_id", endpoint="fpd_ms"
    )

    assert result.treatment_group_coverage == pytest.approx(1.0)
    assert not result.treatment_allocation_imbalanced
    assert result.usable_for_additive_correction


def test_missing_treatment_group_is_flagged():
    frame = _balanced_frame().copy()
    frame.loc[frame["plate_id"] == "P2", "vehicle"] = True

    result = check_additive_correction_assumptions(
        frame, group_column="plate_id", endpoint="fpd_ms"
    )

    assert result.treatment_group_coverage == pytest.approx(0.5)
    assert result.treatment_allocation_imbalanced
    assert not result.usable_for_additive_correction


def test_highly_heterogeneous_additive_shifts_are_flagged():
    frame = pd.DataFrame(
        {
            "plate_id": ["P1", "P1", "P2", "P2", "P3", "P3"],
            "vehicle": [True, False, True, False, True, False],
            "fpd_ms": [10.0, 20.0, 100.0, 110.0, 1000.0, 1010.0],
        }
    )

    result = check_additive_correction_assumptions(
        frame,
        group_column="plate_id",
        endpoint="fpd_ms",
        max_shift_cv_pct=25.0,
    )

    assert result.scale_like_drift_flag
    assert not result.usable_for_additive_correction


def test_missing_columns_fail_loudly():
    with pytest.raises(ValueError, match="Missing columns"):
        check_additive_correction_assumptions(
            _balanced_frame(), group_column="missing", endpoint="fpd_ms"
        )

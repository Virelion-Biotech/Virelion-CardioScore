from __future__ import annotations

import pandas as pd
import pytest

from virelion_cardioscore.analysis.normalization_validation import validate_control_anchor_correction


def _drifted_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "plate_id": ["P1", "P1", "P1", "P2", "P2", "P2"],
            "vehicle": [True, True, False, True, True, False],
            "compound": ["A"] * 6,
            "well": ["V1", "V2", "T1", "V3", "V4", "T2"],
            "fpd_ms": [100.0, 102.0, 130.0, 120.0, 122.0, 150.0],
        }
    )


def test_validation_detects_drift_reduction_and_effect_preservation():
    _, result = validate_control_anchor_correction(
        _drifted_frame(),
        group_column="plate_id",
        endpoint="fpd_ms",
    )

    assert result.control_between_sd_before == pytest.approx((20.0**2 / 2) ** 0.5)
    assert result.control_between_sd_after == pytest.approx(0.0)
    assert result.treatment_effect_rmse == pytest.approx(0.0)
    assert result.passed_drift_reduction is True
    assert result.passed_effect_preservation is True


def test_validation_rejects_missing_endpoint():
    with pytest.raises(ValueError, match="Missing columns"):
        validate_control_anchor_correction(
            _drifted_frame(),
            group_column="plate_id",
            endpoint="missing",
        )


def test_validation_result_is_serializable():
    _, result = validate_control_anchor_correction(
        _drifted_frame(),
        group_column="plate_id",
        endpoint="fpd_ms",
    )

    payload = result.to_dict()
    assert payload["endpoint"] == "fpd_ms"
    assert payload["passed_drift_reduction"] is True
    assert payload["passed_effect_preservation"] is True

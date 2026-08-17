from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from virelion_cardioscore.analysis.normalization import apply_control_anchor_correction


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "plate_id": ["P1", "P1", "P2", "P2", "P1", "P2"],
            "vehicle": [True, True, True, True, False, False],
            "compound": ["A", "A", "A", "A", "A", "A"],
            "well": ["V1", "V2", "V3", "V4", "T1", "T2"],
            "fpd_ms": [100.0, 102.0, 120.0, 122.0, 130.0, 150.0],
            "beat_rate_bpm": [60.0, 61.0, 70.0, 71.0, 80.0, 90.0],
        }
    )


def test_control_anchor_correction_aligns_group_control_means():
    corrected, diagnostic = apply_control_anchor_correction(
        _frame(),
        group_column="plate_id",
        corrected_columns=["fpd_ms"],
        min_controls_per_group=2,
    )

    controls = corrected.loc[corrected["vehicle"]].groupby("plate_id")["fpd_ms"].mean()
    assert controls.loc["P1"] == pytest.approx(111.0)
    assert controls.loc["P2"] == pytest.approx(111.0)
    assert diagnostic.n_groups == 2
    assert diagnostic.n_controls == 4
    assert diagnostic.group_shifts["P1"]["fpd_ms"] == pytest.approx(10.0)
    assert diagnostic.group_shifts["P2"]["fpd_ms"] == pytest.approx(-10.0)
    assert diagnostic.assumption_checks["fpd_ms"]["usable_for_additive_correction"] is True


def test_control_anchor_preserves_within_group_treatment_control_difference():
    corrected, _ = apply_control_anchor_correction(
        _frame(),
        group_column="plate_id",
        corrected_columns=["fpd_ms"],
        min_controls_per_group=2,
    )

    p1_difference = corrected.loc[corrected["well"] == "T1", "fpd_ms"].iloc[0] - corrected.loc[corrected["well"] == "V1", "fpd_ms"].iloc[0]
    p2_difference = corrected.loc[corrected["well"] == "T2", "fpd_ms"].iloc[0] - corrected.loc[corrected["well"] == "V3", "fpd_ms"].iloc[0]
    assert p1_difference == pytest.approx(30.0)
    assert p2_difference == pytest.approx(30.0)


def test_control_anchor_requires_sufficient_controls():
    frame = _frame().drop(index=1).drop(index=3)
    with pytest.raises(ValueError, match="at least 2 vehicle wells per group"):
        apply_control_anchor_correction(
            frame,
            group_column="plate_id",
            corrected_columns=["fpd_ms"],
            min_controls_per_group=2,
        )


def test_control_anchor_rejects_missing_column():
    with pytest.raises(ValueError, match="absent"):
        apply_control_anchor_correction(
            _frame(),
            group_column="missing_plate",
            corrected_columns=["fpd_ms"],
        )


def test_control_anchor_is_reproducible_and_diagnostic_serializes():
    corrected, diagnostic = apply_control_anchor_correction(
        _frame(),
        group_column="plate_id",
        corrected_columns=["fpd_ms", "beat_rate_bpm"],
        min_controls_per_group=2,
    )

    assert np.isfinite(corrected["fpd_ms"]).all()
    payload = diagnostic.to_dict()
    assert payload["corrected_columns"] == ["fpd_ms", "beat_rate_bpm"]
    assert set(payload["group_shifts"]) == {"P1", "P2"}
    assert set(payload["assumption_checks"]) == {"fpd_ms", "beat_rate_bpm"}


def test_control_anchor_fails_closed_when_treatment_is_missing_from_a_group():
    frame = _frame().copy()
    frame.loc[frame["well"] == "T2", "vehicle"] = True

    with pytest.raises(ValueError, match="Treatment observations are missing"):
        apply_control_anchor_correction(
            frame,
            group_column="plate_id",
            corrected_columns=["fpd_ms"],
            min_controls_per_group=2,
            require_all_groups=True,
        )


def test_control_anchor_can_report_but_not_fail_on_assumption_warning():
    frame = _frame().copy()
    frame.loc[frame["well"] == "T2", "vehicle"] = True

    corrected, diagnostic = apply_control_anchor_correction(
        frame,
        group_column="plate_id",
        corrected_columns=["fpd_ms"],
        min_controls_per_group=2,
        require_all_groups=True,
        fail_on_assumption_violation=False,
    )

    assert np.isfinite(corrected["fpd_ms"]).all()
    assert diagnostic.assumption_checks["fpd_ms"]["treatment_allocation_imbalanced"] is True

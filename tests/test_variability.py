from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from virelion_cardioscore.analysis.variability import (
    control_variability,
    standardized_treatment_separation,
)


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "compound": ["A", "A", "A", "A", "B", "B", "B", "B"],
            "vehicle": [True, True, False, False, True, True, False, False],
            "plate_id": ["P1", "P1", "P1", "P1", "P2", "P2", "P2", "P2"],
            "fpd_ms": [100.0, 102.0, 130.0, 135.0, 120.0, 122.0, 150.0, 152.0],
            "beat_rate_bpm": [60.0, 61.0, 70.0, 71.0, 60.0, 61.0, 72.0, 73.0],
        }
    )


def test_control_variability_reports_stable_controls():
    result = control_variability(
        _frame(), endpoint_columns=["fpd_ms"], max_control_cv_pct=20.0
    )

    row = result.iloc[0]
    assert row["n_groups"] == 2
    assert row["n_controls"] == 4
    assert row["status"] == "stable"
    assert row["between_group_sd"] == pytest.approx(np.sqrt(200.0))


def test_control_variability_flags_high_group_variability():
    frame = _frame().copy()
    frame.loc[frame["plate_id"] == "P2", "fpd_ms"] = [200.0, 210.0, 150.0, 152.0]

    result = control_variability(
        frame, endpoint_columns=["fpd_ms"], max_control_cv_pct=10.0
    )

    assert result.iloc[0]["status"] == "high_variability"
    assert result.iloc[0]["control_cv_pct"] > 10.0


def test_control_variability_requires_group_metadata():
    with pytest.raises(ValueError, match="requires one of"):
        control_variability(_frame().drop(columns=["plate_id"]))


def test_control_variability_requires_two_groups_for_between_group_estimate():
    frame = _frame().query("plate_id == 'P1'")
    result = control_variability(frame, endpoint_columns=["fpd_ms"])

    assert result.iloc[0]["status"] == "insufficient_groups"
    assert pd.isna(result.iloc[0]["between_group_sd"])


def test_standardized_treatment_separation_is_group_specific():
    result = standardized_treatment_separation(
        _frame(), endpoint="fpd_ms", group_column="plate_id"
    )

    assert len(result) == 2
    assert (result["n_controls"] == 2).all()
    assert (result["n_treated"] == 2).all()
    assert (result["standardized_separation"] > 0).all()

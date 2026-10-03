"""Tests for control-anchor normalization."""

from __future__ import annotations

import pandas as pd
import pytest

from virelion_cardioscore.analysis.normalization import apply_control_anchor_correction


def _heterogeneous_controls() -> pd.DataFrame:
    rows = []
    for plate, control_value, n_controls in [("P1", 100.0, 2), ("P2", 200.0, 6)]:
        for i in range(n_controls):
            rows.append(
                {
                    "compound": "A",
                    "plate_id": plate,
                    "well": f"{plate}_V{i}",
                    "vehicle": True,
                    "fpd_ms": control_value,
                }
            )
        rows.append(
            {
                "compound": "A",
                "plate_id": plate,
                "well": f"{plate}_T0",
                "vehicle": False,
                "fpd_ms": control_value * 1.2,
            }
        )
    return pd.DataFrame(rows)


def test_control_anchor_target_is_not_weighted_by_control_count():
    df = _heterogeneous_controls()

    corrected, diagnostic = apply_control_anchor_correction(
        df,
        group_column="plate_id",
        corrected_columns=["fpd_ms"],
        min_controls_per_group=2,
        min_treated_per_group=1,
        require_all_groups=True,
        require_treatment_in_all_groups=True,
        # The fixture deliberately has a 100 vs 200 plate shift. Relax the
        # shift guard here because this test isolates equal-weight target
        # construction rather than acceptance/rejection of that shift.
        max_shift_cv_pct=100.0,
    )

    assert diagnostic.target_means["fpd_ms"] == pytest.approx(150.0)
    control_means = corrected.loc[corrected["vehicle"], "fpd_ms"].groupby(corrected.loc[corrected["vehicle"], "plate_id"]).mean()
    assert control_means["P1"] == pytest.approx(150.0)
    assert control_means["P2"] == pytest.approx(150.0)
    assert diagnostic.group_shifts["P1"]["fpd_ms"] == pytest.approx(50.0)
    assert diagnostic.group_shifts["P2"]["fpd_ms"] == pytest.approx(-50.0)

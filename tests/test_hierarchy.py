from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from virelion_cardioscore.analysis.hierarchy import (
    count_independent_units,
    detect_hierarchy,
    hierarchy_columns,
    summarize_experimental_units,
)


def _effects() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "compound": ["A"] * 8,
            "concentration_uM": [1.0] * 8,
            "well": ["W1", "W2", "W3", "W4", "W5", "W6", "W7", "W8"],
            "biological_replicate": ["B1", "B1", "B2", "B2", "B3", "B3", "B4", "B4"],
            "plate_id": ["P1"] * 4 + ["P2"] * 4,
            "fpd_change_pct": [10, 12, 20, 22, 30, 32, 40, 42],
            "beat_rate_change_pct": np.zeros(8),
            "amplitude_change_pct": np.zeros(8),
            "stv_increase": np.zeros(8),
            "triangulation_proxy_change": np.zeros(8),
        }
    )


def test_detect_hierarchy_prefers_explicit_biological_replicate():
    spec = detect_hierarchy(_effects())
    assert spec.biological_unit == "biological_replicate"
    assert spec.plate_unit == "plate_id"
    assert hierarchy_columns(_effects()) == ["biological_replicate", "plate_id"]


def test_summarize_experimental_units_averages_technical_wells_within_biological_unit():
    summary = summarize_experimental_units(_effects())

    assert len(summary) == 4
    assert set(summary["biological_replicate"]) == {"B1", "B2", "B3", "B4"}
    b1 = summary.loc[summary["biological_replicate"] == "B1"].iloc[0]
    assert b1["n_wells"] == 2
    assert b1["fpd_change_pct_mean"] == pytest.approx(11.0)


def test_independent_unit_count_is_not_well_count():
    summary = summarize_experimental_units(_effects())
    counts = count_independent_units(summary)

    assert counts.iloc[0]["n_independent_units"] == 4
    assert counts.iloc[0]["n_independent_units"] < 8


def test_legacy_well_only_data_still_works():
    df = _effects().drop(columns=["biological_replicate", "plate_id"])
    summary = summarize_experimental_units(df)

    assert len(summary) == 8
    assert "well" in summary.columns
    assert summary["n_wells"].eq(1).all()

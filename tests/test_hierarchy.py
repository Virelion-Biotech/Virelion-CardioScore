from __future__ import annotations

import pandas as pd
import pytest

from virelion_cardioscore.analysis.hierarchy import (
    SUPPORTED_SCORING_UNITS,
    aggregate_to_scoring_units,
    count_independent_units,
    summarize_experimental_units,
)


def _effects() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "compound": ["A"] * 8,
            "concentration_uM": [1.0] * 8,
            "well": [f"W{i}" for i in range(8)],
            "biological_replicate": ["B1", "B1", "B2", "B2", "B3", "B3", "B4", "B4"],
            "batch_id": ["X"] * 8,
            "plate_id": ["P1"] * 4 + ["P2"] * 4,
            "site": ["S1"] * 8,
            "vehicle": [False] * 8,
            "fpd_change_pct": [10.0, 12.0, 20.0, 22.0, 30.0, 32.0, 40.0, 42.0],
        }
    )


def test_summarize_experimental_units_averages_technical_wells_within_biological_unit():
    summary = summarize_experimental_units(_effects())
    b1 = summary.loc[summary["biological_replicate"] == "B1"].iloc[0]
    assert b1["n_wells"] == 2
    assert b1["fpd_change_pct_mean"] == pytest.approx(11.0)


def test_independent_unit_count_is_not_well_count():
    summary = summarize_experimental_units(_effects())
    counts = count_independent_units(summary)

    assert counts.iloc[0]["n_independent_units"] == 4
    assert counts.iloc[0]["n_independent_units"] < 8


def test_supported_scoring_units_are_explicit():
    assert SUPPORTED_SCORING_UNITS == ("auto", "well", "biological_replicate", "batch", "plate")


def test_biological_replicate_scoring_aggregates_technical_wells():
    aggregated = aggregate_to_scoring_units(_effects(), scoring_unit="biological_replicate")

    assert len(aggregated) == 4
    assert aggregated.set_index("biological_replicate").loc["B1", "fpd_change_pct"] == pytest.approx(11.0)

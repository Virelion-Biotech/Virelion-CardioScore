from __future__ import annotations

import pandas as pd
import pytest

from virelion_cardioscore.analysis.hierarchy import aggregate_to_scoring_units


def _effects_with_custom_unit_column() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "compound": ["A", "A", "A", "A"],
            "concentration_uM": [1.0, 1.0, 1.0, 1.0],
            "well": ["W1", "W2", "W3", "W4"],
            "bio_id": ["B1", "B1", "B2", "B2"],
            "fpd_change_pct": [10.0, 20.0, 30.0, 40.0],
            "beat_rate_change_pct": [0.0] * 4,
            "amplitude_change_pct": [0.0] * 4,
            "stv_increase": [0.0] * 4,
            "triangulation_proxy_change": [0.0] * 4,
            "max_effect_pct": [10.0, 20.0, 30.0, 40.0],
        }
    )


def test_configured_biological_unit_column_is_used():
    aggregated = aggregate_to_scoring_units(
        _effects_with_custom_unit_column(),
        scoring_unit="biological_replicate",
        biological_unit_column="bio_id",
    )

    assert len(aggregated) == 2
    assert set(aggregated["bio_id"]) == {"B1", "B2"}
    assert aggregated.set_index("bio_id").loc["B1", "fpd_change_pct"] == pytest.approx(15.0)


def test_missing_configured_unit_column_is_rejected():
    with pytest.raises(ValueError, match="requires column 'bio_id'"):
        aggregate_to_scoring_units(
            _effects_with_custom_unit_column(),
            scoring_unit="biological_replicate",
            biological_unit_column="not_present",
        )

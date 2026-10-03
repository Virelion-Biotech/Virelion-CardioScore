from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from virelion_cardioscore.analysis.hierarchy import aggregate_to_scoring_units


def _effects(**overrides) -> pd.DataFrame:
    base = {
        "compound": ["A"] * 4,
        "concentration_uM": [1.0] * 4,
        "well": ["W1", "W2", "W3", "W4"],
        "biological_replicate": ["BR1", "BR1", "BR1", "BR1"],
        "site": ["S1"] * 4,
        "experiment_id": ["E1"] * 4,
        "fpd_change_pct": [1.0, 2.0, 3.0, 4.0],
        "beat_rate_change_pct": [1.0] * 4,
        "amplitude_change_pct": [-1.0] * 4,
        "stv_increase": [0.01] * 4,
        "triangulation_proxy_change": [0.02] * 4,
        "max_effect_pct": [2.0] * 4,
    }
    base.update(overrides)
    return pd.DataFrame(base)


def test_hierarchy_aggregation_fails_on_partial_endpoint_data():
    frame = _effects()
    frame.loc[2, "fpd_change_pct"] = np.nan

    with pytest.raises(ValueError, match="partial endpoint data are not allowed"):
        aggregate_to_scoring_units(frame, scoring_unit="biological_replicate")


def test_hierarchy_aggregation_fails_on_non_numeric_endpoint_data():
    frame = _effects()
    frame["fpd_change_pct"] = frame["fpd_change_pct"].astype(object)
    frame.loc[2, "fpd_change_pct"] = "bad"

    with pytest.raises(ValueError, match="partial endpoint data are not allowed"):
        aggregate_to_scoring_units(frame, scoring_unit="biological_replicate")


def test_biological_replicate_ids_are_namespaced_by_experiment():
    first = _effects()
    second = _effects(
        well=["W5", "W6", "W7", "W8"],
        site=["S1"] * 4,
        experiment_id=["E2"] * 4,
        fpd_change_pct=[10.0, 10.0, 10.0, 10.0],
    )
    frame = pd.concat([first, second], ignore_index=True)

    aggregated = aggregate_to_scoring_units(frame, scoring_unit="biological_replicate")

    assert len(aggregated) == 2
    assert set(aggregated["experiment_id"]) == {"E1", "E2"}
    assert set(aggregated["well"]) == {"biological_replicate:S1:E1:BR1", "biological_replicate:S1:E2:BR1"}

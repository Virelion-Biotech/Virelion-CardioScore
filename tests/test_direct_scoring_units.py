from __future__ import annotations

import pandas as pd
import pytest

from virelion_cardioscore.analysis.cipa_scoring import CardioScoreEngine


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "compound": ["A", "A", "A", "A"],
            "biological_replicate": ["BR1", "BR1", "BR2", "BR2"],
            "site": ["S1", "S1", "S1", "S1"],
            "experiment_id": ["E1", "E1", "E1", "E1"],
            "fpd_change_pct": [20.0, 21.0, 22.0, 23.0],
            "beat_rate_change_pct": [0.0] * 4,
            "amplitude_change_pct": [0.0] * 4,
            "stv_increase": [0.0] * 4,
            "triangulation_proxy": [0.0] * 4,
        }
    )


def test_direct_scoring_reports_unique_independent_units():
    result = CardioScoreEngine().score_feature_table(
        _frame(), independent_unit_column="biological_replicate"
    )
    assert len(result) == 1
    assert result[0].n_wells == 4
    assert result[0].n_independent_units == 2


def test_direct_scoring_namespaces_reused_unit_ids():
    frame = pd.concat(
        [
            _frame(),
            _frame().assign(
                experiment_id="E2",
                fpd_change_pct=[40.0, 41.0, 42.0, 43.0],
            ),
        ],
        ignore_index=True,
    )
    result = CardioScoreEngine().score_feature_table(
        frame, independent_unit_column="biological_replicate"
    )
    assert result[0].n_independent_units == 4


def test_direct_scoring_rejects_missing_independent_unit_ids():
    frame = _frame()
    frame.loc[0, "biological_replicate"] = None
    with pytest.raises(ValueError, match="missing or blank identifiers"):
        CardioScoreEngine().score_feature_table(
            frame, independent_unit_column="biological_replicate"
        )

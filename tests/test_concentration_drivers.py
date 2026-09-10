from __future__ import annotations

import pandas as pd
import pytest

from virelion_cardioscore.analysis.concentration_drivers import (
    DEFAULT_ENDPOINT_THRESHOLDS,
    summarize_concentration_drivers,
)


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "compound": ["A"] * 4,
            "concentration_uM": [0.1, 1.0, 10.0, 100.0],
            "vehicle": [False] * 4,
            "fpd_change_pct": [2.0, 5.0, 8.0, 9.0],
            "beat_rate_change_pct": [1.0, 2.0, 4.0, 5.0],
            "amplitude_change_pct": [-5.0, -10.0, -21.0, -30.0],
            "stv_increase": [0.01, 0.05, 0.10, 0.16],
            "triangulation_proxy_change": [0.01, 0.10, 0.19, 0.21],
        }
    )


def test_default_endpoint_thresholds_are_endpoint_specific():
    assert DEFAULT_ENDPOINT_THRESHOLDS["fpd_change_pct"] == pytest.approx(10.0)
    assert DEFAULT_ENDPOINT_THRESHOLDS["beat_rate_change_pct"] == pytest.approx(15.0)
    assert DEFAULT_ENDPOINT_THRESHOLDS["amplitude_change_pct"] == pytest.approx(20.0)


def test_amplitude_driver_uses_harmful_decrease_direction():
    result = summarize_concentration_drivers(_frame(), endpoint="amplitude_change_pct")
    assert result["n_supporting_concentrations"] == 2
    assert result["direction"] == "decrease"
    assert result["worst_value"] == pytest.approx(-30.0)


def test_driver_requires_endpoint_column():
    with pytest.raises(ValueError, match="missing"):
        summarize_concentration_drivers(_frame().drop(columns=["stv_increase"]), endpoint="stv_increase")

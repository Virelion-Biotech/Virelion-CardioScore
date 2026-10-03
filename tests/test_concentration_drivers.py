from __future__ import annotations

import pandas as pd
import pytest

from virelion_cardioscore.analysis.concentration_drivers import (
    DEFAULT_ENDPOINT_THRESHOLDS,
    concentration_drivers,
)


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "compound": ["A"] * 4,
            "concentration_uM": [0.1, 1.0, 10.0, 100.0],
            "fpd_change_pct_mean": [2.0, 5.0, 8.0, 9.0],
            "beat_rate_change_pct_mean": [1.0, 2.0, 4.0, 5.0],
            "amplitude_change_pct_mean": [-5.0, -10.0, -21.0, -30.0],
            "stv_increase_mean": [0.01, 0.05, 0.10, 0.16],
            "triangulation_proxy_change_mean": [0.01, 0.10, 0.19, 0.21],
        }
    )


def test_default_endpoint_thresholds_are_endpoint_specific():
    assert DEFAULT_ENDPOINT_THRESHOLDS["fpd_change_pct"] == pytest.approx(10.0)
    assert DEFAULT_ENDPOINT_THRESHOLDS["beat_rate_change_pct"] == pytest.approx(15.0)
    assert DEFAULT_ENDPOINT_THRESHOLDS["amplitude_change_pct"] == pytest.approx(20.0)


def test_amplitude_driver_uses_harmful_decrease_direction():
    result = concentration_drivers(_frame(), endpoint_directions={"amplitude_change_pct": "decrease"})
    row = result[result["endpoint"] == "amplitude_change_pct"].iloc[0]
    assert int(row["concentrations_supporting_signal"]) == 2
    assert row["direction"] == "decrease"
    assert row["driver_value"] == pytest.approx(-30.0)


def test_driver_ignores_unavailable_optional_endpoint_column():
    broken = _frame().drop(columns=["stv_increase_mean"])
    result = concentration_drivers(broken)
    assert "stv_increase" not in set(result["endpoint"])


def test_driver_requires_explicitly_requested_endpoint_column():
    broken = _frame().drop(columns=["stv_increase_mean"])
    with pytest.raises(ValueError, match="stv_increase"):
        concentration_drivers(broken, endpoint_directions={"stv_increase": "increase"})

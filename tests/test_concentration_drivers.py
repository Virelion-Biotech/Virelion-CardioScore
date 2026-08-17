import pandas as pd
import pytest

from virelion_cardioscore.analysis.concentration_drivers import concentration_drivers


def _summary(values: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "compound": ["A"] * len(values),
            "concentration_uM": [1.0, 3.0, 10.0, 30.0],
            "fpd_change_pct_mean": values,
            "beat_rate_change_pct_mean": [0.0] * len(values),
            "amplitude_change_pct_mean": [0.0] * len(values),
            "stv_increase_mean": [0.0] * len(values),
            "triangulation_proxy_change_mean": [0.0] * len(values),
        }
    )


def test_driver_identifies_worst_fpd_concentration():
    result = concentration_drivers(_summary([5.0, 12.0, 25.0, 18.0]))
    row = result.loc[result["endpoint"] == "fpd_change_pct"].iloc[0]

    assert row["driver_concentration_uM"] == pytest.approx(10.0)
    assert row["driver_value"] == pytest.approx(25.0)
    assert row["concentrations_supporting_signal"] == 3
    assert row["support_fraction"] == pytest.approx(0.75)


def test_single_terminal_spike_has_low_support_fraction():
    result = concentration_drivers(_summary([0.0, 2.0, 3.0, 100.0]))
    row = result.loc[result["endpoint"] == "fpd_change_pct"].iloc[0]

    assert row["driver_concentration_uM"] == pytest.approx(30.0)
    assert row["concentrations_supporting_signal"] == 1
    assert row["support_fraction"] == pytest.approx(0.25)


def test_amplitude_driver_uses_harmful_decrease_direction():
    summary = _summary([0.0, 0.0, 0.0, 0.0])
    summary["amplitude_change_pct_mean"] = [-5.0, -12.0, -30.0, -20.0]
    result = concentration_drivers(summary)
    row = result.loc[result["endpoint"] == "amplitude_change_pct"].iloc[0]

    assert row["driver_concentration_uM"] == pytest.approx(10.0)
    assert row["driver_value"] == pytest.approx(-30.0)
    assert row["concentrations_supporting_signal"] == 3


def test_stv_uses_fractional_scale_for_threshold():
    summary = _summary([0.0, 0.0, 0.0, 0.0])
    summary["stv_increase_mean"] = [0.05, 0.15, 0.30, 0.05]
    result = concentration_drivers(summary)
    row = result.loc[result["endpoint"] == "stv_increase"].iloc[0]

    assert row["driver_concentration_uM"] == pytest.approx(10.0)
    assert row["concentrations_supporting_signal"] == 2


def test_missing_required_columns_are_rejected():
    with pytest.raises(ValueError, match="missing"):
        concentration_drivers(pd.DataFrame({"compound": ["A"]}))

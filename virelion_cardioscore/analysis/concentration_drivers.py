"""Concentration provenance diagnostics for compound-level endpoint aggregation.

This module does not change CardioScore. It identifies which tested concentration
produced the compound-level endpoint value under the configured direction and
quantifies how broadly the response supports that worst-case observation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


ENDPOINT_DIRECTIONS = {
    "fpd_change_pct": "absolute",
    "beat_rate_change_pct": "absolute",
    "amplitude_change_pct": "decrease",
    "stv_increase": "increase",
    "triangulation_proxy_change": "increase",
}


@dataclass(frozen=True)
class ConcentrationDriver:
    """Provenance for one compound/endpoint worst-case response."""

    compound: str
    endpoint: str
    direction: str
    driver_concentration_uM: float
    driver_value: float
    concentrations_tested: int
    concentrations_supporting_signal: int
    support_fraction: float

    def to_dict(self) -> dict[str, object]:
        return {
            "compound": self.compound,
            "endpoint": self.endpoint,
            "direction": self.direction,
            "driver_concentration_uM": self.driver_concentration_uM,
            "driver_value": self.driver_value,
            "concentrations_tested": self.concentrations_tested,
            "concentrations_supporting_signal": self.concentrations_supporting_signal,
            "support_fraction": self.support_fraction,
        }


def _select_driver(values: pd.Series, direction: str) -> int:
    clean = pd.to_numeric(values, errors="coerce")
    if clean.notna().sum() == 0:
        raise ValueError("No finite endpoint values are available for driver analysis.")
    if direction == "decrease":
        return int(clean.idxmin())
    if direction == "increase":
        return int(clean.idxmax())
    if direction == "absolute":
        return int(clean.abs().idxmax())
    raise ValueError(f"Unsupported endpoint direction: {direction!r}.")


def concentration_drivers(
    concentration_summary: pd.DataFrame,
    *,
    endpoint_directions: dict[str, str] | None = None,
    effect_threshold_pct: float = 10.0,
) -> pd.DataFrame:
    """Identify worst-case concentrations and signal support for each endpoint.

    ``support_fraction`` is the fraction of tested concentrations whose absolute
    endpoint effect reaches ``effect_threshold_pct``. For directional endpoints,
    only the harmful direction is counted.
    """
    if concentration_summary.empty:
        return pd.DataFrame()
    if effect_threshold_pct < 0:
        raise ValueError("effect_threshold_pct must be non-negative.")
    required = {"compound", "concentration_uM"}
    missing = sorted(required - set(concentration_summary.columns))
    if missing:
        raise ValueError(f"Concentration summary is missing columns: {missing}.")

    directions = {**ENDPOINT_DIRECTIONS, **(endpoint_directions or {})}
    rows: list[dict[str, object]] = []
    for compound, group in concentration_summary.groupby("compound", sort=True):
        group = group.sort_values("concentration_uM")
        for endpoint, direction in directions.items():
            column = f"{endpoint}_mean"
            if column not in group.columns:
                continue
            finite = pd.to_numeric(group[column], errors="coerce")
            finite_group = group.loc[finite.notna()].copy()
            finite_values = finite.loc[finite.notna()]
            if finite_values.empty:
                continue

            driver_index = _select_driver(finite_values, direction)
            driver_row = finite_group.loc[driver_index]
            if direction == "decrease":
                harmful = finite_values <= -effect_threshold_pct
            elif direction == "increase":
                harmful = finite_values >= effect_threshold_pct / 100.0 if endpoint in {"stv_increase", "triangulation_proxy_change"} else finite_values >= effect_threshold_pct
            else:
                harmful = finite_values.abs() >= effect_threshold_pct
            support_count = int(harmful.sum())
            n_tested = int(finite_values.size)
            rows.append(
                ConcentrationDriver(
                    compound=str(compound),
                    endpoint=endpoint,
                    direction=direction,
                    driver_concentration_uM=float(driver_row["concentration_uM"]),
                    driver_value=float(driver_row[column]),
                    concentrations_tested=n_tested,
                    concentrations_supporting_signal=support_count,
                    support_fraction=float(support_count / n_tested),
                ).to_dict()
            )
    return pd.DataFrame(rows)

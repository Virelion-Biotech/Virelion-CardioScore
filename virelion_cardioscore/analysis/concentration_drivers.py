"""Concentration provenance diagnostics for compound-level endpoint aggregation.

This module does not change CardioScore. It identifies which tested concentration
produced the compound-level endpoint value under the configured direction and
quantifies how broadly the response supports that worst-case observation.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


ENDPOINT_DIRECTIONS = {
    "fpd_change_pct": "absolute",
    "beat_rate_change_pct": "absolute",
    "amplitude_change_pct": "decrease",
    "stv_increase": "increase",
    "triangulation_proxy_change": "increase",
}

DEFAULT_ENDPOINT_THRESHOLDS = {
    "fpd_change_pct": 10.0,
    "beat_rate_change_pct": 15.0,
    "amplitude_change_pct": 20.0,
    "stv_increase": 0.15,
    "triangulation_proxy_change": 0.20,
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
    endpoint_thresholds: dict[str, float] | None = None,
    effect_threshold_pct: float | None = None,
) -> pd.DataFrame:
    """Identify worst-case concentrations and signal support for each endpoint.

    ``endpoint_thresholds`` should use the same native units as the endpoint
    values. The legacy ``effect_threshold_pct`` argument is retained for API
    compatibility, but when supplied it is only used for endpoints not present
    in ``endpoint_thresholds`` or the defaults.
    """
    if concentration_summary.empty:
        return pd.DataFrame()
    if effect_threshold_pct is not None and effect_threshold_pct < 0:
        raise ValueError("effect_threshold_pct must be non-negative.")
    required = {"compound", "concentration_uM"}
    missing = sorted(required - set(concentration_summary.columns))
    if missing:
        raise ValueError(f"Concentration summary is missing columns: {missing}.")

    directions = {**ENDPOINT_DIRECTIONS, **(endpoint_directions or {})}
    thresholds = {**DEFAULT_ENDPOINT_THRESHOLDS, **(endpoint_thresholds or {})}
    if effect_threshold_pct is not None:
        for endpoint in directions:
            thresholds.setdefault(endpoint, effect_threshold_pct / 100.0 if endpoint in {"stv_increase", "triangulation_proxy_change"} else effect_threshold_pct)

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
            if endpoint not in thresholds:
                raise ValueError(f"No effect threshold is configured for endpoint {endpoint!r}.")
            threshold = float(thresholds[endpoint])
            if threshold < 0:
                raise ValueError(f"Effect threshold for endpoint {endpoint!r} cannot be negative.")
            if direction == "decrease":
                harmful = finite_values <= -threshold
            elif direction == "increase":
                harmful = finite_values >= threshold
            else:
                harmful = finite_values.abs() >= threshold
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

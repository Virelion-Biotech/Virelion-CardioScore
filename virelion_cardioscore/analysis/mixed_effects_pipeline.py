"""Pipeline-facing helpers for optional hierarchical inference."""

from __future__ import annotations

import pandas as pd

from virelion_cardioscore.analysis.mixed_effects import MixedEffectsResult, fit_random_intercept


def fit_compound_concentration_mixed_effects(
    df: pd.DataFrame,
    *,
    group_column: str,
    endpoints: list[str],
    vehicle_column: str = "vehicle",
) -> list[dict]:
    """Fit endpoint-level random-intercept models per compound and concentration.

    Treatment is derived from the vehicle flag: vehicle wells are coded 0 and
    treated wells 1. Concentrations are analyzed separately so distinct doses
    are never pooled into a single treatment effect.
    """
    required = {"compound", "concentration_uM", vehicle_column, group_column}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Mixed-effects pipeline helper is missing columns: {missing}.")

    working = df.copy()
    working["_treatment"] = (~working[vehicle_column].astype(bool)).astype(int)
    rows: list[dict] = []

    for (compound, concentration), subset in working.groupby(
        ["compound", "concentration_uM"], sort=True
    ):
        for endpoint in endpoints:
            if endpoint not in subset.columns:
                continue
            try:
                result: MixedEffectsResult = fit_random_intercept(
                    subset,
                    endpoint=endpoint,
                    treatment_column="_treatment",
                    group_column=group_column,
                )
            except (ValueError, ImportError) as exc:
                rows.append(
                    {
                        "compound": str(compound),
                        "concentration_uM": concentration,
                        "endpoint": endpoint,
                        "status": "not_estimable",
                        "error": str(exc),
                    }
                )
                continue

            row = result.to_dict()
            row.update(
                {
                    "compound": str(compound),
                    "concentration_uM": concentration,
                    "status": "ok" if result.converged else "not_converged",
                    "error": None,
                }
            )
            rows.append(row)

    return rows

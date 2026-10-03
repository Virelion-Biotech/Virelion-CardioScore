"""Pipeline-facing helpers for optional hierarchical inference."""

from __future__ import annotations

import pandas as pd

from virelion_cardioscore.analysis.mixed_effects import MixedEffectsResult, fit_random_intercept
from virelion_cardioscore.utils.coercion import coerce_bool_series


def fit_compound_concentration_mixed_effects(
    df: pd.DataFrame,
    *,
    group_column: str,
    endpoints: list[str],
    vehicle_column: str = "vehicle",
    treatment_column: str = "_treatment",
) -> list[dict]:
    """Fit endpoint-level random-intercept models per compound and concentration.

    Treatment is derived from the vehicle flag unless the configured
    ``treatment_column`` already exists. Concentrations are analyzed separately
    so distinct doses are never pooled into a single treatment effect.
    """
    required = {"compound", "concentration_uM", group_column}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Mixed-effects pipeline helper is missing columns: {missing}.")
    if vehicle_column not in df.columns and treatment_column not in df.columns:
        raise ValueError(
            f"Mixed-effects inference requires either vehicle_column={vehicle_column!r} "
            f"or treatment_column={treatment_column!r}."
        )

    working = df.copy()
    if treatment_column in working.columns:
        numeric_treatment = pd.to_numeric(working[treatment_column], errors="coerce")
        if numeric_treatment.isna().any() or not set(numeric_treatment.dropna().unique()).issubset({0, 1}):
            raise ValueError(f"Treatment column {treatment_column!r} must contain only 0/1 values.")
        working[treatment_column] = numeric_treatment.astype(int)
        if vehicle_column in working.columns:
            vehicle_bool = coerce_bool_series(working[vehicle_column], name=vehicle_column)
            derived_treatment = (~vehicle_bool).astype(int)
            if not working[treatment_column].equals(derived_treatment):
                raise ValueError(
                    f"Treatment column {treatment_column!r} conflicts with {vehicle_column!r}; "
                    "explicit treatment metadata must agree with the vehicle flag."
                )
    else:
        vehicle_bool = coerce_bool_series(working[vehicle_column], name=vehicle_column)
        working[treatment_column] = (~vehicle_bool).astype(int)

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
                    treatment_column=treatment_column,
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

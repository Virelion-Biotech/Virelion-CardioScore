"""Compare conventional and hierarchical treatment-effect estimates."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compare_effect_estimates(
    conventional: pd.DataFrame,
    mixed_effects: pd.DataFrame,
    *,
    endpoint_column: str = "fpd_change_pct_mean",
    mixed_effect_column: str = "treatment_effect",
    effect_unit_column: str = "effect_unit",
) -> pd.DataFrame:
    """Align estimates only when both inputs explicitly declare the same effect unit.

    A treatment effect expressed in milliseconds cannot be compared numerically
    with a percentage-change estimate. Older versions identified rows only by
    endpoint name, which allowed exactly that invalid comparison. Both inputs
    must now carry an explicit ``effect_unit`` column, and every aligned pair
    must declare the same unit.
    """
    required_conventional = {
        "compound",
        "concentration_uM",
        endpoint_column,
        effect_unit_column,
    }
    required_mixed = {
        "compound",
        "concentration_uM",
        "endpoint",
        mixed_effect_column,
        "status",
        effect_unit_column,
    }
    missing_conventional = sorted(required_conventional - set(conventional.columns))
    missing_mixed = sorted(required_mixed - set(mixed_effects.columns))
    if missing_conventional:
        raise ValueError(f"Conventional estimates are missing columns: {missing_conventional}.")
    if missing_mixed:
        raise ValueError(f"Mixed-effects estimates are missing columns: {missing_mixed}.")

    conventional_long = conventional[
        ["compound", "concentration_uM", endpoint_column, effect_unit_column]
    ].copy()
    conventional_long["endpoint"] = endpoint_column
    conventional_long = conventional_long.rename(columns={endpoint_column: "conventional_effect"})

    mixed = mixed_effects[mixed_effects["status"] == "ok"].copy()
    mixed = mixed[
        ["compound", "concentration_uM", "endpoint", mixed_effect_column, effect_unit_column]
    ].rename(columns={mixed_effect_column: "mixed_effect"})

    merged = conventional_long.merge(
        mixed,
        on=["compound", "concentration_uM", "endpoint"],
        how="inner",
        suffixes=("_conventional", "_mixed"),
    )
    if merged.empty:
        return merged.assign(
            absolute_difference=pd.Series(dtype=float),
            relative_difference_pct=pd.Series(dtype=float),
            direction_agreement=pd.Series(dtype=bool),
        )

    units_match = (
        merged[f"{effect_unit_column}_conventional"].astype(str).str.strip().str.lower()
        == merged[f"{effect_unit_column}_mixed"].astype(str).str.strip().str.lower()
    )
    if not units_match.all():
        mismatched = merged.loc[~units_match, [
            "compound",
            "concentration_uM",
            "endpoint",
            f"{effect_unit_column}_conventional",
            f"{effect_unit_column}_mixed",
        ]].to_dict(orient="records")
        raise ValueError(
            "Cannot compare effect estimates with different units: "
            f"{mismatched}"
        )

    merged["effect_unit"] = merged[f"{effect_unit_column}_conventional"]
    merged["absolute_difference"] = (
        merged["mixed_effect"] - merged["conventional_effect"]
    ).abs()
    denominator = merged["conventional_effect"].abs()
    merged["relative_difference_pct"] = np.where(
        denominator > 1e-12,
        merged["absolute_difference"] / denominator * 100.0,
        np.nan,
    )
    merged["direction_agreement"] = (
        np.sign(merged["mixed_effect"]) == np.sign(merged["conventional_effect"])
    )
    return merged


def summarize_effect_concordance(comparison: pd.DataFrame) -> dict:
    """Summarize direction and magnitude agreement across aligned estimates."""
    if comparison.empty:
        return {
            "n_comparisons": 0,
            "direction_agreement_rate": np.nan,
            "mean_absolute_difference": np.nan,
            "median_absolute_difference": np.nan,
            "mean_relative_difference_pct": np.nan,
        }

    relative = pd.to_numeric(comparison["relative_difference_pct"], errors="coerce").dropna()
    absolute = pd.to_numeric(comparison["absolute_difference"], errors="coerce").dropna()
    return {
        "n_comparisons": int(len(comparison)),
        "direction_agreement_rate": float(comparison["direction_agreement"].mean()),
        "mean_absolute_difference": float(absolute.mean()) if not absolute.empty else np.nan,
        "median_absolute_difference": float(absolute.median()) if not absolute.empty else np.nan,
        "mean_relative_difference_pct": float(relative.mean()) if not relative.empty else np.nan,
    }

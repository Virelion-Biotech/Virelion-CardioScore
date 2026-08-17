"""Sensitivity analysis for CardioScore endpoint weights.

This module is diagnostic only: it does not alter the production scoring defaults.
It perturbs one configured endpoint weight at a time, renormalizes the full weight
vector, and reports changes in score and risk class.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import pandas as pd

from virelion_cardioscore.analysis.cipa_scoring import CardioScoreEngine


@dataclass(frozen=True)
class WeightSensitivitySpec:
    """Relative perturbation applied to each endpoint weight in turn."""

    relative_change: float = 0.20
    borderline_change_rate: float = 0.25
    unstable_change_rate: float = 0.25

    def __post_init__(self) -> None:
        if self.relative_change < 0:
            raise ValueError("relative_change must be non-negative.")
        if self.relative_change >= 1.0:
            raise ValueError("relative_change must be less than 1.0.")
        if not 0.0 <= self.borderline_change_rate <= 1.0:
            raise ValueError("borderline_change_rate must be between 0 and 1.")
        if not 0.0 <= self.unstable_change_rate <= 1.0:
            raise ValueError("unstable_change_rate must be between 0 and 1.")
        if self.unstable_change_rate < self.borderline_change_rate:
            raise ValueError("unstable_change_rate must be >= borderline_change_rate.")


def run_weight_sensitivity(
    engine: CardioScoreEngine,
    compounds: dict[str, dict[str, float]],
    *,
    spec: WeightSensitivitySpec | None = None,
) -> pd.DataFrame:
    """Perturb each endpoint weight and compare score/risk class with baseline.

    ``compounds`` maps compound names to complete endpoint-value mappings.
    Endpoint values are validated by the engine, so incomplete/non-finite data
    fail closed rather than being treated as zero liability. The supplied engine
    is never mutated; each perturbation is evaluated on an isolated copy.
    """
    spec = spec or WeightSensitivitySpec()
    base_weights = {name: float(meta["weight"]) for name, meta in engine.endpoints.items()}
    rows: list[dict[str, object]] = []

    for compound, values in compounds.items():
        baseline = engine.score_compound(compound, values)
        for endpoint, base_weight in base_weights.items():
            for direction, multiplier in (("down", 1.0 - spec.relative_change), ("up", 1.0 + spec.relative_change)):
                perturbed = base_weights.copy()
                perturbed[endpoint] = base_weight * multiplier
                total = sum(perturbed.values())
                if total <= 0:
                    raise ValueError("Perturbed endpoint weights must sum to a positive value.")

                isolated_engine = deepcopy(engine)
                isolated_engine.config["endpoints"] = {
                    name: {**meta, "weight": perturbed[name]}
                    for name, meta in isolated_engine.config["endpoints"].items()
                }
                isolated_engine.endpoints = isolated_engine.config["endpoints"]
                result = isolated_engine.score_compound(compound, values)

                rows.append(
                    {
                        "compound": compound,
                        "endpoint_perturbed": endpoint,
                        "direction": direction,
                        "relative_change": spec.relative_change,
                        "baseline_weight": base_weight,
                        "perturbed_weight_raw": perturbed[endpoint],
                        "baseline_score": baseline.score,
                        "perturbed_score": result.score,
                        "score_delta": result.score - baseline.score,
                        "baseline_risk_class": baseline.risk_class,
                        "perturbed_risk_class": result.risk_class,
                        "risk_class_changed": baseline.risk_class != result.risk_class,
                    }
                )

    return pd.DataFrame(rows)


def summarize_weight_sensitivity(
    results: pd.DataFrame,
    *,
    borderline_change_rate: float = 0.25,
    unstable_change_rate: float = 0.25,
) -> pd.DataFrame:
    """Summarize score range and risk-class instability by compound.

    A compound is ``stable`` when no sensitivity perturbation changes its risk
    class. It is ``borderline`` when the change rate is positive but below the
    configured unstable threshold, and ``unstable`` when the change rate meets
    or exceeds that threshold. The thresholds are analysis labels, not validated
    scientific cutoffs.
    """
    required = {
        "compound",
        "baseline_score",
        "perturbed_score",
        "score_delta",
        "risk_class_changed",
    }
    missing = sorted(required - set(results.columns))
    if missing:
        raise ValueError(f"Sensitivity results are missing columns: {missing}.")
    if not 0.0 <= borderline_change_rate <= 1.0:
        raise ValueError("borderline_change_rate must be between 0 and 1.")
    if not 0.0 <= unstable_change_rate <= 1.0:
        raise ValueError("unstable_change_rate must be between 0 and 1.")
    if unstable_change_rate < borderline_change_rate:
        raise ValueError("unstable_change_rate must be >= borderline_change_rate.")

    summary = (
        results.groupby("compound", sort=True)
        .agg(
            baseline_score=("baseline_score", "first"),
            min_perturbed_score=("perturbed_score", "min"),
            max_perturbed_score=("perturbed_score", "max"),
            max_absolute_score_delta=("score_delta", lambda values: float(values.abs().max())),
            n_perturbations=("score_delta", "size"),
            risk_class_change_rate=("risk_class_changed", "mean"),
        )
        .reset_index()
    )
    summary["sensitivity_class"] = "borderline"
    summary.loc[summary["risk_class_change_rate"] == 0.0, "sensitivity_class"] = "stable"
    summary.loc[
        summary["risk_class_change_rate"] >= unstable_change_rate,
        "sensitivity_class",
    ] = "unstable"
    return summary

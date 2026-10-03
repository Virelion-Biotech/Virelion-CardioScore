"""CardioScore classification-stability robustness analysis."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from virelion_cardioscore.analysis.pipeline import CardioScorePipeline


@dataclass(frozen=True)
class ClassificationStabilityResult:
    """Summary of score and risk-class stability under bounded perturbations."""

    compound: str
    baseline_score: float
    baseline_risk_class: str
    n_scenarios: int
    class_stability_rate: float
    mean_absolute_score_delta: float
    max_absolute_score_delta: float

    def to_dict(self) -> dict:
        return {
            "compound": self.compound,
            "baseline_score": self.baseline_score,
            "baseline_risk_class": self.baseline_risk_class,
            "n_scenarios": self.n_scenarios,
            "class_stability_rate": self.class_stability_rate,
            "mean_absolute_score_delta": self.mean_absolute_score_delta,
            "max_absolute_score_delta": self.max_absolute_score_delta,
        }


def perturb_feature_table(
    frame: pd.DataFrame,
    *,
    relative_noise_sd: float = 0.0,
    additive_endpoint_shift: float = 0.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Apply bounded numeric perturbation without changing experimental labels."""
    if relative_noise_sd < 0:
        raise ValueError("relative_noise_sd must be non-negative.")
    if additive_endpoint_shift < 0:
        raise ValueError("additive_endpoint_shift must be non-negative.")

    rng = np.random.default_rng(seed)
    perturbed = frame.copy()
    endpoints = ["fpd_ms", "beat_rate_bpm", "amplitude_uv", "stv", "triangulation_proxy"]
    for column in endpoints:
        if column not in perturbed.columns:
            raise ValueError(f"Feature table is missing endpoint {column!r}.")
        values = pd.to_numeric(perturbed[column], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"Endpoint {column!r} contains non-finite values.")
        scale = np.maximum(np.abs(values), 1.0)
        noise = rng.normal(0.0, relative_noise_sd, size=len(values)) * scale
        if additive_endpoint_shift:
            signs = np.where(values >= 0.0, 1.0, -1.0)
            noise += signs * additive_endpoint_shift
        perturbed[column] = values + noise
    return perturbed


def evaluate_classification_stability(
    frame: pd.DataFrame,
    *,
    pipeline: CardioScorePipeline | None = None,
    relative_noise_sd: float = 0.01,
    additive_endpoint_shift: float = 0.0,
    n_scenarios: int = 20,
    seed: int = 42,
) -> pd.DataFrame:
    """Measure score/class stability under deterministic bounded perturbations."""
    if n_scenarios < 1:
        raise ValueError("n_scenarios must be at least 1.")
    pipeline = pipeline or CardioScorePipeline.from_defaults()
    baseline = pipeline.run(frame)
    if baseline.summary_table.empty:
        return pd.DataFrame(columns=list(ClassificationStabilityResult.__dataclass_fields__))

    baseline_by_compound = {
        str(row["compound"]): row
        for _, row in baseline.summary_table.iterrows()
    }
    observations: dict[str, list[tuple[float, str]]] = {compound: [] for compound in baseline_by_compound}

    for scenario in range(n_scenarios):
        perturbed = perturb_feature_table(
            frame,
            relative_noise_sd=relative_noise_sd,
            additive_endpoint_shift=additive_endpoint_shift,
            seed=seed + scenario,
        )
        result = pipeline.run(perturbed)
        for _, row in result.summary_table.iterrows():
            compound = str(row["compound"])
            if compound in observations:
                observations[compound].append((float(row["cardioscore"]), str(row["risk_class"])))

    rows = []
    for compound, baseline_row in baseline_by_compound.items():
        values = observations[compound]
        if not values:
            continue
        score_deltas = np.abs(np.asarray([score for score, _ in values]) - float(baseline_row["cardioscore"]))
        stability = float(np.mean([risk == str(baseline_row["risk_class"]) for _, risk in values]))
        summary = ClassificationStabilityResult(
            compound=compound,
            baseline_score=float(baseline_row["cardioscore"]),
            baseline_risk_class=str(baseline_row["risk_class"]),
            n_scenarios=len(values),
            class_stability_rate=stability,
            mean_absolute_score_delta=float(np.mean(score_deltas)),
            max_absolute_score_delta=float(np.max(score_deltas)),
        )
        rows.append(summary.to_dict())
    return pd.DataFrame(rows)

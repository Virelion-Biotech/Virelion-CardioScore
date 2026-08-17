"""
Transparent CardioScore engine.

Aggregates multiple electrophysiological endpoints into a continuous risk score
and a categorical Low / Moderate / High classification. Every contribution is
inspectable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import yaml


@dataclass
class EndpointContribution:
    name: str
    raw_value: float
    contribution: float
    weight: float
    description: str = ""


@dataclass
class ScoreResult:
    compound: str
    score: float
    risk_class: str
    risk_color: str
    interpretation: str
    contributions: list[EndpointContribution] = field(default_factory=list)
    max_concentration_uM: Optional[float] = None
    n_wells: int = 0
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "compound": self.compound,
            "cardioscore": round(self.score, 4),
            "risk_class": self.risk_class,
            "interpretation": self.interpretation,
            "max_concentration_uM": self.max_concentration_uM,
            "n_wells": self.n_wells,
            "metadata": self.metadata,
            "contributions": [
                {
                    "endpoint": c.name,
                    "raw_value": round(c.raw_value, 4),
                    "contribution": round(c.contribution, 4),
                    "weight": c.weight,
                    "description": c.description,
                }
                for c in self.contributions
            ],
        }


class CardioScoreEngine:
    """Interpretable multi-endpoint risk scorer."""

    def __init__(
        self,
        endpoint_config_path: Optional[str | Path] = None,
        low_threshold: float = 0.30,
        moderate_threshold: float = 0.60,
    ):
        if endpoint_config_path is None:
            endpoint_config_path = (
                Path(__file__).resolve().parent.parent / "config" / "cipa_endpoints.yaml"
            )
        self.endpoint_config_path = Path(endpoint_config_path)
        self.config = self._load_config()
        self.endpoints = self.config["endpoints"]
        self.risk_categories = self.config["risk_categories"]
        self.low_threshold = low_threshold
        self.moderate_threshold = moderate_threshold

    def _load_config(self) -> dict:
        with open(self.endpoint_config_path) as f:
            return yaml.safe_load(f)

    def _normalize_effect(
        self,
        value: float,
        direction: str,
        threshold: float,
        max_contribution: float = 1.0,
    ) -> float:
        if direction == "absolute":
            excess = max(0.0, abs(value) - threshold)
            scale = threshold if threshold > 0 else 1.0
            return float(np.clip(excess / (3.0 * scale), 0.0, max_contribution))
        if direction == "increase":
            excess = max(0.0, value - threshold)
            scale = threshold if threshold > 0 else 1.0
            return float(np.clip(excess / (3.0 * scale), 0.0, max_contribution))
        if direction == "decrease":
            excess = max(0.0, -value - threshold) if value < 0 else max(0.0, value - threshold)
            scale = threshold if threshold > 0 else 1.0
            return float(np.clip(excess / (3.0 * scale), 0.0, max_contribution))
        raise ValueError(f"Unknown direction: {direction}")

    def score_compound(
        self,
        compound: str,
        endpoint_values: dict[str, float],
        max_concentration_uM: Optional[float] = None,
        n_wells: int = 0,
        dose_response_evidence: Optional[float] = None,
        dose_response_weight: float = 0.0,
    ) -> ScoreResult:
        """Score a compound with optional non-overlapping exposure-response evidence."""
        if dose_response_weight < 0:
            raise ValueError("dose_response_weight must be non-negative")
        if dose_response_evidence is not None and not 0.0 <= dose_response_evidence <= 1.0:
            raise ValueError("dose_response_evidence must be between 0 and 1")
        if dose_response_weight > 0 and dose_response_evidence is None:
            raise ValueError("dose_response_evidence is required when dose_response_weight > 0")

        contributions: list[EndpointContribution] = []
        weighted_sum = 0.0
        total_weight = 0.0

        for name, meta in self.endpoints.items():
            raw = float(endpoint_values.get(name, 0.0))
            weight = float(meta["weight"])
            direction = meta["direction"]
            thresh = float(meta["effect_threshold"])
            max_c = float(meta.get("max_contribution", 1.0))

            contrib = self._normalize_effect(raw, direction, thresh, max_c)
            weighted_sum += weight * contrib
            total_weight += weight

            contributions.append(
                EndpointContribution(
                    name=name,
                    raw_value=raw,
                    contribution=contrib,
                    weight=weight,
                    description=meta.get("description", ""),
                )
            )

        metadata = {}
        if dose_response_weight > 0 and dose_response_evidence is not None:
            weighted_sum += dose_response_weight * dose_response_evidence
            total_weight += dose_response_weight
            contributions.append(
                EndpointContribution(
                    name="dose_response_exposure_evidence",
                    raw_value=float(dose_response_evidence),
                    contribution=float(dose_response_evidence),
                    weight=float(dose_response_weight),
                    description=(
                        "Exposure-response evidence from quality-passing 4PL fits; "
                        "fraction of the tested log-concentration range above the fitted EC50."
                    ),
                )
            )
            metadata["dose_response_evidence"] = float(dose_response_evidence)
            metadata["dose_response_weight"] = float(dose_response_weight)

        score = float(np.clip(weighted_sum / total_weight if total_weight > 0 else 0.0, 0.0, 1.0))

        if score < self.low_threshold:
            cat = self.risk_categories["low"]
        elif score < self.moderate_threshold:
            cat = self.risk_categories["moderate"]
        else:
            cat = self.risk_categories["high"]

        return ScoreResult(
            compound=compound,
            score=score,
            risk_class=cat["label"],
            risk_color=cat["color"],
            interpretation=cat["interpretation"],
            contributions=contributions,
            max_concentration_uM=max_concentration_uM,
            n_wells=n_wells,
            metadata=metadata,
        )

    def score_feature_table(self, df: pd.DataFrame) -> list[ScoreResult]:
        results = []
        for compound, group in df.groupby("compound"):
            endpoint_values = {}
            for ep in self.endpoints:
                if ep in group.columns:
                    endpoint_values[ep] = float(group[ep].abs().max())
                else:
                    endpoint_values[ep] = 0.0

            max_conc = None
            if "concentration_uM" in group.columns:
                treated = group.loc[~group.get("vehicle", False)] if "vehicle" in group.columns else group
                max_conc = float(treated["concentration_uM"].max()) if len(treated) else None
            n_wells = int(group["well"].nunique()) if "well" in group.columns else len(group)

            results.append(
                self.score_compound(
                    compound=str(compound),
                    endpoint_values=endpoint_values,
                    max_concentration_uM=max_conc,
                    n_wells=n_wells,
                )
            )
        return results

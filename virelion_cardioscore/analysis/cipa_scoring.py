"""Transparent CardioScore engine."""

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
    n_independent_units: int = 0
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "compound": self.compound,
            "cardioscore": round(self.score, 4),
            "risk_class": self.risk_class,
            "interpretation": self.interpretation,
            "max_concentration_uM": self.max_concentration_uM,
            "n_wells": self.n_wells,
            "n_independent_units": self.n_independent_units,
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
            endpoint_config_path = Path(__file__).resolve().parent.parent / "config" / "cipa_endpoints.yaml"
        self.endpoint_config_path = Path(endpoint_config_path)
        self.config = self._load_config()
        self.endpoints = self.config["endpoints"]
        self.risk_categories = self.config["risk_categories"]
        self.low_threshold = float(low_threshold)
        self.moderate_threshold = float(moderate_threshold)
        if not 0.0 <= self.low_threshold < self.moderate_threshold <= 1.0:
            raise ValueError("Risk thresholds must satisfy 0 <= low < moderate <= 1.")

        # Keep the endpoint YAML and pipeline threshold configuration from silently disagreeing.
        low_max = self.risk_categories["low"].get("max_score")
        moderate_max = self.risk_categories["moderate"].get("max_score")
        if low_max is not None and not np.isclose(float(low_max), self.low_threshold):
            raise ValueError(
                "risk_categories.low.max_score disagrees with the configured low_threshold."
            )
        if moderate_max is not None and not np.isclose(float(moderate_max), self.moderate_threshold):
            raise ValueError(
                "risk_categories.moderate.max_score disagrees with the configured moderate_threshold."
            )

    def _load_config(self) -> dict:
        with open(self.endpoint_config_path, encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}
        if not isinstance(config.get("endpoints"), dict) or not config["endpoints"]:
            raise ValueError("Endpoint configuration must define a non-empty 'endpoints' mapping.")
        if not isinstance(config.get("risk_categories"), dict):
            raise ValueError("Endpoint configuration must define 'risk_categories'.")
        allowed_directions = {"absolute", "increase", "decrease"}
        for name, meta in config["endpoints"].items():
            if not isinstance(meta, dict):
                raise ValueError(f"Endpoint {name!r} configuration must be a mapping.")
            for key in ("weight", "effect_threshold", "direction"):
                if key not in meta:
                    raise ValueError(f"Endpoint {name!r} is missing required field {key!r}.")
            weight = float(meta["weight"])
            threshold = float(meta["effect_threshold"])
            direction = str(meta["direction"])
            if weight < 0:
                raise ValueError(f"Endpoint {name!r} weight cannot be negative.")
            if threshold < 0:
                raise ValueError(f"Endpoint {name!r} effect_threshold cannot be negative.")
            if direction not in allowed_directions:
                raise ValueError(
                    f"Endpoint {name!r} has unsupported direction {direction!r}; "
                    f"expected one of {sorted(allowed_directions)}."
                )
        if sum(float(meta["weight"]) for meta in config["endpoints"].values()) <= 0:
            raise ValueError("Endpoint weights must sum to a positive value.")
        for category in ("low", "moderate", "high"):
            meta = config["risk_categories"].get(category)
            if not isinstance(meta, dict) or "label" not in meta:
                raise ValueError(f"risk_categories.{category} must define a label.")
        return config

    def _normalize_effect(self, value: float, direction: str, threshold: float, max_contribution: float = 1.0) -> float:
        if not np.isfinite(value):
            return 0.0
        scale = threshold if threshold > 0 else 1.0
        if direction == "absolute":
            excess = max(0.0, abs(value) - threshold)
        elif direction == "increase":
            excess = max(0.0, value - threshold)
        elif direction == "decrease":
            excess = max(0.0, -value - threshold)
        else:
            raise ValueError(f"Unknown direction: {direction}")
        return float(np.clip(excess / (3.0 * scale), 0.0, max_contribution))

    def score_compound(
        self,
        compound: str,
        endpoint_values: dict[str, float],
        max_concentration_uM: Optional[float] = None,
        n_wells: int = 0,
        n_independent_units: int = 0,
        dose_response_evidence: Optional[float] = None,
        dose_response_weight: float = 0.0,
    ) -> ScoreResult:
        if dose_response_weight < 0:
            raise ValueError("dose_response_weight must be non-negative")
        if dose_response_evidence is not None and not 0.0 <= dose_response_evidence <= 1.0:
            raise ValueError("dose_response_evidence must be between 0 and 1")
        if dose_response_weight > 0 and dose_response_evidence is None:
            raise ValueError("dose_response_evidence is required when dose_response_weight > 0")

        missing = sorted(set(self.endpoints) - set(endpoint_values))
        if missing:
            raise ValueError(f"Scoring endpoint values are missing: {missing}")
        non_finite = sorted(
            name for name in self.endpoints
            if not np.isfinite(float(endpoint_values[name]))
        )
        if non_finite:
            raise ValueError(f"Scoring endpoint values must be finite: {non_finite}")

        contributions: list[EndpointContribution] = []
        weighted_sum = 0.0
        total_weight = 0.0
        for name, meta in self.endpoints.items():
            raw = float(endpoint_values[name])
            weight = float(meta["weight"])
            contrib = self._normalize_effect(raw, meta["direction"], float(meta["effect_threshold"]), float(meta.get("max_contribution", 1.0)))
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
                    description="Exposure-response evidence from quality-passing 4PL fits.",
                )
            )
            metadata["dose_response_evidence"] = float(dose_response_evidence)
            metadata["dose_response_weight"] = float(dose_response_weight)

        score = float(np.clip(weighted_sum / total_weight if total_weight > 0 else 0.0, 0.0, 1.0))
        if score <= self.low_threshold:
            cat = self.risk_categories["low"]
        elif score <= self.moderate_threshold:
            cat = self.risk_categories["moderate"]
        else:
            cat = self.risk_categories["high"]
        return ScoreResult(
            compound=compound,
            score=score,
            risk_class=cat["label"],
            risk_color=cat.get("color", ""),
            interpretation=cat.get("interpretation", ""),
            contributions=contributions,
            max_concentration_uM=max_concentration_uM,
            n_wells=n_wells,
            n_independent_units=n_independent_units,
            metadata=metadata,
        )

    def _feature_endpoint_value(self, group: pd.DataFrame, name: str) -> float:
        if name not in group.columns:
            return 0.0
        meta = self.endpoints[name]
        values = pd.to_numeric(group[name], errors="coerce").dropna()
        if values.empty:
            return 0.0
        if meta["direction"] == "absolute":
            return float(values.abs().max())
        if meta["direction"] == "increase":
            return float(values.max())
        if meta["direction"] == "decrease":
            return float(values.min())
        raise ValueError(f"Unknown direction: {meta['direction']}")

    def score_feature_table(self, df: pd.DataFrame) -> list[ScoreResult]:
        required = {"compound", *self.endpoints.keys()}
        missing = sorted(required - set(df.columns))
        if missing:
            raise ValueError(f"Feature table is missing scoring endpoint(s): {missing}")
        results = []
        for compound, group in df.groupby("compound", sort=True):
            endpoint_values = {name: self._feature_endpoint_value(group, name) for name in self.endpoints}
            results.append(self.score_compound(str(compound), endpoint_values, n_wells=len(group), n_independent_units=len(group)))
        return results

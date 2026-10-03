"""Bootstrap inference utilities for CardioScore concentration data.

These functions provide descriptive uncertainty estimates for replicate-level
responses and matched concentration-profile differences. They are inference
helpers, not replacements for a prespecified statistical analysis plan.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BootstrapCI:
    estimate: float
    ci_low: float
    ci_high: float
    confidence: float
    n_observations: int
    n_bootstrap: int
    seed: Optional[int] = None
    n_clusters: Optional[int] = None

    def to_dict(self) -> dict:
        return {
            "estimate": self.estimate,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "confidence": self.confidence,
            "n_observations": self.n_observations,
            "n_bootstrap": self.n_bootstrap,
            "seed": self.seed,
            "n_clusters": self.n_clusters,
        }


@dataclass(frozen=True)
class ProfileDifference:
    concentrations: tuple[float, ...]
    differences: tuple[float, ...]
    ci_low: tuple[float, ...]
    ci_high: tuple[float, ...]
    n_bootstrap: int
    confidence: float

    def to_dict(self) -> dict:
        return {
            "concentrations": list(self.concentrations),
            "differences": list(self.differences),
            "ci_low": list(self.ci_low),
            "ci_high": list(self.ci_high),
            "n_bootstrap": self.n_bootstrap,
            "confidence": self.confidence,
        }


def _validate_bootstrap_args(n_bootstrap: int, confidence: float) -> None:
    if n_bootstrap < 100:
        raise ValueError("n_bootstrap must be at least 100.")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1.")


def bootstrap_ci(
    values: np.ndarray,
    *,
    statistic: Callable[[np.ndarray], float] = np.mean,
    n_bootstrap: int = 2000,
    confidence: float = 0.95,
    seed: Optional[int] = 42,
) -> BootstrapCI:
    """Bootstrap a one-sample statistic using observation-level resampling."""
    _validate_bootstrap_args(n_bootstrap, confidence)
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        raise ValueError("At least two finite observations are required for bootstrap inference.")

    rng = np.random.default_rng(seed)
    estimate = float(statistic(x))
    indices = rng.integers(0, x.size, size=(n_bootstrap, x.size))
    samples = x[indices]
    boot_stats = np.asarray([statistic(sample) for sample in samples], dtype=float)
    alpha = 1.0 - confidence
    low, high = np.quantile(boot_stats, [alpha / 2.0, 1.0 - alpha / 2.0])
    return BootstrapCI(
        estimate=estimate,
        ci_low=float(low),
        ci_high=float(high),
        confidence=confidence,
        n_observations=int(x.size),
        n_bootstrap=n_bootstrap,
        seed=seed,
        n_clusters=None,
    )


def bootstrap_cluster_ci(
    values: np.ndarray,
    clusters: np.ndarray,
    *,
    statistic: Callable[[np.ndarray], float] = np.mean,
    n_bootstrap: int = 2000,
    confidence: float = 0.95,
    seed: Optional[int] = 42,
) -> BootstrapCI:
    """Bootstrap a statistic by resampling independent clusters as units.

    The default mean statistic is evaluated after first averaging observations
    within each cluster, so clusters contribute equally even when they contain
    different numbers of technical observations. Custom statistics are applied
    to the cluster-level values as well. This prevents technical-well imbalance
    from silently changing the inferential weight of independent units.
    """
    _validate_bootstrap_args(n_bootstrap, confidence)
    x = np.asarray(values, dtype=float)
    g = np.asarray(clusters)
    if x.ndim != 1 or g.ndim != 1 or x.size != g.size:
        raise ValueError("values and clusters must be one-dimensional arrays of equal length.")
    finite = np.isfinite(x)
    x = x[finite]
    g = g[finite]
    if x.size < 2:
        raise ValueError("At least two finite observations are required for bootstrap inference.")
    if pd.isna(g).any():
        raise ValueError("clusters cannot contain missing identifiers.")
    unique_clusters = pd.unique(g)
    if len(unique_clusters) < 2:
        raise ValueError("At least two independent clusters are required for cluster bootstrap.")

    cluster_values = np.asarray(
        [float(np.mean(x[g == cluster])) for cluster in unique_clusters],
        dtype=float,
    )
    if not np.isfinite(cluster_values).all():
        raise ValueError("Cluster-level values must be finite for bootstrap inference.")

    rng = np.random.default_rng(seed)
    estimate = float(statistic(cluster_values))
    boot_stats = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        sampled = rng.integers(0, len(cluster_values), size=len(cluster_values))
        boot_stats[i] = float(statistic(cluster_values[sampled]))
    alpha = 1.0 - confidence
    low, high = np.quantile(boot_stats, [alpha / 2.0, 1.0 - alpha / 2.0])
    return BootstrapCI(
        estimate=estimate,
        ci_low=float(low),
        ci_high=float(high),
        confidence=confidence,
        n_observations=int(x.size),
        n_bootstrap=n_bootstrap,
        seed=seed,
        n_clusters=int(len(unique_clusters)),
    )


def bootstrap_profile_difference(
    concentrations: np.ndarray,
    group_a: dict[float, np.ndarray],
    group_b: dict[float, np.ndarray],
    *,
    n_bootstrap: int = 2000,
    confidence: float = 0.95,
    seed: Optional[int] = 42,
) -> ProfileDifference:
    """Estimate matched concentration differences with bootstrap confidence intervals.

    The returned intervals describe uncertainty under replicate resampling. No
    p-value is reported because this percentile bootstrap distribution is centered
    on the observed effect rather than generated under a null hypothesis.
    """
    _validate_bootstrap_args(n_bootstrap, confidence)
    common = sorted(set(group_a).intersection(group_b))
    if not common:
        raise ValueError("No matched concentrations are available for comparison.")

    rng = np.random.default_rng(seed)
    alpha = 1.0 - confidence
    differences: list[float] = []
    ci_low: list[float] = []
    ci_high: list[float] = []

    for concentration in common:
        a = np.asarray(group_a[concentration], dtype=float)
        b = np.asarray(group_b[concentration], dtype=float)
        a = a[np.isfinite(a)]
        b = b[np.isfinite(b)]
        if a.size < 2 or b.size < 2:
            raise ValueError(
                f"At least two finite replicates per group are required at concentration {concentration}."
            )

        observed = float(np.mean(a) - np.mean(b))
        a_idx = rng.integers(0, a.size, size=(n_bootstrap, a.size))
        b_idx = rng.integers(0, b.size, size=(n_bootstrap, b.size))
        boot_diff = a[a_idx].mean(axis=1) - b[b_idx].mean(axis=1)
        low, high = np.quantile(boot_diff, [alpha / 2.0, 1.0 - alpha / 2.0])

        differences.append(observed)
        ci_low.append(float(low))
        ci_high.append(float(high))

    return ProfileDifference(
        concentrations=tuple(float(c) for c in common),
        differences=tuple(differences),
        ci_low=tuple(ci_low),
        ci_high=tuple(ci_high),
        n_bootstrap=n_bootstrap,
        confidence=confidence,
    )

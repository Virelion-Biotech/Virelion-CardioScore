"""Bootstrap inference utilities for CardioScore concentration data.

These functions provide uncertainty estimates for replicate-level responses and
simple matched concentration-profile comparisons. They are inference helpers,
not replacements for a prespecified statistical analysis plan.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np


@dataclass(frozen=True)
class BootstrapCI:
    estimate: float
    ci_low: float
    ci_high: float
    confidence: float
    n_observations: int
    n_bootstrap: int
    seed: Optional[int] = None

    def to_dict(self) -> dict:
        return {
            "estimate": self.estimate,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "confidence": self.confidence,
            "n_observations": self.n_observations,
            "n_bootstrap": self.n_bootstrap,
            "seed": self.seed,
        }


@dataclass(frozen=True)
class ProfileDifference:
    concentrations: tuple[float, ...]
    differences: tuple[float, ...]
    ci_low: tuple[float, ...]
    ci_high: tuple[float, ...]
    p_values: tuple[float, ...]
    n_bootstrap: int
    confidence: float

    def to_dict(self) -> dict:
        return {
            "concentrations": list(self.concentrations),
            "differences": list(self.differences),
            "ci_low": list(self.ci_low),
            "ci_high": list(self.ci_high),
            "p_values": list(self.p_values),
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
    """Bootstrap a one-sample statistic using replicate-level resampling."""
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
    """Compare two replicate-level profiles at matched concentrations.

    Each concentration is resampled independently within group A and group B.
    The reported two-sided p-value is the fraction of bootstrap differences whose
    sign is opposite the observed difference, doubled and clipped to one.
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
    p_values: list[float] = []

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

        if observed == 0.0:
            p_value = 1.0
        else:
            opposite = np.mean(np.sign(boot_diff) == -np.sign(observed))
            p_value = float(min(1.0, max(0.0, 2.0 * opposite)))

        differences.append(observed)
        ci_low.append(float(low))
        ci_high.append(float(high))
        p_values.append(p_value)

    return ProfileDifference(
        concentrations=tuple(float(c) for c in common),
        differences=tuple(differences),
        ci_low=tuple(ci_low),
        ci_high=tuple(ci_high),
        p_values=tuple(p_values),
        n_bootstrap=n_bootstrap,
        confidence=confidence,
    )

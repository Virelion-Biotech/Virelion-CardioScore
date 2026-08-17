"""Robustness sweeps for conventional versus hierarchical effect estimation."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
import pandas as pd

from virelion_cardioscore.analysis.stress_tests import (
    StressTestSpec,
    conventional_treatment_effect,
    make_known_effect_dataset,
    true_treatment_effect,
)


@dataclass(frozen=True)
class RobustnessGrid:
    """Parameter grid for known-effect robustness experiments."""

    treatment_effect: float = 10.0
    offset_magnitudes: tuple[float, ...] = (0.0, 10.0, 20.0, 40.0)
    treatment_group_counts: tuple[int, ...] = (4, 3, 2, 1)
    replicate_counts: tuple[int, ...] = (2, 4, 8)
    noise_sds: tuple[float, ...] = (0.5, 1.0, 5.0)
    n_repeats: int = 10
    n_groups: int = 4
    seed: int = 42
    bias_tolerance: float = 2.0


def _offsets(magnitude: float, n_groups: int) -> tuple[float, ...]:
    """Create symmetric offsets with the requested maximum magnitude."""
    if magnitude == 0 or n_groups == 1:
        return tuple(0.0 for _ in range(n_groups))
    center = (n_groups - 1) / 2.0
    return tuple(float((i - center) * magnitude / max(abs(center), 1.0)) for i in range(n_groups))


def run_robustness_matrix(grid: RobustnessGrid | None = None) -> pd.DataFrame:
    """Run a reproducible stress matrix and return one row per simulation.

    The matrix records conventional-estimator bias against the known generating
    treatment effect. It does not claim that the conventional estimator is a
    reference standard; the ground truth is known only because these datasets
    are synthetic.
    """
    grid = grid or RobustnessGrid()
    if grid.n_groups < 2:
        raise ValueError("Robustness sweeps require at least two groups.")
    if grid.n_repeats < 1:
        raise ValueError("n_repeats must be at least 1.")
    if grid.bias_tolerance < 0:
        raise ValueError("bias_tolerance must be non-negative.")

    rows: list[dict] = []
    scenario_id = 0
    for magnitude, treated_groups_count, replicates, noise_sd, repeat in product(
        grid.offset_magnitudes,
        grid.treatment_group_counts,
        grid.replicate_counts,
        grid.noise_sds,
        range(grid.n_repeats),
    ):
        if not 1 <= treated_groups_count <= grid.n_groups:
            raise ValueError("treatment_group_counts must fall between 1 and n_groups.")

        offsets = _offsets(float(magnitude), grid.n_groups)
        # Assign treatment to the highest-offset groups deliberately: this makes
        # imbalance stress-testable without changing the generating effect.
        treatment_groups = tuple(range(grid.n_groups - treated_groups_count, grid.n_groups))
        seed = int(grid.seed + repeat + scenario_id * 1000)
        spec = StressTestSpec(
            treatment_effect=grid.treatment_effect,
            group_offsets=offsets,
            controls_per_group=replicates,
            treated_per_group=replicates,
            groups_with_treatment=treatment_groups,
            noise_sd=noise_sd,
            seed=seed,
        )
        data = make_known_effect_dataset(spec)
        estimate = conventional_treatment_effect(data)
        truth = true_treatment_effect(spec)
        bias = estimate - truth
        rows.append(
            {
                "scenario_id": scenario_id,
                "repeat": repeat,
                "offset_magnitude": float(magnitude),
                "treated_groups": int(treated_groups_count),
                "treatment_group_fraction": float(treated_groups_count / grid.n_groups),
                "replicates_per_group": int(replicates),
                "noise_sd": float(noise_sd),
                "true_effect": float(truth),
                "conventional_effect": float(estimate),
                "conventional_bias": float(bias),
                "absolute_bias": float(abs(bias)),
                "within_tolerance": bool(abs(bias) <= grid.bias_tolerance),
            }
        )
        scenario_id += 1

    return pd.DataFrame(rows)


def summarize_robustness_matrix(results: pd.DataFrame, *, bias_tolerance: float = 2.0) -> pd.DataFrame:
    """Summarize bias, tolerance recovery, and scenario counts by stress level."""
    required = {
        "offset_magnitude",
        "treated_groups",
        "replicates_per_group",
        "noise_sd",
        "conventional_bias",
        "absolute_bias",
    }
    missing = sorted(required - set(results.columns))
    if missing:
        raise ValueError(f"Robustness results are missing columns: {missing}.")

    working = results.copy()
    working["within_tolerance"] = working["absolute_bias"] <= bias_tolerance
    grouped = working.groupby(
        ["offset_magnitude", "treated_groups", "replicates_per_group", "noise_sd"],
        sort=True,
        dropna=False,
    )
    return grouped.agg(
        n_simulations=("conventional_bias", "size"),
        mean_bias=("conventional_bias", "mean"),
        median_bias=("conventional_bias", "median"),
        mean_absolute_bias=("absolute_bias", "mean"),
        max_absolute_bias=("absolute_bias", "max"),
        recovery_rate=("within_tolerance", "mean"),
    ).reset_index()

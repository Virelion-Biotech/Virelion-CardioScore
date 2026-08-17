"""Synthetic stress-test generators for hierarchical CardioScore analysis."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class StressTestSpec:
    """Specification for a known-ground-truth plate/batch stress test."""

    treatment_effect: float = 10.0
    group_offsets: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0)
    group_scales: tuple[float, ...] | None = None
    controls_per_group: int = 4
    treated_per_group: int = 4
    groups_with_treatment: tuple[int, ...] | None = None
    noise_sd: float = 1.0
    seed: int = 42


def make_known_effect_dataset(spec: StressTestSpec | None = None) -> pd.DataFrame:
    """Generate endpoint data with known treatment effect and plate drift."""
    spec = spec or StressTestSpec()
    if len(spec.group_offsets) < 2:
        raise ValueError("Stress tests require at least two groups.")
    if spec.group_scales is not None and len(spec.group_scales) != len(spec.group_offsets):
        raise ValueError("group_scales must match group_offsets in length.")
    if spec.controls_per_group < 1 or spec.treated_per_group < 1:
        raise ValueError("Each group must contain at least one control and treated observation.")

    n_groups = len(spec.group_offsets)
    treated_groups = (
        set(range(n_groups))
        if spec.groups_with_treatment is None
        else set(spec.groups_with_treatment)
    )
    if not treated_groups.issubset(set(range(n_groups))):
        raise ValueError("groups_with_treatment contains an invalid group index.")

    rng = np.random.default_rng(spec.seed)
    rows: list[dict] = []
    well_id = 1
    for group_index, offset in enumerate(spec.group_offsets):
        scale = 1.0 if spec.group_scales is None else spec.group_scales[group_index]
        plate = f"P{group_index + 1}"
        for _ in range(spec.controls_per_group):
            baseline = 100.0 * scale + offset
            rows.append(
                {
                    "compound": "StressCompound",
                    "concentration_uM": 1.0,
                    "well": f"W{well_id}",
                    "vehicle": True,
                    "plate_id": plate,
                    "fpd_ms": baseline + rng.normal(0, spec.noise_sd),
                }
            )
            well_id += 1

        if group_index in treated_groups:
            for _ in range(spec.treated_per_group):
                baseline = 100.0 * scale + offset
                response = baseline + spec.treatment_effect * scale
                rows.append(
                    {
                        "compound": "StressCompound",
                        "concentration_uM": 1.0,
                        "well": f"W{well_id}",
                        "vehicle": False,
                        "plate_id": plate,
                        "fpd_ms": response + rng.normal(0, spec.noise_sd),
                    }
                )
                well_id += 1

    return pd.DataFrame(rows)


def conventional_treatment_effect(df: pd.DataFrame, *, endpoint: str = "fpd_ms") -> float:
    """Estimate treated-minus-control effect by pooled observations."""
    controls = pd.to_numeric(df.loc[df["vehicle"], endpoint], errors="coerce").dropna()
    treated = pd.to_numeric(df.loc[~df["vehicle"], endpoint], errors="coerce").dropna()
    if controls.empty or treated.empty:
        raise ValueError("Both control and treated observations are required.")
    return float(treated.mean() - controls.mean())


def true_treatment_effect(spec: StressTestSpec | None = None) -> float:
    """Return the known generating treatment effect for a stress-test spec."""
    return float((spec or StressTestSpec()).treatment_effect)

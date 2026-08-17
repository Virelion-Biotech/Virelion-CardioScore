"""
Synthetic MEA dataset generator for demonstration and testing.

Produces realistic-looking multi-well, multi-concentration field-potential
feature tables that the CardioScore pipeline can consume end-to-end without
requiring proprietary MEA files.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class SyntheticMEADataset:
    """Container for a synthetic multi-compound MEA experiment."""

    features: pd.DataFrame
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        required = {"compound", "concentration_uM", "well", "vehicle"}
        missing = required - set(self.features.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    @property
    def compounds(self) -> list[str]:
        return sorted(self.features["compound"].unique().tolist())

    def get_compound(self, name: str) -> pd.DataFrame:
        return self.features[self.features["compound"] == name].copy()


def _sigmoid(x: np.ndarray, ec50: float, hill: float = 1.2) -> np.ndarray:
    return 1.0 / (1.0 + (ec50 / np.maximum(x, 1e-9)) ** hill)


def generate_synthetic_mea(
    n_compounds: int = 4,
    n_concentrations: int = 6,
    n_wells_per_conc: int = 4,
    seed: int = 42,
    include_toxic: bool = True,
) -> SyntheticMEADataset:
    rng = np.random.default_rng(seed)
    concentrations = np.logspace(-2, 2, n_concentrations)

    profiles = []
    for i in range(n_compounds):
        name = f"Compound_{chr(65 + i)}"
        if include_toxic and i == n_compounds - 1:
            profiles.append({
                "name": name, "fpd_ec50": 3.0, "rate_ec50": 8.0,
                "amp_ec50": 5.0, "stv_ec50": 2.5, "tri_ec50": 4.0, "tox": 1.0,
            })
        else:
            profiles.append({
                "name": name,
                "fpd_ec50": rng.uniform(20, 80),
                "rate_ec50": rng.uniform(30, 100),
                "amp_ec50": rng.uniform(40, 120),
                "stv_ec50": rng.uniform(25, 90),
                "tri_ec50": rng.uniform(30, 100),
                "tox": rng.uniform(0.1, 0.45),
            })

    rows = []
    for prof in profiles:
        for conc in concentrations:
            for well_idx in range(n_wells_per_conc):
                well = f"W{well_idx + 1:02d}"
                fpd_eff = _sigmoid(np.array([conc]), prof["fpd_ec50"])[0] * prof["tox"]
                rate_eff = _sigmoid(np.array([conc]), prof["rate_ec50"])[0] * prof["tox"] * 0.7
                amp_eff = _sigmoid(np.array([conc]), prof["amp_ec50"])[0] * prof["tox"] * 0.8
                stv_eff = _sigmoid(np.array([conc]), prof["stv_ec50"])[0] * prof["tox"]
                tri_eff = _sigmoid(np.array([conc]), prof["tri_ec50"])[0] * prof["tox"] * 0.9

                fpd_ms = 280.0 * (1.0 + 0.25 * fpd_eff) + rng.normal(0, 6)
                beat_rate = 55.0 * (1.0 - 0.30 * rate_eff) + rng.normal(0, 2.5)
                amplitude = 180.0 * (1.0 - 0.40 * amp_eff) + rng.normal(0, 8)
                stv = 0.04 + 0.12 * stv_eff + abs(rng.normal(0, 0.008))
                triangulation = 0.18 + 0.25 * tri_eff + abs(rng.normal(0, 0.02))

                noise_sd = abs(rng.normal(8, 3))
                n_electrodes = int(rng.integers(6, 12))
                beat_detection_rate = float(np.clip(rng.normal(0.92, 0.05), 0.6, 1.0))

                rows.append({
                    "compound": prof["name"],
                    "concentration_uM": float(conc),
                    "well": well,
                    "vehicle": False,
                    "fpd_ms": float(fpd_ms),
                    "beat_rate_bpm": float(beat_rate),
                    "amplitude_uv": float(amplitude),
                    "stv": float(stv),
                    "triangulation_proxy": float(triangulation),
                    "noise_sd_uv": float(noise_sd),
                    "n_electrodes": n_electrodes,
                    "beat_detection_rate": beat_detection_rate,
                })

        for well_idx in range(n_wells_per_conc):
            well = f"V{well_idx + 1:02d}"
            rows.append({
                "compound": prof["name"],
                "concentration_uM": 0.0,
                "well": well,
                "vehicle": True,
                "fpd_ms": float(280.0 + rng.normal(0, 5)),
                "beat_rate_bpm": float(55.0 + rng.normal(0, 2)),
                "amplitude_uv": float(180.0 + rng.normal(0, 7)),
                "stv": float(0.04 + abs(rng.normal(0, 0.005))),
                "triangulation_proxy": float(0.18 + abs(rng.normal(0, 0.015))),
                "noise_sd_uv": float(abs(rng.normal(7, 2))),
                "n_electrodes": int(rng.integers(7, 12)),
                "beat_detection_rate": float(np.clip(rng.normal(0.95, 0.03), 0.7, 1.0)),
            })

    df = pd.DataFrame(rows)
    metadata = {
        "n_compounds": n_compounds,
        "n_concentrations": n_concentrations,
        "n_wells_per_conc": n_wells_per_conc,
        "seed": seed,
        "generator": "virelion_cardioscore.io.synthetic",
        "note": "Synthetic data for demonstration only. Not real biological recordings.",
    }
    return SyntheticMEADataset(features=df, metadata=metadata)


def load_synthetic_dataset(
    n_compounds: int = 4,
    n_concentrations: int = 6,
    seed: int = 42,
) -> SyntheticMEADataset:
    return generate_synthetic_mea(
        n_compounds=n_compounds,
        n_concentrations=n_concentrations,
        seed=seed,
        include_toxic=True,
    )

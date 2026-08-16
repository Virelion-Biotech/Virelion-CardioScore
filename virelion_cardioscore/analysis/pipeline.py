"""
End-to-end CardioScore pipeline.

Orchestrates quality control, vehicle normalization, concentration-response
summaries, scoring, and report generation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import yaml

from virelion_cardioscore.analysis.cipa_scoring import CardioScoreEngine, ScoreResult
from virelion_cardioscore.io.synthetic import SyntheticMEADataset


@dataclass
class PipelineResult:
    scores: list[ScoreResult]
    feature_table: pd.DataFrame
    summary_table: pd.DataFrame
    config: dict = field(default_factory=dict)
    qc_log: list[str] = field(default_factory=list)

    def to_html(self, path: str | Path) -> None:
        from virelion_cardioscore.reporting.report_generator import write_html_report
        write_html_report(self, path)

    def to_csv(self, path: str | Path) -> None:
        self.summary_table.to_csv(path, index=False)

    def to_json(self, path: str | Path) -> None:
        import json
        payload = {
            "scores": [s.to_dict() for s in self.scores],
            "summary": self.summary_table.to_dict(orient="records"),
        }
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)


class CardioScorePipeline:
    """High-level orchestrator for the CardioScore workflow."""

    def __init__(self, config: dict):
        self.config = config
        self.engine = CardioScoreEngine(
            low_threshold=config.get("scoring", {}).get("low_threshold", 0.30),
            moderate_threshold=config.get("scoring", {}).get("moderate_threshold", 0.60),
        )
        self.qc_log: list[str] = []

    @classmethod
    def from_config(cls, path: str | Path) -> "CardioScorePipeline":
        path = Path(path)
        with open(path) as f:
            cfg = yaml.safe_load(f)
        return cls(cfg)

    @classmethod
    def from_defaults(cls) -> "CardioScorePipeline":
        default = Path(__file__).resolve().parent.parent / "config" / "default.yaml"
        return cls.from_config(default)

    def apply_qc(self, df: pd.DataFrame) -> pd.DataFrame:
        qc = self.config.get("quality_control", {})
        min_elec = qc.get("min_electrodes_per_well", 4)
        max_noise = qc.get("max_noise_sd_uv", 25.0)
        min_bdr = qc.get("min_beat_detection_rate", 0.7)

        before = len(df)
        mask = (
            (df["n_electrodes"] >= min_elec)
            & (df["noise_sd_uv"] <= max_noise)
            & (df["beat_detection_rate"] >= min_bdr)
        )
        rejected = df[~mask]
        kept = df[mask].copy()

        if qc.get("log_rejections", True) and len(rejected) > 0:
            for _, row in rejected.iterrows():
                self.qc_log.append(
                    f"Rejected {row['compound']} {row['well']} "
                    f"(noise={row['noise_sd_uv']:.1f}, elec={row['n_electrodes']}, "
                    f"bdr={row['beat_detection_rate']:.2f})"
                )
        self.qc_log.append(f"QC: kept {len(kept)}/{before} wells")
        return kept

    def compute_effects(self, df: pd.DataFrame) -> pd.DataFrame:
        records = []
        for compound, group in df.groupby("compound"):
            vehicle = group[group["vehicle"] == True]  # noqa: E712
            treated = group[group["vehicle"] == False]

            if vehicle.empty:
                self.qc_log.append(f"Warning: no vehicle wells for {compound}")
                continue

            v_fpd = vehicle["fpd_ms"].mean()
            v_rate = vehicle["beat_rate_bpm"].mean()
            v_amp = vehicle["amplitude_uv"].mean()
            v_stv = vehicle["stv"].mean()
            v_tri = vehicle["triangulation_proxy"].mean()

            for _, row in treated.iterrows():
                records.append(
                    {
                        "compound": compound,
                        "concentration_uM": row["concentration_uM"],
                        "well": row["well"],
                        "vehicle": False,
                        "fpd_ms": row["fpd_ms"],
                        "beat_rate_bpm": row["beat_rate_bpm"],
                        "amplitude_uv": row["amplitude_uv"],
                        "stv": row["stv"],
                        "triangulation_proxy": row["triangulation_proxy"],
                        "fpd_change_pct": 100.0 * (row["fpd_ms"] - v_fpd) / v_fpd,
                        "beat_rate_change_pct": 100.0 * (row["beat_rate_bpm"] - v_rate) / v_rate,
                        "amplitude_change_pct": 100.0 * (row["amplitude_uv"] - v_amp) / v_amp,
                        "stv_increase": (row["stv"] - v_stv) / max(v_stv, 1e-6),
                        "triangulation_proxy_change": (row["triangulation_proxy"] - v_tri) / max(v_tri, 1e-6),
                    }
                )
        return pd.DataFrame(records)

    def run(self, dataset: SyntheticMEADataset | pd.DataFrame) -> PipelineResult:
        self.qc_log = []
        if isinstance(dataset, SyntheticMEADataset):
            df = dataset.features.copy()
        else:
            df = dataset.copy()

        df = self.apply_qc(df)
        effects = self.compute_effects(df)

        agg_rows = []
        for compound, group in effects.groupby("compound"):
            agg_rows.append(
                {
                    "compound": compound,
                    "fpd_change_pct": group["fpd_change_pct"].abs().max(),
                    "beat_rate_change_pct": group["beat_rate_change_pct"].abs().max(),
                    "amplitude_change_pct": group["amplitude_change_pct"].min(),
                    "stv_increase": group["stv_increase"].max(),
                    "triangulation_proxy": group["triangulation_proxy_change"].max(),
                    "max_concentration_uM": group["concentration_uM"].max(),
                    "n_wells": group["well"].nunique(),
                }
            )
        agg = pd.DataFrame(agg_rows)

        scores = []
        for _, row in agg.iterrows():
            endpoint_values = {
                "fpd_change_pct": row["fpd_change_pct"],
                "beat_rate_change_pct": row["beat_rate_change_pct"],
                "amplitude_change_pct": row["amplitude_change_pct"],
                "stv_increase": row["stv_increase"],
                "triangulation_proxy": row["triangulation_proxy"],
            }
            scores.append(
                self.engine.score_compound(
                    compound=row["compound"],
                    endpoint_values=endpoint_values,
                    max_concentration_uM=row["max_concentration_uM"],
                    n_wells=int(row["n_wells"]),
                )
            )

        summary = pd.DataFrame(
            [
                {
                    "compound": s.compound,
                    "cardioscore": round(s.score, 4),
                    "risk_class": s.risk_class,
                    "max_concentration_uM": s.max_concentration_uM,
                    "n_wells": s.n_wells,
                    "interpretation": s.interpretation,
                }
                for s in scores
            ]
        ).sort_values("cardioscore", ascending=False)

        return PipelineResult(
            scores=scores,
            feature_table=effects,
            summary_table=summary,
            config=self.config,
            qc_log=self.qc_log,
        )

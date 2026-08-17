"""
End-to-end CardioScore pipeline.

Orchestrates quality control, vehicle normalization, replicate-aware
concentration summaries, optional concentration-response fitting, scoring,
and report generation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from virelion_cardioscore.analysis.cipa_scoring import CardioScoreEngine, ScoreResult
from virelion_cardioscore.analysis.dose_response import DoseResponseFit, fit_concentration_series
from virelion_cardioscore.io.synthetic import SyntheticMEADataset


@dataclass
class PipelineResult:
    scores: list[ScoreResult]
    feature_table: pd.DataFrame
    summary_table: pd.DataFrame
    concentration_table: pd.DataFrame = field(default_factory=pd.DataFrame)
    dose_response_fits: dict[str, list[DoseResponseFit]] = field(default_factory=dict)
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
            "concentration_summary": self.concentration_table.to_dict(orient="records"),
            "dose_response_fits": {
                compound: [fit.to_dict() for fit in fits]
                for compound, fits in self.dose_response_fits.items()
            },
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
        reject_irregular = qc.get("reject_wells_with_arrhythmia_proxy", False)
        irregularity_threshold = qc.get("arrhythmia_proxy_max_stv")

        before = len(df)
        mask = (
            (df["n_electrodes"] >= min_elec)
            & (df["noise_sd_uv"] <= max_noise)
            & (df["beat_detection_rate"] >= min_bdr)
        )

        if reject_irregular:
            if irregularity_threshold is None:
                self.qc_log.append(
                    "Warning: arrhythmia-proxy rejection is enabled, but "
                    "quality_control.arrhythmia_proxy_max_stv is not configured; "
                    "no irregularity rejection was applied. STV is treated only "
                    "as an optional irregularity proxy, not an arrhythmia detector."
                )
            else:
                mask &= df["stv"] <= float(irregularity_threshold)

        rejected = df[~mask]
        kept = df[mask].copy()

        if qc.get("log_rejections", True) and len(rejected) > 0:
            for _, row in rejected.iterrows():
                details = (
                    f"noise={row['noise_sd_uv']:.1f}, elec={row['n_electrodes']}, "
                    f"bdr={row['beat_detection_rate']:.2f}"
                )
                if reject_irregular and irregularity_threshold is not None and row["stv"] > irregularity_threshold:
                    details += f", stv={row['stv']:.3f}"
                self.qc_log.append(
                    f"Rejected {row['compound']} {row['well']} ({details})"
                )
        self.qc_log.append(f"QC: kept {len(kept)}/{before} wells")
        return kept

    def compute_effects(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate vehicle-normalized effects for each treated well."""
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

            if v_fpd == 0 or v_rate == 0 or v_amp == 0:
                self.qc_log.append(
                    f"Warning: zero vehicle baseline encountered for {compound}; "
                    "affected percentage effects were not calculated."
                )

            for _, row in treated.iterrows():
                fpd_change = np.nan if v_fpd == 0 else 100.0 * (row["fpd_ms"] - v_fpd) / v_fpd
                rate_change = np.nan if v_rate == 0 else 100.0 * (row["beat_rate_bpm"] - v_rate) / v_rate
                amp_change = np.nan if v_amp == 0 else 100.0 * (row["amplitude_uv"] - v_amp) / v_amp
                stv_increase = (row["stv"] - v_stv) / max(v_stv, 1e-6)
                tri_change = (row["triangulation_proxy"] - v_tri) / max(v_tri, 1e-6)

                finite_effects = [
                    abs(x)
                    for x in [fpd_change, rate_change, amp_change]
                    if pd.notna(x)
                ]
                finite_effects.extend([
                    abs(stv_increase) * 100.0,
                    abs(tri_change) * 100.0,
                ])
                max_effect_pct = max(finite_effects) if finite_effects else np.nan

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
                        "fpd_change_pct": fpd_change,
                        "beat_rate_change_pct": rate_change,
                        "amplitude_change_pct": amp_change,
                        "stv_increase": stv_increase,
                        "triangulation_proxy_change": tri_change,
                        "max_effect_pct": max_effect_pct,
                    }
                )
        return pd.DataFrame(records)

    @staticmethod
    def summarize_concentrations(
        effects: pd.DataFrame,
        replicate_aggregation: str = "mean",
    ) -> pd.DataFrame:
        """Summarize replicates within each compound-concentration group."""
        if effects.empty:
            return pd.DataFrame()
        if replicate_aggregation not in {"mean", "median"}:
            raise ValueError(
                "Unsupported replicate_aggregation: "
                f"{replicate_aggregation!r}. Expected 'mean' or 'median'."
            )

        endpoint_columns = [
            "fpd_change_pct",
            "beat_rate_change_pct",
            "amplitude_change_pct",
            "stv_increase",
            "triangulation_proxy_change",
        ]
        grouped = effects.groupby(["compound", "concentration_uM"], sort=True)
        rows = []
        for (compound, concentration), group in grouped:
            row = {
                "compound": compound,
                "concentration_uM": concentration,
                "n_replicates": int(group["well"].nunique()),
            }
            for endpoint in endpoint_columns:
                values = pd.to_numeric(group[endpoint], errors="coerce").dropna()
                aggregator = values.mean if replicate_aggregation == "mean" else values.median
                row[f"{endpoint}_mean"] = float(aggregator()) if len(values) else np.nan
                row[f"{endpoint}_sd"] = float(values.std(ddof=1)) if len(values) > 1 else np.nan
                row[f"{endpoint}_sem"] = (
                    float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else np.nan
                )
            row["max_effect_pct_mean"] = max(
                [
                    abs(row["fpd_change_pct_mean"]) if pd.notna(row["fpd_change_pct_mean"]) else 0.0,
                    abs(row["beat_rate_change_pct_mean"]) if pd.notna(row["beat_rate_change_pct_mean"]) else 0.0,
                    abs(row["amplitude_change_pct_mean"]) if pd.notna(row["amplitude_change_pct_mean"]) else 0.0,
                    abs(row["stv_increase_mean"]) * 100.0 if pd.notna(row["stv_increase_mean"]) else 0.0,
                    abs(row["triangulation_proxy_change_mean"]) * 100.0 if pd.notna(row["triangulation_proxy_change_mean"]) else 0.0,
                ]
            )
            rows.append(row)
        return pd.DataFrame(rows)

    @staticmethod
    def aggregate_compound_effects(
        concentration_summary: pd.DataFrame,
        concentration_aggregation: str = "max_absolute_effect",
    ) -> pd.DataFrame:
        """Aggregate concentration-level summaries into compound-level scoring inputs."""
        if concentration_summary.empty:
            return pd.DataFrame()
        if concentration_aggregation != "max_absolute_effect":
            raise ValueError(
                "Unsupported concentration_aggregation: "
                f"{concentration_aggregation!r}. Expected 'max_absolute_effect'."
            )

        rows = []
        for compound, group in concentration_summary.groupby("compound"):
            def max_abs(column: str) -> float:
                values = pd.to_numeric(group[column], errors="coerce").dropna()
                return float(values.abs().max()) if len(values) else 0.0

            def max_positive(column: str) -> float:
                values = pd.to_numeric(group[column], errors="coerce").dropna()
                return float(values.max()) if len(values) else 0.0

            def min_value(column: str) -> float:
                values = pd.to_numeric(group[column], errors="coerce").dropna()
                return float(values.min()) if len(values) else 0.0

            rows.append(
                {
                    "compound": compound,
                    "fpd_change_pct": max_abs("fpd_change_pct_mean"),
                    "beat_rate_change_pct": max_abs("beat_rate_change_pct_mean"),
                    "amplitude_change_pct": min_value("amplitude_change_pct_mean"),
                    "stv_increase": max_positive("stv_increase_mean"),
                    "triangulation_proxy": max_positive("triangulation_proxy_change_mean"),
                    "max_concentration_uM": float(group["concentration_uM"].max()),
                    "n_wells": int(group["n_replicates"].sum()),
                    "concentrations_tested": int(group["concentration_uM"].nunique()),
                    "max_effect_pct": float(group["max_effect_pct_mean"].max()),
                }
            )
        return pd.DataFrame(rows)

    def fit_dose_response(self, concentration_summary: pd.DataFrame) -> dict[str, list[DoseResponseFit]]:
        """Fit 4PL curves for each compound when explicitly enabled."""
        concentration_cfg = self.config.get("concentration_response", {})
        if not concentration_cfg.get("fit_curve", False) or concentration_summary.empty:
            return {}

        min_points = int(concentration_cfg.get("fit_min_concentrations", 4))
        min_r_squared = float(concentration_cfg.get("fit_min_r_squared", 0.80))
        min_monotonicity = float(concentration_cfg.get("fit_min_monotonicity", 0.80))
        ec50_boundary_factor = float(concentration_cfg.get("fit_ec50_boundary_factor", 2.0))
        max_ec50_uncertainty_fold = float(
            concentration_cfg.get("fit_max_ec50_uncertainty_fold", 100.0)
        )
        results: dict[str, list[DoseResponseFit]] = {}
        for compound, group in concentration_summary.groupby("compound"):
            fits = fit_concentration_series(
                group,
                min_points=min_points,
                min_r_squared=min_r_squared,
                min_monotonicity=min_monotonicity,
                ec50_boundary_factor=ec50_boundary_factor,
                max_ec50_uncertainty_fold=max_ec50_uncertainty_fold,
            )
            results[str(compound)] = fits
            failed = [fit for fit in fits if not fit.success]
            poor_quality = [fit for fit in fits if fit.success and not fit.quality_pass]
            if failed:
                self.qc_log.append(
                    f"Dose-response fitting: {compound} has {len(failed)} endpoint fit(s) "
                    "that did not converge or lacked sufficient data."
                )
            if poor_quality:
                self.qc_log.append(
                    f"Dose-response fitting: {compound} has {len(poor_quality)} endpoint fit(s) "
                    "that failed one or more configured quality gates."
                )
        return results

    @staticmethod
    def dose_response_exposure_evidence(
        fits: list[DoseResponseFit],
        min_concentration_uM: float,
        max_concentration_uM: float,
        endpoint_weights: dict[str, float],
    ) -> Optional[float]:
        """Summarize quality-passing EC50 fits as exposure evidence in [0, 1]."""
        if min_concentration_uM <= 0 or max_concentration_uM <= min_concentration_uM:
            return None
        usable = []
        log_span = np.log10(max_concentration_uM / min_concentration_uM)
        for fit in fits:
            if not fit.quality_pass or fit.ec50 is None or fit.ec50 <= 0:
                continue
            coverage = np.clip(
                np.log10(max_concentration_uM / fit.ec50) / log_span,
                0.0,
                1.0,
            )
            weight = float(endpoint_weights.get(fit.endpoint, 0.0))
            if weight > 0:
                usable.append((weight, float(coverage)))
        if not usable:
            return None
        total_weight = sum(weight for weight, _ in usable)
        return float(sum(weight * coverage for weight, coverage in usable) / total_weight)

    def run(self, dataset: SyntheticMEADataset | pd.DataFrame) -> PipelineResult:
        self.qc_log = []
        if isinstance(dataset, SyntheticMEADataset):
            df = dataset.features.copy()
        else:
            df = dataset.copy()

        df = self.apply_qc(df)
        effects = self.compute_effects(df)

        concentration_cfg = self.config.get("concentration_response", {})
        replicate_aggregation = concentration_cfg.get("replicate_aggregation", "mean")
        concentration_aggregation = concentration_cfg.get(
            "concentration_aggregation", "max_absolute_effect"
        )
        concentration_summary = self.summarize_concentrations(
            effects,
            replicate_aggregation=replicate_aggregation,
        )

        min_concentrations = int(concentration_cfg.get("min_concentrations", 1))
        effect_threshold_pct = float(concentration_cfg.get("effect_threshold_pct", 0.0))

        if not concentration_summary.empty:
            for compound, group in concentration_summary.groupby("compound"):
                n_concentrations = group["concentration_uM"].nunique()
                if n_concentrations < min_concentrations:
                    self.qc_log.append(
                        f"Warning: {compound} has {n_concentrations} treated concentration(s); "
                        f"configured minimum is {min_concentrations}. Score retained, "
                        "but concentration-response coverage is limited."
                    )

        dose_response_fits = self.fit_dose_response(concentration_summary)
        dose_response_cfg = self.config.get("scoring", {})
        dose_response_weight = float(dose_response_cfg.get("dose_response_weight", 0.0))

        agg = self.aggregate_compound_effects(
            concentration_summary,
            concentration_aggregation=concentration_aggregation,
        )
        if not agg.empty:
            agg["effect_detected"] = agg["max_effect_pct"] >= effect_threshold_pct

        scores = []
        endpoint_weights = {name: float(meta["weight"]) for name, meta in self.engine.endpoints.items()}
        for _, row in agg.iterrows():
            compound = str(row["compound"])
            fits = dose_response_fits.get(compound, [])
            treated_concentrations = concentration_summary.loc[
                concentration_summary["compound"] == compound, "concentration_uM"
            ]
            min_concentration = float(treated_concentrations.min()) if not treated_concentrations.empty else np.nan
            max_concentration = float(treated_concentrations.max()) if not treated_concentrations.empty else np.nan
            exposure_evidence = self.dose_response_exposure_evidence(
                fits,
                min_concentration,
                max_concentration,
                endpoint_weights,
            )

            endpoint_values = {
                "fpd_change_pct": row["fpd_change_pct"],
                "beat_rate_change_pct": row["beat_rate_change_pct"],
                "amplitude_change_pct": row["amplitude_change_pct"],
                "stv_increase": row["stv_increase"],
                "triangulation_proxy": row["triangulation_proxy"],
            }
            scores.append(
                self.engine.score_compound(
                    compound=compound,
                    endpoint_values=endpoint_values,
                    max_concentration_uM=row["max_concentration_uM"],
                    n_wells=int(row["n_wells"]),
                    dose_response_evidence=exposure_evidence,
                    dose_response_weight=dose_response_weight,
                )
            )

        summary_rows = []
        for s in scores:
            agg_row = agg.loc[agg["compound"] == s.compound].iloc[0]
            fits = dose_response_fits.get(s.compound, [])
            successful_quality_fits = sum(fit.quality_pass for fit in fits)
            exposure_evidence = s.metadata.get("dose_response_evidence")
            summary_rows.append(
                {
                    "compound": s.compound,
                    "cardioscore": round(s.score, 4),
                    "risk_class": s.risk_class,
                    "max_concentration_uM": s.max_concentration_uM,
                    "n_wells": s.n_wells,
                    "concentrations_tested": int(agg_row["concentrations_tested"]),
                    "max_effect_pct": round(float(agg_row["max_effect_pct"]), 2),
                    "effect_detected": bool(agg_row["effect_detected"]),
                    "dose_response_fit_endpoints": successful_quality_fits,
                    "dose_response_exposure_evidence": (
                        round(float(exposure_evidence), 4) if exposure_evidence is not None else np.nan
                    ),
                    "interpretation": s.interpretation,
                }
            )

        summary = pd.DataFrame(summary_rows)
        if not summary.empty:
            summary = summary.sort_values("cardioscore", ascending=False)

        return PipelineResult(
            scores=scores,
            feature_table=effects,
            summary_table=summary,
            concentration_table=concentration_summary,
            dose_response_fits=dose_response_fits,
            config=self.config,
            qc_log=self.qc_log,
        )

"""End-to-end CardioScore pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from virelion_cardioscore.analysis.cipa_scoring import CardioScoreEngine, ScoreResult
from virelion_cardioscore.analysis.dose_response import DoseResponseFit, fit_concentration_series
from virelion_cardioscore.analysis.hierarchy import aggregate_to_scoring_units
from virelion_cardioscore.analysis.normalization import apply_control_anchor_correction
from virelion_cardioscore.analysis.statistics import bootstrap_ci
from virelion_cardioscore.analysis.variability import control_variability, standardized_treatment_separation
from virelion_cardioscore.io.synthetic import SyntheticMEADataset


@dataclass
class PipelineResult:
    scores: list[ScoreResult]
    feature_table: pd.DataFrame
    summary_table: pd.DataFrame
    concentration_table: pd.DataFrame = field(default_factory=pd.DataFrame)
    inference_table: pd.DataFrame = field(default_factory=pd.DataFrame)
    variability_table: pd.DataFrame = field(default_factory=pd.DataFrame)
    variability_before_correction: pd.DataFrame = field(default_factory=pd.DataFrame)
    separation_table: pd.DataFrame = field(default_factory=pd.DataFrame)
    normalization_diagnostic: dict = field(default_factory=dict)
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
            "inference": self.inference_table.to_dict(orient="records"),
            "variability": self.variability_table.to_dict(orient="records"),
            "variability_before_correction": self.variability_before_correction.to_dict(orient="records"),
            "treatment_separation": self.separation_table.to_dict(orient="records"),
            "normalization": self.normalization_diagnostic,
            "dose_response_fits": {
                compound: [fit.to_dict() for fit in fits]
                for compound, fits in self.dose_response_fits.items()
            },
        }
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)


class CardioScorePipeline:
    """High-level orchestrator for the CardioScore workflow."""

    def __init__(self, config: dict):
        self.config = config
        scoring_cfg = config.get("scoring", {})
        endpoint_config = scoring_cfg.get("endpoint_config", "cipa_endpoints.yaml")
        endpoint_path = Path(endpoint_config)
        if not endpoint_path.is_absolute():
            base_dir = Path(config.get("_config_dir", Path(__file__).resolve().parents[1] / "config"))
            endpoint_path = base_dir / endpoint_path
            if not endpoint_path.exists():
                endpoint_path = Path(__file__).resolve().parents[1] / "config" / endpoint_config
        if not endpoint_path.exists():
            raise ValueError(f"Configured endpoint configuration does not exist: {endpoint_path}")
        self.engine = CardioScoreEngine(
            endpoint_config_path=endpoint_path,
            low_threshold=scoring_cfg.get("low_threshold", 0.30),
            moderate_threshold=scoring_cfg.get("moderate_threshold", 0.60),
        )
        self.qc_log: list[str] = []

    @classmethod
    def from_config(cls, path: str | Path) -> "CardioScorePipeline":
        path = Path(path)
        with open(path, encoding="utf-8") as handle:
            cfg = yaml.safe_load(handle) or {}
        cfg = dict(cfg)
        cfg["_config_dir"] = str(path.resolve().parent)
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

    def _control_group_columns(self, df: pd.DataFrame) -> list[str]:
        scope = self.config.get("control_normalization", {}).get("scope", "compound")
        aliases = {
            "compound": ["compound"],
            "plate": ["plate_id"],
            "batch": ["batch_id" if "batch_id" in df.columns else "experiment_id"],
            "biological_replicate": ["biological_replicate"],
            "global": [],
        }
        if scope not in aliases:
            raise ValueError(
                f"Unsupported control_normalization.scope: {scope!r}. Expected 'compound', 'plate', 'batch', 'biological_replicate', or 'global'."
            )
        columns = aliases[scope]
        missing = [column for column in columns if column not in df.columns]
        if missing:
            raise ValueError(f"control_normalization.scope={scope!r} requires metadata column(s) {missing!r}, but they are not present in the dataset.")
        for column in columns:
            if df[column].isna().any() or df[column].astype(str).str.strip().eq("").any():
                raise ValueError(f"control_normalization.scope={scope!r} cannot use grouping column {column!r} with missing or blank identifiers.")
        return columns

    def compute_effects(self, df: pd.DataFrame) -> pd.DataFrame:
        control_cfg = self.config.get("control_normalization", {})
        normalize = bool(self.config.get("scoring", {}).get("normalize_by_vehicle", True))
        if not normalize:
            required = {"compound", "concentration_uM", "well", "fpd_change_pct", "beat_rate_change_pct", "amplitude_change_pct", "stv_increase", "triangulation_proxy_change"}
            missing = sorted(required - set(df.columns))
            if missing:
                raise ValueError(f"normalize_by_vehicle=false requires precomputed effect columns: {missing}")
            effects = df.copy()
            effects["vehicle"] = False
            effects["max_effect_pct"] = effects[["fpd_change_pct", "beat_rate_change_pct", "amplitude_change_pct"]].abs().max(axis=1)
            return effects
        records = []
        optional_metadata = ["biological_replicate", "batch_id", "experiment_id", "plate_id"]
        scope = control_cfg.get("scope", "compound")
        control_columns = self._control_group_columns(df)
        require_match = bool(control_cfg.get("require_matching_control", True))
        grouped = df.groupby(control_columns, dropna=False, sort=True) if control_columns else [((), df)]
        for group_key, group in grouped:
            if not isinstance(group_key, tuple):
                group_key = (group_key,)
            vehicle = group[group["vehicle"].astype(bool)]
            treated = group[~group["vehicle"].astype(bool)]
            if vehicle.empty:
                message = f"No matching vehicle control for normalization scope={scope!r} group={group_key!r}."
                if require_match:
                    self.qc_log.append(f"Warning: {message} Treated wells excluded from effect calculation.")
                else:
                    self.qc_log.append(f"Warning: {message} Group skipped; no unnormalized fallback is supported.")
                continue
            v_fpd = vehicle["fpd_ms"].mean()
            v_rate = vehicle["beat_rate_bpm"].mean()
            v_amp = vehicle["amplitude_uv"].mean()
            v_stv = vehicle["stv"].mean()
            v_tri = vehicle["triangulation_proxy"].mean()
            for _, row in treated.iterrows():
                fpd_change = np.nan if v_fpd == 0 else 100.0 * (row["fpd_ms"] - v_fpd) / v_fpd
                rate_change = np.nan if v_rate == 0 else 100.0 * (row["beat_rate_bpm"] - v_rate) / v_rate
                amp_change = np.nan if v_amp == 0 else 100.0 * (row["amplitude_uv"] - v_amp) / v_amp
                stv_increase = (row["stv"] - v_stv) / max(abs(v_stv), 1e-6)
                tri_change = (row["triangulation_proxy"] - v_tri) / max(abs(v_tri), 1e-6)
                finite_effects = [abs(x) for x in [fpd_change, rate_change, amp_change] if pd.notna(x)]
                finite_effects.extend([abs(stv_increase) * 100.0, abs(tri_change) * 100.0])
                record = {
                    "compound": str(row["compound"]),
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
                    "max_effect_pct": max(finite_effects) if finite_effects else np.nan,
                }
                for column in optional_metadata:
                    if column in row.index:
                        record[column] = row[column]
                records.append(record)
        return pd.DataFrame(records)

    def prepare_scoring_effects(self, effects: pd.DataFrame) -> pd.DataFrame:
        unit_cfg = self.config.get("experimental_units", {})
        scoring_unit = unit_cfg.get("scoring_unit", "well")
        if scoring_unit != "well" and unit_cfg.get("fall_back_to_well", False):
            required_map = {
                "biological_replicate": unit_cfg.get("biological_unit_column"),
                "batch": unit_cfg.get("batch_unit_column"),
                "plate": unit_cfg.get("plate_unit_column"),
            }
            requested_column = required_map.get(scoring_unit)
            if requested_column and requested_column not in effects.columns:
                self.qc_log.append(
                    f"Experimental-unit column {requested_column!r} is absent; falling back to well-level scoring."
                )
                scoring_unit = "well"
        try:
            prepared = aggregate_to_scoring_units(
                effects,
                scoring_unit=scoring_unit,
                biological_unit_column=unit_cfg.get("biological_unit_column"),
                batch_unit_column=unit_cfg.get("batch_unit_column"),
                plate_unit_column=unit_cfg.get("plate_unit_column"),
            )
        except ValueError as exc:
            raise ValueError(f"Experimental-unit configuration is invalid: {exc}") from exc
        if scoring_unit != "well":
            technical_wells = int(effects["well"].nunique()) if "well" in effects else len(effects)
            scoring_units = int(prepared["well"].nunique()) if "well" in prepared else len(prepared)
            self.qc_log.append(f"Experimental units: scoring at {scoring_unit} level; collapsed {technical_wells} technical wells into {scoring_units} independent units.")
        return prepared

    def run_variability_diagnostics(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        cfg = self.config.get("variability", {})
        if not cfg.get("enabled", False):
            return pd.DataFrame(), pd.DataFrame()
        group_column = cfg.get("group_column")
        variability_table = control_variability(df, group_column=group_column, max_control_cv_pct=float(cfg.get("max_control_cv_pct", 20.0)))
        if variability_table.empty:
            return variability_table, pd.DataFrame()
        resolved_group = str(variability_table.iloc[0]["group_column"])
        separation_frames = []
        for endpoint in ["fpd_ms", "beat_rate_bpm", "amplitude_uv", "stv", "triangulation_proxy"]:
            if endpoint in df.columns:
                separation_frames.append(standardized_treatment_separation(df.assign(vehicle=df["vehicle"].astype(bool)), endpoint=endpoint, group_column=resolved_group))
        separation_table = pd.concat(separation_frames, ignore_index=True) if separation_frames else pd.DataFrame()
        for _, row in variability_table.iterrows():
            if row["status"] in {"high_variability", "insufficient_groups"}:
                self.qc_log.append(f"Warning: {row['endpoint']} {row['message']}")
        return variability_table, separation_table

    def apply_control_anchor_normalization(self, df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        cfg = self.config.get("variability", {}).get("correction", {})
        if not cfg.get("enabled", False):
            return df, {}
        if not self.config.get("variability", {}).get("enabled", False):
            raise ValueError("variability.correction.enabled requires variability.enabled=true.")
        assumptions = cfg.get("assumptions", {})
        corrected, diagnostic = apply_control_anchor_correction(
            df,
            group_column=self.config.get("variability", {}).get("group_column"),
            corrected_columns=cfg.get("corrected_columns"),
            min_controls_per_group=int(cfg.get("min_controls_per_group", 2)),
            require_all_groups=bool(cfg.get("require_all_groups", True)),
            min_treated_per_group=int(assumptions.get("min_treated_per_group", 1)),
            require_treatment_in_all_groups=bool(assumptions.get("require_treatment_in_all_groups", True)),
            max_shift_cv_pct=float(assumptions.get("max_shift_cv_pct", 50.0)),
            fail_on_assumption_violation=bool(assumptions.get("fail_closed", True)),
        )
        self.qc_log.append(f"Control normalization: applied control-anchored recentering across {diagnostic.n_groups} groups using {diagnostic.n_controls} vehicle wells.")
        return corrected, diagnostic.to_dict()

    @staticmethod
    def summarize_concentrations(effects: pd.DataFrame, replicate_aggregation: str = "mean") -> pd.DataFrame:
        if effects.empty:
            return pd.DataFrame()
        if replicate_aggregation not in {"mean", "median"}:
            raise ValueError(f"Unsupported replicate_aggregation: {replicate_aggregation!r}. Expected 'mean' or 'median'.")
        endpoint_columns = ["fpd_change_pct", "beat_rate_change_pct", "amplitude_change_pct", "stv_increase", "triangulation_proxy_change"]
        rows = []
        for (compound, concentration), group in effects.groupby(["compound", "concentration_uM"], sort=True):
            row = {"compound": compound, "concentration_uM": concentration, "n_replicates": int(group["well"].nunique()), "n_technical_wells": int(group["n_wells"].sum()) if "n_wells" in group.columns else int(group["well"].nunique())}
            for endpoint in endpoint_columns:
                values = pd.to_numeric(group[endpoint], errors="coerce").dropna()
                aggregator = values.mean if replicate_aggregation == "mean" else values.median
                row[f"{endpoint}_mean"] = float(aggregator()) if len(values) else np.nan
                row[f"{endpoint}_sd"] = float(values.std(ddof=1)) if len(values) > 1 else np.nan
                row[f"{endpoint}_sem"] = float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else np.nan
            row["max_effect_pct_mean"] = max([abs(row["fpd_change_pct_mean"]), abs(row["beat_rate_change_pct_mean"]), abs(row["amplitude_change_pct_mean"]), abs(row["stv_increase_mean"]) * 100.0, abs(row["triangulation_proxy_change_mean"]) * 100.0])
            rows.append(row)
        return pd.DataFrame(rows)

    @staticmethod
    def bootstrap_concentration_inference(effects: pd.DataFrame, *, n_bootstrap: int = 2000, confidence: float = 0.95, seed: int = 42) -> pd.DataFrame:
        if effects.empty:
            return pd.DataFrame()
        endpoint_columns = ["fpd_change_pct", "beat_rate_change_pct", "amplitude_change_pct", "stv_increase", "triangulation_proxy_change"]
        rows = []
        for group_index, ((compound, concentration), group) in enumerate(effects.groupby(["compound", "concentration_uM"], sort=True)):
            row = {"compound": compound, "concentration_uM": concentration, "n_replicates": int(group["well"].nunique())}
            for endpoint in endpoint_columns:
                values = pd.to_numeric(group[endpoint], errors="coerce").to_numpy(dtype=float)
                values = values[np.isfinite(values)]
                if len(values) < 2:
                    row[f"{endpoint}_ci_low"] = np.nan
                    row[f"{endpoint}_ci_high"] = np.nan
                    continue
                result = bootstrap_ci(values, n_bootstrap=n_bootstrap, confidence=confidence, seed=seed + group_index)
                row[f"{endpoint}_ci_low"] = result.ci_low
                row[f"{endpoint}_ci_high"] = result.ci_high
            rows.append(row)
        return pd.DataFrame(rows)

    @staticmethod
    def aggregate_compound_effects(
        concentration_summary: pd.DataFrame,
        concentration_aggregation: str = "max_absolute_effect",
        endpoint_directions: dict[str, str] | None = None,
    ) -> pd.DataFrame:
        if concentration_summary.empty:
            return pd.DataFrame()
        if concentration_aggregation != "max_absolute_effect":
            raise ValueError(f"Unsupported concentration_aggregation: {concentration_aggregation!r}. Expected 'max_absolute_effect'.")
        endpoint_directions = endpoint_directions or {
            "fpd_change_pct": "absolute",
            "beat_rate_change_pct": "absolute",
            "amplitude_change_pct": "decrease",
            "stv_increase": "increase",
            "triangulation_proxy_change": "increase",
        }
        rows = []
        for compound, group in concentration_summary.groupby("compound"):
            def aggregate_endpoint(column: str) -> float:
                values = pd.to_numeric(group[column], errors="coerce").dropna()
                if values.empty:
                    return 0.0
                endpoint = column.removesuffix("_mean")
                direction = endpoint_directions.get(endpoint, "absolute")
                if direction == "decrease":
                    return float(values.min())
                if direction == "increase":
                    return float(values.max())
                if direction == "absolute":
                    return float(values.abs().max())
                raise ValueError(f"Unsupported endpoint direction: {direction!r} for {endpoint!r}.")

            technical_wells = int(group["n_technical_wells"].sum()) if "n_technical_wells" in group.columns else int(group["n_replicates"].sum())
            independent_units = int(group["n_replicates"].sum())
            rows.append({
                "compound": compound,
                "fpd_change_pct": aggregate_endpoint("fpd_change_pct_mean"),
                "beat_rate_change_pct": aggregate_endpoint("beat_rate_change_pct_mean"),
                "amplitude_change_pct": aggregate_endpoint("amplitude_change_pct_mean"),
                "stv_increase": aggregate_endpoint("stv_increase_mean"),
                "triangulation_proxy": aggregate_endpoint("triangulation_proxy_change_mean"),
                "max_concentration_uM": float(group["concentration_uM"].max()),
                "n_wells": technical_wells,
                "n_independent_units": independent_units,
                "concentrations_tested": int(group["concentration_uM"].nunique()),
                "max_effect_pct": float(group["max_effect_pct_mean"].max()),
                "n_technical_wells": technical_wells,
            })
        return pd.DataFrame(rows)

    def fit_dose_response(self, concentration_summary: pd.DataFrame) -> dict[str, list[DoseResponseFit]]:
        cfg = self.config.get("concentration_response", {})
        if not cfg.get("fit_curve", False) or concentration_summary.empty:
            return {}
        results: dict[str, list[DoseResponseFit]] = {}
        for compound, group in concentration_summary.groupby("compound"):
            fits = fit_concentration_series(group, min_points=int(cfg.get("fit_min_concentrations", 4)), min_r_squared=float(cfg.get("fit_min_r_squared", 0.80)), min_monotonicity=float(cfg.get("fit_min_monotonicity", 0.80)), ec50_boundary_factor=float(cfg.get("fit_ec50_boundary_factor", 2.0)), max_ec50_uncertainty_fold=float(cfg.get("fit_max_ec50_uncertainty_fold", 100.0)))
            results[str(compound)] = fits
        return results

    @staticmethod
    def dose_response_exposure_evidence(fits: list[DoseResponseFit], min_concentration_uM: float, max_concentration_uM: float, endpoint_weights: dict[str, float]) -> float | None:
        if min_concentration_uM <= 0 or max_concentration_uM <= min_concentration_uM:
            return None
        log_span = np.log10(max_concentration_uM / min_concentration_uM)
        usable = []
        for fit in fits:
            if not fit.quality_pass or fit.ec50 is None or fit.ec50 <= 0:
                continue
            coverage = np.clip(np.log10(max_concentration_uM / fit.ec50) / log_span, 0.0, 1.0)
            weight = float(endpoint_weights.get(fit.endpoint, 0.0))
            if weight > 0:
                usable.append((weight, float(coverage)))
        if not usable:
            return None
        total_weight = sum(weight for weight, _ in usable)
        return float(sum(weight * coverage for weight, coverage in usable) / total_weight)

    def run(self, dataset: SyntheticMEADataset | pd.DataFrame) -> PipelineResult:
        self.qc_log = []
        df = dataset.features.copy() if isinstance(dataset, SyntheticMEADataset) else dataset.copy()
        df = self.apply_qc(df)
        variability_cfg = self.config.get("variability", {})
        variability_before, _ = self.run_variability_diagnostics(df)
        normalization_diagnostic: dict = {}
        if variability_cfg.get("correction", {}).get("enabled", False):
            df, normalization_diagnostic = self.apply_control_anchor_normalization(df)
        variability_after, separation_table = self.run_variability_diagnostics(df)
        effects = self.compute_effects(df)
        scoring_effects = self.prepare_scoring_effects(effects)
        concentration_cfg = self.config.get("concentration_response", {})
        concentration_summary = self.summarize_concentrations(scoring_effects, replicate_aggregation=concentration_cfg.get("replicate_aggregation", "mean"))
        min_concentrations = int(concentration_cfg.get("min_concentrations", 3))
        for compound, group in concentration_summary.groupby("compound"):
            n_concentrations = int(group["concentration_uM"].nunique())
            if n_concentrations < min_concentrations:
                self.qc_log.append(
                    f"Warning: {compound} has {n_concentrations} tested concentration(s); "
                    f"configured minimum is {min_concentrations}. No concentrations were silently excluded."
                )
        inference_cfg = self.config.get("inference", {})
        inference_table = pd.DataFrame()
        if inference_cfg.get("enabled", False) and not scoring_effects.empty:
            inference_table = self.bootstrap_concentration_inference(scoring_effects, n_bootstrap=int(inference_cfg.get("n_bootstrap", 2000)), confidence=float(inference_cfg.get("confidence", 0.95)), seed=int(inference_cfg.get("seed", 42)))
        dose_response_fits = self.fit_dose_response(concentration_summary)
        scoring_endpoint_directions = {
            name: str(meta["direction"])
            for name, meta in self.engine.endpoints.items()
        }
        dose_response_weight = float(self.config.get("scoring", {}).get("dose_response_weight", 0.0))
        agg = self.aggregate_compound_effects(
            concentration_summary,
            endpoint_directions={
                "fpd_change_pct": scoring_endpoint_directions.get("fpd_change_pct", "absolute"),
                "beat_rate_change_pct": scoring_endpoint_directions.get("beat_rate_change_pct", "absolute"),
                "amplitude_change_pct": scoring_endpoint_directions.get("amplitude_change_pct", "decrease"),
                "stv_increase": scoring_endpoint_directions.get("stv_increase", "increase"),
                "triangulation_proxy_change": scoring_endpoint_directions.get("triangulation_proxy_change", "increase"),
            },
        )
        if not agg.empty and not scoring_effects.empty:
            compound_unit_counts = scoring_effects.groupby("compound", sort=False)["well"].nunique()
            agg["n_independent_units"] = agg["compound"].map(compound_unit_counts).fillna(0).astype(int)
        if not agg.empty:
            agg["effect_detected"] = agg["max_effect_pct"] >= float(concentration_cfg.get("effect_threshold_pct", 0.0))
        scores = []
        endpoint_weights = {name: float(meta["weight"]) for name, meta in self.engine.endpoints.items()}
        for _, row in agg.iterrows():
            compound = str(row["compound"])
            fits = dose_response_fits.get(compound, [])
            concentrations = concentration_summary.loc[concentration_summary["compound"] == compound, "concentration_uM"]
            min_concentration = float(concentrations.min()) if not concentrations.empty else np.nan
            max_concentration = float(concentrations.max()) if not concentrations.empty else np.nan
            exposure_evidence = self.dose_response_exposure_evidence(fits, min_concentration, max_concentration, endpoint_weights)
            scores.append(self.engine.score_compound(
                compound=compound,
                endpoint_values={
                    "fpd_change_pct": row["fpd_change_pct"],
                    "beat_rate_change_pct": row["beat_rate_change_pct"],
                    "amplitude_change_pct": row["amplitude_change_pct"],
                    "stv_increase": row["stv_increase"],
                    "triangulation_proxy": row["triangulation_proxy"],
                },
                max_concentration_uM=row["max_concentration_uM"],
                n_wells=int(row["n_wells"]),
                n_independent_units=int(row["n_independent_units"]),
                dose_response_evidence=exposure_evidence,
                dose_response_weight=dose_response_weight,
            ))
        summary_rows = []
        fit_curve_enabled = bool(concentration_cfg.get("fit_curve", False))
        max_ec50_uncertainty_fold = float(concentration_cfg.get("fit_max_ec50_uncertainty_fold", 100.0))
        for s in scores:
            agg_row = agg.loc[agg["compound"] == s.compound].iloc[0]
            row = {
                "compound": s.compound,
                "cardioscore": round(s.score, 4),
                "risk_class": s.risk_class,
                "max_concentration_uM": s.max_concentration_uM,
                "n_wells": s.n_wells,
                "n_independent_units": s.n_independent_units,
                "n_technical_wells": int(agg_row["n_technical_wells"]),
                "concentrations_tested": int(agg_row["concentrations_tested"]),
                "max_effect_pct": round(float(agg_row["max_effect_pct"]), 2),
                "effect_detected": bool(agg_row["effect_detected"]),
                "interpretation": s.interpretation,
            }
            if fit_curve_enabled:
                fits = dose_response_fits.get(s.compound, [])
                successful = [fit for fit in fits if fit.success]
                monotonicities = [fit.monotonicity for fit in successful if fit.monotonicity is not None]
                row["dose_response_fit_endpoints"] = len(successful)
                row["dose_response_boundary_flags"] = sum(1 for fit in successful if fit.ec50_boundary_flag)
                row["dose_response_high_uncertainty"] = sum(
                    1
                    for fit in successful
                    if fit.ec50_uncertainty_fold is not None and fit.ec50_uncertainty_fold > max_ec50_uncertainty_fold
                )
                row["dose_response_mean_monotonicity"] = (
                    round(float(np.mean(monotonicities)), 4) if monotonicities else None
                )
            summary_rows.append(row)
        summary = pd.DataFrame(summary_rows)
        if not summary.empty:
            summary = summary.sort_values("cardioscore", ascending=False)
        return PipelineResult(
            scores=scores,
            feature_table=effects,
            summary_table=summary,
            concentration_table=concentration_summary,
            inference_table=inference_table,
            variability_table=variability_after,
            variability_before_correction=variability_before,
            separation_table=separation_table,
            normalization_diagnostic=normalization_diagnostic,
            dose_response_fits=dose_response_fits,
            config=self.config,
            qc_log=self.qc_log,
        )

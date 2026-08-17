"""Opt-in pipeline wrapper that adds hierarchical mixed-effects inference."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from virelion_cardioscore.analysis.mixed_effects_pipeline import fit_compound_concentration_mixed_effects
from virelion_cardioscore.analysis.pipeline import CardioScorePipeline, PipelineResult
from virelion_cardioscore.io.synthetic import SyntheticMEADataset


class HierarchicalPipelineResult:
    """Delegate normal CardioScore results while adding mixed-effects output."""

    def __init__(self, base: PipelineResult, mixed_effects_table: pd.DataFrame):
        self.base = base
        self.mixed_effects_table = mixed_effects_table

    def __getattr__(self, name: str):
        return getattr(self.base, name)

    def to_json(self, path: str | Path) -> None:
        path = Path(path)
        payload = {
            "scores": [score.to_dict() for score in self.base.scores],
            "summary": self.base.summary_table.to_dict(orient="records"),
            "concentration_summary": self.base.concentration_table.to_dict(orient="records"),
            "inference": self.base.inference_table.to_dict(orient="records"),
            "variability": self.base.variability_table.to_dict(orient="records"),
            "variability_before_correction": self.base.variability_before_correction.to_dict(orient="records"),
            "treatment_separation": self.base.separation_table.to_dict(orient="records"),
            "normalization": self.base.normalization_diagnostic,
            "mixed_effects": self.mixed_effects_table.to_dict(orient="records"),
            "dose_response_fits": {
                compound: [fit.to_dict() for fit in fits]
                for compound, fits in self.base.dose_response_fits.items()
            },
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def to_html(self, path: str | Path) -> None:
        path = Path(path)
        self.base.to_html(path)
        if self.mixed_effects_table.empty:
            return
        html = path.read_text(encoding="utf-8")

        columns = [
            "compound", "concentration_uM", "endpoint", "status",
            "treatment_effect", "treatment_se", "treatment_pvalue", "icc",
            "group_column", "n_groups", "n_observations",
        ]
        present = [column for column in columns if column in self.mixed_effects_table.columns]
        header = "".join(f"<th>{column}</th>" for column in present)
        rows = []
        for _, row in self.mixed_effects_table.iterrows():
            cells = []
            for column in present:
                value = row[column]
                if pd.isna(value):
                    value = "—"
                elif isinstance(value, float):
                    value = f"{value:.4g}"
                cells.append(f"<td>{value}</td>")
            rows.append(f"<tr>{''.join(cells)}</tr>")

        card = f"""
        <div class=\"card\">
          <h2 style=\"margin-top:0\">Hierarchical Mixed-Effects Inference</h2>
          <p class=\"meta\">Inference only; these models use the same QC/normalization population as the base run and do not alter CardioScore.</p>
          <table>
            <thead><tr>{header}</tr></thead>
            <tbody>{''.join(rows)}</tbody>
          </table>
        </div>
        """
        html = html.replace("</body>", f"{card}</body>")
        path.write_text(html, encoding="utf-8")


class HierarchicalCardioScorePipeline:
    """Run the standard pipeline plus optional mixed-effects inference."""

    def __init__(self, config: dict):
        self.config = config
        self.base = CardioScorePipeline(config)

    @classmethod
    def from_config(cls, path: str | Path) -> "HierarchicalCardioScorePipeline":
        return cls(CardioScorePipeline.from_config(path).config)

    @classmethod
    def from_defaults(cls) -> "HierarchicalCardioScorePipeline":
        return cls(CardioScorePipeline.from_defaults().config)

    def run(self, dataset: SyntheticMEADataset | pd.DataFrame) -> HierarchicalPipelineResult:
        base_result = self.base.run(dataset)
        cfg = self.config.get("mixed_effects", {})
        if not cfg.get("enabled", False):
            return HierarchicalPipelineResult(base_result, pd.DataFrame())

        raw = dataset.features.copy() if isinstance(dataset, SyntheticMEADataset) else dataset.copy()
        analysis_df = self.base.apply_qc(raw)
        if self.config.get("variability", {}).get("correction", {}).get("enabled", False):
            analysis_df, _ = self.base.apply_control_anchor_normalization(analysis_df)

        group_column = cfg.get("group_column")
        if group_column is None:
            for candidate in ("plate_id", "batch_id", "experiment_id"):
                if candidate in analysis_df.columns:
                    group_column = candidate
                    break
        if group_column is None:
            raise ValueError(
                "mixed_effects.enabled requires a genuine grouping column such as "
                "plate_id, batch_id, or experiment_id."
            )
        if analysis_df[group_column].isna().any() or analysis_df[group_column].astype(str).str.strip().eq("").any():
            raise ValueError(f"Mixed-effects grouping column {group_column!r} contains missing or blank identifiers.")

        endpoints = list(cfg.get("endpoints", []))
        rows = fit_compound_concentration_mixed_effects(
            analysis_df,
            group_column=group_column,
            endpoints=endpoints,
            vehicle_column=cfg.get("vehicle_column", "vehicle"),
            treatment_column=cfg.get("treatment_column", "_treatment"),
        )
        return HierarchicalPipelineResult(base_result, pd.DataFrame(rows))

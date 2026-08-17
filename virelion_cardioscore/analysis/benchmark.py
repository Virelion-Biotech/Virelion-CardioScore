"""Reproducible benchmark runner for CardioScore reference datasets."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from virelion_cardioscore.analysis.pipeline import CardioScorePipeline


_ALLOWED_EVIDENCE_LEVELS = {
    "raw_mea_dataset",
    "processed_mea_summary",
    "published_mea_summary",
}
_EXECUTABLE_EVIDENCE_LEVELS = {"raw_mea_dataset", "processed_mea_summary"}


@dataclass(frozen=True)
class BenchmarkComparison:
    dataset: str
    compound: str
    expected_score: float
    observed_score: float
    score_error: float
    score_tolerance: float
    expected_risk_class: str
    observed_risk_class: str
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "compound": self.compound,
            "expected_score": self.expected_score,
            "observed_score": self.observed_score,
            "score_error": self.score_error,
            "score_tolerance": self.score_tolerance,
            "expected_risk_class": self.expected_risk_class,
            "observed_risk_class": self.observed_risk_class,
            "passed": self.passed,
        }


def _resolve_path(base_dir: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base_dir / path


def run_benchmark_manifest(manifest_path: str | Path) -> list[BenchmarkComparison]:
    """Run all benchmark entries declared by a YAML/JSON manifest."""
    manifest_path = Path(manifest_path)
    with manifest_path.open(encoding="utf-8") as handle:
        if manifest_path.suffix.lower() in {".yaml", ".yml"}:
            manifest = yaml.safe_load(handle) or {}
        else:
            manifest = json.load(handle)

    entries = manifest.get("datasets", [])
    if not isinstance(entries, list) or not entries:
        raise ValueError("Benchmark manifest must contain a non-empty 'datasets' list.")

    results: list[BenchmarkComparison] = []
    base_dir = manifest_path.resolve().parent
    for entry in entries:
        name = str(entry.get("name", "unnamed"))
        evidence_level = str(entry.get("evidence_level", "raw_mea_dataset"))
        if evidence_level not in _ALLOWED_EVIDENCE_LEVELS:
            raise ValueError(
                f"Benchmark dataset {name!r} has unsupported evidence_level {evidence_level!r}."
            )
        if evidence_level not in _EXECUTABLE_EVIDENCE_LEVELS:
            raise ValueError(
                f"Benchmark dataset {name!r} is {evidence_level!r}; full numerical benchmark "
                "requires raw_mea_dataset or processed_mea_summary evidence."
            )
        features_path = _resolve_path(base_dir, str(entry["features"]))
        config_path = _resolve_path(base_dir, str(entry["config"]))
        tolerance = float(entry.get("score_tolerance", 0.01))
        if tolerance < 0:
            raise ValueError(f"Benchmark dataset {name!r} score_tolerance cannot be negative.")
        expected = entry.get("expected", [])
        if not isinstance(expected, list) or not expected:
            raise ValueError(f"Benchmark dataset {name!r} has no expected reference rows.")

        result = CardioScorePipeline.from_config(config_path).run(pd.read_csv(features_path))
        observed = {
            str(row["compound"]): row
            for row in result.summary_table.to_dict(orient="records")
        }
        for reference in expected:
            compound = str(reference["compound"])
            if compound not in observed:
                raise ValueError(f"Benchmark dataset {name!r} is missing observed compound {compound!r}.")
            row = observed[compound]
            expected_score = float(reference["cardioscore"])
            observed_score = float(row["cardioscore"])
            expected_class = str(reference["risk_class"])
            observed_class = str(row["risk_class"])
            error = abs(observed_score - expected_score)
            passed = error <= tolerance and observed_class == expected_class
            results.append(
                BenchmarkComparison(
                    dataset=name,
                    compound=compound,
                    expected_score=expected_score,
                    observed_score=observed_score,
                    score_error=error,
                    score_tolerance=tolerance,
                    expected_risk_class=expected_class,
                    observed_risk_class=observed_class,
                    passed=passed,
                )
            )
    return results


def benchmark_summary(results: list[BenchmarkComparison]) -> dict[str, Any]:
    passed = sum(item.passed for item in results)
    return {
        "n_comparisons": len(results),
        "n_passed": passed,
        "n_failed": len(results) - passed,
        "pass_rate": (passed / len(results)) if results else 0.0,
    }

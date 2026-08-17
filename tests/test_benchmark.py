from pathlib import Path

import pandas as pd
import pytest
import yaml

from virelion_cardioscore.analysis.benchmark import benchmark_summary, run_benchmark_manifest


def test_reference_benchmark_detects_known_score_and_class(tmp_path: Path):
    features = pd.DataFrame(
        [
            {
                "compound": "Known",
                "well": "V1",
                "concentration_uM": 0.0,
                "vehicle": True,
                "fpd_ms": 280.0,
                "beat_rate_bpm": 55.0,
                "amplitude_uv": 180.0,
                "stv": 0.04,
                "triangulation_proxy": 0.18,
                "noise_sd_uv": 5.0,
                "n_electrodes": 8,
                "beat_detection_rate": 0.95,
            },
            {
                "compound": "Known",
                "well": "W1",
                "concentration_uM": 1.0,
                "vehicle": False,
                "fpd_ms": 392.0,
                "beat_rate_bpm": 55.0,
                "amplitude_uv": 180.0,
                "stv": 0.04,
                "triangulation_proxy": 0.18,
                "noise_sd_uv": 5.0,
                "n_electrodes": 8,
                "beat_detection_rate": 0.95,
            },
        ]
    )
    features_path = tmp_path / "known.csv"
    features.to_csv(features_path, index=False)

    repo_root = Path(__file__).parents[1]
    config_path = repo_root / "virelion_cardioscore" / "config" / "default.yaml"
    manifest = {
        "datasets": [
            {
                "name": "known_effect",
                "features": str(features_path),
                "config": str(config_path),
                "evidence_level": "processed_mea_summary",
                "score_tolerance": 1e-6,
                "expected": [
                    {"compound": "Known", "cardioscore": 0.30, "risk_class": "Moderate"}
                ],
            }
        ]
    }
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(yaml.safe_dump(manifest), encoding="utf-8")

    results = run_benchmark_manifest(manifest_path)

    assert len(results) == 1
    assert results[0].passed
    assert results[0].observed_score == 0.30
    assert results[0].observed_risk_class == "Moderate"
    assert benchmark_summary(results) == {
        "n_comparisons": 1,
        "n_passed": 1,
        "n_failed": 0,
        "pass_rate": 1.0,
    }


def test_benchmark_rejects_published_summary_as_full_numerical_reference(tmp_path: Path):
    repo_root = Path(__file__).parents[1]
    config_path = repo_root / "virelion_cardioscore" / "config" / "default.yaml"
    features_path = tmp_path / "placeholder.csv"
    pd.DataFrame().to_csv(features_path, index=False)
    manifest = {
        "datasets": [
            {
                "name": "published_only",
                "features": str(features_path),
                "config": str(config_path),
                "evidence_level": "published_mea_summary",
                "expected": [{"compound": "Known", "cardioscore": 0.4, "risk_class": "Moderate"}],
            }
        ]
    }
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(yaml.safe_dump(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="full numerical benchmark"):
        run_benchmark_manifest(manifest_path)


def test_benchmark_summary_reports_failures():
    class Dummy:
        passed = False

    summary = benchmark_summary([Dummy(), Dummy()])
    assert summary["n_comparisons"] == 2
    assert summary["n_failed"] == 2
    assert summary["pass_rate"] == 0.0

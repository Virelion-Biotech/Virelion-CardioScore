from __future__ import annotations

import json

import pandas as pd

from virelion_cardioscore.analysis.hierarchical_pipeline import HierarchicalCardioScorePipeline


def _dataset() -> pd.DataFrame:
    rows = []
    for plate, baseline in [("P1", 100.0), ("P2", 120.0), ("P3", 110.0)]:
        for concentration in [1.0, 10.0]:
            for replicate in range(2):
                rows.append(
                    {
                        "plate_id": plate,
                        "compound": "A",
                        "concentration_uM": concentration,
                        "vehicle": True,
                        "well": f"{plate}_V_{concentration}_{replicate}",
                        "fpd_ms": baseline,
                        "beat_rate_bpm": 60.0,
                        "amplitude_uv": 100.0,
                        "stv": 0.1,
                        "triangulation_proxy": 0.1,
                        "n_electrodes": 4,
                        "noise_sd_uv": 5.0,
                        "beat_detection_rate": 0.95,
                    }
                )
                rows.append(
                    {
                        "plate_id": plate,
                        "compound": "A",
                        "concentration_uM": concentration,
                        "vehicle": False,
                        "well": f"{plate}_T_{concentration}_{replicate}",
                        "fpd_ms": baseline + (10.0 if concentration == 1.0 else 25.0),
                        "beat_rate_bpm": 60.0,
                        "amplitude_uv": 100.0,
                        "stv": 0.1,
                        "triangulation_proxy": 0.1,
                        "n_electrodes": 4,
                        "noise_sd_uv": 5.0,
                        "beat_detection_rate": 0.95,
                    }
                )
    return pd.DataFrame(rows)


def test_hierarchical_pipeline_is_off_by_default(tmp_path):
    pipeline = HierarchicalCardioScorePipeline.from_defaults()
    result = pipeline.run(_dataset())
    assert result.mixed_effects_table.empty


def test_hierarchical_pipeline_keeps_concentrations_separate(tmp_path):
    pipeline = HierarchicalCardioScorePipeline.from_defaults()
    pipeline.config["mixed_effects"]["enabled"] = True
    pipeline.config["mixed_effects"]["group_column"] = "plate_id"
    pipeline.config["mixed_effects"]["endpoints"] = ["fpd_ms"]

    result = pipeline.run(_dataset())
    assert not result.mixed_effects_table.empty
    ok = result.mixed_effects_table[result.mixed_effects_table["status"] != "not_estimable"]
    assert set(ok["concentration_uM"]) == {1.0, 10.0}
    assert set(ok["endpoint"]) == {"fpd_ms"}


def test_hierarchical_pipeline_json_export(tmp_path):
    pipeline = HierarchicalCardioScorePipeline.from_defaults()
    pipeline.config["mixed_effects"]["enabled"] = True
    pipeline.config["mixed_effects"]["group_column"] = "plate_id"
    pipeline.config["mixed_effects"]["endpoints"] = ["fpd_ms"]

    result = pipeline.run(_dataset())
    output = tmp_path / "result.json"
    result.to_json(output)
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert "mixed_effects" in payload
    assert payload["mixed_effects"]

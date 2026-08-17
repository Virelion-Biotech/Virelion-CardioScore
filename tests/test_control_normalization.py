from __future__ import annotations

import pandas as pd
import pytest

from virelion_cardioscore.analysis.pipeline import CardioScorePipeline


def _shared_vehicle_frame() -> pd.DataFrame:
    rows = [
        {
            "compound": "A", "well": "A01", "concentration_uM": 1.0,
            "vehicle": False, "fpd_ms": 120.0, "beat_rate_bpm": 60.0,
            "amplitude_uv": 100.0, "stv": 0.1, "triangulation_proxy": 0.1,
            "n_electrodes": 4, "noise_sd_uv": 5.0, "beat_detection_rate": 0.95,
            "plate_id": "P1",
        },
        {
            "compound": "B", "well": "B01", "concentration_uM": 1.0,
            "vehicle": False, "fpd_ms": 130.0, "beat_rate_bpm": 60.0,
            "amplitude_uv": 100.0, "stv": 0.1, "triangulation_proxy": 0.1,
            "n_electrodes": 4, "noise_sd_uv": 5.0, "beat_detection_rate": 0.95,
            "plate_id": "P1",
        },
        {
            "compound": "Control", "well": "V01", "concentration_uM": 0.0,
            "vehicle": True, "fpd_ms": 100.0, "beat_rate_bpm": 60.0,
            "amplitude_uv": 100.0, "stv": 0.1, "triangulation_proxy": 0.1,
            "n_electrodes": 4, "noise_sd_uv": 5.0, "beat_detection_rate": 0.95,
            "plate_id": "P1",
        },
    ]
    return pd.DataFrame(rows)


def test_plate_scope_uses_shared_vehicle_control_across_compounds():
    pipeline = CardioScorePipeline.from_defaults()
    pipeline.config["control_normalization"]["scope"] = "plate"

    effects = pipeline.compute_effects(_shared_vehicle_frame())

    assert set(effects["compound"]) == {"A", "B"}
    assert effects.loc[effects["compound"] == "A", "fpd_change_pct"].iloc[0] == pytest.approx(20.0)
    assert effects.loc[effects["compound"] == "B", "fpd_change_pct"].iloc[0] == pytest.approx(30.0)


def test_compound_scope_preserves_legacy_missing_control_behavior():
    frame = _shared_vehicle_frame()
    frame.loc[frame["compound"] == "Control", "compound"] = "C"

    pipeline = CardioScorePipeline.from_defaults()
    effects = pipeline.compute_effects(frame)

    assert effects.empty
    assert any("No matching vehicle control" in msg for msg in pipeline.qc_log)


def test_missing_control_metadata_is_rejected_for_plate_scope():
    pipeline = CardioScorePipeline.from_defaults()
    pipeline.config["control_normalization"]["scope"] = "plate"

    with pytest.raises(ValueError, match="requires metadata column"):
        pipeline.compute_effects(_shared_vehicle_frame().drop(columns=["plate_id"]))


def test_invalid_control_scope_is_rejected():
    pipeline = CardioScorePipeline.from_defaults()
    pipeline.config["control_normalization"]["scope"] = "unknown"

    with pytest.raises(ValueError, match="Unsupported control_normalization.scope"):
        pipeline.compute_effects(_shared_vehicle_frame())

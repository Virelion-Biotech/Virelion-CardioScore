from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from virelion_cardioscore.analysis.pipeline import CardioScorePipeline
from virelion_cardioscore.analysis.variability import (
    control_variability,
    standardized_treatment_separation,
)


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "compound": ["A", "A", "A", "A", "B", "B", "B", "B"],
            "vehicle": [True, True, False, False, True, True, False, False],
            "plate_id": ["P1", "P1", "P1", "P1", "P2", "P2", "P2", "P2"],
            "fpd_ms": [100.0, 102.0, 130.0, 135.0, 120.0, 122.0, 150.0, 152.0],
            "beat_rate_bpm": [60.0, 61.0, 60.0, 61.0, 60.0, 61.0, 62.0, 63.0],
        }
    )


def test_control_variability_reports_stable_controls():
    result = control_variability(
        _frame(), endpoint_columns=["fpd_ms"], max_control_cv_pct=20.0
    )

    row = result.iloc[0]
    assert row["n_groups"] == 2
    assert row["n_controls"] == 4
    assert row["status"] == "stable"
    assert row["between_group_sd"] == pytest.approx(np.sqrt(200.0))


def test_control_variability_flags_high_group_variability():
    frame = _frame().copy()
    frame.loc[frame["plate_id"] == "P2", "fpd_ms"] = [200.0, 210.0, 150.0, 152.0]

    result = control_variability(
        frame, endpoint_columns=["fpd_ms"], max_control_cv_pct=10.0
    )

    assert result.iloc[0]["status"] == "high_variability"
    assert result.iloc[0]["control_cv_pct"] > 10.0


def test_control_variability_requires_group_metadata():
    with pytest.raises(ValueError, match="requires one of"):
        control_variability(_frame().drop(columns=["plate_id"]))


def test_control_variability_requires_two_groups_for_between_group_estimate():
    frame = _frame().query("plate_id == 'P1'")
    result = control_variability(frame, endpoint_columns=["fpd_ms"])

    assert result.iloc[0]["status"] == "insufficient_groups"
    assert pd.isna(result.iloc[0]["between_group_sd"])


def test_standardized_treatment_separation_is_group_specific():
    frame = _frame().copy()
    # Vehicle wells at concentration 0, treated wells at a nonzero dose --
    # matching real feature-table data (io.raw_trace / io.synthetic always
    # populate concentration_uM this way).
    frame["concentration_uM"] = [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0]

    result = standardized_treatment_separation(
        frame, endpoint="fpd_ms", group_column="plate_id"
    )

    assert len(result) == 2
    assert (result["n_controls"] == 2).all()
    assert (result["n_treated"] == 2).all()
    assert (result["standardized_separation"] > 0).all()


def test_pipeline_exposes_variability_diagnostics_without_scoring_change(tmp_path):
    frame = _frame().copy()
    frame["n_electrodes"] = 4
    frame["noise_sd_uv"] = 5.0
    frame["beat_detection_rate"] = 0.95
    frame["amplitude_uv"] = 100.0
    frame["stv"] = 0.1
    frame["triangulation_proxy"] = 0.1
    frame["concentration_uM"] = [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0]
    frame["well"] = [f"W{i}" for i in range(len(frame))]

    pipeline = CardioScorePipeline.from_defaults()
    pipeline.config["variability"]["enabled"] = True
    pipeline.config["variability"]["group_column"] = "plate_id"
    pipeline.config["variability"]["max_control_cv_pct"] = 20.0

    result = pipeline.run(frame)

    assert not result.variability_table.empty
    assert not result.separation_table.empty
    assert {"status", "control_cv_pct", "between_group_sd"}.issubset(result.variability_table.columns)
    assert len(result.scores) > 0
    # Guards the standardized_treatment_separation bug where grouping by
    # concentration_uM silo'd vehicle (conc=0) and treated (conc=1) wells
    # apart, making a real control-vs-treated comparison impossible.
    assert (result.separation_table["n_controls"] > 0).all()
    assert (result.separation_table["n_treated"] > 0).all()

    output = tmp_path / "result.json"
    result.to_json(output)
    payload = json.loads(output.read_text())
    assert "variability" in payload
    assert "treatment_separation" in payload
    assert len(payload["variability"]) == len(result.variability_table)

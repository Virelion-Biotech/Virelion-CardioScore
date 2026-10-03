"""Informative-dropout accounting and no-zero-fill regression tests.

A compound whose cells stop beating at high concentration must never look cleaner because QC
removed its worst wells, and an endpoint that cannot be computed must never be scored as "no effect".
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from virelion_cardioscore.analysis.pipeline import CardioScorePipeline

CONCENTRATIONS = [0.1, 1.0, 10.0, 100.0]
EFFECT_BY_CONCENTRATION = [0.02, 0.10, 0.60, 1.50]  # strongly proarrhythmic at high concentration
ENDPOINTS = ["fpd_ms", "beat_rate_bpm", "amplitude_uv", "stv", "triangulation_proxy"]


def make_frame(
    n_dead_top_concentrations: int = 0,
    *,
    compound: str = "DRUG",
    vehicle_dead: int = 0,
    treated_per_concentration: int = 4,
    dead_mode: str = "collapse",
) -> pd.DataFrame:
    """Vehicle plus four treated concentrations; the top N concentrations optionally lose signal.

    dead_mode="collapse": endpoints NaN, no electrodes, no beat detection (cells stopped beating).
    dead_mode="noise": wells stay measurable but are too noisy (a purely technical failure).
    """
    rng = np.random.default_rng(0)
    rows = []
    for i in range(6):
        rows.append(
            {
                "compound": compound,
                "well": f"V{i}",
                "concentration_uM": 0.0,
                "vehicle": True,
                "site": "S1",
                "cell_type": "C",
                "concentration_index": 0,
                "fpd_ms": 400 + rng.normal(0, 5),
                "beat_rate_bpm": 60 + rng.normal(0, 1),
                "amplitude_uv": 200 + rng.normal(0, 3),
                "stv": 5 + rng.normal(0, 0.2),
                "triangulation_proxy": 1.0 + rng.normal(0, 0.01),
                "n_electrodes": 12,
                "noise_sd_uv": 8.0,
                "beat_detection_rate": 0.95,
            }
        )
    for index, concentration in enumerate(CONCENTRATIONS):
        effect = EFFECT_BY_CONCENTRATION[index]
        for replicate in range(treated_per_concentration):
            rows.append(
                {
                    "compound": compound,
                    "well": f"T{index}{replicate}",
                    "concentration_uM": concentration,
                    "vehicle": False,
                    "site": "S1",
                    "cell_type": "C",
                    "concentration_index": index + 1,
                    "fpd_ms": 400 * (1 + effect) + rng.normal(0, 5),
                    "beat_rate_bpm": 60 + rng.normal(0, 1),
                    "amplitude_uv": 200 * (1 - 0.2 * effect),
                    "stv": 5 * (1 + effect),
                    "triangulation_proxy": 1.0 * (1 + effect),
                    "n_electrodes": 12,
                    "noise_sd_uv": 8.0,
                    "beat_detection_rate": 0.95,
                }
            )
    frame = pd.DataFrame(rows)
    dead_concentrations = CONCENTRATIONS[len(CONCENTRATIONS) - n_dead_top_concentrations :]
    dead = frame["concentration_uM"].isin(dead_concentrations) & ~frame["vehicle"]
    if dead_mode == "collapse":
        frame.loc[dead, ENDPOINTS] = np.nan
        frame.loc[dead, "n_electrodes"] = 0
        frame.loc[dead, "beat_detection_rate"] = 0.0
    else:
        frame.loc[dead, "noise_sd_uv"] = 99.0
    if vehicle_dead:
        vehicle_index = frame.index[frame["vehicle"]][:vehicle_dead]
        frame.loc[vehicle_index, ENDPOINTS] = np.nan
        frame.loc[vehicle_index, "n_electrodes"] = 0
        frame.loc[vehicle_index, "beat_detection_rate"] = 0.0
    return frame


def run(frame: pd.DataFrame):
    return CardioScorePipeline.from_defaults().run(frame)


def test_clean_data_reports_no_dropout():
    result = run(make_frame())
    row = result.summary_table.iloc[0]
    assert row["risk_class"] == "High"
    assert not bool(row["informative_dropout"]) and row["dropout_concentrations_uM"] == ""
    assert result.exclusion_table.empty and result.qc_rejections.empty
    assert not result.dropout_table["informative_dropout"].any()


def test_top_concentration_collapse_is_flagged_not_silent():
    # Regression: the same proarrhythmic compound dropped from High to Low with only a log line.
    result = run(make_frame(1))
    row = result.summary_table.iloc[0]
    assert bool(row["informative_dropout"])
    assert row["dropout_concentrations_uM"] == "100"
    assert row["n_wells_rejected_signal_loss"] == 4 and row["max_dropout_fraction"] == 1.0
    assert (
        row["concentrations_tested"] == 3
    )  # surviving concentrations only, and now reported as such
    top = result.dropout_table[result.dropout_table["concentration_uM"] == 100.0].iloc[0]
    assert top["n_wells_input"] == 4 and top["n_signal_loss"] == 4 and top["n_kept"] == 0
    assert any("informative dropout" in line for line in result.qc_log)


def test_compound_that_loses_too_many_concentrations_stays_visible():
    result = run(make_frame(2))
    assert result.summary_table.empty
    row = result.exclusion_table.iloc[0]
    assert row["compound"] == "DRUG" and row["reason"] == "insufficient_concentrations"
    assert "informative dropout at 10;100 uM" in row["detail"]
    assert result.provenance["excluded_compounds"] == ["DRUG"]


def test_compound_with_every_treated_well_rejected_is_reported():
    frame = pd.concat(
        [make_frame(0, compound="STABLE"), make_frame(4, compound="DEAD")], ignore_index=True
    )
    result = run(frame)
    assert list(result.summary_table["compound"]) == ["STABLE"]
    row = result.exclusion_table.set_index("compound").loc["DEAD"]
    assert row["reason"] == "all_treated_wells_rejected_by_qc"
    assert "informative dropout at 0.1;1;10;100 uM" in row["detail"]


def test_dataset_where_no_treated_well_survives_reports_instead_of_crashing():
    # Regression: this used to raise a bare KeyError('compound').
    result = run(make_frame(4))
    assert result.summary_table.empty
    assert result.exclusion_table.iloc[0]["reason"] == "all_treated_wells_rejected_by_qc"


def test_rejections_carry_reason_codes():
    result = run(make_frame(1))
    rejections = result.qc_rejections
    assert len(rejections) == 4 and rejections["signal_loss"].all()
    assert all(
        "low_beat_detection" in reasons and "too_few_electrodes" in reasons
        for reasons in rejections["reasons"]
    )
    assert set(rejections["compound"]) == {"DRUG"} and not rejections["vehicle"].any()


def test_technical_noise_rejections_are_not_signal_loss():
    result = run(make_frame(1, dead_mode="noise"))
    assert set(result.qc_rejections["reasons"]) == {"high_noise"}
    assert not result.qc_rejections["signal_loss"].any()
    assert not result.dropout_table["informative_dropout"].any()
    assert not bool(result.summary_table.iloc[0]["informative_dropout"])


def test_dropout_shared_with_vehicle_wells_is_not_informative():
    # Half of the vehicle wells fail as well, and half of the top-concentration wells fail: no excess.
    frame = make_frame(vehicle_dead=3, treated_per_concentration=4)
    top = frame.index[(frame["concentration_uM"] == 100.0) & ~frame["vehicle"]][:2]
    frame.loc[top, ENDPOINTS] = np.nan
    frame.loc[top, "n_electrodes"] = 0
    frame.loc[top, "beat_detection_rate"] = 0.0
    result = run(frame)
    top_row = result.dropout_table[result.dropout_table["concentration_uM"] == 100.0].iloc[0]
    assert top_row["dropout_fraction"] == 0.5 and top_row["vehicle_dropout_fraction"] == 0.5
    assert not bool(top_row["informative_dropout"])


def test_dropout_thresholds_are_configurable_and_validated():
    pipeline = CardioScorePipeline.from_defaults()
    pipeline.config["quality_control"]["informative_dropout_min_fraction"] = 1.01
    with pytest.raises(Exception, match="informative_dropout_min_fraction"):
        CardioScorePipeline(pipeline.config)
    strict = CardioScorePipeline.from_defaults()
    strict.config["quality_control"]["informative_dropout_min_fraction"] = 1.0
    strict.config["quality_control"]["informative_dropout_min_excess"] = 1.0
    frame = make_frame(1)
    top = frame.index[(frame["concentration_uM"] == 100.0) & ~frame["vehicle"]][:3]  # 3 of 4 lost
    frame = make_frame(0)
    frame.loc[top, ENDPOINTS] = np.nan
    frame.loc[top, "n_electrodes"] = 0
    frame.loc[top, "beat_detection_rate"] = 0.0
    assert not strict.run(frame).dropout_table["informative_dropout"].any()
    assert (
        run(frame).dropout_table["informative_dropout"].any()
    )  # default rule: 0.75 >= 0.5 and excess


def test_aggregation_never_zero_fills_a_missing_endpoint():
    # Regression: an endpoint with no finite values used to aggregate to 0.0 ("no effect").
    concentration_summary = pd.DataFrame(
        {
            "compound": ["X"] * 3,
            "concentration_uM": [1.0, 3.0, 10.0],
            "fpd_change_pct_mean": [5.0, 20.0, 40.0],
            "beat_rate_change_pct_mean": [1.0, 2.0, 3.0],
            "amplitude_change_pct_mean": [0.0, -5.0, -10.0],
            "stv_increase_mean": [np.nan] * 3,
            "triangulation_proxy_change_mean": [0.0, 0.1, 0.3],
            "max_effect_pct_mean": [5.0, 20.0, 40.0],
            "n_replicates": [3] * 3,
        }
    )
    for mode in ("mean_harmful_effect", "max_absolute_effect"):
        out = CardioScorePipeline.aggregate_compound_effects(
            concentration_summary, concentration_aggregation=mode
        )
        assert np.isnan(out["stv_increase"].iloc[0])
        assert np.isfinite(out["fpd_change_pct"].iloc[0])


def test_uncomputable_endpoint_stops_the_run_loudly():
    frame = make_frame()
    # Vehicle FPD of exactly 0 makes the FPD effect undefined for every treated well of this compound.
    frame.loc[frame["vehicle"], "fpd_ms"] = 0.0
    with pytest.raises(ValueError, match="partial endpoint data are not allowed"):
        run(frame)


def test_multiple_compounds_only_the_dropout_one_is_flagged():
    frame = pd.concat(
        [make_frame(0, compound="STABLE"), make_frame(1, compound="COLLAPSING")], ignore_index=True
    )
    summary = run(frame).summary_table.set_index("compound")
    assert not bool(summary.loc["STABLE", "informative_dropout"])
    assert bool(summary.loc["COLLAPSING", "informative_dropout"])


def test_json_export_contains_dropout_and_exclusions(tmp_path):
    result = run(make_frame(2))
    path = tmp_path / "result.json"
    result.to_json(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["exclusions"][0]["reason"] == "insufficient_concentrations"
    assert any(row["informative_dropout"] for row in payload["dropout"])
    assert len(payload["qc_rejections"]) == 8
"""Unified Colab runner integration tests for strict MC_Data / Cardio PyMEA intake."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

RUNNER_PATH = Path(__file__).resolve().parents[1] / "scripts" / "validation" / "run_colab_validation.py"
_spec = importlib.util.spec_from_file_location("run_colab_validation_mcs", RUNNER_PATH)
runner = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(runner)


def write_mcs(path: Path, *, n: int = 3000) -> None:
    rng = np.random.default_rng(3)
    electrodes = ["El F7", "El F8", "El G7"]
    pad = lambda text: f"{text:<12}"  # noqa: E731
    lines = [
        "MC_DataTool ASCII conversion",
        "",
        "\t".join(pad(name) for name in ["t", *electrodes]),
        "\t".join(pad(unit) for unit in ["[ms]", *["[µV]"] * len(electrodes)]),
    ]
    for i in range(n):
        values = [pad(f"{value:.3f}") for value in rng.normal(0, 5, len(electrodes))]
        lines.append("\t".join([pad(f"{i:.3f}"), *values]))
    path.write_bytes(("\r\n".join(lines) + "\r\n").encode("latin-1"))


def test_mcs_selection_is_explicit_and_validated():
    assert runner.mcs_selection_from_env({}) == ([], 0.0, 30.0)
    env = {
        "CARDIOSCORE_MCS_ELECTRODES": "El F7, El F8",
        "CARDIOSCORE_MCS_START_S": "1.5",
        "CARDIOSCORE_MCS_DURATION_S": "20",
    }
    assert runner.mcs_selection_from_env(env) == (["El F7", "El F8"], 1.5, 20.0)
    with pytest.raises(ValueError, match="START_S"):
        runner.mcs_selection_from_env({"CARDIOSCORE_MCS_START_S": "-1"})
    with pytest.raises(ValueError, match="DURATION_S"):
        runner.mcs_selection_from_env({"CARDIOSCORE_MCS_DURATION_S": "0"})


def test_mcs_discovery_ignores_unrelated_text(tmp_path):
    recording = tmp_path / "Day9.txt"
    readme = tmp_path / "README.txt"
    write_mcs(recording)
    readme.write_text("Cardio PyMEA release notes only\n", encoding="utf-8")
    matches = runner.find_mcs_ascii_sources([readme, recording])
    assert len(matches) == 1
    assert matches[0][0] == recording
    assert matches[0][1].electrodes == ("El F7", "El F8", "El G7")


def test_mcs_stage_blocks_without_inventing_electrodes(tmp_path):
    recording = tmp_path / "Day9.txt"
    write_mcs(recording)
    derived = tmp_path / "derived"
    results = tmp_path / "results"
    derived.mkdir()
    results.mkdir()
    state = runner.run_mcs_ascii_signal(
        recording,
        derived,
        results,
        electrodes=[],
        start_s=0.0,
        duration_s=1.0,
    )
    assert state["status"] == "blocked"
    assert state["available_electrodes"] == ["El F7", "El F8", "El G7"]
    assert not (results / "raw_mea_lineage.json").exists()


def test_mcs_stage_is_deterministic_and_signal_only(tmp_path, monkeypatch):
    recording = tmp_path / "Day9.txt"
    write_mcs(recording)
    derived = tmp_path / "derived"
    results = tmp_path / "results"
    derived.mkdir()
    results.mkdir()
    monkeypatch.setattr(runner, "PIN", "a" * 40)

    state = runner.run_mcs_ascii_signal(
        recording,
        derived,
        results,
        electrodes=["El F8", "El F7"],
        start_s=0.5,
        duration_s=1.0,
    )
    assert state["status"] == "complete"
    assert state["scope"] == "signal_component_only"
    assert state["verified_rebuild"] is True

    lineage = json.loads((results / "raw_mea_lineage.json").read_text(encoding="utf-8"))
    assert lineage["verified_rebuild"] is True
    assert lineage["output_sha256"] == lineage["rebuild_sha256"]
    assert lineage["selection"]["electrodes"] == ["El F8", "El F7"]
    assert lineage["drug_response_metadata_invented"] is False
    assert lineage["eligible_for_locked_drug_risk_scoring"] is False

    out = derived / "mcs_signal_window"
    assert np.load(out / "time_s.npy").shape == (1000,)
    assert np.load(out / "voltage_uv.npy").shape == (1000, 2)

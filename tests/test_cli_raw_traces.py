"""Tests for the `cardioscore run --raw-traces` CLI path."""

from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from virelion_cardioscore.cli import main


def _default_config_path() -> Path:
    return Path(__file__).resolve().parent.parent / "virelion_cardioscore" / "config" / "default.yaml"


def test_cli_run_with_raw_traces(two_compound_plate, tmp_path):
    runner = CliRunner()
    out_dir = tmp_path / "out"
    result = runner.invoke(
        main,
        [
            "run",
            "--config", str(_default_config_path()),
            "--raw-traces", str(two_compound_plate),
            "--output-dir", str(out_dir),
        ],
    )
    assert result.exit_code == 0, result.output
    assert (out_dir / "summary.csv").exists()
    assert (out_dir / "scores.json").exists()
    assert (out_dir / "report.html").exists()
    assert "Compound_Toxic" in result.output


def test_cli_run_rejects_both_features_and_raw_traces(two_compound_plate, tmp_path):
    runner = CliRunner()
    dummy_features = tmp_path / "features.csv"
    dummy_features.write_text("compound,concentration_uM,well,vehicle\n")

    result = runner.invoke(
        main,
        [
            "run",
            "--config", str(_default_config_path()),
            "--raw-traces", str(two_compound_plate),
            "--features", str(dummy_features),
            "--output-dir", str(tmp_path / "out"),
        ],
    )
    assert result.exit_code != 0
    assert "only one of" in result.output.lower()


def test_cli_run_with_malformed_raw_traces_shows_clean_error(tmp_path):
    bad_csv = tmp_path / "bad.csv"
    bad_csv.write_text("compound,well\nA,W01\n")

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "run",
            "--config", str(_default_config_path()),
            "--raw-traces", str(bad_csv),
            "--output-dir", str(tmp_path / "out"),
        ],
    )
    assert result.exit_code != 0
    assert "missing required column" in result.output.lower()
    # Should be a clean ClickException, not a raw traceback dumped on the user.
    assert "Traceback" not in result.output

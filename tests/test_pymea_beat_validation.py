"""Blinded annotation sheet + scoring, exercised end to end on a synthetic file in the real layout."""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from virelion_cardioscore.validation.signal_truth import make_truth_recording

_path = Path(__file__).resolve().parents[1] / "scripts" / "validation" / "pymea_beat_validation.py"
_spec = importlib.util.spec_from_file_location("pymea_beat_validation", _path)
tool = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tool)

DURATION_S = 90
ELECTRODES = ["El A1", "El A2", "El B1", "El B2"]


@pytest.fixture(scope="module")
def recording(tmp_path_factory):
    """Two electrodes carry beats (~1.2 s apart), two are pure noise; written in the MC_DataTool layout."""
    rng = np.random.default_rng(0)
    truths = {
        "El A1": make_truth_recording(
            11, duration_s=DURATION_S, amplitude_uv=800.0, noise_sd_uv=3.0, bpm=50.0
        ),
        "El B1": make_truth_recording(
            12, duration_s=DURATION_S, amplitude_uv=600.0, noise_sd_uv=3.0, bpm=50.0
        ),
    }
    columns = []
    for name in ELECTRODES:
        columns.append(
            truths[name].trace_uv if name in truths else rng.normal(0, 3.0, DURATION_S * 1000)
        )
    pad = lambda text: f"{text:<12}"  # noqa: E731
    lines = [
        "MC_DataTool ASCII conversion",
        "",
        "\t".join(pad(n) for n in ["t", *ELECTRODES]),
        "\t".join(pad(u) for u in ["[ms]", *["[µV]"] * 4]),
    ]
    for i in range(DURATION_S * 1000):
        lines.append("\t".join([pad(f"{i:.3f}"), *(pad(f"{col[i]:.2f}") for col in columns)]))
    directory = tmp_path_factory.mktemp("pymea")
    source = directory / "Day9.txt"
    source.write_bytes(("\r\n".join(lines) + "\r\n").encode("latin-1"))
    return {"source": source, "truths": truths, "dir": directory}


def make(recording, name="sheet", seed=7):
    out = recording["dir"] / name
    manifest = tool.make_sheet(
        recording["source"], out, n_windows=8, window_s=10.0, seed=seed, duration_s=None
    )
    return out, manifest


def annotate_from_truth(recording, out, manifest, path, *, drop=0):
    rows = []
    for window in manifest["windows"]:
        truth = recording["truths"].get(window["electrode"])
        beats = []
        if truth is not None:
            beats = [
                round(t - window["start_s"], 3)
                for t in truth.beat_times_s
                if window["start_s"] <= t < window["start_s"] + 10.0
            ]
        rows.append(
            {
                "window_id": window["window_id"],
                "png_file": "",
                "window_s": 10,
                "done": 1,
                "unsure": 0,
                "beat_times_s": ";".join(map(str, beats[drop:])),
                "annotator": "A",
                "notes": "",
            }
        )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tool.SHEET_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_sheet_is_deterministic_blinded_and_complete(recording):
    out, manifest = make(recording, "s1", seed=7)
    _, again = make(recording, "s2", seed=7)
    assert [(w["window_id"], w["electrode"], w["start_s"]) for w in manifest["windows"]] == [
        (w["window_id"], w["electrode"], w["start_s"]) for w in again["windows"]
    ]
    assert (
        len(list((out / "plots").glob("*.png"))) == 8
        and len(list((out / "windows").glob("*.npy"))) == 8
    )
    rows = list(csv.DictReader((out / "annotation_sheet.csv").open(encoding="utf-8")))
    assert all(
        row["beat_times_s"] == "" and row["done"] == "0" for row in rows
    )  # empty: nothing pre-filled
    assert manifest["source_sha256"] == tool._sha256(
        recording["source"]
    )  # original, unmodified source
    assert make(recording, "s3", seed=8)[1]["windows"] != manifest["windows"]


def test_perfect_annotations_score_near_perfect(recording):
    out, manifest = make(recording, "s4")
    annotations = annotate_from_truth(recording, out, manifest, out / "A.csv")
    results = tool.score(out, [annotations], 0.05, None, tool.FilterConfig())
    scored = results["vs_annotator"]["A.csv"]
    assert (
        scored["recall"] == 1.0
        and scored["f1"] >= 0.95
        and scored["timing_median_abs_error_ms"] < 5
    )


def test_missed_annotations_and_noise_windows_are_scored_honestly(recording):
    out, manifest = make(recording, "s5")
    annotations = annotate_from_truth(
        recording, out, manifest, out / "A.csv", drop=1
    )  # annotator skips first beat per window
    scored = tool.score(out, [annotations], 0.05, None, tool.FilterConfig())["vs_annotator"][
        "A.csv"
    ]
    assert (
        scored["precision"] < 1.0 and scored["fp"] > 0
    )  # detector beats the annotator never marked count as FP


def test_inter_annotator_agreement_and_criteria(recording):
    out, manifest = make(recording, "s6")
    first = annotate_from_truth(recording, out, manifest, out / "A.csv")
    second = annotate_from_truth(recording, out, manifest, out / "B.csv")
    criteria = out / "criteria.yaml"
    criteria.write_text(
        "beat_detection:\n  status: draft\n  f1_min: 0.95\n  precision_min: 0.95\n  recall_min: 0.95\n  timing_median_abs_error_ms_max: 10\n",
        encoding="utf-8",
    )
    results = tool.score(out, [first, second], 0.05, criteria, tool.FilterConfig())
    assert results["inter_annotator"]["f1"] == 1.0
    assert results["criteria"]["passed"] is True and results["criteria"]["status"] == "draft"


def test_unfinished_sheets_and_tampered_windows_are_rejected(recording):
    out, manifest = make(recording, "s7")
    with pytest.raises(ValueError, match="not marked done=1"):
        tool.score(out, [out / "annotation_sheet.csv"], 0.05, None, tool.FilterConfig())
    good = annotate_from_truth(recording, out, manifest, out / "A.csv")
    np.save(out / "windows" / "w00.npy", np.zeros((10, 2)))
    with pytest.raises(ValueError, match="changed since the sheet was made"):
        tool.score(out, [good], 0.05, None, tool.FilterConfig())


def test_cli_round_trip(recording, capsys):
    out = recording["dir"] / "cli"
    assert (
        tool.main(
            [
                "sheet",
                str(recording["source"]),
                "--out-dir",
                str(out),
                "--n-windows",
                "4",
                "--window-s",
                "10",
            ]
        )
        == 0
    )
    manifest = json.loads((out / "sheet_manifest.json").read_text(encoding="utf-8"))
    annotate_from_truth(recording, out, manifest, out / "A.csv")
    assert (
        tool.main(
            [
                "score",
                "--sheet-dir",
                str(out),
                "--annotations",
                str(out / "A.csv"),
                "--out",
                str(out / "results.json"),
            ]
        )
        == 0
    )
    assert "vs_annotator" in json.loads((out / "results.json").read_text(encoding="utf-8"))

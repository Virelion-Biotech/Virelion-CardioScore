"""MC_Data ASCII reader and the Cardio PyMEA adapter."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from virelion_cardioscore.io.mcs_ascii import SourceFormatError, read_window, sniff_layout

_path = Path(__file__).resolve().parents[1] / "scripts" / "validation" / "adapt_cardio_pymea.py"
_spec = importlib.util.spec_from_file_location("adapt_cardio_pymea", _path)
adapter = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(adapter)

ELECTRODES = ["F7", "F8", "G7"]


def write_recording(
    path: Path,
    *,
    fs_hz: float = 1000.0,
    n: int = 3000,
    header: bool = True,
    names=None,
    bad_time=False,
):
    step_ms = 1000.0 / fs_hz
    rng = np.random.default_rng(0)
    lines = ["MC_Data export", "Channel units: uV"] if header else []
    if header:
        lines.append("\t".join(["Time [ms]", *(names or ELECTRODES)]))
    for i in range(n):
        t = i * step_ms if not (bad_time and i == 400) else 399 * step_ms
        lines.append(
            "\t".join([f"{t:.3f}", *(f"{v:.3f}" for v in rng.normal(0, 5, len(ELECTRODES)))])
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_layout_is_found_from_the_data_not_assumed(tmp_path):
    source = tmp_path / "rec.txt"
    write_recording(source)
    layout = sniff_layout(source)
    assert layout.delimiter == "\t" and layout.header_lines == 3
    assert layout.columns == ("Time [ms]", *ELECTRODES) and layout.electrodes == tuple(ELECTRODES)
    assert layout.fs_hz == pytest.approx(1000.0)


def test_read_window_returns_seconds_and_selected_electrodes(tmp_path):
    source = tmp_path / "rec.txt"
    write_recording(source)
    layout = sniff_layout(source)
    time_s, voltage = read_window(
        source, layout, electrodes=["G7", "F7"], start_s=0.5, duration_s=1.0, chunk_rows=700
    )
    assert (
        time_s[0] == pytest.approx(0.5)
        and time_s[-1] == pytest.approx(1.499)
        and voltage.shape == (1000, 2)
    )
    full_time, full = read_window(
        source, layout, electrodes=ELECTRODES, start_s=0.0, duration_s=3.0
    )
    assert np.array_equal(voltage[:, 0], full[500:1500, 2]) and np.array_equal(
        voltage[:, 1], full[500:1500, 0]
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"header": False}, "Column names"),
        ({"names": ["F7"]}, "do not line up"),
        ({"names": ["F7", "F7", "F8"]}, "Duplicate"),
        ({"fs_hz": 5.0}, "implausible"),
    ],
)
def test_format_mismatches_fail_loudly(tmp_path, kwargs, message):
    source = tmp_path / "rec.txt"
    write_recording(source, **kwargs)
    with pytest.raises(SourceFormatError, match=message):
        sniff_layout(source)


def test_timestamp_problems_and_unknown_electrodes_are_rejected(tmp_path):
    source = tmp_path / "rec.txt"
    write_recording(source, bad_time=True)
    layout = sniff_layout(source)
    with pytest.raises(SourceFormatError, match="strictly increasing"):
        read_window(source, layout, electrodes=["F7"], start_s=0.0, duration_s=1.0)
    with pytest.raises(SourceFormatError, match="Unknown"):
        read_window(source, layout, electrodes=["Z9"], start_s=0.0, duration_s=1.0)


def test_adapter_records_hashes_and_a_verified_rebuild(tmp_path):
    source = tmp_path / "rec.txt"
    write_recording(source)
    out = tmp_path / "out"
    code = adapter.main(
        [
            str(source),
            "--electrodes",
            "F7",
            "F8",
            "--start-s",
            "0",
            "--duration-s",
            "2",
            "--out-dir",
            str(out),
        ]
    )
    assert code == 0
    lineage = json.loads((out / "lineage.json").read_text(encoding="utf-8"))
    assert (
        lineage["verified_rebuild"] is True
        and lineage["output_sha256"] == lineage["rebuild_sha256"]
    )
    assert lineage["source_sha256"] == adapter.file_sha256(source) and lineage["n_samples"] == 2000
    assert np.load(out / "voltage_uv.npy").shape == (2000, 2)
    assert "compound" not in json.dumps(
        lineage
    )  # untreated data: no treatment metadata is invented


# --- the real Cardio PyMEA layout: title, blank line, names row, units row, fixed-width latin-1 ------

REAL_ELECTRODES = ["El F7", "El F8", "El F12", "El E9"]


def write_real_layout(
    path: Path, *, n: int = 4000, newline: str = "\r\n", units=None, widen_after: int | None = None
):
    rng = np.random.default_rng(1)
    units = units or ["[ms]", *["[µV]"] * len(REAL_ELECTRODES)]
    pad = lambda text: f"{text:<12}"  # noqa: E731 - mirrors the fixed 12-character fields of MC_DataTool
    lines = [
        "MC_DataTool ASCII conversion",
        "",
        "\t".join(pad(name) for name in ["t", *REAL_ELECTRODES]),
        "\t".join(pad(unit) for unit in units),
    ]
    for i in range(n):
        ragged = (
            widen_after is not None and i >= widen_after
        )  # padding stops: record length now varies
        cell = (lambda text: text) if ragged else pad
        values = [cell(f"{rng.normal(0, 10.0):.2f}") for _ in REAL_ELECTRODES]
        lines.append("\t".join([cell(f"{i * 1.0:.3f}"), *values]))
    path.write_bytes((newline.join(lines) + newline).encode("latin-1"))


@pytest.mark.parametrize("newline", ["\r\n", "\n"])
def test_real_layout_is_read_unmodified_including_units_row(tmp_path, newline):
    source = tmp_path / "Day9.txt"
    write_real_layout(source, newline=newline)
    before = source.read_bytes()
    layout = sniff_layout(source)
    assert layout.header_lines == 4 and layout.units == ("[ms]", *["[µV]"] * 4)
    assert layout.columns == ("t", *REAL_ELECTRODES) and layout.fs_hz == pytest.approx(1000.0)
    assert layout.record_bytes is not None and layout.header_bytes > 0
    read_window(source, layout, electrodes=["El F8"], start_s=0.0, duration_s=1.0)
    assert (
        source.read_bytes() == before
    )  # no preprocessing, no sed: the source hash stays the archive's hash


def test_seek_path_matches_streaming_path_and_is_really_used(tmp_path):
    source = tmp_path / "Day9.txt"
    write_real_layout(source)
    layout = sniff_layout(source)
    from virelion_cardioscore.io import mcs_ascii

    usecols = [0, 1, 3]
    assert mcs_ascii._seek_frame(source, layout, usecols, 2.0, 1.0) is not None
    for electrodes in (["El F7", "El F12"], ["El F12", "El F7"]):
        fast = read_window(source, layout, electrodes=electrodes, start_s=2.0, duration_s=1.0)
        slow = read_window(
            source, layout, electrodes=electrodes, start_s=2.0, duration_s=1.0, allow_seek=False
        )
        assert np.array_equal(fast[0], slow[0]) and np.array_equal(fast[1], slow[1])
        assert fast[0][0] == pytest.approx(2.0) and fast[1].shape == (1000, 2)


def test_seek_falls_back_to_streaming_when_the_file_is_not_really_fixed_width(tmp_path):
    source = tmp_path / "Day9.txt"
    write_real_layout(
        source, widen_after=1000
    )  # padded (fixed-width) at the top, ragged afterwards
    layout = sniff_layout(source)
    from virelion_cardioscore.io import mcs_ascii

    assert layout.record_bytes is not None
    assert (
        mcs_ascii._seek_frame(source, layout, [0, 1], 2.0, 1.0) is None
    )  # position check catches it
    fast = read_window(source, layout, electrodes=["El F7"], start_s=2.0, duration_s=1.0)
    slow = read_window(
        source, layout, electrodes=["El F7"], start_s=2.0, duration_s=1.0, allow_seek=False
    )
    assert np.array_equal(fast[1], slow[1]) and fast[0][0] == pytest.approx(2.0)


@pytest.mark.parametrize(
    ("units", "message"),
    [
        (["[s]", "[µV]", "[µV]", "[µV]", "[µV]"], "only \\[ms\\]"),
        (["[ms]", "[µV]", "[mV]", "[µV]", "[µV]"], "Unrecognized voltage unit"),
        (["[ms]", "[µV]", "[µV]"], "Units row has 3 entries"),
    ],
)
def test_unexpected_units_are_rejected_not_assumed(tmp_path, units, message):
    source = tmp_path / "Day9.txt"
    write_real_layout(source, units=units)
    with pytest.raises(SourceFormatError, match=message):
        sniff_layout(source)


def test_adapter_hashes_the_original_file_and_verifies_the_rebuild(tmp_path):
    source = tmp_path / "Day9.txt"
    write_real_layout(source)
    out = tmp_path / "out"
    code = adapter.main(
        [
            str(source),
            "--electrodes",
            "El F7",
            "El F8",
            "--start-s",
            "1",
            "--duration-s",
            "2",
            "--out-dir",
            str(out),
        ]
    )
    lineage = json.loads((out / "lineage.json").read_text(encoding="utf-8"))
    assert code == 0 and lineage["verified_rebuild"] is True
    assert lineage["source_sha256"] == adapter.file_sha256(source)
    assert (
        lineage["layout"]["units"] == ["[ms]", *["[µV]"] * 4]
        and lineage["layout"]["seek_capable"] is True
    )

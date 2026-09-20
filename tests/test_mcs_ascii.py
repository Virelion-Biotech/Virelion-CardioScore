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
"""The raw-source inspector must describe files without altering or over-reading them."""

from __future__ import annotations

import importlib.util
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_path = Path(__file__).resolve().parents[1] / "scripts" / "validation" / "inspect_raw_source.py"
_spec = importlib.util.spec_from_file_location("inspect_raw_source", _path)
inspector = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(inspector)


def test_tab_delimited_recording_with_preamble(tmp_path):
    path = tmp_path / "rec.txt"
    rows = ["# Axion export", "Time (s)\tE1\tE2"] + [
        f"{i * 0.0001:.4f}\t{i % 7}\t{i % 5}" for i in range(50)
    ]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    info = inspector.inspect_file(path)
    assert info["kind"] == "text" and info["delimiter"] == "\t"
    assert info["column_names_guess"] == ["Time (s)", "E1", "E2"]
    assert info["first_numeric_row_index"] == 2 and info["line_count"] == 52
    assert info["first_column_monotonic_in_sample"] is True
    assert info["first_column_median_step"] == pytest.approx(0.0001)
    assert len(info["sha256"]) == 64 and info["bytes"] == path.stat().st_size


def test_source_file_is_not_modified(tmp_path):
    path = tmp_path / "a.csv"
    path.write_text("t,v\n0,1\n1,2\n", encoding="utf-8")
    before = path.read_bytes(), path.stat().st_mtime_ns
    inspector.build_inventory([tmp_path])
    assert (path.read_bytes(), path.stat().st_mtime_ns) == before
    assert [p.name for p in tmp_path.iterdir()] == ["a.csv"]


def test_mat_v5_variables_and_nested_struct(tmp_path):
    scipy_io = pytest.importorskip("scipy.io")
    path = tmp_path / "plate1_baseline.mat"
    scipy_io.savemat(
        path,
        {
            "fs": 12500.0,
            "data": np.zeros((4, 100)),
            "meta": {"well": "A1", "fpd_ms": np.arange(3.0)},
        },
    )
    info = inspector.inspect_file(path)
    assert info["format"] == "mat_v5_or_earlier"
    names = {v["name"] for v in info["variables"]}
    assert {"fs", "data", "meta"} <= names
    assert info["structure"]["data"]["shape"] == [4, 100]
    assert set(info["structure"]["meta"]["fields"]) == {"well", "fpd_ms"}


def test_hdf5_and_v73_layout_is_listed_without_loading_data(tmp_path):
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "rec.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("wells/A1/voltage", data=np.zeros((8, 1000), dtype="float32"))
    info = inspector.inspect_file(path)
    entry = next(i for i in info["items"] if i["path"] == "wells/A1/voltage")
    assert entry["shape"] == [8, 1000] and entry["dtype"] == "float32"


def test_spreadsheet_headers_and_label_guard(tmp_path):
    pytest.importorskip("openpyxl")
    path = tmp_path / "ref.xlsx"
    pd.DataFrame({"Drug_Name": ["a", "b"], "risk": ["H", "L"], "ddFPDc": [1.0, 2.0]}).to_excel(
        path, index=False
    )
    info = inspector.inspect_file(path)
    assert info["sheets"]["Sheet1"]["columns"] == ["Drug_Name", "risk", "ddFPDc"]
    assert (
        info["label_like_names"] == ["risk"] and "label-like" in info["label_blind_warning"].lower()
    )


def test_zip_members_are_listed_and_unsafe_paths_flagged_never_extracted(tmp_path):
    path = tmp_path / "bundle.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("ok/readme.txt", "hello")
        archive.writestr("../escape.txt", "bad")
    info = inspector.inspect_file(path)
    assert info["n_members"] == 2 and info["unsafe_members"] == ["../escape.txt"]
    assert not (tmp_path.parent / "escape.txt").exists()


def test_unknown_and_corrupt_files_are_reported_not_fatal(tmp_path):
    (tmp_path / "blob.bin").write_bytes(bytes(range(64)))
    (tmp_path / "broken.mat").write_bytes(b"not a mat file")
    inventory = inspector.build_inventory([tmp_path])
    by_name = {entry["file"]: entry for entry in inventory["files"]}
    assert by_name["blob.bin"]["kind"] == "unknown" and by_name["blob.bin"][
        "first_bytes_hex"
    ].startswith("0001")
    assert "error" in by_name["broken.mat"]


def test_cli_writes_json_and_rejects_missing_paths(tmp_path):
    (tmp_path / "x.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    out = tmp_path / "inv.json"
    assert inspector.main([str(tmp_path / "x.csv"), "--out", str(out)]) == 0
    assert json.loads(out.read_text(encoding="utf-8"))["n_files"] == 1
    assert inspector.main([str(tmp_path / "missing"), "--out", str(out)]) == 1
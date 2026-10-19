from pathlib import Path

import numpy as np
import pytest

from virelion_cardioscore.io.mcs_hdf5 import MCSHDF5Error, read_mcs_hdf5_traces


h5py = pytest.importorskip("h5py")


def _info_dtype():
    return np.dtype(
        [
            ("ChannelID", "<i4"),
            ("RowIndex", "<i4"),
            ("GroupID", "<i4"),
            ("ElectrodeGroup", "<i4"),
            ("Label", "S32"),
            ("RawDataType", "S16"),
            ("Unit", "S8"),
            ("Exponent", "<i4"),
            ("ADZero", "<i4"),
            ("Tick", "<i8"),
            ("ConversionFactor", "<i8"),
            ("ADCBits", "<i4"),
        ]
    )


def _write_fixture(path: Path, *, stamps=None):
    info = np.zeros(2, dtype=_info_dtype())
    for i in range(2):
        info[i]["ChannelID"] = i + 10
        info[i]["RowIndex"] = i
        info[i]["Label"] = f"E{i+1}".encode()
        info[i]["RawDataType"] = b"Int"
        info[i]["Unit"] = b"V"
        info[i]["Exponent"] = -12
        info[i]["ADZero"] = 100
        info[i]["Tick"] = 100
        info[i]["ConversionFactor"] = 8670
        info[i]["ADCBits"] = 24
    data = np.array([[100, 200, 0, 300], [100, 50, 150, 100]], dtype=np.int32)
    if stamps is None:
        stamps = np.array([[200000, 0, 3]], dtype=np.int64)
    with h5py.File(path, "w") as h:
        stream = h.require_group("Data/Recording_0/AnalogStream/Stream_0")
        stream.create_dataset("ChannelData", data=data)
        stream.create_dataset("ChannelDataTimeStamps", data=stamps)
        stream.create_dataset("InfoChannel", data=info)


def test_mcs_formula_and_sampling_rate(tmp_path):
    path = tmp_path / "fixture.h5"
    _write_fixture(path)
    fs_hz, traces, metadata = read_mcs_hdf5_traces(path)
    assert fs_hz == 10000.0
    assert list(traces) == ["E1", "E2"]
    expected = (np.array([100, 200, 0, 300], dtype=float) - 100.0) * 8670e-12 * 1e6
    assert np.allclose(traces["E1"], expected)
    assert metadata[0].multiplier_to_uv == pytest.approx(0.00867)


def test_windowing_uses_seconds(tmp_path):
    path = tmp_path / "fixture.h5"
    _write_fixture(path)
    _, traces, _ = read_mcs_hdf5_traces(path, start_s=0.0001, duration_s=0.0002)
    assert traces["E1"].shape == (2,)


def test_discontinuous_segments_fail_closed(tmp_path):
    path = tmp_path / "fixture.h5"
    stamps = np.array([[0, 0, 1], [500, 2, 3]], dtype=np.int64)
    _write_fixture(path, stamps=stamps)
    with pytest.raises(MCSHDF5Error, match="Discontinuous MCS time segments"):
        read_mcs_hdf5_traces(path)

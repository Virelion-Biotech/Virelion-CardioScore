"""Strict Multi Channel Systems HDF5 raw-signal ingestion.

This adapter implements the published MCS HDF5 Raw Data Definition for analog
streams. Raw ADC integers are converted with the manufacturer formula::

    physical_value = (raw - ADZero) * ConversionFactor * 10**Exponent

in ``InfoChannel.Unit``. ``Tick`` is interpreted in microseconds, so
``fs_hz = 1_000_000 / Tick``.

The adapter is intentionally strict: it refuses unsupported units, inconsistent
per-channel sampling rates, malformed RowIndex values, and discontinuous sample
segments rather than silently guessing or interpolating.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

from virelion_cardioscore.io.raw_trace import RawWellRecording


class MCSHDF5Error(ValueError):
    """Raised when an MCS HDF5 stream cannot be interpreted without guessing."""


@dataclass(frozen=True)
class MCSChannelMetadata:
    channel_id: int
    row_index: int
    label: str
    unit: str
    exponent: int
    ad_zero: int
    tick_us: int
    conversion_factor: int
    adc_bits: int
    raw_data_type: str

    @property
    def fs_hz(self) -> float:
        if self.tick_us <= 0:
            raise MCSHDF5Error(f"Invalid Tick={self.tick_us} for channel {self.channel_id}")
        return 1_000_000.0 / float(self.tick_us)

    @property
    def multiplier_to_uv(self) -> float:
        if self.unit == "V":
            return float(self.conversion_factor) * (10.0 ** self.exponent) * 1e6
        if self.unit == "mV":
            return float(self.conversion_factor) * (10.0 ** self.exponent) * 1e3
        if self.unit in {"uV", "µV", "μV"}:
            return float(self.conversion_factor) * (10.0 ** self.exponent)
        raise MCSHDF5Error(
            f"Unsupported MCS physical unit {self.unit!r}; refusing implicit conversion to uV"
        )


def _decode(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.bytes_):
        return bytes(value).decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        return _decode(value.item())
    return value


def _stream_path(recording_index: int, stream_index: int) -> str:
    return f"Data/Recording_{recording_index}/AnalogStream/Stream_{stream_index}"


def _read_info_rows(info_array: np.ndarray) -> list[MCSChannelMetadata]:
    required = {
        "ChannelID",
        "RowIndex",
        "Label",
        "RawDataType",
        "Unit",
        "Exponent",
        "ADZero",
        "Tick",
        "ConversionFactor",
        "ADCBits",
    }
    names = set(info_array.dtype.names or ())
    missing = sorted(required - names)
    if missing:
        raise MCSHDF5Error(f"InfoChannel missing required field(s): {missing}")

    rows: list[MCSChannelMetadata] = []
    for row in info_array:
        rows.append(
            MCSChannelMetadata(
                channel_id=int(row["ChannelID"]),
                row_index=int(row["RowIndex"]),
                label=str(_decode(row["Label"])),
                raw_data_type=str(_decode(row["RawDataType"])),
                unit=str(_decode(row["Unit"])),
                exponent=int(row["Exponent"]),
                ad_zero=int(row["ADZero"]),
                tick_us=int(row["Tick"]),
                conversion_factor=int(row["ConversionFactor"]),
                adc_bits=int(row["ADCBits"]),
            )
        )
    return rows


def _validate_segments(stamps: np.ndarray, n_samples: int, tick_us: int) -> None:
    stamps = np.asarray(stamps)
    if stamps.ndim != 2 or stamps.shape[1] != 3 or stamps.shape[0] < 1:
        raise MCSHDF5Error(
            f"ChannelDataTimeStamps must have shape (n_segments, 3), got {stamps.shape}"
        )
    ordered = stamps[np.argsort(stamps[:, 1])]
    expected_first = 0
    previous_last_time: int | None = None
    for start_time, first_idx, last_idx in ordered:
        first_idx = int(first_idx)
        last_idx = int(last_idx)
        start_time = int(start_time)
        if first_idx != expected_first:
            raise MCSHDF5Error(
                "Discontinuous ChannelData index segments are not supported without resampling"
            )
        if last_idx < first_idx:
            raise MCSHDF5Error("ChannelDataTimeStamps contains last_idx < first_idx")
        if previous_last_time is not None:
            expected_start_time = previous_last_time + tick_us
            if start_time != expected_start_time:
                raise MCSHDF5Error(
                    "Discontinuous MCS time segments are not supported without resampling"
                )
        previous_last_time = start_time + (last_idx - first_idx) * tick_us
        expected_first = last_idx + 1
    if expected_first != n_samples:
        raise MCSHDF5Error(
            f"Timestamp segments cover {expected_first} samples but ChannelData has {n_samples}"
        )


def read_mcs_hdf5_traces(
    path: str | Path,
    *,
    recording_index: int = 0,
    stream_index: int = 0,
    start_s: float = 0.0,
    duration_s: Optional[float] = None,
) -> tuple[float, dict[str, np.ndarray], list[MCSChannelMetadata]]:
    """Read one MCS analog stream and return ``(fs_hz, traces_uv, metadata)``.

    ``traces_uv`` is keyed by channel label when unique, otherwise by ChannelID.
    Sample-time offsets are not preserved because CardioScore consumes uniformly
    sampled traces; continuity is validated before the offset is discarded.
    """
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "Reading MCS HDF5 requires h5py; install virelion-cardioscore[hdf5] or h5py"
        ) from exc

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if start_s < 0:
        raise MCSHDF5Error("start_s cannot be negative")
    if duration_s is not None and duration_s <= 0:
        raise MCSHDF5Error("duration_s must be positive when provided")

    stream_path = _stream_path(recording_index, stream_index)
    with h5py.File(path, "r") as handle:
        if stream_path not in handle:
            raise MCSHDF5Error(f"Missing analog stream {stream_path!r}")
        stream = handle[stream_path]
        for name in ("ChannelData", "ChannelDataTimeStamps", "InfoChannel"):
            if name not in stream:
                raise MCSHDF5Error(f"Missing {stream_path}/{name}")

        data = stream["ChannelData"]
        if data.ndim != 2:
            raise MCSHDF5Error(f"ChannelData must be 2D, got shape {data.shape}")
        metadata = _read_info_rows(np.asarray(stream["InfoChannel"][()]))
        if not metadata:
            raise MCSHDF5Error("InfoChannel contains no channels")

        fs_values = {round(row.fs_hz, 12) for row in metadata}
        if len(fs_values) != 1:
            raise MCSHDF5Error(
                f"Per-channel Tick values imply inconsistent sampling rates: {sorted(fs_values)}"
            )
        fs_hz = float(next(iter(fs_values)))
        tick_us = metadata[0].tick_us
        _validate_segments(
            np.asarray(stream["ChannelDataTimeStamps"][()]), int(data.shape[1]), tick_us
        )

        row_indices = [row.row_index for row in metadata]
        if len(set(row_indices)) != len(row_indices):
            raise MCSHDF5Error("InfoChannel RowIndex values are not unique")
        if min(row_indices) < 0 or max(row_indices) >= data.shape[0]:
            raise MCSHDF5Error(
                f"InfoChannel RowIndex outside ChannelData rows 0..{data.shape[0] - 1}"
            )

        start = int(round(start_s * fs_hz))
        stop = int(data.shape[1])
        if duration_s is not None:
            stop = min(stop, start + int(round(duration_s * fs_hz)))
        if start >= int(data.shape[1]) or start >= stop:
            raise MCSHDF5Error("Requested time window does not overlap the recording")

        labels = [row.label for row in metadata]
        labels_unique = len(labels) == len(set(labels)) and all(label.strip() for label in labels)
        traces: dict[str, np.ndarray] = {}
        for row in metadata:
            raw = np.asarray(data[row.row_index, start:stop], dtype=np.float64)
            trace_uv = (raw - float(row.ad_zero)) * row.multiplier_to_uv
            key = row.label if labels_unique else str(row.channel_id)
            traces[key] = trace_uv

    return fs_hz, traces, metadata


def load_mcs_hdf5_recording(
    path: str | Path,
    *,
    compound: str,
    well: str,
    concentration_uM: float,
    vehicle: bool,
    start_s: float = 0.0,
    duration_s: Optional[float] = None,
    recording_index: int = 0,
    stream_index: int = 0,
    metadata: Optional[dict[str, object]] = None,
) -> RawWellRecording:
    """Load an MCS HDF5 stream directly into CardioScore's raw recording model."""
    fs_hz, traces, channels = read_mcs_hdf5_traces(
        path,
        recording_index=recording_index,
        stream_index=stream_index,
        start_s=start_s,
        duration_s=duration_s,
    )
    meta = dict(metadata or {})
    meta.update(
        {
            "source_format": "MCS HDF5",
            "source_file": Path(path).name,
            "recording_index": recording_index,
            "stream_index": stream_index,
            "channel_count": len(channels),
        }
    )
    return RawWellRecording(
        compound=str(compound),
        well=str(well),
        concentration_uM=float(concentration_uM),
        vehicle=bool(vehicle),
        fs_hz=float(fs_hz),
        electrode_traces=traces,
        metadata=meta,
    )

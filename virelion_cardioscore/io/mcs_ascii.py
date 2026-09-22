"""Reader for Multichannel Systems MC_Data ASCII (*.txt) MEA exports.

Documented layout (Cardio PyMEA / MC_Data): a column-wise text file whose first column is time in
milliseconds and whose remaining columns are electrodes, in microvolts. The real exports start with a
title line, a blank line, a column-name row and a units row (``[ms]``, ``[µV]`` ...), in latin-1.
Nothing about the layout is assumed beyond that: the header is located from the data, column names
must line up with the data columns (never invented), units are verified when present, timestamps must
be strictly increasing and uniformly spaced, and any mismatch raises ``SourceFormatError`` instead of
being repaired. The source file is only ever read, never edited.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ADAPTER_VERSION = "mcs-ascii-1.1"
ENCODING = "latin-1"
MAX_TIMESTAMP_JITTER = 0.01
PLAUSIBLE_FS_HZ = (500.0, 100_000.0)
SEEK_MARGIN_ROWS = 5
VOLTAGE_UNITS = {"[µV]", "[uV]", "[μV]"}
_UNIT_TOKEN = re.compile(r"^\[.*\]$")


class SourceFormatError(ValueError):
    """The file does not match the documented MC_Data ASCII layout."""


@dataclass(frozen=True)
class McsAsciiLayout:
    delimiter: str
    header_lines: int
    columns: tuple[str, ...]  # first entry is the time column
    fs_hz: float
    header_bytes: int = 0
    record_bytes: int | None = None  # constant bytes per data line when the file is fixed-width
    units: tuple[str, ...] | None = (
        None  # None: the file has no units row (units then follow the documentation)
    )

    @property
    def electrodes(self) -> tuple[str, ...]:
        return self.columns[1:]


def _tokens(line: str, delimiter: str) -> list[str]:
    parts = line.split() if delimiter == " " else line.split(delimiter)
    return [token.strip() for token in parts if token.strip() != ""]


def _is_numeric(token: str) -> bool:
    try:
        float(token.replace(",", "."))
    except ValueError:
        return False
    return True


def sniff_layout(
    path: str | Path, *, max_header_lines: int = 200, time_unit: str = "ms"
) -> McsAsciiLayout:
    if time_unit != "ms":
        raise SourceFormatError("Only millisecond time columns are documented for MC_Data exports.")
    raw_lines: list[bytes] = []
    with Path(path).open("rb") as handle:
        for _ in range(max_header_lines + 60):
            line = handle.readline()
            if not line:
                break
            raw_lines.append(line)
    lines = [raw.decode(ENCODING).rstrip("\r\n") for raw in raw_lines]
    body = [line for line in lines if line.strip()]
    if not body:
        raise SourceFormatError("File is empty.")
    try:
        delimiter = csv.Sniffer().sniff("\n".join(body[-20:]), delimiters="\t,; ").delimiter
    except csv.Error as exc:
        raise SourceFormatError(f"Cannot determine the column delimiter: {exc}") from exc
    first_data = next(
        (
            i
            for i, line in enumerate(lines)
            if line.strip() and all(_is_numeric(t) for t in _tokens(line, delimiter))
        ),
        None,
    )
    if first_data is None or first_data > max_header_lines:
        raise SourceFormatError("No numeric data row found near the top of the file.")
    n_columns = len(_tokens(lines[first_data], delimiter))
    if n_columns < 2:
        raise SourceFormatError("Expected a time column plus at least one electrode column.")

    header = [line for line in lines[:first_data] if line.strip()]
    units: tuple[str, ...] | None = None
    if header and all(_UNIT_TOKEN.match(t) for t in _tokens(header[-1], delimiter)):
        units = tuple(_tokens(header[-1], delimiter))
        header = header[:-1]
        if len(units) != n_columns:
            raise SourceFormatError(
                f"Units row has {len(units)} entries but the data has {n_columns} columns."
            )
        if units[0] != "[ms]":
            raise SourceFormatError(f"Time unit is {units[0]!r}; only [ms] is supported.")
        bad = sorted({u for u in units[1:] if u not in VOLTAGE_UNITS})
        if bad:
            raise SourceFormatError(
                f"Unrecognized voltage unit(s) {bad}; only microvolts are supported."
            )
    names = _tokens(header[-1], delimiter) if header else []
    if len(names) != n_columns:
        raise SourceFormatError(
            f"Column names ({len(names)}) do not line up with data columns ({n_columns}); "
            "names are not guessed. Inspect the file header with inspect_raw_source.py."
        )
    if len(set(names)) != len(names):
        raise SourceFormatError("Duplicate column names.")

    data_rows = [
        (raw, line)
        for raw, line in zip(raw_lines[first_data:], lines[first_data:], strict=True)
        if line.strip()
    ]
    times = [float(_tokens(line, delimiter)[0].replace(",", ".")) for _, line in data_rows[:50]]
    steps = np.diff(times)
    if len(steps) < 2 or (steps <= 0).any():
        raise SourceFormatError("Time column is not strictly increasing at the top of the file.")
    fs_hz = 1000.0 / float(np.median(steps))
    if not PLAUSIBLE_FS_HZ[0] <= fs_hz <= PLAUSIBLE_FS_HZ[1]:
        raise SourceFormatError(
            f"Implied sampling rate {fs_hz:.1f} Hz is implausible; the time unit may not be ms."
        )
    lengths = {
        len(raw) for raw, _ in data_rows[:-1] if len(data_rows) > 1
    }  # the last sampled line may be cut short
    record_bytes = lengths.pop() if len(lengths) == 1 and len(data_rows) > 3 else None
    return McsAsciiLayout(
        delimiter=delimiter,
        header_lines=first_data,
        columns=tuple(names),
        fs_hz=fs_hz,
        header_bytes=sum(len(raw) for raw in raw_lines[:first_data]),
        record_bytes=record_bytes,
        units=units,
    )


def _read_kwargs(layout: McsAsciiLayout, usecols: list[int]) -> dict:
    return {
        "sep": layout.delimiter if layout.delimiter != " " else r"\s+",
        "header": None,
        "usecols": usecols,
        "dtype": np.float64,
        "encoding": ENCODING,
        "engine": "c" if layout.delimiter != " " else "python",
    }


def _seek_frame(
    path: Path, layout: McsAsciiLayout, usecols: list[int], start_s: float, duration_s: float
):
    """Jump straight to the window in a fixed-width file; None if the position cannot be verified."""
    if not layout.record_bytes or start_s <= 0:
        return None
    step_s = 1.0 / layout.fs_hz
    first_row = max(0, int(round(start_s * layout.fs_hz)) - SEEK_MARGIN_ROWS)
    n_rows = int(round(duration_s * layout.fs_hz)) + 2 * SEEK_MARGIN_ROWS
    with path.open("rb") as handle:
        handle.seek(layout.header_bytes + first_row * layout.record_bytes)
        try:
            frame = pd.read_csv(handle, nrows=n_rows, **_read_kwargs(layout, usecols))
        except (ValueError, pd.errors.ParserError):
            return None  # landed mid-line (record length is not constant): fall back to streaming
    if frame.empty or abs(frame.iloc[0, 0] / 1000.0 - first_row * step_s) > 0.25 * step_s:
        return None  # not actually fixed-width where we landed: fall back to streaming
    return frame


def read_window(
    path: str | Path,
    layout: McsAsciiLayout,
    *,
    electrodes: list[str],
    start_s: float = 0.0,
    duration_s: float,
    chunk_rows: int = 200_000,
    allow_seek: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (time_s, voltage_uv[n_samples, n_electrodes]) for a time window.

    Fixed-width files are entered by byte offset (verified against the timestamp found there); anything
    else, or a failed verification, streams the file from the top. Both paths return identical data.
    """
    path = Path(path)
    unknown = [name for name in electrodes if name not in layout.electrodes]
    if unknown or not electrodes:
        raise SourceFormatError(f"Unknown or empty electrode selection: {unknown or electrodes}")
    end_s = start_s + duration_s
    requested = [layout.columns.index(name) for name in electrodes]
    # pandas returns columns in file order, not in the order requested; reorder explicitly so every
    # voltage column stays paired with the electrode name it was requested under.
    file_order = sorted(set(requested))
    usecols = [0, *file_order]
    column_of = {index: position for position, index in enumerate(file_order)}
    reorder = [column_of[index] for index in requested]

    seeked = _seek_frame(path, layout, usecols, start_s, duration_s) if allow_seek else None
    chunks = (
        [seeked]
        if seeked is not None
        else pd.read_csv(
            path,
            skiprows=layout.header_lines,
            chunksize=chunk_rows,
            **_read_kwargs(layout, usecols),
        )
    )
    times: list[np.ndarray] = []
    values: list[np.ndarray] = []
    for chunk in chunks:
        time_s = chunk.iloc[:, 0].to_numpy() / 1000.0
        keep = (time_s >= start_s) & (time_s < end_s)
        if keep.any():
            times.append(time_s[keep])
            values.append(chunk.iloc[:, 1:].to_numpy()[keep][:, reorder])
        if time_s.size and time_s[-1] >= end_s:
            break
    if not times:
        raise SourceFormatError("The requested window contains no samples.")
    time_arr = np.concatenate(times)
    voltage = np.concatenate(values)
    if not np.isfinite(voltage).all():
        raise SourceFormatError("Non-finite voltage values in the requested window.")
    steps = np.diff(time_arr)
    if (steps <= 0).any():
        raise SourceFormatError(
            "Timestamps are not strictly increasing in the window (duplicates or reversals)."
        )
    if np.max(np.abs(steps - np.median(steps))) > MAX_TIMESTAMP_JITTER * np.median(steps):
        raise SourceFormatError(
            "Timestamps are not uniformly spaced in the window (gaps or jitter over 1%)."
        )
    return time_arr, voltage

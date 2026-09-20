"""Reader for Multichannel Systems MC_Data ASCII (*.txt) MEA exports.

Documented layout (Cardio PyMEA / MC_Data): a column-wise text file whose first column is time in
milliseconds and whose remaining columns are electrodes, in microvolts. Nothing about the layout is
assumed beyond that: the header is located from the data, column names must line up with the data
columns (never invented), timestamps must be strictly increasing and uniformly spaced, and any
mismatch raises ``SourceFormatError`` instead of being repaired.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ADAPTER_VERSION = "mcs-ascii-1.0"
MAX_TIMESTAMP_JITTER = 0.01
PLAUSIBLE_FS_HZ = (500.0, 100_000.0)


class SourceFormatError(ValueError):
    """The file does not match the documented MC_Data ASCII layout."""


@dataclass(frozen=True)
class McsAsciiLayout:
    delimiter: str
    header_lines: int
    columns: tuple[str, ...]  # first entry is the time column
    fs_hz: float

    @property
    def electrodes(self) -> tuple[str, ...]:
        return self.columns[1:]


def _tokens(line: str, delimiter: str) -> list[str]:
    return [
        token.strip()
        for token in (line.split(delimiter) if delimiter != " " else line.split())
        if token.strip() != ""
    ]


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
    lines: list[str] = []
    with Path(path).open("r", encoding="utf-8", errors="replace") as handle:
        for _ in range(max_header_lines + 60):
            line = handle.readline()
            if not line:
                break
            lines.append(line.rstrip("\r\n"))
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
    names = _tokens(header[-1], delimiter) if header else []
    if len(names) != n_columns:
        raise SourceFormatError(
            f"Column names ({len(names)}) do not line up with data columns ({n_columns}); "
            "names are not guessed. Inspect the file header with inspect_raw_source.py."
        )
    if len(set(names)) != len(names):
        raise SourceFormatError("Duplicate column names.")
    times = [
        float(_tokens(line, delimiter)[0].replace(",", "."))
        for line in lines[first_data : first_data + 50]
        if line.strip()
    ]
    steps = np.diff(times)
    if len(steps) < 2 or (steps <= 0).any():
        raise SourceFormatError("Time column is not strictly increasing at the top of the file.")
    fs_hz = 1000.0 / float(np.median(steps))
    if not PLAUSIBLE_FS_HZ[0] <= fs_hz <= PLAUSIBLE_FS_HZ[1]:
        raise SourceFormatError(
            f"Implied sampling rate {fs_hz:.1f} Hz is implausible; the time unit may not be ms."
        )
    return McsAsciiLayout(
        delimiter=delimiter, header_lines=first_data, columns=tuple(names), fs_hz=fs_hz
    )


def read_window(
    path: str | Path,
    layout: McsAsciiLayout,
    *,
    electrodes: list[str],
    start_s: float = 0.0,
    duration_s: float,
    chunk_rows: int = 200_000,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (time_s, voltage_uv[n_samples, n_electrodes]) for a time window; streams the file."""
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
    reader = pd.read_csv(
        path,
        sep=layout.delimiter if layout.delimiter != " " else r"\s+",
        skiprows=layout.header_lines,
        header=None,
        usecols=usecols,
        decimal=".",
        chunksize=chunk_rows,
        dtype=np.float64,
        engine="c" if layout.delimiter != " " else "python",
    )
    times: list[np.ndarray] = []
    values: list[np.ndarray] = []
    for chunk in reader:
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
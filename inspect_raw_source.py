#!/usr/bin/env python3
"""Inventory raw MEA source files without interpreting or modifying them.

For each file this records name, size and SHA-256 plus a structural summary (text layout, MAT/HDF5
variables, spreadsheet headers, ZIP members). It never loads a large file fully, never writes next
to the source, and flags label-like columns so reference labels are not mixed into feature
development by accident. The JSON output is the "inventory" a source receipt needs and is small
enough to share.

    python inspect_raw_source.py /path/to/data_dir_or_file [...] --out source_inventory.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

INSPECTOR_VERSION = "1.0"
LABEL_LIKE = ("risk", "label", "class", "category", "tdp", "reference", "outcome")
TEXT_SUFFIXES = {".txt", ".csv", ".tsv", ".dat", ".asc", ".tab"}
MAX_STRUCT_DEPTH = 3


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_number(token: str) -> bool:
    try:
        float(token.replace(",", "."))
    except ValueError:
        return False
    return True


def inspect_text(
    path: Path, sample_lines: int = 40, count_lines_up_to_mb: int = 4096
) -> dict[str, Any]:
    with path.open("rb") as handle:
        raw = handle.read(1 << 16)
    encoding = "utf-8"
    try:
        raw.decode("utf-8")
    except UnicodeDecodeError:
        encoding = "latin-1"
    lines: list[str] = []
    with path.open("r", encoding=encoding, errors="replace") as handle:
        for _ in range(sample_lines):
            line = handle.readline()
            if not line:
                break
            lines.append(line.rstrip("\r\n"))
    sample = "\n".join(lines)
    delimiter = None
    try:
        delimiter = csv.Sniffer().sniff(sample, delimiters="\t,;| ").delimiter
    except csv.Error:
        delimiter = "\t" if "\t" in sample else ("," if "," in sample else None)
    split = (lambda line: line.split(delimiter)) if delimiter else (lambda line: line.split())
    widths = sorted({len(split(line)) for line in lines if line.strip()})
    first_data = next(
        (
            i
            for i, line in enumerate(lines)
            if line.strip() and all(_is_number(t.strip()) for t in split(line) if t.strip())
        ),
        None,
    )
    header_lines = lines[:first_data] if first_data is not None else []
    columns = [t.strip() for t in split(header_lines[-1])] if header_lines else []
    info: dict[str, Any] = {
        "kind": "text",
        "encoding_guess": encoding,
        "delimiter": delimiter,
        "distinct_field_counts_in_sample": widths,
        "first_numeric_row_index": first_data,
        "preamble_or_header_lines": header_lines[:20],
        "column_names_guess": columns,
        "first_lines": lines[:10],
    }
    if first_data is not None:
        numeric = [split(line) for line in lines[first_data:] if line.strip()]
        try:
            times = [float(row[0].strip().replace(",", ".")) for row in numeric]
            steps = sorted(b - a for a, b in zip(times, times[1:], strict=False) if b > a)
            if steps:
                info["first_column_median_step"] = steps[len(steps) // 2]
                info["first_column_monotonic_in_sample"] = all(
                    b > a for a, b in zip(times, times[1:], strict=False)
                )
        except (ValueError, IndexError):
            pass
    if path.stat().st_size <= count_lines_up_to_mb * (1 << 20):
        with path.open("rb") as handle:
            info["line_count"] = sum(
                block.count(b"\n") for block in iter(lambda: handle.read(1 << 22), b"")
            )
    return info


def _describe_value(value: Any, depth: int = 0) -> Any:
    import numpy as np

    if isinstance(value, np.ndarray):
        return {"type": "ndarray", "shape": list(value.shape), "dtype": str(value.dtype)}
    if hasattr(value, "_fieldnames"):
        if depth >= MAX_STRUCT_DEPTH:
            return {"type": "struct", "fields": list(value._fieldnames)}
        return {
            "type": "struct",
            "fields": {
                name: _describe_value(getattr(value, name), depth + 1) for name in value._fieldnames
            },
        }
    if isinstance(value, dict):
        return {
            k: _describe_value(v, depth + 1)
            for k, v in value.items()
            if not str(k).startswith("__")
        }
    return {"type": type(value).__name__}


def inspect_mat(path: Path, max_load_mb: int = 300) -> dict[str, Any]:
    info: dict[str, Any] = {"kind": "mat"}
    try:
        import scipy.io

        info["variables"] = [
            {"name": name, "shape": list(shape), "class": cls}
            for name, shape, cls in scipy.io.whosmat(str(path))
        ]
        info["format"] = "mat_v5_or_earlier"
        if path.stat().st_size <= max_load_mb * (1 << 20):
            loaded = scipy.io.loadmat(str(path), squeeze_me=True, struct_as_record=False)
            info["structure"] = _describe_value(
                {k: v for k, v in loaded.items() if not k.startswith("__")}
            )
        else:
            info["structure"] = f"not loaded (file larger than {max_load_mb} MB)"
        return info
    except NotImplementedError:
        info["format"] = "mat_v7.3_hdf5"
    except ImportError as exc:
        return {**info, "error": f"scipy unavailable: {exc}"}
    except Exception as exc:  # corrupt or unusual file: report, do not crash the inventory
        return {**info, "error": f"{type(exc).__name__}: {exc}"}
    return {**info, **inspect_hdf5(path)}


def inspect_hdf5(path: Path) -> dict[str, Any]:
    try:
        import h5py
    except ImportError as exc:
        return {"kind": "hdf5", "error": f"h5py unavailable: {exc}"}
    items: list[dict[str, Any]] = []
    with h5py.File(path, "r") as handle:

        def visit(name: str, node: Any) -> None:
            if isinstance(node, h5py.Dataset):
                items.append({"path": name, "shape": list(node.shape), "dtype": str(node.dtype)})
            else:
                items.append({"path": name, "group": True})

        handle.visititems(visit)
    return {
        "kind": "hdf5",
        "n_items": len(items),
        "items": items[:500],
        "truncated": len(items) > 500,
    }


def inspect_spreadsheet(path: Path) -> dict[str, Any]:
    try:
        import pandas as pd

        sheets = pd.read_excel(path, sheet_name=None, nrows=5)
    except Exception as exc:
        return {"kind": "spreadsheet", "error": f"{type(exc).__name__}: {exc}"}
    return {
        "kind": "spreadsheet",
        "sheets": {
            name: {
                "columns": [str(c) for c in frame.columns],
                "first_rows": frame.astype(str).values.tolist(),
            }
            for name, frame in sheets.items()
        },
    }


def inspect_zip(path: Path) -> dict[str, Any]:
    members = []
    with zipfile.ZipFile(path) as archive:
        for entry in archive.infolist():
            unsafe = entry.filename.startswith(("/", "\\")) or ".." in Path(entry.filename).parts
            members.append(
                {"name": entry.filename, "bytes": entry.file_size, "unsafe_path": unsafe}
            )
    return {
        "kind": "zip",
        "n_members": len(members),
        "members": members[:500],
        "truncated": len(members) > 500,
        "unsafe_members": [m["name"] for m in members if m["unsafe_path"]],
    }


def label_like_columns(info: dict[str, Any]) -> list[str]:
    names: list[str] = list(info.get("column_names_guess") or [])
    for sheet in (info.get("sheets") or {}).values():
        names.extend(sheet.get("columns", []))
    for variable in info.get("variables") or []:
        names.append(variable["name"])
    for item in info.get("items") or []:
        names.append(str(item.get("path", "")))
    return sorted({n for n in names if any(token in n.lower() for token in LABEL_LIKE)})


def inspect_file(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    entry: dict[str, Any] = {
        "file": path.name,
        "relative_path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    try:
        if suffix in TEXT_SUFFIXES:
            entry.update(inspect_text(path))
        elif suffix == ".mat":
            entry.update(inspect_mat(path))
        elif suffix in {".h5", ".hdf5"}:
            entry.update(inspect_hdf5(path))
        elif suffix in {".xlsx", ".xls"}:
            entry.update(inspect_spreadsheet(path))
        elif suffix == ".zip":
            entry.update(inspect_zip(path))
        else:
            with path.open("rb") as handle:
                entry.update({"kind": "unknown", "first_bytes_hex": handle.read(64).hex()})
    except Exception as exc:
        entry["error"] = f"{type(exc).__name__}: {exc}"
    labels = label_like_columns(entry)
    if labels:
        entry["label_like_names"] = labels
        entry["label_blind_warning"] = (
            "Label-like names present. Keep them out of feature development until the pre-registration is frozen."
        )
    return entry


def build_inventory(paths: list[Path]) -> dict[str, Any]:
    files: list[Path] = []
    for path in paths:
        files.extend(sorted(p for p in path.rglob("*") if p.is_file()) if path.is_dir() else [path])
    return {
        "inspector_version": INSPECTOR_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_files": len(files),
        "files": [inspect_file(path) for path in files],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=Path("source_inventory.json"))
    args = parser.parse_args(argv)
    missing = [str(p) for p in args.paths if not p.exists()]
    if missing:
        print(f"Not found: {missing}", file=sys.stderr)
        return 1
    inventory = build_inventory(args.paths)
    args.out.write_text(json.dumps(inventory, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"Wrote {args.out} ({inventory['n_files']} file(s))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

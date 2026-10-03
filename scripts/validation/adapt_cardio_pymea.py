#!/usr/bin/env python3
"""Extract a time window of selected electrodes from a Cardio PyMEA (MC_Data ASCII) recording.

Writes time_s.npy, voltage_uv.npy, electrodes.json and lineage.json. The extraction is run twice from
the source file and lineage.json records ``verified_rebuild`` only if both runs give identical data
hashes. No compound, concentration or vehicle metadata is invented: this dataset is untreated, so the
output feeds signal-level validation, not the drug-response pipeline.

    python scripts/validation/adapt_cardio_pymea.py recording.txt --electrodes F7 F8 --start-s 0 --duration-s 30 --out-dir out
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from virelion_cardioscore.io.mcs_ascii import ADAPTER_VERSION, read_window, sniff_layout  # noqa: E402


def array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array, dtype="<f8").tobytes()).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def extract_once(source: Path, electrodes: list[str], start_s: float, duration_s: float):
    layout = sniff_layout(source)
    time_s, voltage = read_window(
        source, layout, electrodes=electrodes, start_s=start_s, duration_s=duration_s
    )
    return layout, time_s, voltage


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("source", type=Path)
    parser.add_argument("--electrodes", nargs="+", required=True)
    parser.add_argument("--start-s", type=float, default=0.0)
    parser.add_argument("--duration-s", type=float, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    layout, time_s, voltage = extract_once(
        args.source, args.electrodes, args.start_s, args.duration_s
    )
    _, time_2, voltage_2 = extract_once(args.source, args.electrodes, args.start_s, args.duration_s)
    hashes = {
        "time_s": (array_sha256(time_s), array_sha256(time_2)),
        "voltage_uv": (array_sha256(voltage), array_sha256(voltage_2)),
    }
    verified = all(first == second for first, second in hashes.values())

    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.out_dir / "time_s.npy", time_s)
    np.save(args.out_dir / "voltage_uv.npy", voltage)
    (args.out_dir / "electrodes.json").write_text(
        json.dumps(args.electrodes) + "\n", encoding="utf-8"
    )
    lineage = {
        "adapter": "adapt_cardio_pymea.py",
        "adapter_version": ADAPTER_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_file": args.source.name,
        "source_sha256": file_sha256(args.source),
        "layout": {
            "delimiter": layout.delimiter,
            "header_lines": layout.header_lines,
            "header_bytes": layout.header_bytes,
            "fs_hz": layout.fs_hz,
            "n_columns": len(layout.columns),
            "units": list(layout.units) if layout.units else None,
            "seek_capable": layout.record_bytes is not None,
        },
        "selection": {
            "electrodes": args.electrodes,
            "start_s": args.start_s,
            "duration_s": args.duration_s,
        },
        "n_samples": int(time_s.size),
        "output_sha256": {name: first for name, (first, _) in hashes.items()},
        "rebuild_sha256": {name: second for name, (_, second) in hashes.items()},
        "verified_rebuild": verified,
    }
    (args.out_dir / "lineage.json").write_text(
        json.dumps(lineage, indent=2) + "\n", encoding="utf-8"
    )
    print(f"verified_rebuild={verified}; wrote {args.out_dir}")
    return 0 if verified else 2


if __name__ == "__main__":
    raise SystemExit(main())

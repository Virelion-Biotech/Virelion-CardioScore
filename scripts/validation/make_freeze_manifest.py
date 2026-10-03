#!/usr/bin/env python3
"""Create validation/freeze_manifest.json from a fully resolved pre-registration.

Refuses to write anything while the pre-registration has unresolved fields, is not marked
``status: frozen``, or disagrees with the shipped scoring configuration.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from virelion_cardioscore.validation.freeze import FreezeError, build_freeze_manifest  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-sha", required=True, help="40-character commit SHA of the code being frozen")
    parser.add_argument("--prereg", type=Path, default=REPO_ROOT / "validation" / "preregistration.yaml")
    parser.add_argument("--config-dir", type=Path, default=REPO_ROOT / "virelion_cardioscore" / "config")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "validation" / "freeze_manifest.json")
    args = parser.parse_args(argv)
    try:
        manifest = build_freeze_manifest(
            prereg_path=args.prereg, config_dir=args.config_dir, package_sha=args.package_sha
        )
    except FreezeError as exc:
        print(exc, file=sys.stderr)
        return 1
    args.out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
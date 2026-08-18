#!/usr/bin/env python3
"""CI-safe checks for the external validation contract.

This always validates the checked-in template and never requires external data.
If a real manifest is supplied, it additionally checks that it is locked and that
its referenced source asset exists with the declared SHA-256.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from virelion_cardioscore.validation.integrity import verify_sha256
from virelion_cardioscore.validation.manifest import ValidationManifest, resolve_relative


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=None)
    args = parser.parse_args()

    template = Path("benchmarks/external_validation_manifest.example.yaml")
    if not template.is_file():
        raise SystemExit("Missing external validation manifest template")
    ValidationManifest.from_yaml(template)

    if args.manifest is None:
        print("Validation contract OK; no external validation asset supplied.")
        return 0

    manifest_path = args.manifest.resolve()
    manifest = ValidationManifest.from_yaml(manifest_path)
    asset = resolve_relative(manifest_path.parent, manifest.asset_path)
    if not asset.is_file():
        raise SystemExit(f"Validation asset not found: {asset}")
    if not verify_sha256(asset, manifest.asset_sha256):
        raise SystemExit(f"Validation asset SHA-256 mismatch: {asset}")
    print("Validation manifest and asset integrity OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

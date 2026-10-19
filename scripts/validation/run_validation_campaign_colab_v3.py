#!/usr/bin/env python3
"""Incremental v3 launcher for the CardioScore public validation campaign.

This launcher reuses the v2 campaign implementation pinned at b1def2d and changes
only public Dryad file acquisition. Dryad's metadata API currently exposes
/api/v2/files/<id>/download links that may require API authentication, while
published landing pages expose anonymous /downloads/file_stream/<id> links.
The v3 acquisition path therefore tries the public file-stream routes first,
verifies published size/SHA-256 when available, and falls back to the API link.
All analysis/scoring functions remain byte-for-byte those of the pinned v2 runner.
"""
from __future__ import annotations

import hashlib
import os
import re
import urllib.request
from pathlib import Path
from typing import Any

BASE_RUNNER_COMMIT = "b1def2d24302827f6b391e1c5054606dbd9f04e1"
BASE_RUNNER_URL = (
    "https://raw.githubusercontent.com/Virelion-Biotech/Virelion-CardioScore/"
    f"{BASE_RUNNER_COMMIT}/scripts/validation/run_validation_campaign_colab.py"
)


def load_base() -> dict[str, Any]:
    source = urllib.request.urlopen(BASE_RUNNER_URL).read()
    print("Base campaign runner commit:", BASE_RUNNER_COMMIT)
    print("Base campaign runner SHA-256:", hashlib.sha256(source).hexdigest())
    ns: dict[str, Any] = {"__name__": "cardioscore_campaign_v2_base"}
    exec(compile(source, "run_validation_campaign_colab.py", "exec"), ns)
    return ns


def install_public_dryad_patch(base: dict[str, Any]) -> None:
    """Replace only dryad_full_dataset; preserve every analysis function unchanged."""

    def dryad_file_id(entry: dict) -> str | None:
        value = entry.get("id")
        if value is not None and str(value).isdigit():
            return str(value)
        href = str((((entry.get("_links") or {}).get("stash:download") or {}).get("href") or ""))
        m = re.search(r"/files/(\d+)(?:/download)?(?:\?.*)?$", href)
        return m.group(1) if m else None

    def download_public_dryad(
        target: Path,
        *,
        candidates: list[str],
        landing_url: str,
        expected_sha256: str | None,
        expected_bytes: int | None,
    ) -> str:
        digest = base["digest"]
        run = base["run"]

        if target.exists():
            size_ok = expected_bytes is None or target.stat().st_size == expected_bytes
            sha_ok = expected_sha256 is None or digest(target) == expected_sha256
            if size_ok and sha_ok:
                print("✓ cached + verified:", target.name)
                return "drive_cache"
            print("Cached Dryad file failed verification; replacing", target.name)
            target.unlink()

        target.parent.mkdir(parents=True, exist_ok=True)
        part = target.with_suffix(target.suffix + ".part")
        errors: list[str] = []
        ua = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/155 Safari/537.36"
        for url in dict.fromkeys(candidates):
            if not url:
                continue
            for resume in (True, False):
                if not resume:
                    part.unlink(missing_ok=True)
                cmd = [
                    "curl", "-L", "--fail", "--retry", "3", "--retry-delay", "3",
                    "--connect-timeout", "30", "-A", ua, "-e", landing_url,
                ]
                if resume:
                    cmd += ["-C", "-"]
                cmd += ["-o", str(part), url]
                try:
                    run(cmd)
                    if expected_bytes is not None and part.stat().st_size != expected_bytes:
                        raise RuntimeError(
                            f"Dryad byte-size mismatch for {target.name}: "
                            f"{part.stat().st_size} != {expected_bytes}"
                        )
                    if expected_sha256 is not None:
                        actual = digest(part)
                        if actual != expected_sha256:
                            raise RuntimeError(
                                f"Dryad SHA-256 mismatch for {target.name}: "
                                f"{actual} != {expected_sha256}"
                            )
                    os.replace(part, target)
                    return url
                except Exception as exc:
                    errors.append(f"{url} (resume={resume}): {type(exc).__name__}: {exc}")
        part.unlink(missing_ok=True)
        raise RuntimeError(
            f"All public/API Dryad download routes failed for {target.name}:\n" + "\n".join(errors)
        )

    def patched_dryad_full_dataset(root: Path, dataset: str, doi: str) -> dict:
        atomic_json = base["atomic_json"]
        digest = base["digest"]
        now = base["now"]
        absolute = base["_absolute_dryad_href"]
        api_files = base["_dryad_api_files"]
        landing_files = base["_dryad_landing_files"]

        raw = root / "raw" / dataset
        extracted = raw / "files"
        raw.mkdir(parents=True, exist_ok=True)
        extracted.mkdir(parents=True, exist_ok=True)
        marker = raw / "SOURCE_UNAVAILABLE.json"
        landing_url = f"https://datadryad.org/dataset/doi:{doi}"

        method = "dryad_rest_metadata_public_file_stream"
        try:
            entries, provenance = api_files(doi)
        except Exception as api_exc:
            try:
                entries = landing_files(doi)
                if not entries:
                    raise RuntimeError("Landing page exposed no downloadable file links.")
                provenance = {"landing_url": landing_url, "api_error": str(api_exc)}
                method = "dryad_landing_public_file_stream"
            except Exception as landing_exc:
                atomic_json(marker, {
                    "doi": doi,
                    "checked_at_utc": now(),
                    "api_error": str(api_exc),
                    "landing_fallback_error": str(landing_exc),
                })
                return {
                    "status": "blocked",
                    "reason": "published_files_unavailable",
                    "doi": doi,
                    "outputs": [{"path": str(marker)}],
                }

        downloaded: list[Path] = []
        receipts: list[dict] = []
        for entry in entries:
            rel = Path(str(entry.get("path") or ""))
            if not rel.name or rel.is_absolute() or ".." in rel.parts:
                raise RuntimeError(f"Unsafe/invalid Dryad path: {rel}")
            target = extracted / rel

            href = str((((entry.get("_links") or {}).get("stash:download") or {}).get("href") or ""))
            file_id = dryad_file_id(entry)
            candidates: list[str] = []
            if file_id:
                candidates.extend([
                    f"https://datadryad.org/downloads/file_stream/{file_id}",
                    f"https://datadryad.org/stash/downloads/file_stream/{file_id}",
                ])
            if href:
                candidates.append(absolute(href))
            if not candidates:
                raise RuntimeError(f"Dryad file has no resolvable download route: {rel}")

            digest_value = str(entry.get("digest") or "").lower() or None
            digest_type = str(entry.get("digestType") or "").lower()
            expected_sha256 = digest_value if digest_type in {"sha-256", "sha256"} else None
            size_value = entry.get("size")
            try:
                expected_bytes = int(size_value) if size_value is not None else None
            except (TypeError, ValueError):
                expected_bytes = None

            used_url = download_public_dryad(
                target,
                candidates=candidates,
                landing_url=landing_url,
                expected_sha256=expected_sha256,
                expected_bytes=expected_bytes,
            )
            downloaded.append(target)
            receipts.append({
                "path": str(rel),
                "bytes": target.stat().st_size,
                "sha256": digest(target),
                "dryad_file_id": file_id,
                "dryad_digest": digest_value,
                "dryad_digest_type": entry.get("digestType"),
                "download_url_used": used_url,
            })

        marker.unlink(missing_ok=True)
        receipt = raw / "acquisition.json"
        atomic_json(receipt, {
            "doi": doi,
            "source": "Dryad",
            "method": method,
            "acquired_at_utc": now(),
            **provenance,
            "files": receipts,
        })
        return {
            "outputs": [{"path": str(receipt)}, {"path": str(extracted)}],
            "n_files": len(downloaded),
            "raw_dir": str(extracted),
            "download_method": method,
        }

    base["dryad_full_dataset"] = patched_dryad_full_dataset


def main() -> int:
    base = load_base()
    install_public_dryad_patch(base)
    print("Dryad public-download compatibility patch: active")
    return int(base["main"]())


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Authenticated Dryad compatibility layer for the resumable CardioScore validation campaign.

Dryad public metadata remain anonymous, but current file-download endpoints require an OAuth2
bearer token. This wrapper obtains a short-lived token from a Dryad API account ID/secret,
downloads files without forwarding the bearer token to object storage, verifies published
SHA-256 digests, and then delegates all analysis/scoring to the unchanged v2 campaign runner.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import urllib.parse
import urllib.request
from pathlib import Path
from datetime import datetime, timezone

BASE_RUNNER_COMMIT = "b1def2d24302827f6b391e1c5054606dbd9f04e1"
BASE_RUNNER_URL = (
    "https://raw.githubusercontent.com/Virelion-Biotech/Virelion-CardioScore/"
    f"{BASE_RUNNER_COMMIT}/scripts/validation/run_validation_campaign_colab.py"
)
DRYAD_TOKEN_URL = "https://datadryad.org/oauth/token"
DRYAD_API = "https://datadryad.org/api/v2"

DRYAD_RETRY_STAGES = {
    "patel_2019_optopacing:acquire",
    "patel_2019_optopacing:inventory",
    "patel_2019_optopacing:analysis",
    "ballouz_vandenberg_2021:acquire",
    "ballouz_vandenberg_2021:inventory",
    "ballouz_vandenberg_2021:analysis",
    "daley_2024_microtissue:acquire",
    "daley_2024_microtissue:inventory",
    "daley_2024_microtissue:analysis",
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _secret(name: str) -> str | None:
    value = os.environ.get(name)
    if value:
        return value.strip()
    try:
        from google.colab import userdata
        value = userdata.get(name)
        return str(value).strip() if value else None
    except Exception:
        return None


def dryad_token() -> str:
    import requests

    client_id = _secret("DRYAD_ID")
    client_secret = _secret("DRYAD_SECRET")
    if not client_id or not client_secret:
        raise RuntimeError(
            "Dryad file downloads now require an API account. Add DRYAD_ID and DRYAD_SECRET "
            "to Colab Secrets (key icon in the left sidebar) and enable notebook access. "
            "Do not paste the secret into notebook source or chat."
        )

    response = requests.post(
        DRYAD_TOKEN_URL,
        data={
            "client_id": client_id,
            "client_secret": client_secret,
            "grant_type": "client_credentials",
        },
        headers={"Content-Type": "application/x-www-form-urlencoded;charset=UTF-8"},
        timeout=60,
    )
    if response.status_code != 200:
        raise RuntimeError(
            f"Dryad OAuth token request failed: HTTP {response.status_code}: "
            f"{response.text[:300]}"
        )
    token = response.json().get("access_token")
    if not token:
        raise RuntimeError("Dryad OAuth response contained no access_token.")

    test = requests.get(
        f"{DRYAD_API}/test",
        headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
        timeout=60,
    )
    if test.status_code != 200:
        raise RuntimeError(
            f"Dryad token validation failed: HTTP {test.status_code}: {test.text[:300]}"
        )
    print("✓ Dryad API token obtained and validated (token value not displayed).")
    return token


def _auth_get_json(url: str, token: str) -> dict:
    import requests

    r = requests.get(
        url,
        headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
        timeout=60,
    )
    r.raise_for_status()
    return r.json()


def _dryad_files_authenticated(doi: str, token: str) -> tuple[list[dict], dict]:
    encoded = urllib.parse.quote("doi:" + doi, safe="")
    dataset_url = f"{DRYAD_API}/datasets/{encoded}"
    dataset = _auth_get_json(dataset_url, token)

    version_href = ((dataset.get("_links") or {}).get("stash:version") or {}).get("href")
    if not version_href:
        versions = _auth_get_json(dataset_url + "/versions", token)
        rows = ((versions.get("_embedded") or {}).get("stash:versions") or [])
        if not rows:
            raise RuntimeError("Dryad dataset exposes no published version.")
        chosen = max(rows, key=lambda x: int(x.get("id", -1)))
        version_href = (
            ((chosen.get("_links") or {}).get("self") or {}).get("href")
            or f"/api/v2/versions/{chosen['id']}"
        )

    version_url = (
        version_href if str(version_href).startswith("http")
        else "https://datadryad.org" + str(version_href)
    )
    version = _auth_get_json(version_url, token)
    files_href = ((version.get("_links") or {}).get("stash:files") or {}).get("href")
    if not files_href:
        files_href = str(version_href).rstrip("/") + "/files"

    files_url = (
        files_href if str(files_href).startswith("http")
        else "https://datadryad.org" + str(files_href)
    )

    entries: list[dict] = []
    seen: set[str] = set()
    while files_url and files_url not in seen:
        seen.add(files_url)
        payload = _auth_get_json(files_url, token)
        entries.extend(((payload.get("_embedded") or {}).get("stash:files") or []))
        next_href = ((payload.get("_links") or {}).get("next") or {}).get("href")
        files_url = (
            next_href if next_href and str(next_href).startswith("http")
            else ("https://datadryad.org" + str(next_href) if next_href else "")
        )

    if not entries:
        raise RuntimeError("Dryad API returned zero files.")
    return entries, {"dataset_url": dataset_url, "version_url": version_url}


def _download_dryad_file(
    url: str,
    target: Path,
    token: str,
    *,
    expected_size: int | None,
    expected_sha256: str | None,
) -> None:
    """Fetch the authenticated Dryad redirect, then download storage bytes without the token."""
    import requests

    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        size_ok = expected_size is None or target.stat().st_size == expected_size
        hash_ok = expected_sha256 is None or sha256_file(target) == expected_sha256
        if size_ok and hash_ok:
            print("✓ cached + verified:", target.name)
            return
        target.unlink()

    # Do NOT forward the bearer token to the object-storage host.
    r = requests.get(
        url,
        headers={"Authorization": f"Bearer {token}", "Accept": "application/octet-stream"},
        allow_redirects=False,
        timeout=60,
    )
    if r.status_code == 401:
        raise RuntimeError("Dryad token was rejected or expired.")
    if r.status_code not in {301, 302, 303, 307, 308}:
        raise RuntimeError(
            f"Dryad download endpoint returned HTTP {r.status_code}; "
            f"expected a redirect to object storage. Body: {r.text[:300]}"
        )
    storage_url = r.headers.get("Location")
    if not storage_url:
        raise RuntimeError("Dryad download response contained no Location redirect.")

    part = target.with_suffix(target.suffix + ".part")
    headers = {}
    mode = "wb"
    if part.exists() and part.stat().st_size > 0:
        headers["Range"] = f"bytes={part.stat().st_size}-"
        mode = "ab"

    with requests.get(storage_url, headers=headers, stream=True, timeout=120) as sr:
        # If Range was ignored, start from scratch instead of duplicating bytes.
        if mode == "ab" and sr.status_code == 200:
            mode = "wb"
        if sr.status_code not in {200, 206}:
            raise RuntimeError(f"Dryad storage download failed: HTTP {sr.status_code}")
        with part.open(mode) as f:
            for chunk in sr.iter_content(chunk_size=8 << 20):
                if chunk:
                    f.write(chunk)

    if expected_size is not None and part.stat().st_size != expected_size:
        actual = part.stat().st_size
        part.unlink(missing_ok=True)
        raise RuntimeError(
            f"Dryad byte-size mismatch for {target.name}: {actual} != {expected_size}"
        )
    if expected_sha256 is not None:
        actual = sha256_file(part)
        if actual != expected_sha256:
            part.unlink(missing_ok=True)
            raise RuntimeError(
                f"Dryad SHA-256 mismatch for {target.name}: {actual} != {expected_sha256}"
            )
    os.replace(part, target)


def sha256_file(path: Path, chunk: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def make_authenticated_dryad(base: dict):
    def dryad_full_dataset(root: Path, dataset: str, doi: str) -> dict:
        token = dryad_token()
        raw = root / "raw" / dataset
        files_dir = raw / "files"
        files_dir.mkdir(parents=True, exist_ok=True)
        unavailable = raw / "SOURCE_UNAVAILABLE.json"

        try:
            entries, provenance = _dryad_files_authenticated(doi, token)
        except Exception as exc:
            base["atomic_json"](
                unavailable,
                {
                    "doi": doi,
                    "checked_at_utc": now(),
                    "reason": "dryad_metadata_unavailable_even_with_authenticated_api",
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )
            return {
                "status": "blocked",
                "reason": "published_files_unavailable",
                "doi": doi,
                "outputs": [{"path": str(unavailable)}],
            }

        receipts = []
        for entry in entries:
            rel = Path(str(entry.get("path") or ""))
            if not rel.name or rel.is_absolute() or ".." in rel.parts:
                raise RuntimeError(f"Unsafe/invalid Dryad path: {rel}")

            href = (((entry.get("_links") or {}).get("stash:download") or {}).get("href"))
            if not href:
                self_href = (((entry.get("_links") or {}).get("self") or {}).get("href"))
                if not self_href:
                    raise RuntimeError(f"Dryad file has no download/self link: {rel}")
                fid = str(self_href).rstrip("/").rsplit("/", 1)[-1]
                href = f"/api/v2/files/{fid}/download"

            url = href if str(href).startswith("http") else "https://datadryad.org" + str(href)
            digest_value = str(entry.get("digest") or "").lower() or None
            digest_type = str(entry.get("digestType") or "").lower()
            expected_sha256 = digest_value if digest_type in {"sha-256", "sha256"} else None
            expected_size = int(entry["size"]) if entry.get("size") is not None else None

            target = files_dir / rel
            _download_dryad_file(
                url,
                target,
                token,
                expected_size=expected_size,
                expected_sha256=expected_sha256,
            )
            receipts.append(
                {
                    "path": str(rel),
                    "bytes": target.stat().st_size,
                    "sha256": sha256_file(target),
                    "dryad_digest": digest_value,
                    "dryad_digest_type": entry.get("digestType"),
                    "api_download_url": url,
                }
            )

        unavailable.unlink(missing_ok=True)
        receipt = raw / "acquisition.json"
        base["atomic_json"](
            receipt,
            {
                "doi": doi,
                "source": "Dryad",
                "method": "authenticated_dryad_api_v2",
                "acquired_at_utc": now(),
                **provenance,
                "files": receipts,
            },
        )
        return {
            "outputs": [{"path": str(receipt)}, {"path": str(files_dir)}],
            "n_files": len(receipts),
            "raw_dir": str(files_dir),
            "download_method": "authenticated_dryad_api_v2",
        }

    return dryad_full_dataset


def main() -> int:
    source = urllib.request.urlopen(BASE_RUNNER_URL).read()
    print("Base campaign runner commit:", BASE_RUNNER_COMMIT)
    print("Base campaign runner SHA-256:", hashlib.sha256(source).hexdigest())
    print("Dryad authenticated-download patch: active")

    ns = {"__name__": "cardioscore_campaign_base_v2"}
    exec(compile(source, "run_validation_campaign_colab.py", "exec"), ns)

    # Reuse all completed v2/v3 work. Retry only the Dryad-dependent stages.
    ns["FORCE"].update(DRYAD_RETRY_STAGES)
    ns["dryad_full_dataset"] = make_authenticated_dryad(ns)

    return int(ns["main"]())


if __name__ == "__main__":
    raise SystemExit(main())

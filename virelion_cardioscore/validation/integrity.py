"""Immutable external-asset provenance and archive inventory helpers."""
from __future__ import annotations

import hashlib
import json
import tarfile
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any


@dataclass(frozen=True)
class InventoryEntry:
    path: str
    size_bytes: int
    is_file: bool
    sha256: str | None = None


@dataclass(frozen=True)
class AssetManifest:
    source_id: str
    source_url: str
    acquisition_date: str
    filename: str
    byte_size: int
    sha256: str
    evidence_level: str = "raw_mea_dataset"
    inventory: tuple[InventoryEntry, ...] = field(default_factory=tuple)
    parser_version: str = ""
    schema_version: str = ""
    notes: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["inventory"] = [asdict(item) for item in self.inventory]
        payload["notes"] = list(self.notes)
        return payload

    def write_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_sha256(path: str | Path, expected_sha256: str) -> bool:
    expected = expected_sha256.strip().lower()
    if len(expected) != 64 or any(ch not in "0123456789abcdef" for ch in expected):
        raise ValueError("expected_sha256 must be a 64-character hexadecimal SHA-256 digest")
    return sha256_file(path) == expected


def _safe_member_name(name: str) -> str:
    candidate = PurePosixPath(name)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"Unsafe archive member path: {name!r}")
    return str(candidate)


def _hash_zip_member(archive: zipfile.ZipFile, info: zipfile.ZipInfo) -> str:
    digest = hashlib.sha256()
    with archive.open(info, "r") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_tar_member(archive: tarfile.TarFile, member: tarfile.TarInfo) -> str | None:
    if not member.isfile():
        return None
    stream = archive.extractfile(member)
    if stream is None:
        return None
    digest = hashlib.sha256()
    with stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inventory_archive(path: str | Path) -> list[InventoryEntry]:
    """Inventory ZIP/TAR members without extracting or modifying the archive."""
    source = Path(path)
    if zipfile.is_zipfile(source):
        with zipfile.ZipFile(source) as archive:
            entries = []
            for info in archive.infolist():
                safe = _safe_member_name(info.filename)
                is_file = not info.is_dir()
                entries.append(
                    InventoryEntry(
                        path=safe,
                        size_bytes=info.file_size,
                        is_file=is_file,
                        sha256=_hash_zip_member(archive, info) if is_file else None,
                    )
                )
            return entries
    if tarfile.is_tarfile(source):
        with tarfile.open(source, "r:*") as archive:
            entries = []
            for member in archive.getmembers():
                safe = _safe_member_name(member.name)
                entries.append(
                    InventoryEntry(
                        path=safe,
                        size_bytes=int(member.size),
                        is_file=member.isfile(),
                        sha256=_hash_tar_member(archive, member),
                    )
                )
            return entries
    raise ValueError(f"Unsupported archive format: {source}")


def build_asset_manifest(
    path: str | Path,
    *,
    source_id: str,
    source_url: str,
    acquisition_date: str,
    evidence_level: str = "raw_mea_dataset",
    parser_version: str = "",
    schema_version: str = "",
    notes: list[str] | None = None,
) -> AssetManifest:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    inventory = tuple(inventory_archive(source)) if zipfile.is_zipfile(source) or tarfile.is_tarfile(source) else tuple()
    return AssetManifest(
        source_id=source_id,
        source_url=source_url,
        acquisition_date=acquisition_date,
        filename=source.name,
        byte_size=source.stat().st_size,
        sha256=sha256_file(source),
        evidence_level=evidence_level,
        inventory=inventory,
        parser_version=parser_version,
        schema_version=schema_version,
        notes=tuple(notes or []),
    )

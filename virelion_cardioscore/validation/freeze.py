"""Pre-registration and freeze-manifest checks for locked external validation.

A locked external result is only meaningful if the analysis plan and the scoring configuration were
fixed *before* the external data were scored. This module makes that commitment explicit and
auditable:

* ``validation/preregistration.yaml`` states the primary analysis, success criteria and the
  scoring choices being committed to. Every field that still needs a human decision is ``null`` or a
  placeholder, and freezing is refused until none remain.
* ``validation/freeze_manifest.json`` records the SHA-256 of the pre-registration and of the frozen
  configuration files, plus the CardioScore package commit they belong to.
* ``verify_freeze`` re-derives all of that at run time, so a changed weight, threshold, dropout rule
  or analysis plan makes the locked stage refuse to produce a headline result.

Both files are committed to git, so the commit history (not this code) is the evidence of *when* the
plan was frozen. This module cannot prove that nobody looked at the external data earlier.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import yaml

FREEZE_SCHEMA_VERSION = 1
FROZEN_CONFIG_FILES = ("default.yaml", "cipa_endpoints.yaml")
PLACEHOLDER_TOKENS = ("TODO", "REPLACE_ME", "TBD")
_SHA1 = re.compile(r"^[0-9a-f]{40}$")


class FreezeError(ValueError):
    """Raised when a pre-registration cannot be frozen or a freeze cannot be verified."""


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path) -> str:
    return sha256_bytes(Path(path).read_bytes())


def parse_preregistration(payload: bytes | str) -> dict[str, Any]:
    data = yaml.safe_load(payload) or {}
    if not isinstance(data, dict):
        raise FreezeError("Pre-registration must be a YAML mapping.")
    return data


def unresolved_fields(node: Any, prefix: str = "") -> list[str]:
    """Dotted paths of values that are still ``null`` or contain a placeholder token."""
    found: list[str] = []
    if isinstance(node, Mapping):
        for key, value in node.items():
            found.extend(unresolved_fields(value, f"{prefix}.{key}" if prefix else str(key)))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            found.extend(unresolved_fields(value, f"{prefix}[{index}]"))
    elif node is None or (isinstance(node, str) and any(token in node for token in PLACEHOLDER_TOKENS)):
        found.append(prefix or "<root>")
    return found


def _config_commitments(prereg: Mapping[str, Any]) -> Mapping[str, Any]:
    commitments = prereg.get("scoring_commitments")
    if not isinstance(commitments, Mapping):
        raise FreezeError("Pre-registration must contain a scoring_commitments mapping.")
    return commitments


def consistency_problems(prereg: Mapping[str, Any], config_dir: str | Path) -> list[str]:
    """Compare the values the pre-registration commits to with the shipped configuration."""
    config_dir = Path(config_dir)
    default = yaml.safe_load((config_dir / "default.yaml").read_text(encoding="utf-8")) or {}
    endpoints = yaml.safe_load((config_dir / "cipa_endpoints.yaml").read_text(encoding="utf-8")) or {}
    commitments = _config_commitments(prereg)
    qc = default.get("quality_control", {})
    concentration = default.get("concentration_response", {})
    scoring = default.get("scoring", {})
    expected = {
        "concentration_aggregation": concentration.get("concentration_aggregation"),
        "min_concentrations": concentration.get("min_concentrations"),
        "require_min_concentrations_for_scoring": concentration.get("require_min_concentrations_for_scoring"),
        "informative_dropout_min_fraction": qc.get("informative_dropout_min_fraction"),
        "informative_dropout_min_excess": qc.get("informative_dropout_min_excess"),
        "low_threshold": scoring.get("low_threshold"),
        "moderate_threshold": scoring.get("moderate_threshold"),
        "normalize_by_vehicle": scoring.get("normalize_by_vehicle"),
        "dose_response_weight": scoring.get("dose_response_weight"),
    }
    problems = []
    for key, actual in expected.items():
        if key not in commitments:
            problems.append(f"scoring_commitments.{key} is missing (configuration has {actual!r}).")
        elif commitments[key] != actual:
            problems.append(
                f"scoring_commitments.{key}={commitments[key]!r} but the frozen configuration has {actual!r}."
            )
    weights = commitments.get("endpoint_weights")
    actual_weights = {name: meta.get("weight") for name, meta in (endpoints.get("endpoints") or {}).items()}
    if weights != actual_weights:
        problems.append(f"scoring_commitments.endpoint_weights={weights!r} but cipa_endpoints.yaml has {actual_weights!r}.")
    return problems


def _sha1_ok(value: Any) -> bool:
    return isinstance(value, str) and bool(_SHA1.match(value))


def build_freeze_manifest(
    *,
    prereg_path: str | Path,
    config_dir: str | Path,
    package_sha: str,
    frozen_at_utc: str | None = None,
) -> dict[str, Any]:
    """Create the freeze manifest, refusing while anything is unresolved or inconsistent."""
    prereg_path = Path(prereg_path)
    config_dir = Path(config_dir)
    prereg = parse_preregistration(prereg_path.read_bytes())
    problems: list[str] = []
    if prereg.get("status") != "frozen":
        problems.append("Pre-registration status must be 'frozen' (currently " f"{prereg.get('status')!r}).")
    problems.extend(f"Unresolved field: {path}" for path in unresolved_fields(prereg))
    problems.extend(consistency_problems(prereg, config_dir))
    if not _sha1_ok(package_sha):
        problems.append("package_sha must be a full 40-character lowercase commit SHA.")
    if problems:
        raise FreezeError("Cannot freeze:\n- " + "\n- ".join(problems))
    return {
        "schema_version": FREEZE_SCHEMA_VERSION,
        "status": "frozen",
        "validation_version": prereg["validation_version"],
        "frozen_at_utc": frozen_at_utc or datetime.now(timezone.utc).isoformat(),
        "package_sha": package_sha,
        "preregistration_sha256": sha256_file(prereg_path),
        "config_sha256": {name: sha256_file(config_dir / name) for name in FROZEN_CONFIG_FILES},
    }


def verify_freeze(
    manifest: Mapping[str, Any],
    *,
    prereg_bytes: bytes,
    config_dir: str | Path,
    package_sha: str,
) -> list[str]:
    """Return every reason the freeze does not hold (an empty list means it verified)."""
    config_dir = Path(config_dir)
    problems: list[str] = []
    if manifest.get("schema_version") != FREEZE_SCHEMA_VERSION:
        problems.append(f"Unsupported freeze schema_version {manifest.get('schema_version')!r}.")
    if manifest.get("status") != "frozen":
        problems.append("Freeze manifest status is not 'frozen'.")
    if manifest.get("package_sha") != package_sha:
        problems.append(
            f"Freeze manifest is for package {manifest.get('package_sha')!r}, but this run uses {package_sha!r}."
        )
    if manifest.get("preregistration_sha256") != sha256_bytes(prereg_bytes):
        problems.append("Pre-registration changed after it was frozen (SHA-256 mismatch).")
    frozen_hashes = manifest.get("config_sha256")
    if not isinstance(frozen_hashes, Mapping):
        problems.append("Freeze manifest has no config_sha256 mapping.")
        frozen_hashes = {}
    for name in FROZEN_CONFIG_FILES:
        path = config_dir / name
        if not path.is_file():
            problems.append(f"Installed package has no {name}.")
        elif frozen_hashes.get(name) != sha256_file(path):
            problems.append(f"{name} differs from the frozen configuration (SHA-256 mismatch).")
    try:
        prereg = parse_preregistration(prereg_bytes)
    except (FreezeError, yaml.YAMLError) as exc:
        return [*problems, f"Pre-registration cannot be parsed: {exc}"]
    if prereg.get("status") != "frozen":
        problems.append("Pre-registration status is not 'frozen'.")
    problems.extend(f"Unresolved field: {path}" for path in unresolved_fields(prereg))
    try:
        problems.extend(consistency_problems(prereg, config_dir))
    except (FreezeError, OSError, yaml.YAMLError) as exc:
        problems.append(f"Consistency check failed: {exc}")
    return problems


def load_manifest(payload: bytes | str) -> dict[str, Any]:
    data = json.loads(payload)
    if not isinstance(data, dict):
        raise FreezeError("Freeze manifest must be a JSON object.")
    return data
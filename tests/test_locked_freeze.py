"""Pre-registration, freeze verification and the locked external stage's primary analysis."""

from __future__ import annotations

import copy
import importlib.util
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from virelion_cardioscore.validation.freeze import (
    FreezeError,
    build_freeze_manifest,
    consistency_problems,
    parse_preregistration,
    sha256_file,
    unresolved_fields,
    verify_freeze,
)
from virelion_cardioscore.validation.metrics import binary_auroc_bootstrap, primary_outcome

ROOT = Path(__file__).resolve().parents[1]
PREREG = ROOT / "validation" / "preregistration.yaml"
CONFIG_DIR = ROOT / "virelion_cardioscore" / "config"
PACKAGE_SHA = "a" * 40

_spec = importlib.util.spec_from_file_location(
    "cardioscore_colab_runner_locked", ROOT / "scripts" / "validation" / "run_colab_validation.py"
)
runner = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(runner)


def resolved_prereg() -> dict:
    prereg = parse_preregistration(PREREG.read_bytes())
    prereg["status"] = "frozen"
    prereg["declarations"]["development_history"].update(
        confirmed_by="Test Owner", confirmed_on="2026-09-21"
    )
    prereg["declarations"]["label_blind_development"].update(
        confirmed_by="Test Owner", confirmed_on="2026-09-21"
    )
    prereg["primary_dataset"].update(
        doi="10.0000/example",
        source_url="https://example.org/data",
        raw_traces_publicly_available=True,
    )
    return prereg


@pytest.fixture()
def frozen(tmp_path):
    """A resolved pre-registration, a copy of the shipped config, and the matching freeze manifest."""
    config_dir = tmp_path / "config"
    shutil.copytree(CONFIG_DIR, config_dir)
    prereg_path = tmp_path / "preregistration.yaml"
    prereg_path.write_text(yaml.safe_dump(resolved_prereg(), sort_keys=False), encoding="utf-8")
    manifest = build_freeze_manifest(
        prereg_path=prereg_path, config_dir=config_dir, package_sha=PACKAGE_SHA
    )
    return {
        "dir": tmp_path,
        "config_dir": config_dir,
        "prereg_path": prereg_path,
        "manifest": manifest,
    }


def verify(frozen, **overrides):
    arguments = {
        "prereg_bytes": frozen["prereg_path"].read_bytes(),
        "config_dir": frozen["config_dir"],
        "package_sha": PACKAGE_SHA,
    }
    arguments.update(overrides)
    return verify_freeze(frozen["manifest"], **arguments)


# --- the shipped draft ------------------------------------------------------------------------


def test_shipped_draft_matches_the_shipped_configuration_but_cannot_be_frozen(tmp_path):
    prereg = parse_preregistration(PREREG.read_bytes())
    assert consistency_problems(prereg, CONFIG_DIR) == []
    unresolved = unresolved_fields(prereg)
    assert (
        "primary_dataset.doi" in unresolved
        and "declarations.development_history.confirmed_by" in unresolved
    )
    with pytest.raises(FreezeError, match="Unresolved field: primary_dataset.doi"):
        build_freeze_manifest(prereg_path=PREREG, config_dir=CONFIG_DIR, package_sha=PACKAGE_SHA)


def test_freeze_requires_status_frozen_and_a_full_sha(tmp_path):
    prereg = resolved_prereg()
    prereg["status"] = "draft"
    path = tmp_path / "p.yaml"
    path.write_text(yaml.safe_dump(prereg), encoding="utf-8")
    with pytest.raises(FreezeError, match="status must be 'frozen'"):
        build_freeze_manifest(prereg_path=path, config_dir=CONFIG_DIR, package_sha=PACKAGE_SHA)
    prereg["status"] = "frozen"
    path.write_text(yaml.safe_dump(prereg), encoding="utf-8")
    with pytest.raises(FreezeError, match="40-character"):
        build_freeze_manifest(prereg_path=path, config_dir=CONFIG_DIR, package_sha="main")


def test_unresolved_fields_finds_nulls_and_placeholders():
    assert unresolved_fields(
        {"a": {"b": None, "c": "ok", "d": ["x", "REPLACE_ME"]}, "e": None}
    ) == [
        "a.b",
        "a.d[1]",
        "e",
    ]


# --- verification ------------------------------------------------------------------------------


def test_a_resolved_freeze_verifies(frozen):
    assert verify(frozen) == []
    assert frozen["manifest"]["package_sha"] == PACKAGE_SHA


def test_editing_the_preregistration_after_freezing_is_detected(frozen):
    edited = yaml.safe_load(frozen["prereg_path"].read_text(encoding="utf-8"))
    edited["primary_analysis"]["outcome_rule"]["success"]["auroc_min"] = 0.5
    problems = verify(frozen, prereg_bytes=yaml.safe_dump(edited).encode("utf-8"))
    assert any("Pre-registration changed after it was frozen" in problem for problem in problems)


@pytest.mark.parametrize(
    ("file_name", "old", "new"),
    [
        ("default.yaml", "  low_threshold: 0.30", "  low_threshold: 0.25"),
        ("cipa_endpoints.yaml", "    weight: 0.30", "    weight: 0.35"),
        (
            "default.yaml",
            "informative_dropout_min_fraction: 0.5",
            "informative_dropout_min_fraction: 0.9",
        ),
    ],
)
def test_changed_scoring_configuration_breaks_the_freeze(frozen, file_name, old, new):
    path = frozen["config_dir"] / file_name
    text = path.read_text(encoding="utf-8")
    assert old in text
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    problems = verify(frozen)
    assert any(
        f"{file_name} differs from the frozen configuration" in problem for problem in problems
    )
    assert any("scoring_commitments" in problem for problem in problems)


def test_wrong_package_commit_and_unfrozen_manifest_are_rejected(frozen):
    assert any("is for package" in problem for problem in verify(frozen, package_sha="b" * 40))
    frozen["manifest"] = {**frozen["manifest"], "status": "draft"}
    assert any("status is not 'frozen'" in problem for problem in verify(frozen))


def test_unresolved_field_reintroduced_after_freezing_is_rejected(frozen):
    prereg = yaml.safe_load(frozen["prereg_path"].read_text(encoding="utf-8"))
    prereg["primary_dataset"]["doi"] = None
    payload = yaml.safe_dump(prereg).encode("utf-8")
    problems = verify(frozen, prereg_bytes=payload)
    assert any("Unresolved field: primary_dataset.doi" in problem for problem in problems)


# --- primary analysis helpers --------------------------------------------------------------------


def test_auroc_bootstrap_is_deterministic_and_resamples_compounds():
    positive = [True] * 6 + [False] * 4
    scores = [0.9, 0.8, 0.7, 0.75, 0.6, 0.55, 0.5, 0.4, 0.3, 0.65]
    first = binary_auroc_bootstrap(positive, scores, n_bootstrap=500, seed=7)
    second = binary_auroc_bootstrap(positive, scores, n_bootstrap=500, seed=7)
    assert first == second
    assert (
        first["n"] == 10
        and first["n_positive"] == 6
        and 0 <= first["ci_lower"] <= first["auroc"] <= first["ci_upper"] <= 1
    )
    perfect = binary_auroc_bootstrap(
        [True, True, False, False], [0.9, 0.8, 0.2, 0.1], n_bootstrap=100, seed=1
    )
    assert perfect["auroc"] == 1.0


def test_auroc_bootstrap_rejects_degenerate_input():
    with pytest.raises(ValueError, match="both positive and negative"):
        binary_auroc_bootstrap([True, True], [0.1, 0.2], n_bootstrap=10, seed=1)
    with pytest.raises(ValueError, match="finite"):
        binary_auroc_bootstrap([True, False], [np.nan, 0.2], n_bootstrap=10, seed=1)


def test_primary_outcome_applies_the_preregistered_rule():
    rule = resolved_prereg()["primary_analysis"]["outcome_rule"]
    assert primary_outcome({"auroc": 0.85, "ci_lower": 0.65}, rule) == "success"
    assert primary_outcome({"auroc": 0.85, "ci_lower": 0.55}, rule) == "inconclusive"
    assert primary_outcome({"auroc": 0.75, "ci_lower": 0.52}, rule) == "inconclusive"
    assert primary_outcome({"auroc": 0.65, "ci_lower": 0.30}, rule) == "failure"
    assert primary_outcome({"auroc": 0.95, "ci_lower": 0.45}, rule) == "failure"


# --- runner: locating and verifying the freeze ----------------------------------------------------


def write_freeze_dir(frozen) -> Path:
    directory = frozen["dir"] / "freeze"
    directory.mkdir()
    shutil.copy(frozen["prereg_path"], directory / "preregistration.yaml")
    (directory / "freeze_manifest.json").write_text(
        json.dumps(frozen["manifest"]), encoding="utf-8"
    )
    return directory


def test_runner_fetches_freeze_files_from_the_pinned_commit(frozen):
    requested = []
    files = {
        "validation/preregistration.yaml": frozen["prereg_path"].read_bytes(),
        "validation/freeze_manifest.json": json.dumps(frozen["manifest"]).encode("utf-8"),
    }

    def fetch(url):
        requested.append(url)
        return files[url.split("/", 6)[-1]]

    revision = "c" * 40
    prereg_bytes, manifest, source = runner.load_freeze_files({}, fetch, revision)
    assert (
        manifest["package_sha"] == PACKAGE_SHA
        and prereg_bytes == files["validation/preregistration.yaml"]
    )
    assert source == f"github:{runner.REPO}@{revision}"
    assert all(f"/{revision}/validation/" in url for url in requested)


def test_runner_refuses_to_locate_a_freeze_when_unpinned():
    info, prereg = runner.verify_locked_freeze({}, None, "unpinned-main")
    assert info["verified"] is False and prereg is None
    assert "not SHA-pinned" in info["problems"][0]


def test_runner_verifies_a_local_override_and_reports_problems(frozen, monkeypatch):
    directory = write_freeze_dir(frozen)
    environ = {"CARDIOSCORE_FREEZE_DIR": str(directory)}
    # The runner verifies against the *installed* package configuration; make the freeze match it.
    manifest = build_freeze_manifest(
        prereg_path=frozen["prereg_path"], config_dir=CONFIG_DIR, package_sha=PACKAGE_SHA
    )
    (directory / "freeze_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    info, prereg = runner.verify_locked_freeze(environ, package_sha=PACKAGE_SHA)
    assert info["verified"] is True and info["source"].startswith("local_override:")
    assert prereg["primary_analysis"]["metric"] == "auroc"
    info, prereg = runner.verify_locked_freeze(environ, package_sha="b" * 40)
    assert info["verified"] is False and prereg is None
    assert any("is for package" in problem for problem in info["problems"])


def test_feature_schema_allows_missing_endpoints_only_on_no_signal_wells():
    from virelion_cardioscore.validation.manifest import validate_feature_schema

    frame = pd.DataFrame(compound_rows("X", 0.5, 0, dead_top=1))
    validate_feature_schema(
        frame
    )  # dead wells (no electrodes, no beats) may carry missing endpoints
    live = frame.index[(frame["concentration_uM"] == 1.0) & ~frame["vehicle"]][0]
    broken = frame.copy()
    broken.loc[live, "stv"] = np.nan
    with pytest.raises(ValueError, match="stv must be numeric and finite"):
        validate_feature_schema(broken)
    infinite = frame.copy()
    infinite.loc[frame.index[frame["n_electrodes"] == 0][0], "fpd_ms"] = np.inf
    with pytest.raises(ValueError, match="fpd_ms must be numeric and finite"):
        validate_feature_schema(infinite)


# --- runner: the locked stage end to end ---------------------------------------------------------

ENDPOINTS = ["fpd_ms", "beat_rate_bpm", "amplitude_uv", "stv", "triangulation_proxy"]
CONCENTRATIONS = [0.1, 1.0, 10.0, 100.0]
# compound -> (reference risk, effect scale at the top concentration)
COMPOUNDS = {
    "LOW_A": ("low", 0.00),
    "LOW_B": ("low", 0.03),
    "INT_A": ("intermediate", 0.9),
    "INT_B": ("intermediate", 1.3),
    "HIGH_A": ("high", 3.0),
    "HIGH_B": ("high", 4.0),
}


def compound_rows(name: str, top_effect: float, seed: int, *, dead_top: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(6):
        rows.append(
            {
                "compound": name, "well": f"{name}-V{i}", "concentration_uM": 0.0, "vehicle": True,
                "site": 1, "cell_type": "C", "concentration_index": 0,
                "fpd_ms": 400 + rng.normal(0, 4), "beat_rate_bpm": 60 + rng.normal(0, 1),
                "amplitude_uv": 200 + rng.normal(0, 3), "stv": 5 + rng.normal(0, 0.2),
                "triangulation_proxy": 1.0 + rng.normal(0, 0.01),
                "n_electrodes": 12, "noise_sd_uv": 8.0, "beat_detection_rate": 0.95,
            }
        )  # fmt: skip
    for index, concentration in enumerate(CONCENTRATIONS):
        effect = top_effect * (0.25, 0.5, 0.75, 1.0)[index]
        dead = index >= len(CONCENTRATIONS) - dead_top
        for replicate in range(4):
            row = {
                "compound": name, "well": f"{name}-T{index}{replicate}", "concentration_uM": concentration,
                "vehicle": False, "site": 1, "cell_type": "C", "concentration_index": index + 1,
                "fpd_ms": 400 * (1 + effect) + rng.normal(0, 4), "beat_rate_bpm": 60 + rng.normal(0, 1),
                "amplitude_uv": 200 * (1 - 0.2 * effect), "stv": 5 * (1 + effect),
                "triangulation_proxy": 1.0 * (1 + effect),
                "n_electrodes": 12, "noise_sd_uv": 8.0, "beat_detection_rate": 0.95,
            }  # fmt: skip
            if dead:
                row.update({column: np.nan for column in ENDPOINTS})
                row.update(n_electrodes=0, beat_detection_rate=0.0)
            rows.append(row)
    return rows


def build_locked_inputs(tmp_path: Path, *, dead: dict[str, int]) -> dict[str, Path]:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    rows = []
    for seed, (name, (_, top_effect)) in enumerate(COMPOUNDS.items()):
        rows.extend(compound_rows(name, top_effect, seed, dead_top=dead.get(name, 0)))
    features_path = tmp_path / "canonical_features.csv"
    pd.DataFrame(rows).to_csv(features_path, index=False)
    reference_path = tmp_path / "reference.csv"
    pd.DataFrame(
        {"compound": list(COMPOUNDS), "reference_risk": [risk for risk, _ in COMPOUNDS.values()]}
    ).to_csv(reference_path, index=False)
    source_path = input_dir / "source.bin"
    source_path.write_bytes(b"synthetic source")
    receipt_path = tmp_path / "source_receipt.json"
    receipt_path.write_text(
        json.dumps(
            {
                "sha256": sha256_file(source_path), "source_filename": "source.bin", "source_id": "synthetic",
                "source_url": "https://example.org/synthetic",
            }
        ),
        encoding="utf-8",
    )  # fmt: skip
    manifest_path = tmp_path / "derivation_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "verified_rebuild": True, "source_sha256": sha256_file(source_path),
                "canonical_features_sha256": sha256_file(features_path),
                "reference_sha256": sha256_file(reference_path), "config_sha256": runner.config_sha256(),
                "reference_provenance": {"source_url": "https://example.org/ref", "source_id": "ref"},
            }
        ),
        encoding="utf-8",
    )  # fmt: skip
    return {
        "assets": {
            "features": features_path, "reference": reference_path,
            "receipt": receipt_path, "manifest": manifest_path,
        },
        "input_dir": input_dir,
    }  # fmt: skip


def small_prereg() -> dict:
    prereg = copy.deepcopy(resolved_prereg())
    prereg["primary_analysis"]["confidence_interval"]["n_bootstrap"] = 300
    return prereg


def run_locked(tmp_path, *, dead):
    built = build_locked_inputs(tmp_path, dead=dead)
    results = tmp_path / "results"
    results.mkdir()
    info = {"verified": True, "source": "test"}
    stage = runner.run_locked_external(
        built["assets"], results, built["input_dir"], small_prereg(), info
    )
    payload = json.loads((results / "locked_external_validation.json").read_text(encoding="utf-8"))
    return stage, payload, results


def test_locked_stage_reports_the_preregistered_primary_analysis(tmp_path):
    stage, payload, results = run_locked(tmp_path, dead={})
    assert stage["status"] == "complete" and stage["n_excluded"] == 0
    primary = payload["primary_analysis"]
    assert primary["auroc"] == 1.0 and primary["outcome"] in {"success", "inconclusive"}
    assert stage["primary_auroc"] == 1.0 and payload["freeze"]["verified"] is True
    assert payload["informative_dropout_compounds"] == [] and payload["dropout_sensitivity"] is None
    assert (results / "locked_dropout_by_concentration.csv").is_file()
    assert (results / "locked_exclusions.csv").is_file()
    assert payload["metrics"]["n"] == 6


def test_locked_stage_flags_informative_dropout_and_runs_the_sensitivity_analysis(tmp_path):
    # HIGH_B loses its top concentration (cells stop beating) but is still scored from three concentrations.
    stage, payload, results = run_locked(tmp_path, dead={"HIGH_B": 1})
    assert stage["n_informative_dropout_compounds"] == 1
    assert payload["informative_dropout_compounds"] == ["HIGH_B"]
    sensitivity = payload["dropout_sensitivity"]
    assert sensitivity["compounds_reclassified_as_high"] == ["HIGH_B"]
    assert sensitivity["primary_analysis"]["auroc"] >= payload["primary_analysis"]["auroc"] - 1e-12
    dropout = pd.read_csv(results / "locked_dropout_by_concentration.csv")
    assert bool(
        dropout.loc[
            (dropout["compound"] == "HIGH_B") & (dropout["concentration_uM"] == 100.0),
            "informative_dropout",
        ].iloc[0]
    )


def test_locked_stage_blocks_and_explains_an_unscoreable_reference_compound(tmp_path):
    stage, payload, results = run_locked(tmp_path, dead={"HIGH_A": 2})
    assert stage["status"] == "blocked_by_exclusions" and stage["n_excluded"] == 1
    assert payload["metrics"] is None and payload["primary_analysis"] is None
    reason = payload["exclusion_reasons"]["HIGH_A"]
    assert (
        reason["reason"] == "insufficient_concentrations"
        and "informative dropout" in reason["detail"]
    )
    exclusions = pd.read_csv(results / "locked_exclusions.csv")
    assert list(exclusions["compound"]) == ["HIGH_A"]


def test_stale_package_pin_blocks_cleanly_instead_of_crashing(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "virelion_cardioscore.validation.freeze":
            raise ImportError("No module named 'virelion_cardioscore.validation.freeze'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    info, prereg = runner.verify_locked_freeze({}, None, "c" * 40)
    assert info["verified"] is False and prereg is None
    assert "Set PIN to a commit that contains" in info["problems"][0]

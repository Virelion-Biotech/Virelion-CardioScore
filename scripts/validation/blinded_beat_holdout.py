#!/usr/bin/env python3
"""Prepare an unscored temporal holdout; verify the frozen method before human scoring."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import yaml


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_tool(repo):
    spec = importlib.util.spec_from_file_location(
        "beat_review_tool", repo / "scripts/validation/pymea_beat_validation.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def method_hashes(repo):
    paths = sorted((repo / "virelion_cardioscore").rglob("*.py"))
    paths += sorted((repo / "virelion_cardioscore/config").glob("*.yaml"))
    paths += [repo / "scripts/validation/pymea_beat_validation.py"]
    return {str(p.relative_to(repo)): sha(p) for p in paths}


def prepare(args):
    repo, out = args.repo.resolve(), args.out.resolve()
    if out.exists():
        raise ValueError("Choose a new study directory; existing evidence is never overwritten.")
    tool = load_tool(repo)
    development = [json.loads(p.read_text()) for p in args.development_manifest]
    source_hashes = {m["source_sha256"] for m in development}
    if len(source_hashes) != 1:
        raise ValueError("Development manifests must describe this same source recording.")
    from virelion_cardioscore.io.mcs_ascii import sniff_layout

    layout = sniff_layout(args.source)
    intervals = []
    for m in development:
        intervals += [(w["start_s"], w["start_s"] + w["window_s"]) for w in m["windows"]]
        intervals += [(s, e) for _, s, e in m.get("excluded_regions", [])]
    exclusions = [
        (electrode, max(0., s - 2.), e + 2.)
        for electrode in layout.electrodes for s, e in intervals
    ]
    # This code/configuration freeze is written before sampling or viewing any new window.
    out.mkdir(parents=True)
    freeze = {
        "schema_version": 1, "status": "frozen",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "detector_commit": args.detector_commit,
        "preparation_script_sha256": sha(__file__),
        "method_files": method_hashes(repo),
        "development_manifest_hashes": {p.name: sha(p) for p in args.development_manifest},
        "expected_source_sha256": next(iter(source_hashes)),
        "seed": args.seed, "n_windows": args.n_windows, "window_s": 10.,
        "development_guard_s": 2., "exclude_development_times_across_all_electrodes": True,
        "new_windows_time_disjoint_across_all_electrodes": True,
        "sampling_uses_detector_or_quality_outputs": False,
        "primary_reference": "annotator_A.csv", "secondary_reference": "annotator_B.csv",
        "reference_standard": "blinded_independent_human_annotation",
        "evidence_scope": "within_recording_temporal_holdout_component_validation",
        "independent_biological_recording": False, "human_validation_complete": False,
        "criteria": {"tolerance_ms": 50, "f1_min": .95, "precision_min": .95,
                     "recall_min": .95, "timing_median_abs_error_ms_max": 10},
    }
    # Check the detector/filter/config snapshot against the published immutable revision.
    for name, digest in freeze["method_files"].items():
        if not name.startswith("virelion_cardioscore/"):
            continue
        published = subprocess.check_output(
            ["git", "show", args.detector_commit + ":" + name], cwd=repo
        )
        if hashlib.sha256(published).hexdigest() != digest:
            raise ValueError("Method differs from published frozen revision: " + name)
    (out / "code_freeze.json").write_text(json.dumps(freeze, indent=2) + "\n")
    criteria = {"schema_version": 1, "status": "frozen",
                "validation_version": "pymea-beat-v2-temporal-holdout",
                "validation_code_sha": args.detector_commit,
                "reference_standard": freeze["reference_standard"],
                "evidence_scope": freeze["evidence_scope"],
                "beat_detection": {"status": "frozen", **freeze["criteria"]}}
    (out / "criteria_v2.yaml").write_text(yaml.safe_dump(criteria, sort_keys=False))
    review = out / "reviewer_kit"
    manifest = tool.make_sheet(
        args.source, review, n_windows=args.n_windows, window_s=10., seed=args.seed,
        duration_s=None, exclude=exclusions, disjoint_time_windows=True,
    )
    if manifest["source_sha256"] != freeze["expected_source_sha256"]:
        raise ValueError("Source hash differs from prior development source; do not score.")
    template = (review / "annotation_sheet.csv").read_bytes()
    for label in ("A", "B"):
        (review / ("annotator_" + label + ".csv")).write_bytes(template)
    (review / "README.md").write_text(
        "# Independent human beat review\n\n"
        "Two reviewers work separately using annotator_A.csv and annotator_B.csv. "
        "Inspect only the raw PNGs or cached raw arrays; do not run a detector or view "
        "development predictions before locking your labels. Every plot is unfiltered voltage.\n\n"
        "Enter beat times in seconds relative to window start, separated by semicolons. "
        "Mark depolarization events rather than every waveform extremum. "
        "Use unsure=1 when the trace cannot support a defensible label; do not guess. "
        "Set done=1 only after reviewing a window. An empty beat_times_s with done=1 "
        "and unsure=0 explicitly means no beats. Enter your name in annotator. "
        "Keep every row and retain window identifiers. Do not copy the other reviewer's labels.\n\n"
        "Annotator A is the prespecified primary reference; B is a secondary comparison. "
        "Disagreements and uncertain windows must be reported. These windows are from "
        "one previously used recording, so they do not provide independent biological replication.\n"
    )
    fixed = [review / "sheet_manifest.json", out / "code_freeze.json", out / "criteria_v2.yaml"]
    fixed += list((review / "windows").glob("*.npy")) + list((review / "plots").glob("*.png"))
    receipt = {"files": {str(p.relative_to(out)): sha(p) for p in fixed},
               "annotations_initially_blank": True, "predictions_generated": False}
    (out / "study_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"prepared": args.n_windows, "predictions_generated": False,
                      "scope": freeze["evidence_scope"]}))


def score(args):
    repo, out = args.repo.resolve(), args.study.resolve()
    freeze = json.loads((out / "code_freeze.json").read_text())
    receipt = json.loads((out / "study_receipt.json").read_text())
    for name, digest in receipt["files"].items():
        if sha(out / name) != digest:
            raise ValueError("Frozen study file changed: " + name)
    if method_hashes(repo) != freeze["method_files"]:
        raise ValueError("Method changed since sampling; restore the frozen code before scoring.")
    annotations = [out / "reviewer_kit" / ("annotator_" + label + ".csv") for label in ("A", "B")]
    for path in annotations:
        rows = list(csv.DictReader(path.open(newline="")))
        if not rows or any(r["done"].strip() != "1" or not r["annotator"].strip() for r in rows):
            raise ValueError("All rows need completed human review and reviewer identity: " + path.name)
    result = load_tool(repo).score(
        out / "reviewer_kit", annotations, .05, out / "criteria_v2.yaml",
        load_tool(repo).FilterConfig(),
    )
    result["evidence_scope"] = freeze["evidence_scope"]
    result["independent_biological_recording"] = False
    result["annotation_sha256"] = {p.name: sha(p) for p in annotations}
    (out / "human_results.json").write_text(json.dumps(result, indent=2, default=float) + "\n")
    print("Wrote human_results.json; interpret within-recording component performance only.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--repo", type=Path, required=True)
    prep.add_argument("--source", type=Path, required=True)
    prep.add_argument("--development-manifest", type=Path, action="append", required=True)
    prep.add_argument("--out", type=Path, required=True)
    prep.add_argument("--detector-commit", required=True)
    prep.add_argument("--seed", type=int, default=20261020)
    prep.add_argument("--n-windows", type=int, default=30)
    scoring = sub.add_parser("score")
    scoring.add_argument("--repo", type=Path, required=True)
    scoring.add_argument("--study", type=Path, required=True)
    args = parser.parse_args()
    prepare(args) if args.command == "prepare" else score(args)


if __name__ == "__main__":
    main()

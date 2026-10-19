#!/usr/bin/env python3
"""Fail-closed interpretation of CardioScore v6 public adapter challenges.

This script does not modify detector settings, endpoint definitions, weights, or risk labels.
It converts exploratory Ballouz and SiMulTox outputs into an explicit evidence-readiness audit.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

MIN_BEAT_DETECTION_RATE = 0.70
MIN_BEATS = 2


def _bool(x) -> bool:
    return bool(x) if not pd.isna(x) else False


def audit_ballouz(results: Path) -> dict:
    summary = json.loads((results / "ballouz_vandenberg_2021" / "ballouz_exploratory_summary.json").read_text())
    paired = pd.read_csv(results / "ballouz_vandenberg_2021" / "ballouz_exploratory_paired_effects.csv")
    per_plate = []
    for plate, g in paired.groupby("plate"):
        per_plate.append({
            "plate": int(plate),
            "n_paired_wells": int(len(g)),
            "median_cis_minus_baseline_ms": float(g["cis_minus_baseline_ms"].median()),
            "fraction_cis_prolonged": float(g["cis_prolonged"].astype(bool).mean()),
            "median_nif_minus_cis_ms": float(g["nif_minus_cis_ms"].median()),
            "fraction_nif_reversed_from_cis": float(g["nif_reversed_from_cis"].astype(bool).mean()),
            "fraction_nif_closer_to_baseline": float(g["nif_closer_to_baseline"].astype(bool).mean()),
        })

    directional_screen = {
        "cis_median_is_prolongation": float(summary["median_cis_minus_baseline_ms"]) > 0,
        "cis_majority_of_paired_wells_prolonged": float(summary["fraction_cis_prolonged"]) > 0.5,
        "nif_median_is_reversal": float(summary["median_nif_minus_cis_ms"]) < 0,
        "nif_majority_of_paired_wells_reversed": float(summary["fraction_nif_reversed_from_cis"]) > 0.5,
        "nif_majority_closer_to_baseline": float(summary["fraction_nif_closer_to_baseline"]) > 0.5,
    }
    recovered = all(directional_screen.values())
    return {
        "dataset": "ballouz_vandenberg_2021",
        "evidence_role": "exploratory_source_adapter_challenge_only",
        "n_paired_wells": int(summary["n_paired_wells_with_valid_fpd_all_conditions"]),
        "directional_screen": directional_screen,
        "screen_result": "directional_pattern_recovered" if recovered else "directional_pattern_not_robustly_recovered",
        "per_plate": per_plate,
        "guardrail": (
            "The >50% majority checks are descriptive audit screens, not preregistered inferential thresholds. "
            "Failure means the current exploratory FPD picker is not validated by this challenge; it does not refute the source biology."
        ),
    }


def _qc_row(row: pd.Series) -> dict:
    rate = float(row["beat_detection_rate"])
    n_beats = int(row["n_beats"])
    passed = rate >= MIN_BEAT_DETECTION_RATE and n_beats >= MIN_BEATS
    return {
        "condition": str(row["condition"]),
        "beat_detection_rate": rate,
        "n_beats": n_beats,
        "qc_pass": passed,
        "qc_rule": f"beat_detection_rate >= {MIN_BEAT_DETECTION_RATE:.2f} and n_beats >= {MIN_BEATS}",
    }


def audit_simultox(results: Path) -> dict:
    single = pd.read_csv(results / "simultox_2024" / "simultox_singlewell_cardioscore_features.csv")
    six = pd.read_csv(results / "simultox_2024" / "simultox_6well_control_channel_features.csv")
    qc = [_qc_row(row) for _, row in single.iterrows()]
    by_condition = {item["condition"]: item for item in qc}
    both_pass = all(by_condition.get(name, {}).get("qc_pass", False) for name in ("Control", "E-4031_1uM"))

    endpoint_differences = None
    if both_pass:
        indexed = single.set_index("condition")
        c, t = indexed.loc["Control"], indexed.loc["E-4031_1uM"]
        endpoint_differences = {
            key: (None if pd.isna(c[key]) or pd.isna(t[key]) else float(t[key] - c[key]))
            for key in ("fpd_ms", "beat_rate_bpm", "amplitude_uv", "stv_ibi_variability", "triangulation_proxy")
        }

    six_qc = (six["beat_detection_rate"] >= MIN_BEAT_DETECTION_RATE) & (six["n_beats"] >= MIN_BEATS)
    return {
        "dataset": "simultox_2024",
        "evidence_role": "format_and_signal_compatibility_challenge_not_independent_risk_validation",
        "singlewell_qc": qc,
        "both_conditions_endpoint_interpretable": bool(both_pass),
        "qc_gated_endpoint_differences": endpoint_differences,
        "sixwell_control": {
            "n_channels": int(len(six)),
            "n_channels_passing_signal_qc": int(six_qc.sum()),
            "fraction_channels_passing_signal_qc": float(six_qc.mean()),
            "median_beat_detection_rate_all_channels": float(six["beat_detection_rate"].median()),
        },
        "screen_result": (
            "endpoint_comparison_descriptively_available" if both_pass
            else "treated_condition_not_endpoint_interpretable_under_signal_qc"
        ),
        "guardrail": (
            "When a condition fails signal QC, FPD, triangulation, STV, amplitude and beat-rate differences are not used as pharmacology evidence. "
            "With one recording per condition, loss of detected beating cannot distinguish biological cessation/arrhythmia from detector incompatibility."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=Path, help="Campaign results directory")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    audit = {
        "schema_version": 1,
        "status": "complete",
        "ballouz": audit_ballouz(args.results_dir),
        "simultox": audit_simultox(args.results_dir),
        "overall_guardrail": "These adapter challenges do not constitute clinical or regulatory validation and do not alter the frozen CardioScore scoring model.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

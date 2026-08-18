#!/usr/bin/env python3
"""Audit the user-supplied processed Blinova 2018 workbook without modifying it.

This script deliberately performs validation/diagnostics only. It does not rename
or repair source values. Known aliases/anomalies are reported explicitly.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

EXPECTED_COLUMNS = [
    "Drug_Name", "Cell_type", "risk", "Platform", "Type_of_EADs",
    "conc", "EAD", "ddFPDc", "site",
]
KNOWN_DRUG_ALIAS = {"D,l,Sotalol": "D,l Sotalol"}
EXPECTED_RISK = {"L", "M", "H"}
PUBLISHED_PLATFORMS = {"AXN", "CLY", "ECR", "AMD", "MCS"}
EXPECTED_CELL_TYPES = {"CDI", "AXG"}
EXPECTED_SITES = set(range(1, 11))
EXPECTED_CONCENTRATIONS = {1, 2, 3, 4}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("xlsx", type=Path)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    df = pd.read_excel(args.xlsx)
    issues: list[dict[str, object]] = []

    if list(df.columns) != EXPECTED_COLUMNS:
        issues.append({"code": "COLUMN_SCHEMA", "columns": list(df.columns)})

    raw_drugs = set(df["Drug_Name"].dropna().astype(str))
    if "D,l,Sotalol" in raw_drugs:
        issues.append({
            "code": "KNOWN_DRUG_ALIAS",
            "raw_value": "D,l,Sotalol",
            "canonical_candidate": "D,l Sotalol",
            "action": "report_only",
        })

    missing_risk = df["risk"].isna()
    if missing_risk.any():
        issues.append({
            "code": "MISSING_RISK",
            "rows": int(missing_risk.sum()),
            "records": df.loc[missing_risk].to_dict(orient="records"),
            "action": "report_only",
        })

    unexpected_platforms = sorted(
        set(df["Platform"].dropna().astype(str)) - PUBLISHED_PLATFORMS
    )
    if unexpected_platforms:
        issues.append({
            "code": "UNEXPECTED_PLATFORM_CODE",
            "values": unexpected_platforms,
            "published_platforms": sorted(PUBLISHED_PLATFORMS),
            "action": "report_only",
        })

    ead_missing_ddfpd = int(df.loc[df["EAD"].eq(1), "ddFPDc"].isna().sum())
    if ead_missing_ddfpd:
        issues.append({
            "code": "EAD_WITH_MISSING_DDFPD",
            "rows": ead_missing_ddfpd,
            "note": (
                "Paper states ddFPDc/APD90c is not calculated when >=50% wells are "
                "arrhythmic; this workbook does not expose the full threshold "
                "calculation directly."
            ),
            "action": "retain_missing_and_audit",
        })

    report = {
        "source_file": str(args.xlsx),
        "sha256": sha256(args.xlsx),
        "rows": int(len(df)),
        "columns": list(df.columns),
        "unique_drugs_raw": int(df["Drug_Name"].nunique()),
        "unique_drugs_after_known_sotalol_alias": int(
            df["Drug_Name"].replace(KNOWN_DRUG_ALIAS).nunique()
        ),
        "sites": sorted(df["site"].dropna().astype(int).unique().tolist()),
        "cell_types": sorted(df["Cell_type"].dropna().astype(str).unique().tolist()),
        "concentrations": sorted(df["conc"].dropna().astype(int).unique().tolist()),
        "platforms": sorted(df["Platform"].dropna().astype(str).unique().tolist()),
        "risk_values": sorted(df["risk"].dropna().astype(str).unique().tolist()),
        "missing_risk_rows": int(df["risk"].isna().sum()),
        "missing_ddFPDc_rows": int(df["ddFPDc"].isna().sum()),
        "ead_rows": int(df["EAD"].eq(1).sum()),
        "issues": issues,
    }

    print(json.dumps(report, indent=2, default=str))
    if args.json:
        args.json.write_text(
            json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

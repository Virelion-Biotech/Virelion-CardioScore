"""Regression tests for the Colab validation runner's Blinova stage.

The synthetic workbook reproduces every quirk documented in
docs/cipa_blinova_workbook_audit.md (two sotalol spellings, one blank risk label,
an undocumented platform code, missing ddFPDc). If a quirk that is already known
can stop the stage, these tests fail in CI instead of in a Colab run.
"""

from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

RUNNER_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "validation"
    / "run_colab_validation.py"
)
_spec = importlib.util.spec_from_file_location("cardioscore_colab_runner", RUNNER_PATH)
runner = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(runner)

DRUGS = [
    "Astemizole", "Azimilide", "Bepridil", "Chlorpromazine", "Cisapride", "Clarithromycin",
    "Clozapine", "D,l Sotalol", "Diltiazem", "Disopyramide", "Dofetilide", "Domperidone",
    "Droperidol", "Ibutilide", "Loratadine", "Metoprolol", "Mexiletine", "Nifedipine",
    "Nitrendipine", "Ondansetron", "Pimozide", "Quinidine", "Ranolazine", "Risperidone",
    "Tamoxifen", "Terfenadine", "Vandetanib", "Verapamil",
]  # fmt: skip
RISKS = ["H", "M", "L"]
RISK_OVERRIDES = {"D,l Sotalol": "H"}


def make_frame() -> pd.DataFrame:
    rows = []
    for i, drug in enumerate(DRUGS):
        for site in range(1, 11):
            for conc in (1, 2):
                rows.append(
                    {
                        "Drug_Name": drug,
                        "Cell_type": "CDI",
                        "risk": RISK_OVERRIDES.get(drug, RISKS[i % 3]),
                        "Platform": "AXN",
                        "Type_of_EADs": np.nan,
                        "conc": conc,
                        "EAD": 0,
                        "ddFPDc": np.nan if (site + conc) % 4 == 0 else 5.0 + site,
                        "site": site,
                    }
                )
    df = pd.DataFrame(rows)
    df["Type_of_EADs"] = df["Type_of_EADs"].astype(object)
    return df


def make_messy_frame() -> pd.DataFrame:
    extra = {
        "Drug_Name": "D,l,Sotalol",
        "Cell_type": "CDI",
        "risk": np.nan,
        "Platform": "ACA",
        "Type_of_EADs": np.nan,
        "conc": 1,
        "EAD": 0,
        "ddFPDc": 28.2979749885,
        "site": 6,
    }
    return pd.concat([make_frame(), pd.DataFrame([extra])], ignore_index=True)


def audit(df: pd.DataFrame, tmp_path: Path) -> dict:
    return runner.audit_blinova_frame(df, tmp_path)


def codes(result: dict) -> set[str]:
    return {f["code"] for f in result["flags"]}


def test_messy_workbook_completes_and_flags_known_quirks(tmp_path):
    result = audit(make_messy_frame(), tmp_path)
    assert result["status"] == "complete"
    assert result["n_compounds"] == 28
    assert {"MISSING_RISK_ROWS", "UNDOCUMENTED_PLATFORM_CODE"} <= codes(result)
    assert result["n_ddFPDc_missing"] == 140

    by_code = {f["code"]: f for f in result["flags"]}
    assert by_code["MISSING_RISK_ROWS"]["n_rows"] == 1
    assert by_code["MISSING_RISK_ROWS"]["rows_by_compound"] == {"sotalol": 1}
    assert by_code["UNDOCUMENTED_PLATFORM_CODE"]["rows_by_code"] == {"ACA": 1}


def test_missing_risk_is_not_injected_into_source_rows(tmp_path):
    result = audit(make_messy_frame(), tmp_path)
    semantic = pd.read_csv(result["semantic_summary_path"])
    row = semantic[semantic["risk_missing"]]
    assert len(row) == 1
    assert row["risk"].isna().all()
    assert row["Drug_Name"].iloc[0] == "D,l,Sotalol"
    assert row["compound"].iloc[0] == "sotalol"

    reference = pd.read_csv(result["reference_path"])
    assert reference.loc[
        reference["compound"] == "sotalol", "reference_risk"
    ].tolist() == ["high"]
    assert list(reference.columns) == ["compound", "reference_risk"]
    assert reference["compound"].is_unique


def test_undocumented_platform_is_kept_but_not_relabeled(tmp_path):
    result = audit(make_messy_frame(), tmp_path)
    semantic = pd.read_csv(result["semantic_summary_path"])
    aca = semantic[semantic["Platform"] == "ACA"]
    assert len(aca) == 1 and not aca["platform_documented"].iloc[0]
    assert semantic["platform_documented"].sum() == len(semantic) - 1


def test_both_sotalol_spellings_map_to_one_compound(tmp_path):
    df = make_messy_frame()
    result = audit(df, tmp_path)
    semantic = pd.read_csv(result["semantic_summary_path"])
    assert set(semantic.loc[semantic["compound"] == "sotalol", "Drug_Name"]) == {
        "D,l Sotalol",
        "D,l,Sotalol",
    }


def test_flags_file_is_written_and_json_safe(tmp_path):
    result = audit(make_messy_frame(), tmp_path)
    payload = json.loads(Path(result["flags_path"]).read_text(encoding="utf-8"))
    assert payload["n_compounds"] == 28
    assert payload["flags"] == result["flags"]
    json.dumps(result, allow_nan=False)


def test_clean_workbook_has_no_flags(tmp_path):
    result = audit(make_frame(), tmp_path)
    assert result["flags"] == [] and result["n_flags"] == 0


def test_unknown_name_variant_is_not_fuzzy_merged(tmp_path):
    df = make_messy_frame()
    df.loc[df.index[df["Drug_Name"] == "Verapamil"][0], "Drug_Name"] = "Sotalol HCl"
    with pytest.raises(ValueError, match="28-drug panel"):
        audit(df, tmp_path)


def test_extra_compound_fails_panel_check(tmp_path):
    df = make_frame()
    df.loc[0, "Drug_Name"] = "Mystery drug"
    with pytest.raises(ValueError, match="28-drug panel"):
        audit(df, tmp_path)


def test_missing_drug_name_fails(tmp_path):
    df = make_frame()
    df.loc[3, "Drug_Name"] = np.nan
    with pytest.raises(ValueError, match="Drug_Name is missing"):
        audit(df, tmp_path)


def test_missing_required_column_fails(tmp_path):
    with pytest.raises(ValueError, match="missing required columns"):
        audit(make_frame().drop(columns=["Platform"]), tmp_path)


def test_malformed_ddfpdc_text_fails_but_missing_is_allowed(tmp_path):
    df = make_frame()
    assert df["ddFPDc"].isna().any()
    df["ddFPDc"] = df["ddFPDc"].astype(object)
    df.loc[5, "ddFPDc"] = "n/a"
    with pytest.raises(ValueError, match="ddFPDc contains malformed"):
        audit(df, tmp_path)


def test_unknown_risk_label_fails(tmp_path):
    df = make_frame()
    df.loc[0, "risk"] = "X"
    with pytest.raises(ValueError, match="Unexpected Blinova risk label"):
        audit(df, tmp_path)


def test_conflicting_risk_labels_within_compound_fail(tmp_path):
    df = make_frame()
    idx = df.index[df["Drug_Name"] == "Astemizole"][0]
    df.loc[idx, "risk"] = "L" if df.loc[idx, "risk"] != "L" else "H"
    with pytest.raises(ValueError, match="not one-to-one"):
        audit(df, tmp_path)


def test_compound_with_no_labeled_row_fails(tmp_path):
    df = make_frame()
    df.loc[df["Drug_Name"] == "Loratadine", "risk"] = np.nan
    with pytest.raises(ValueError, match="no labeled risk row"):
        audit(df, tmp_path)


def test_typed_event_without_ead_flag_fails(tmp_path):
    df = make_frame()
    df.loc[0, "Type_of_EADs"] = "B"
    with pytest.raises(ValueError, match="internally inconsistent"):
        audit(df, tmp_path)


def test_typed_arrhythmia_in_no_event_compound_is_flagged_not_fatal(tmp_path):
    df = make_frame()
    idx = df.index[df["Drug_Name"] == "Terfenadine"][:3]
    df.loc[idx, "Type_of_EADs"] = "A"
    df.loc[idx, "EAD"] = 1
    result = audit(df, tmp_path)
    flag = next(f for f in result["flags"] if f["code"] == "PUBLISHED_CLAIM_MISMATCH")
    assert flag["compound"] == "terfenadine"
    assert flag["rows_abcd_events"] == 3 and flag["datasets_with_abcd_events"] == 2


def test_untyped_ead_flag_in_no_event_compound_is_flagged_separately(tmp_path):
    df = make_frame()
    idx = df.index[df["Drug_Name"] == "Verapamil"][:4]
    df.loc[idx, "EAD"] = 1
    result = audit(df, tmp_path)
    assert "PUBLISHED_CLAIM_MISMATCH" not in codes(result)
    flag = next(f for f in result["flags"] if f["code"] == "EAD_FLAG_WITHOUT_ARRHYTHMIA_TYPE")
    assert flag["compound"] == "verapamil" and flag["rows_EAD_eq_1"] == 4


def test_quiescence_events_are_kept_separate_from_arrhythmia(tmp_path):
    df = make_frame()
    idx = df.index[df["Drug_Name"] == "Cisapride"][:2]
    df.loc[idx, "Type_of_EADs"] = "Q"
    df.loc[idx, "EAD"] = 1
    result = audit(df, tmp_path)
    assert result["n_Q_events"] == 2 and result["n_abcd_events"] == 0


def test_xlsx_round_trip_through_run_blinova(tmp_path):
    pytest.importorskip("openpyxl")
    path = tmp_path / "Blinova_etal_2018_data.xlsx"
    make_messy_frame().to_excel(path, index=False)
    derived = tmp_path / "derived"
    derived.mkdir()
    result = runner.run_blinova(path, derived)
    assert result["status"] == "complete" and result["n_compounds"] == 28
    assert {"MISSING_RISK_ROWS", "UNDOCUMENTED_PLATFORM_CODE"} <= codes(result)


def test_runner_revision_defaults_to_unpinned_and_validates_sha():
    assert runner.resolve_runner_revision({}) == ("unpinned-main", None)
    sha, digest = "a" * 40, "b" * 64
    env = {"CARDIOSCORE_RUNNER_SHA": sha, "CARDIOSCORE_RUNNER_SOURCE_SHA256": digest}
    assert runner.resolve_runner_revision(env) == (sha, digest)
    with pytest.raises(ValueError, match="40-character"):
        runner.resolve_runner_revision({"CARDIOSCORE_RUNNER_SHA": "main"})
    with pytest.raises(ValueError, match="64-character"):
        runner.resolve_runner_revision({"CARDIOSCORE_RUNNER_SOURCE_SHA256": "abc"})


def test_internal_call_arities_match_signatures():
    tree = ast.parse(RUNNER_PATH.read_text(encoding="utf-8"))
    funcs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    problems = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            fn = funcs.get(node.func.id)
            if fn is None or fn.args.kwarg or fn.args.vararg:
                continue
            total = len(fn.args.args)
            required = total - len(fn.args.defaults)
            given = len(node.args) + len(node.keywords)
            if not required <= given <= total:
                problems.append((node.func.id, node.lineno, given, required, total))
    assert not problems, problems

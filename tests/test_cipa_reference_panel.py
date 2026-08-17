from pathlib import Path

import yaml


def test_cipa_reference_panel_has_28_unique_drugs_and_expected_category_counts():
    path = Path(__file__).parents[1] / "benchmarks" / "cipa_28_reference_panel.yaml"
    panel = yaml.safe_load(path.read_text(encoding="utf-8"))

    categories = panel["risk_categories"]
    assert len(categories["high"]) == 8
    assert len(categories["intermediate"]) == 11
    assert len(categories["low"]) == 9

    all_drugs = categories["high"] + categories["intermediate"] + categories["low"]
    assert len(all_drugs) == 28
    assert len(set(all_drugs)) == 28

    design = panel["study_design"]
    assert design["drugs"] == 28
    assert design["independent_sites"] == 10
    assert design["site_cell_type_combinations"] == 15

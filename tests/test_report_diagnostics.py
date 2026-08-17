import json

from virelion_cardioscore.analysis.pipeline import CardioScorePipeline
from virelion_cardioscore.io.synthetic import load_synthetic_dataset
from virelion_cardioscore.reporting.diagnostics import concentration_driver_table, enrich_html_report, enrich_json_report


def test_concentration_driver_table_identifies_tested_drivers():
    result = CardioScorePipeline.from_defaults().run(
        load_synthetic_dataset(n_compounds=2, n_concentrations=4, seed=42)
    )

    drivers = concentration_driver_table(result)

    assert not drivers.empty
    assert {
        "compound",
        "endpoint",
        "driver_concentration_uM",
        "support_fraction",
    }.issubset(drivers.columns)
    assert drivers["support_fraction"].between(0.0, 1.0).all()


def test_json_report_contains_concentration_provenance(tmp_path):
    result = CardioScorePipeline.from_defaults().run(
        load_synthetic_dataset(n_compounds=1, n_concentrations=4, seed=42)
    )
    path = tmp_path / "scores.json"
    result.to_json(path)
    enrich_json_report(result, path)

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert "concentration_drivers" in payload
    assert payload["diagnostics"]["concentration_driver_provenance"]["does_not_modify_score"] is True


def test_html_report_contains_concentration_provenance(tmp_path):
    result = CardioScorePipeline.from_defaults().run(
        load_synthetic_dataset(n_compounds=1, n_concentrations=4, seed=42)
    )
    path = tmp_path / "report.html"
    result.to_html(path)
    enrich_html_report(result, path)

    html = path.read_text(encoding="utf-8")
    assert "Concentration Provenance" in html
    assert "Supporting concentrations" in html

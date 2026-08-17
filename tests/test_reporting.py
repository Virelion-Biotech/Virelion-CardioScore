from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from virelion_cardioscore.reporting.report_generator import write_html_report


def test_html_report_includes_variability_diagnostics(tmp_path):
    result = SimpleNamespace(
        summary_table=pd.DataFrame(
            {
                "compound": ["A"],
                "cardioscore": [0.42],
                "risk_class": ["Moderate"],
                "max_concentration_uM": [10.0],
                "n_wells": [4],
            }
        ),
        scores=[],
        variability_table=pd.DataFrame(
            {
                "endpoint": ["fpd_ms"],
                "group_column": ["plate_id"],
                "n_groups": [2],
                "n_controls": [4],
                "control_cv_pct": [8.5],
                "between_group_sd": [12.0],
                "status": ["stable"],
            }
        ),
        # write_html_report's normalization card reads these too -- match
        # PipelineResult's real defaults (empty dict / empty DataFrame) so
        # this stand-in behaves like an unconfigured pipeline run rather
        # than raising an AttributeError before the assertion is reached.
        normalization_diagnostic={},
        variability_before_correction=pd.DataFrame(),
        separation_table=pd.DataFrame(
            {
                "compound": ["A"],
                "plate_id": ["P1"],
                "endpoint": ["fpd_ms"],
                "n_controls": [2],
                "n_treated": [2],
                "standardized_separation": [2.1],
            }
        ),
        qc_log=[],
    )

    output = tmp_path / "report.html"
    write_html_report(result, output)
    html = output.read_text(encoding="utf-8")

    assert "Control Stability &amp; Plate/Batch Diagnostics" in html
    assert "fpd_ms" in html
    assert "8.50%" in html
    assert "Exploratory treatment/control separation" in html
    assert "2.100" in html

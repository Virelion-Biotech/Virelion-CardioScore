from __future__ import annotations

import pandas as pd

from virelion_cardioscore.reporting.report_generator import write_html_report


def test_html_report_surfaces_normalization_before_after(tmp_path):
    class DummyResult:
        summary_table = pd.DataFrame(
            [{
                "compound": "A",
                "cardioscore": 0.4,
                "risk_class": "Moderate",
                "max_concentration_uM": 10.0,
                "n_wells": 4,
            }]
        )
        scores = []
        qc_log = []
        normalization_diagnostic = {
            "group_column": "plate_id",
            "n_groups": 2,
            "n_controls": 4,
            "corrected_columns": ["fpd_ms"],
        }
        variability_before_correction = pd.DataFrame(
            [{
                "endpoint": "fpd_ms",
                "group_column": "plate_id",
                "n_groups": 2,
                "n_controls": 4,
                "control_cv_pct": 10.0,
                "between_group_sd": 10.0,
                "status": "high_variability",
            }]
        )
        variability_table = pd.DataFrame(
            [{
                "endpoint": "fpd_ms",
                "group_column": "plate_id",
                "n_groups": 2,
                "n_controls": 4,
                "control_cv_pct": 1.0,
                "between_group_sd": 0.0,
                "status": "stable",
            }]
        )
        separation_table = pd.DataFrame()

    path = tmp_path / "report.html"
    write_html_report(DummyResult(), path)
    html = path.read_text(encoding="utf-8")

    assert "Control-Anchored Normalization" in html
    assert "Between-group SD before" in html
    assert "0.00" in html
    assert "1.00%" in html

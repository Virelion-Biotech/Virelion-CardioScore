from __future__ import annotations

import pandas as pd
import pytest

from virelion_cardioscore.analysis.inference_comparison import (
    compare_effect_estimates,
    summarize_effect_concordance,
)


def test_compare_effect_estimates_aligns_and_reports_disagreement():
    conventional = pd.DataFrame(
        {
            "compound": ["A", "A"],
            "concentration_uM": [1.0, 10.0],
            "fpd_change_pct_mean": [20.0, -5.0],
        }
    )
    mixed = pd.DataFrame(
        {
            "compound": ["A", "A"],
            "concentration_uM": [1.0, 10.0],
            "endpoint": ["fpd_change_pct_mean", "fpd_change_pct_mean"],
            "treatment_effect": [18.0, -4.0],
            "status": ["ok", "ok"],
        }
    )

    comparison = compare_effect_estimates(conventional, mixed)

    assert len(comparison) == 2
    assert comparison["direction_agreement"].all()
    assert comparison.loc[0, "absolute_difference"] == pytest.approx(2.0)

    summary = summarize_effect_concordance(comparison)
    assert summary["n_comparisons"] == 2
    assert summary["direction_agreement_rate"] == pytest.approx(1.0)


def test_compare_effect_estimates_ignores_non_estimable_models():
    conventional = pd.DataFrame(
        {
            "compound": ["A"],
            "concentration_uM": [1.0],
            "fpd_change_pct_mean": [20.0],
        }
    )
    mixed = pd.DataFrame(
        {
            "compound": ["A"],
            "concentration_uM": [1.0],
            "endpoint": ["fpd_change_pct_mean"],
            "treatment_effect": [None],
            "status": ["not_estimable"],
        }
    )

    comparison = compare_effect_estimates(conventional, mixed)
    assert comparison.empty


def test_compare_effect_estimates_requires_expected_schema():
    with pytest.raises(ValueError, match="Conventional estimates are missing"):
        compare_effect_estimates(
            pd.DataFrame({"compound": ["A"]}),
            pd.DataFrame(
                {
                    "compound": ["A"],
                    "concentration_uM": [1.0],
                    "endpoint": ["fpd_change_pct_mean"],
                    "treatment_effect": [2.0],
                    "status": ["ok"],
                }
            ),
        )

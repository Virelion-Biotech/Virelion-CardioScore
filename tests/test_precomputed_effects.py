import pandas as pd

from virelion_cardioscore.analysis.pipeline import CardioScorePipeline


def test_precomputed_effect_max_includes_stv_and_triangulation():
    df = pd.DataFrame(
        {
            "compound": ["STV_only"],
            "concentration_uM": [1.0],
            "well": ["A01"],
            "fpd_change_pct": [0.0],
            "beat_rate_change_pct": [0.0],
            "amplitude_change_pct": [0.0],
            "stv_increase": [0.8],
            "triangulation_proxy_change": [0.0],
        }
    )
    pipeline = CardioScorePipeline.from_defaults()
    effects = pipeline.compute_effects(df)
    assert effects.loc[0, "max_effect_pct"] == 80.0


def test_precomputed_effect_max_includes_triangulation():
    df = pd.DataFrame(
        {
            "compound": ["Tri_only"],
            "concentration_uM": [1.0],
            "well": ["A01"],
            "fpd_change_pct": [0.0],
            "beat_rate_change_pct": [0.0],
            "amplitude_change_pct": [0.0],
            "stv_increase": [0.0],
            "triangulation_proxy_change": [0.65],
        }
    )
    pipeline = CardioScorePipeline.from_defaults()
    effects = pipeline.compute_effects(df)
    assert effects.loc[0, "max_effect_pct"] == 65.0

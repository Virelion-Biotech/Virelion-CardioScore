from __future__ import annotations

import pytest

from virelion_cardioscore.analysis.dose_response import _fitted_harmful_effect_magnitude


def test_absolute_direction_uses_response_excursion_not_baseline_magnitude():
    # A +8% fitted baseline with a +12% upper asymptote represents a 4-point
    # concentration-induced excursion, not 12 points of induced harm.
    assert _fitted_harmful_effect_magnitude(
        8.0,
        12.0,
        "fpd_change_pct",
    ) == pytest.approx(4.0)


def test_absolute_direction_handles_shortening_symmetrically():
    assert _fitted_harmful_effect_magnitude(
        8.0,
        -12.0,
        "fpd_change_pct",
    ) == pytest.approx(20.0)

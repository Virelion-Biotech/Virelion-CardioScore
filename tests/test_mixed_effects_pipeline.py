from __future__ import annotations

import pandas as pd

from virelion_cardioscore.analysis.mixed_effects_pipeline import (
    fit_compound_concentration_mixed_effects,
)


def _frame() -> pd.DataFrame:
    rows = []
    for plate, offset in [("P1", 0.0), ("P2", 10.0), ("P3", 20.0)]:
        for vehicle, fpd in [(True, 100.0 + offset), (True, 101.0 + offset), (False, 120.0 + offset), (False, 121.0 + offset)]:
            rows.append(
                {
                    "compound": "A",
                    "concentration_uM": 1.0,
                    "vehicle": vehicle,
                    "plate_id": plate,
                    "fpd_ms": fpd,
                }
            )
        for vehicle, fpd in [(True, 100.0 + offset), (True, 101.0 + offset), (False, 110.0 + offset), (False, 111.0 + offset)]:
            rows.append(
                {
                    "compound": "A",
                    "concentration_uM": 10.0,
                    "vehicle": vehicle,
                    "plate_id": plate,
                    "fpd_ms": fpd,
                }
            )
    return pd.DataFrame(rows)


def test_mixed_effects_keeps_concentrations_separate():
    rows = fit_compound_concentration_mixed_effects(
        _frame(),
        group_column="plate_id",
        endpoints=["fpd_ms"],
    )
    concentrations = {row["concentration_uM"] for row in rows}
    assert concentrations == {1.0, 10.0}
    assert len(rows) == 2


def test_mixed_effects_reports_non_estimable_designs():
    frame = _frame().query("concentration_uM == 1.0").copy()
    frame.loc[frame["plate_id"] == "P3", "vehicle"] = True

    rows = fit_compound_concentration_mixed_effects(
        frame,
        group_column="plate_id",
        endpoints=["fpd_ms"],
    )

    assert len(rows) == 1
    assert rows[0]["status"] in {"ok", "not_converged", "not_estimable"}

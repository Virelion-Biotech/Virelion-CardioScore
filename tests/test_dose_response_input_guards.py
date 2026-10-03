from __future__ import annotations

import numpy as np
import pytest

from virelion_cardioscore.analysis.dose_response import fit_4pl


def _valid_inputs() -> tuple[np.ndarray, np.ndarray]:
    x = np.logspace(-1, 2, 6)
    y = 80.0 / (1.0 + (10.0 / x) ** 1.5)
    return x, y


def test_4pl_rejects_nonfinite_concentrations_or_responses():
    x, y = _valid_inputs()
    x[2] = np.nan
    with pytest.raises(ValueError, match="only finite values"):
        fit_4pl(x, y)


def test_4pl_rejects_nonpositive_concentrations():
    x, y = _valid_inputs()
    x[2] = 0.0
    with pytest.raises(ValueError, match="strictly positive concentrations"):
        fit_4pl(x, y)


def test_4pl_rejects_duplicate_concentrations():
    x, y = _valid_inputs()
    x[2] = x[1]
    with pytest.raises(ValueError, match="one response per concentration"):
        fit_4pl(x, y)


def test_4pl_rejects_invalid_standard_errors():
    x, y = _valid_inputs()
    sem = np.ones_like(x)
    sem[3] = 0.0
    with pytest.raises(ValueError, match="strictly positive values"):
        fit_4pl(x, y, response_sem=sem)

from __future__ import annotations

import pandas as pd
import pytest

from virelion_cardioscore.utils.coercion import coerce_bool_series


def test_coerce_bool_series_handles_string_false_correctly():
    values = pd.Series(["True", "False", "yes", "no", "1", "0"])
    result = coerce_bool_series(values, name="vehicle")
    assert result.tolist() == [True, False, True, False, True, False]
    assert result.dtype == bool


def test_coerce_bool_series_rejects_ambiguous_values():
    with pytest.raises(ValueError, match="unsupported boolean value"):
        coerce_bool_series(pd.Series(["maybe"]), name="vehicle")


def test_coerce_bool_series_rejects_missing_values():
    with pytest.raises(ValueError, match="missing value"):
        coerce_bool_series(pd.Series([True, None]), name="vehicle")

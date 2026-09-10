"""Strict coercion helpers for analysis metadata."""

from __future__ import annotations

from typing import Any

import pandas as pd


_TRUE_VALUES = {True, 1, 1.0, "true", "1", "1.0", "yes"}
_FALSE_VALUES = {False, 0, 0.0, "false", "0", "0.0", "no"}


def coerce_bool_series(series: pd.Series, *, name: str = "value") -> pd.Series:
    """Convert supported boolean encodings and reject ambiguous values.

    In particular, strings such as ``"False"`` must not pass through Python's
    generic ``bool()`` conversion, where any non-empty string becomes ``True``.
    """
    def convert(value: Any) -> bool:
        if pd.isna(value):
            raise ValueError(f"{name} contains a missing value.")
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"true", "1", "1.0", "yes"}:
                return True
            if normalized in {"false", "0", "0.0", "no"}:
                return False
            raise ValueError(
                f"{name} contains unsupported boolean value {value!r}; "
                "expected true/false, 1/0, or yes/no."
            )
        if value in _TRUE_VALUES:
            return True
        if value in _FALSE_VALUES:
            return False
        raise ValueError(
            f"{name} contains unsupported boolean value {value!r}; "
            "expected true/false, 1/0, or yes/no."
        )

    return series.map(convert).astype(bool)

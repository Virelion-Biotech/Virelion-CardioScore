"""Strict coercion helpers for analysis metadata."""

from __future__ import annotations

from typing import Any

import pandas as pd


_TRUE_VALUES = {"true", "1", "1.0", "yes"}
_FALSE_VALUES = {"false", "0", "0.0", "no"}


def coerce_bool_series(series: pd.Series, *, name: str = "value") -> pd.Series:
    """Convert supported boolean encodings and reject ambiguous values."""
    def convert(value: Any) -> bool:
        if pd.isna(value):
            raise ValueError(f"{name} contains a missing value.")
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in _TRUE_VALUES:
                return True
            if normalized in _FALSE_VALUES:
                return False
            raise ValueError(
                f"{name} contains unsupported boolean value {value!r}; "
                "expected true/false, 1/0, or yes/no."
            )
        if isinstance(value, (int, float)) and value in {0, 1}:
            return bool(value)
        raise ValueError(
            f"{name} contains unsupported boolean value {value!r}; "
            "expected true/false, 1/0, or yes/no."
        )

    return series.map(convert).astype(bool)

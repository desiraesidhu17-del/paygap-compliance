from __future__ import annotations

import pandas as pd

REQUIRED_COLUMNS = [
    "Gender",
    "OrdinaryPay",
    "SpecialSalary",
    "HoursWorked",
    "OTHours",
    "OTPay",
    "Bonus",
]

NUMERIC_COLUMNS = [
    "OrdinaryPay",
    "SpecialSalary",
    "HoursWorked",
    "OTHours",
    "OTPay",
    "Bonus",
]


def _format_missing(columns: list[str]) -> str:
    if not columns:
        return ""
    if len(columns) == 1:
        return columns[0]
    return ", ".join(columns[:-1]) + f", and {columns[-1]}"


def coerce_df(df: pd.DataFrame) -> pd.DataFrame:
    """Validate required columns and coerce numeric fields to floats."""
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        missing_str = _format_missing(missing)
        raise ValueError(f"Missing columns: {missing_str}")

    coerced = df.copy()

    for column in NUMERIC_COLUMNS:
        coerced[column] = pd.to_numeric(coerced[column], errors="coerce")

    coerced["Gender"] = coerced["Gender"].astype(str)

    return coerced

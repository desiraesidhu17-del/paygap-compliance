import pandas as pd
import pytest

from services.schema import coerce_df


def test_missing_cols_raises():
    df = pd.DataFrame({"Gender": ["Man"]})

    with pytest.raises(ValueError) as exc:
        coerce_df(df)

    message = str(exc.value)
    assert "Missing columns" in message
    assert "OrdinaryPay" in message


def test_successful_coercion():
    df = pd.DataFrame(
        {
            "Gender": ["Man", "Woman"],
            "OrdinaryPay": ["52000", 48000],
            "SpecialSalary": [0, "1000"],
            "HoursWorked": ["2080", "2070"],
            "OTHours": [0, "5"],
            "OTPay": ["0", "300"],
            "Bonus": ["1250", "not available"],
        }
    )

    result = coerce_df(df)

    numeric_columns = [
        "OrdinaryPay",
        "SpecialSalary",
        "HoursWorked",
        "OTHours",
        "OTPay",
        "Bonus",
    ]

    for column in numeric_columns:
        assert pd.api.types.is_numeric_dtype(result[column])

    assert result.loc[0, "OrdinaryPay"] == 52000
    assert pd.isna(result.loc[1, "Bonus"])
    assert result.loc[0, "Gender"] == "Man"

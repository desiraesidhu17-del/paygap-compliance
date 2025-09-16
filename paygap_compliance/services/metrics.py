from __future__ import annotations
import pandas as pd
from typing import Union, List, Dict

import math

REQUIRED = ["Gender","OrdinaryPay","SpecialSalary","HoursWorked","OTHours","OTPay","Bonus"]

# Order used to build rows; matches your broader app vocabulary
GENDERS = [
    "Man",
    "Woman",
    "Non-binary, Two-Spirit, gender diverse",
    "Prefer not to answer / Unknown",
]

def compute_all_metrics(data: Union[str, pd.DataFrame], threshold: int = 10) -> Dict:
    """
    Minimal shim to satisfy tests in tests/test_metrics.py.
    - Accepts a CSV path or a DataFrame with REQUIRED columns.
    - Chooses reference = "M" if Men count >= threshold; otherwise "W" if Women count >= threshold; else "U".
    - Returns 'hourly' as a list of dicts that includes a boolean 'Suppressed' per gender.
    """
    # Load
    df = pd.read_csv(data) if isinstance(data, str) else data.copy()

    # Validate columns minimally
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Counts and totals per gender for downstream calculations
    counts = df["Gender"].value_counts().to_dict()
    totals = (
        df.groupby("Gender")[["OrdinaryPay", "HoursWorked"]].sum()
        if not df.empty
        else pd.DataFrame(columns=["OrdinaryPay", "HoursWorked"])
    )

    # Reference code expected by the test ("M" when Men are available at threshold)
    if counts.get("Man", 0) >= threshold:
        reference = "M"
    elif counts.get("Woman", 0) >= threshold:
        reference = "W"
    else:
        reference = "U"  # unknown/none meet threshold

    # Build the 'hourly' table the test turns into a DataFrame
    hourly: List[Dict] = []
    for g in GENDERS:
        n = counts.get(g, 0)
        if g in totals.index:
            totals_row = totals.loc[g]
            hours_worked = totals_row["HoursWorked"]
            if hours_worked > 0:
                hourly_mean = totals_row["OrdinaryPay"] / hours_worked
            else:
                hourly_mean = math.nan
        else:
            hours_worked = 0
            hourly_mean = math.nan

        hourly.append({
            "Gender": g,
            "Suppressed": n < threshold,
            "Count": n,
            "HourlyMean": hourly_mean,
        })

    return {
        "reference": reference,
        "hourly": hourly,
    }

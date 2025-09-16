# paygap_compliance/services/ledger_to_totals_transform.py
import argparse
import re
import pandas as pd
import numpy as np

CANON_CODES = {"REG", "OT", "BONUS", "COMM", "OTHER"}

DEFAULT_CODE_MAP = {
    "R": "REG", "REGULAR": "REG", "HOURLY": "REG", "SALARY": "REG", "BASE": "REG",
    "OVT": "OT", "OVERTIME": "OT", "OT1.5": "OT", "OT2.0": "OT", "OT": "OT",
    "BON": "BONUS", "ANNUAL_BONUS": "BONUS", "BONUS": "BONUS",
    "COMMISSION": "COMM", "COMM_QTR": "COMM", "COMM": "COMM",
    "RSU": "OTHER", "STOCK": "OTHER", "SPIFF": "OTHER",
}

def _normalize_code(raw_code: str, description: str, code_map: dict) -> str:
    raw = "" if pd.isna(raw_code) else str(raw_code).strip().upper()
    if raw in CANON_CODES:
        return raw
    if raw in code_map:
        return code_map[raw]
    desc = "" if pd.isna(description) else str(description).upper()
    if "OVERTIME" in raw or " OT" in f" {raw}" or "OVERTIME" in desc or " OT" in f" {desc}":
        return "OT"
    if "COMMISSION" in raw or "COMM" in raw or "COMMISSION" in desc or "COMM" in desc:
        return "COMM"
    if "BONUS" in raw or "BON " in f"{raw} " or "BONUS" in desc or "BON " in f"{desc} ":
        return "BONUS"
    if any(k in raw or k in desc for k in ("REG", "HOURLY", "SALARY", "REGULAR", "BASE")):
        return "REG"
    return "OTHER"

def _norm_province(s: pd.Series) -> pd.Series:
    norm = (
        s.fillna("")
         .astype(str)
         .str.upper()
         .str.replace(r"[^A-Z]", "", regex=True)
    )
    return norm.replace({"BRITISHCOLUMBIA": "BC"})

def transform_ledger_to_totals(
    ledger_csv: str,
    output_csv: str,
    year: int,
    province_filter: str = "BC",
    code_map: dict | None = None,
) -> pd.DataFrame:
    # Read all as string; coerce numeric cols explicitly
    df = pd.read_csv(ledger_csv, dtype=str)

    # Basic guardrails
    if not any(c.lower() == "employeeid" for c in df.columns):
        raise ValueError("Ledger is missing an EmployeeID column.")

    for col in ("Hours", "Amount", "FTE"):
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # Year filter
    if "ReportingYear" in df.columns:
        df = df[df["ReportingYear"].astype(str) == str(year)]
    elif "PayDate" in df.columns:
        parsed = pd.to_datetime(df["PayDate"], errors="coerce")
        df = df[parsed.dt.year == year]

    # Province filter
    if "WorkProvince" in df.columns and province_filter:
        want = re.sub(r"[^A-Z]", "", str(province_filter).upper())
        df = df[_norm_province(df["WorkProvince"]) == want]

    if df.empty:
        raise ValueError("No rows after filtering by year/province.")

    # Canonical earning code, robust row-wise
    cmap = {k.upper(): v for k, v in (code_map or DEFAULT_CODE_MAP).items()}
    df["CanonCode"] = df.apply(
        lambda r: _normalize_code(
            r.get("EarningCode", ""), r.get("EarningDescription", ""), cmap
        ),
        axis=1
    )

    # Aggregate by employee + canon code
    sum_amt = df.groupby(["EmployeeID", "CanonCode"], dropna=False)["Amount"].sum().unstack(fill_value=0.0)
    sum_hrs = df.groupby(["EmployeeID", "CanonCode"], dropna=False)["Hours"].sum().unstack(fill_value=0.0)

    # Carry forward last-known categorical columns
    def last_non_null(series: pd.Series) -> str:
        s = series.dropna()
        return "" if s.empty else s.iloc[-1]

    carry_cols = [
        "Gender","Department","JobTitle","EmploymentType","FTE",
        "WorkCity","WorkProvince","PayrollSystem","WorkEmail",
        "PreferredName","LegalFirstName","LegalLastName",
        "HireDate","TerminationDate","OnLeaveFlag",
    ]
    carried = {c: df.groupby("EmployeeID")[c].apply(last_non_null) for c in carry_cols if c in df.columns}
    carried_df = pd.DataFrame(carried)

    # Build totals
    totals = pd.DataFrame(index=sum_amt.index).reset_index()

    def _col_sum(code): return sum_amt[code] if code in sum_amt.columns else 0.0
    def _hrs_sum(code): return sum_hrs[code] if code in sum_hrs.columns else 0.0

    totals["HourlyPayTotal"]         = _col_sum("REG")
    totals["OvertimePayTotal"]       = _col_sum("OT")
    totals["BonusPayTotal"]          = _col_sum("BONUS")
    totals["CommissionPayTotal"]     = _col_sum("COMM")
    totals["OtherIncentivePayTotal"] = _col_sum("OTHER")
    totals["RegularHoursPaid"]       = _hrs_sum("REG")
    totals["OvertimeHoursPaid"]      = _hrs_sum("OT")

    # NEW: implied hourly rate for downstream analytics
    with np.errstate(divide="ignore", invalid="ignore"):
        hourly = np.where(totals["RegularHoursPaid"] > 0,
                          totals["HourlyPayTotal"] / totals["RegularHoursPaid"],
                          np.nan)
    totals["Hourly"] = hourly

    # Merge carried attributes
    if not carried_df.empty:
        totals = totals.merge(carried_df, left_on="EmployeeID", right_index=True, how="left")

    # Derived flags/period
    totals["ReceivedBonusFlag"]    = ((totals.get("BonusPayTotal", 0) + totals.get("CommissionPayTotal", 0)) > 0).map({True: "Y", False: "N"})
    totals["ReportingPeriodStart"] = f"{year}-01-01"
    totals["ReportingPeriodEnd"]   = f"{year}-12-31"
    totals["EarningCurrency"]      = "CAD"

    for c in ("HourlyPayTotal","OvertimePayTotal","BonusPayTotal","CommissionPayTotal",
              "OtherIncentivePayTotal","RegularHoursPaid","OvertimeHoursPaid","FTE","Hourly"):
        if c in totals.columns:
            totals[c] = pd.to_numeric(totals[c], errors="coerce").fillna(0.0)

    # Final column order (+ Hourly so the app can auto-map it as the rate)
    desired = [
        "EmployeeID","LegalFirstName","LegalLastName","PreferredName","WorkEmail",
        "Department","JobTitle","EmploymentType","FTE","WorkCity","WorkProvince",
        "HireDate","TerminationDate","OnLeaveFlag","Gender",
        "ReportingPeriodStart","ReportingPeriodEnd",
        "HourlyPayTotal","RegularHoursPaid","OvertimePayTotal","OvertimeHoursPaid",
        "BonusPayTotal","CommissionPayTotal","OtherIncentivePayTotal",
        "ReceivedBonusFlag","PayrollSystem","EarningCurrency",
        "Hourly",  # <-- keep at the end for clarity
    ]
    for c in desired:
        if c not in totals.columns:
            totals[c] = ""
    totals = totals[desired]
    totals.to_csv(output_csv, index=False)
    return totals

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ledger_csv")
    ap.add_argument("output_csv")
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--province", default="BC")
    args = ap.parse_args()
    transform_ledger_to_totals(args.ledger_csv, args.output_csv, year=args.year, province_filter=args.province)
    print(f"✅ Wrote {args.output_csv}")

if __name__ == "__main__":
    main()

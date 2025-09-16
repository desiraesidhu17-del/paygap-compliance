# app.py
# -----------------------------------------------------------------------------
# Flask app for BC Pay Transparency — PDF-optimized tables (no header wraps),
# dtype-safe table rendering, centered em-dashes, and landscape PDF output.
# -----------------------------------------------------------------------------

import os
import re
import tempfile
from uuid import uuid4
from datetime import datetime, date
from io import StringIO, BytesIO
from typing import Optional, Dict

import numpy as np
import pandas as pd
from flask import (
    Flask, render_template, request, redirect, url_for,
    abort, flash, send_file, Response
)

# Optional Playwright (browser "Print to PDF" still works without it)
try:
    from playwright.sync_api import sync_playwright
except Exception:
    sync_playwright = None

# -----------------------------------------------------------------------------
# Import shim for the ledger→totals transformer
# -----------------------------------------------------------------------------
import sys
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from paygap_compliance.services.ledger_to_totals_transform import transform_ledger_to_totals
from paygap_compliance.routes.report import report_bp

# -----------------------------------------------------------------------------
# Flask setup + in-memory store
# -----------------------------------------------------------------------------
app = Flask(__name__, static_folder="static", static_url_path="/static")
app.secret_key = os.environ.get("SECRET_KEY", "dev")

app.register_blueprint(report_bp)

REPORTS: dict[int, dict] = {}
NEXT_ID = 1

# Strong print styles to keep columns aligned and prevent header wraps
PRINT_STYLES = r"""
/* Keep colors in print */
@media print { html,body { -webkit-print-color-adjust:exact; print-color-adjust:exact } }

/* Tables: fixed layout for consistent columns; compact padding in print */
.table-wrap{ overflow:auto }
table.gov, table.dataframe{
  width:100%; border-collapse:collapse; table-layout:fixed;
  background:#fff; border:1px solid #e2e8f0; font-size:13px;
}
table.gov thead th, table.dataframe thead th{
  background:#f8fafc; padding:10px 10px; border-bottom:1px solid #e2e8f0;
  font-weight:700; text-align:left;
  /* prevent mid-word breaks and wraps that misalign columns */
  white-space:nowrap; word-break:normal; overflow-wrap:normal; hyphens:manual;
}
table.gov tbody td, table.dataframe tbody td{
  padding:8px 10px; border-bottom:1px solid #e2e8f0; vertical-align:top; white-space:normal;
}
table.gov tbody tr:last-child td, table.dataframe tbody tr:last-child td{ border-bottom:none }

/* Numeric alignment + centered em-dash */
table.gov .num, table.dataframe .num{ display:inline-block; min-width:3.4ch; text-align:right }
table.gov .suppressed-dash, table.dataframe .suppressed-dash{ display:inline-block; min-width:3.4ch; text-align:center }

/* Avoid splitting header/rows across pages; repeat header each page */
@media print{
  @page{ size:A4 landscape; margin:12mm }
  .container{ margin:0; padding:0 }
  .toolbar, .no-print{ display:none!important }
  table.gov thead, table.dataframe thead{ display:table-header-group }
  table.gov tfoot, table.dataframe tfoot{ display:table-footer-group }
  table.gov tr, table.dataframe tr{ break-inside:avoid; page-break-inside:avoid }
}
"""

def _inject_print_css(html: str) -> str:
    """Inject PRINT_STYLES at the end of <head>."""
    head_lower = html.lower()
    tag = "</head>"
    idx = head_lower.rfind(tag)
    block = f"<style>{PRINT_STYLES}</style>"
    if idx != -1:
        return html[:idx] + block + html[idx:]
    return block + html

# -----------------------------------------------------------------------------
# Gender canon + helpers
# -----------------------------------------------------------------------------
GENDER_CANONICAL = {
    "W": "Woman", "WOMAN": "Woman", "WOMEN": "Woman", "F": "Woman", "FEMALE": "Woman",
    "M": "Man", "MAN": "Man", "MEN": "Man", "MALE": "Man",
    "X": "Non-binary, Two-Spirit, gender diverse",
    "NB": "Non-binary, Two-Spirit, gender diverse",
    "NON-BINARY": "Non-binary, Two-Spirit, gender diverse",
    "TWO-SPIRIT": "Non-binary, Two-Spirit, gender diverse",
    "GENDER DIVERSE": "Non-binary, Two-Spirit, gender diverse",
    "U": "Prefer not to answer / Unknown",
    "UNKNOWN": "Prefer not to answer / Unknown",
    "PREFER NOT TO ANSWER": "Prefer not to answer / Unknown",
}
GENDER_ORDER = [
    "Woman",
    "Man",
    "Non-binary, Two-Spirit, gender diverse",
    "Prefer not to answer / Unknown",
]

def safe_date(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return ""
    try:
        y, m, d = [int(x) for x in s.split("-")]
        date(y, m, d)
        return s
    except Exception:
        return ""

def canon_gender(val) -> str:
    if val is None:
        return "Prefer not to answer / Unknown"
    s = str(val).strip()
    return GENDER_CANONICAL.get(s, GENDER_CANONICAL.get(s.upper(), s))

def map_gender_labels(df: pd.DataFrame, col: str = "Gender") -> pd.DataFrame:
    if df is None or col not in df.columns:
        return df
    return df.assign(**{col: df[col].map(canon_gender)})

def enforce_gender_row_order(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or "Gender" not in df.columns:
        return df
    out = df.set_index("Gender").reindex(GENDER_ORDER).reset_index()
    if "N" in out.columns:
        out = out[out["N"].fillna(0).astype(float) > 0]
    elif "Count" in out.columns:
        out = out[out["Count"].fillna(0).astype(float) > 0]
    else:
        out = out.dropna(how="all")
    return out

def get_col_ci(df: pd.DataFrame, name: str) -> Optional[str]:
    lower = name.lower()
    for c in df.columns:
        if c.lower() == lower:
            return c
    return None

# -----------------------------------------------------------------------------
# Upload shape inference (ledger vs totals)
# -----------------------------------------------------------------------------
def derive_bc_headcount_range(employees: str | int | None) -> Optional[str]:
    try:
        n = int(str(employees).strip())
    except Exception:
        return None
    if n < 0:
        return None
    if n < 50:
        return "Under 50"
    if n <= 299:
        return "50–299"
    if n <= 999:
        return "300–999"
    return "1,000+"

def _infer_mode_from_df(df: pd.DataFrame) -> str:
    cols_lower = {c.lower() for c in df.columns}
    if {"paydate", "earningcode"}.issubset(cols_lower): return "ledger"
    if {"hourlypaytotal", "overtimepaytotal"}.intersection(cols_lower): return "totals"
    emp_col = get_col_ci(df, "EmployeeID")
    if emp_col and df[emp_col].duplicated().any(): return "ledger"
    return "totals"

# -----------------------------------------------------------------------------
# Table rendering (suppression + formatting)
# -----------------------------------------------------------------------------
def render_table(
    df: Optional[pd.DataFrame],
    *,
    money_cols: tuple[str, ...] = (),
    pct_cols:   tuple[str, ...] = (),
    int_cols:   tuple[str, ...] = (),
    rename: dict | None = None,
    classes: tuple[str, ...] = ("dataframe", "gov"),
    na_text: str = "—",
    suppress_col: str | None = None,
    collapse_suppressed: bool = False,
    auto_right_align: bool = True,
) -> str:
    if df is None or len(df) == 0:
        return ""
    d = df.copy()
    if rename:
        d = d.rename(columns=rename)

    # Canonicalize genders & row order (no-ops if "Gender" missing)
    d = map_gender_labels(d)
    d = enforce_gender_row_order(d)

    label_html = '<span class="suppressed">Suppressed (n&lt;10)</span>'
    dash_html  = '<span class="num suppressed-dash">—</span>'

    def _is_html(v) -> bool:
        return isinstance(v, str) and v.lstrip().startswith("<")

    def _wrap_num(s: str) -> str:
        return f'<span class="num">{s}</span>'

    def _fmt_money(v):
        if _is_html(v): return v
        try: return _wrap_num(f"${float(v):,.2f}")
        except Exception: return na_text

    def _fmt_pct(v):
        if _is_html(v): return v
        try: return _wrap_num(f"{float(v):.1f}%")
        except Exception: return na_text

    def _fmt_int(v):
        if _is_html(v):
            return v
        try:
            iv = int(pd.to_numeric(v, errors="coerce"))
            return _wrap_num(f"{iv:,d}")
        except Exception:
            return na_text

    def _fmt_num(v):
        if _is_html(v): return v
        try: return _wrap_num(f"{float(v):,.2f}")
        except Exception: return na_text

    # Suppression handling
    if suppress_col and suppress_col in d.columns:
        vals = d[suppress_col]
        if vals.dtype == bool:
            sup_mask = vals
        else:
            v = vals.astype(str).str.strip().str.lower()
            sup_mask = v.isin(["true", "1", "yes", "y", "suppressed"])
        if sup_mask.any():
            d2 = d.copy()
            for c in d2.columns:
                if c == suppress_col: continue
                if d2[c].dtype != object: d2[c] = d2[c].astype(object)
            for c in d2.columns:
                if c == suppress_col: continue
                if c == "Gender":
                    d2.loc[sup_mask, c] = label_html
                else:
                    d2.loc[sup_mask, c] = dash_html
            d2[suppress_col] = np.where(sup_mask, "Suppressed", "Published")
            if collapse_suppressed:
                kept = d2.loc[~sup_mask].copy()
                synth = {}
                for c in d2.columns:
                    if c == "Gender": synth[c] = label_html
                    elif c == suppress_col: synth[c] = "Suppressed"
                    else: synth[c] = dash_html
                d = pd.concat([kept, pd.DataFrame([synth])], ignore_index=True)
            else:
                d = d2

    # Force ints for "N"/"Count" so we don't get 90.0 artifacts
    for c in ("N", "Count"):
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce").fillna(pd.NA).astype("Int64")

    # Column formatters
    fmt: dict[str, callable] = {}
    for c in money_cols:
        if c in d.columns: fmt[c] = _fmt_money
    for c in pct_cols:
        if c in d.columns: fmt[c] = _fmt_pct
    for c in int_cols:
        if c in d.columns: fmt[c] = _fmt_int

    if auto_right_align:
        from pandas.api.types import is_numeric_dtype
        protected = set(money_cols) | set(pct_cols) | set(int_cols) | {"Gender"}
        for c in d.columns:
            if c in protected: continue
            try:
                if is_numeric_dtype(d[c]): fmt.setdefault(c, _fmt_num)
            except Exception:
                pass

    return d.to_html(
        index=False,
        classes=list(classes),
        border=0,
        na_rep=na_text,
        formatters=fmt,
        escape=False,
    )

# -----------------------------------------------------------------------------
# Auto-mapping
# -----------------------------------------------------------------------------
COLUMN_ALIASES = {
    "gender": ["Gender", "Sex", "Worker Gender", "Employee Gender"],
    "hourly": ["Hourly", "Hourly Rate", "Rate of Pay", "Hourly Wage", "Wage", "Pay Rate", "HourlyPayTotal"],
    "overtime_pay": ["Overtime Pay", "Overtime_Pay", "OT Pay", "O/T Earnings", "Overtime Earnings", "OT_Earnings", "OvertimePayTotal"],
    "overtime_hours": ["Overtime Hours", "Overtime_Hours", "OT Hours", "O/T Hours", "OT_Hours", "OvertimeHoursPaid"],
    "bonus": ["Bonus", "Bonus Amount", "Incentive", "Variable Pay", "Variable_Pay", "Variable Compensation", "BonusPayTotal"],
    "commission": ["Commission", "Commission Amount", "CommissionPayTotal"],
    "received_bonus": ["Received Bonus", "Received_Bonus", "Bonus Paid", "Any Bonus", "Bonus Flag", "ReceivedBonusFlag"],
}

def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (s or "").lower())

def find_column(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    if df is None: return None
    normmap = {_norm(c): c for c in df.columns}
    for cand in candidates:
        key = _norm(cand)
        if key in normmap: return normmap[key]
    return None

def auto_map_columns(df: pd.DataFrame) -> dict:
    mapping = {}
    for key, aliases in COLUMN_ALIASES.items():
        col = find_column(df, aliases)
        if col: mapping[key] = col
    return mapping

def apply_auto_mapping(df: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    work = df.copy()
    if "gender" in mapping and "Gender" not in work.columns:
        work["Gender"] = work[mapping["gender"]]
    if "hourly" in mapping and "hourly" not in work.columns:
        work["hourly"] = work[mapping["hourly"]]
    if "overtime_pay" in mapping and "overtime_pay" not in work.columns:
        work["overtime_pay"] = work[mapping["overtime_pay"]]
    if "overtime_hours" in mapping and "overtime_hours" not in work.columns:
        work["overtime_hours"] = work[mapping["overtime_hours"]]
    bonus_nums = []
    if "bonus" in mapping:
        bonus_nums.append(pd.to_numeric(work[mapping["bonus"]], errors="coerce"))
    if "commission" in mapping:
        bonus_nums.append(pd.to_numeric(work[mapping["commission"]], errors="coerce"))
    if bonus_nums and "bonus" not in work.columns:
        work["bonus"] = sum(bonus_nums).fillna(0)
    if "received_bonus" in mapping and "received_bonus" not in work.columns:
        work["received_bonus"] = work[mapping["received_bonus"]]
    return work

def _to_boolish(s: pd.Series) -> pd.Series:
    v = s.astype(str).str.strip().str.lower()
    truey  = {"1","true","t","yes","y"}
    falsey = {"0","false","f","no","n"}
    out = pd.Series(pd.NA, index=s.index, dtype="boolean")
    out = out.mask(v.isin(truey), True)
    out = out.mask(v.isin(falsey), False)
    return out.fillna(False)

# -----------------------------------------------------------------------------
# Metric builders
# -----------------------------------------------------------------------------
def _with_gaps(agg: pd.DataFrame, ref_gender: str) -> pd.DataFrame:
    if agg is None or "Gender" not in agg.columns: return agg
    out = agg.copy()
    ref_row = out.loc[out["Gender"] == ref_gender]
    ref_suppressed = False
    if ref_row.empty:
        ref_mean = ref_median = None
    else:
        ref_mean = float(ref_row["Mean"].iloc[0]) if "Mean" in ref_row else None
        ref_median = float(ref_row["Median"].iloc[0]) if "Median" in ref_row else None
        if "N" in ref_row:
            try:
                ref_suppressed = int(ref_row["N"].iloc[0]) < 10
            except Exception:
                ref_suppressed = True

    def calc_gap(val, ref):
        try:
            v = float(val); r = float(ref)
            if r == 0 or pd.isna(v) or pd.isna(r): return None
            return round(((r - v) / r) * 100, 1)
        except Exception:
            return None

    if ref_row.empty or ref_suppressed:
        out["Gap (mean)"] = None
        out["Gap (median)"] = None
        return out

    out["Gap (mean)"] = out.apply(
        lambda row: "Ref" if row["Gender"] == ref_gender else calc_gap(row.get("Mean"), ref_mean),
        axis=1
    )
    out["Gap (median)"] = out.apply(
        lambda row: "Ref" if row["Gender"] == ref_gender else calc_gap(row.get("Median"), ref_median),
        axis=1
    )
    return out

def _agg_mean_median_count(df: pd.DataFrame, gender_col: str, value_col: str, ref_gender: str) -> pd.DataFrame:
    agg = (
        df.assign(_g=df[gender_col].map(canon_gender))
          .groupby("_g", dropna=False, observed=False)[value_col]
          .agg(["mean", "median", "count"]).reset_index()
          .rename(columns={"_g": "Gender", "mean": "Mean", "median": "Median", "count": "N"})
    )
    agg["Suppressed"] = agg["N"] < 10
    agg = _with_gaps(agg, ref_gender)
    return enforce_gender_row_order(agg)

def build_gender_distribution(df: pd.DataFrame, gender_col: str = "Gender") -> Optional[pd.DataFrame]:
    if gender_col not in df.columns:
        for alt in ["gender", "GENDER", "sex", "Sex"]:
            if alt in df.columns: gender_col = alt; break
        else: return None
    out = (
        df.assign(_g=df[gender_col].map(canon_gender))
          .groupby("_g", dropna=False, observed=False)
          .size().reset_index(name="Count")
          .rename(columns={"_g": "Gender"})
          .assign(**{"Included?": lambda d: np.where(d["Count"] >= 10, "Yes", "No")})
    )
    return enforce_gender_row_order(out)

def build_hourly(df: pd.DataFrame, ref_gender: str = "Man", gender_col: str = "Gender") -> Optional[pd.DataFrame]:
    if gender_col not in df.columns: return None
    pay_cols = [c for c in df.columns if c.lower() in {
        "hourly", "hourly_pay", "wage", "pay", "pay rate", "rate of pay", "hourlypaytotal"
    }]
    if not pay_cols: return None
    return _agg_mean_median_count(df, gender_col, pay_cols[0], ref_gender)

def build_overtime(df: pd.DataFrame, ref_gender: str = "Man", gender_col: str = "Gender") -> Optional[pd.DataFrame]:
    if gender_col not in df.columns: return None
    ot_cols = [c for c in df.columns if c.lower() in {
        "overtime_pay", "ot pay", "ot_pay", "otpay", "o/t earnings", "overtime earnings", "overtimepaytotal"
    }]
    if not ot_cols: return None
    return _agg_mean_median_count(df, gender_col, ot_cols[0], ref_gender)

def build_bonus(df: pd.DataFrame, ref_gender: str = "Man", gender_col: str = "Gender") -> Optional[pd.DataFrame]:
    if gender_col not in df.columns: return None
    b_cols = [c for c in df.columns if c.lower() in {
        "bonus", "bonus_pay", "incentive", "variable pay", "variable_pay", "commission",
        "bonuspaytotal", "commissionpaytotal"
    }]
    if not b_cols: return None
    return _agg_mean_median_count(df, gender_col, b_cols[0], ref_gender)

def build_participation(df: pd.DataFrame, gender_col: str = "Gender") -> Optional[pd.DataFrame]:
    if gender_col not in df.columns: return None
    flag = [c for c in df.columns if c.lower() in ["received_bonus", "any_bonus", "bonus_flag", "bonus paid"]]
    b_cols = [c for c in df.columns if c.lower() in ["bonus", "bonus_pay", "incentive", "variable pay", "variable_pay", "commission"]]
    if flag:
        work = df.assign(_p=_to_boolish(df[flag[0]]))
    elif b_cols:
        work = df.assign(_p=pd.to_numeric(df[b_cols[0]], errors="coerce").fillna(0) > 0)
    else:
        return None
    agg = (
        work.assign(_g=work[gender_col].map(canon_gender))
            .groupby("_g", dropna=False, observed=False)
            .agg(Proportion=("_p", "mean"), N=("_p", "size")).reset_index()
            .rename(columns={"_g":"Gender"})
    )
    agg["Proportion"] = (agg["Proportion"] * 100).round(1)
    agg["Suppressed"] = agg["N"] < 10
    return enforce_gender_row_order(agg)

def build_quartiles(df: pd.DataFrame, gender_col: str = "Gender") -> Optional[pd.DataFrame]:
    pay_cols = [c for c in df.columns if c.lower() in ["hourly", "hourly_pay", "wage", "pay", "pay rate", "rate of pay"]]
    if not pay_cols: return None
    col = pay_cols[0]
    if gender_col not in df.columns: return None

    w = df[[gender_col, col]].copy()
    w[col] = pd.to_numeric(w[col], errors="coerce")
    w = w.dropna(subset=[col])
    if len(w) < 4: return None

    ranks = w[col].rank(method="first")
    qlabels = ["Q1 (lowest)", "Q2", "Q3", "Q4 (highest)"]
    try:
        w["Quartile"] = pd.qcut(ranks, 4, labels=qlabels)
    except Exception:
        q = np.linspace(0, 1, 5)
        bins = np.quantile(ranks, q)
        bins = np.unique(bins)
        if len(bins) < 5: return None
        w["Quartile"] = pd.cut(ranks, bins=bins, labels=qlabels, include_lowest=True)

    w["Gender"] = w[gender_col].map(canon_gender)

    counts = w.groupby("Gender", dropna=False, observed=False).size().reset_index(name="N")
    pivot = (
        w.pivot_table(index="Gender", columns="Quartile", values=col, aggfunc="count", fill_value=0, observed=False)
         .reindex(columns=qlabels, fill_value=0)
    )
    pct = pivot.div(pivot.sum(axis=1).replace(0, pd.NA), axis=0) * 100
    pct = pct.round(1).reset_index()

    out = counts.merge(pct, on="Gender", how="left")
    out["Suppressed"] = out["N"] < 10
    return enforce_gender_row_order(out)

def build_report_tables(
    raw_frames: Dict[str, Optional[pd.DataFrame]],
    ref_gender: str = "Man",
    *,
    collapse_suppressed: bool = False,
) -> Dict[str, str]:
    gender        = raw_frames.get("gender")
    hourly        = raw_frames.get("hourly")
    overtime      = raw_frames.get("overtime")
    bonus         = raw_frames.get("bonus")
    participation = raw_frames.get("participation")
    quartiles     = raw_frames.get("quartiles")

    def _reorder_and_rename(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
        if df is None: return None
        wanted = ["Gender", "Mean", "Median", "Gap (mean)", "Gap (median)", "N", "Suppressed"]
        present = [c for c in wanted if c in df.columns]
        df2 = df[present].copy() if present else df.copy()
        # Shorter labels → no header wrap
        return df2.rename(columns={
            "Mean": "Mean ($)",
            "Median": "Median ($)",
            "Gap (mean)": "Gap mean (%)",
            "Gap (median)": "Gap median (%)",
        })

    if hourly   is not None: hourly   = _reorder_and_rename(hourly)
    if overtime is not None: overtime = _reorder_and_rename(overtime)
    if bonus    is not None: bonus    = _reorder_and_rename(bonus)

    def _reorder_quartiles(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
        if df is None: return None
        cols = ["Gender", "Q1 (lowest)", "Q2", "Q3", "Q4 (highest)", "N", "Suppressed"]
        present = [c for c in cols if c in df.columns]
        return df[present].copy()

    if quartiles is not None:
        quartiles = _reorder_quartiles(quartiles)

    gender_table = render_table(gender, rename={"Included": "Included?"}, int_cols=("Count",), collapse_suppressed=collapse_suppressed) or ""
    hourly_table = render_table(hourly,   money_cols=("Mean ($)", "Median ($)"), pct_cols=("Gap mean (%)","Gap median (%)"), int_cols=("N",), suppress_col="Suppressed", collapse_suppressed=collapse_suppressed) or ""
    overtime_pay = render_table(overtime, money_cols=("Mean ($)", "Median ($)"), pct_cols=("Gap mean (%)","Gap median (%)"), int_cols=("N",), suppress_col="Suppressed", collapse_suppressed=collapse_suppressed) or ""
    bonus_table  = render_table(bonus,    money_cols=("Mean ($)", "Median ($)"), pct_cols=("Gap mean (%)","Gap median (%)"), int_cols=("N",), suppress_col="Suppressed", collapse_suppressed=collapse_suppressed) or ""
    participation_table = render_table(participation, pct_cols=("Proportion receiving bonus pay",), int_cols=("N",), rename={"Proportion":"Proportion receiving bonus pay"}, suppress_col="Suppressed", collapse_suppressed=collapse_suppressed) or ""
    quartiles_table     = render_table(quartiles, pct_cols=("Q1 (lowest)","Q2","Q3","Q4 (highest)"), int_cols=("N",), suppress_col="Suppressed", collapse_suppressed=collapse_suppressed) or ""

    return {
        "gender_table":        gender_table,
        "hourly_table":        hourly_table,
        "overtime_pay":        overtime_pay,
        "bonus_table":         bonus_table,
        "participation_table": participation_table,
        "quartiles_table":     quartiles_table,
    }

# -----------------------------------------------------------------------------
# Summaries for the review page
# -----------------------------------------------------------------------------
def summarize_gender_counts(base_df: pd.DataFrame) -> dict:
    if "Gender" not in base_df.columns:
        return {"total": 0, "by_gender": {}, "suppressed": []}
    work = base_df.assign(_g=base_df["Gender"].map(canon_gender))
    counts = work.groupby("_g", dropna=False, observed=False).size().to_dict()
    suppressed = [g for g, n in counts.items() if n < 10]
    total = int(sum(counts.values()))
    ordered = {g: counts.get(g, 0) for g in GENDER_ORDER if g in counts}
    return {"total": total, "by_gender": ordered, "suppressed": suppressed}

def summarize_tables(tables_raw: dict) -> dict:
    out = {}
    for key, df in tables_raw.items():
        if df is None or not isinstance(df, pd.DataFrame) or len(df) == 0:
            out[key] = {"available": False, "published_rows": 0, "suppressed_rows": 0}
            continue
        col = "Suppressed" if "Suppressed" in df.columns else ("Status" if "Status" in df.columns else None)
        if col is None:
            out[key] = {"available": True, "published_rows": len(df), "suppressed_rows": 0}
            continue
        vals = df[col].astype(str).str.lower()
        sup_mask = vals.eq("suppressed") if col == "Status" else vals.isin(["true", "1", "yes", "y"])
        out[key] = {"available": True, "published_rows": int((~sup_mask).sum()), "suppressed_rows": int(sup_mask.sum())}
    return out

def _sanitize_numeric_cols(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty: return frame
    cols = [c for c in frame.columns if c.lower() in {
        "hourly","hourly_pay","wage","pay","pay rate","rate of pay",
        "overtime_pay","bonus","bonus_pay","incentive","variable pay","variable_pay",
        "commission","bonuspaytotal","commissionpaytotal","overtimepaytotal",
        "regularhourspaid","overtimehourspaid",
    }]
    out = frame.copy()
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out

# -----------------------------------------------------------------------------
# Internal helper: stage data and jump to /review/<id>
# -----------------------------------------------------------------------------
def _stage_and_redirect_to_review(base_df: pd.DataFrame, form_like):
    base_df = _sanitize_numeric_cols(base_df)
    ref_gender = (form_like.get("reference_gender") or "Man").strip() or "Man"

    df_gender        = build_gender_distribution(base_df)
    df_hourly        = build_hourly(base_df,   ref_gender=ref_gender)
    df_overtime      = build_overtime(base_df, ref_gender=ref_gender)
    df_bonus         = build_bonus(base_df,    ref_gender=ref_gender)
    df_participation = build_participation(base_df)
    df_quartiles     = build_quartiles(base_df)

    rendered = build_report_tables(
        {"gender":df_gender,"hourly":df_hourly,"overtime":df_overtime,"bonus":df_bonus,"participation":df_participation,"quartiles":df_quartiles},
        ref_gender=ref_gender,
        collapse_suppressed=True,
    )

    global NEXT_ID
    report_meta = {
        "id": NEXT_ID,
        "legal_name":       (form_like.get("legal_name") or "").strip(),
        "trade_name":       (form_like.get("trade_name") or "").strip(),
        "naics":            (form_like.get("naics") or "").strip(),
        "employees":        (form_like.get("employees") or "").strip(),
        "reference_gender": ref_gender,
        "start_date":       safe_date(form_like.get("start_date")),
        "end_date":         safe_date(form_like.get("end_date")),
        "posted":           form_like.get("posted", "None"),
        "contact":          (form_like.get("contact") or "").strip(),
        "year":             form_like.get("year") or datetime.now().year,
        "public_url":       None,
        "mailing_address":   (form_like.get("mailing_address") or "").strip(),
        "bc_headcount_range":(form_like.get("bc_headcount_range") or "").strip(),
    }

    REPORTS[NEXT_ID] = {
        "meta":   report_meta,
        "tables": rendered,
        "raw": {
            "df": base_df,
            "gender": df_gender, "hourly": df_hourly, "overtime": df_overtime,
            "bonus": df_bonus, "participation": df_participation, "quartiles": df_quartiles,
        },
    }
    rid = NEXT_ID
    NEXT_ID += 1
    return redirect(url_for("review", report_id=rid))

# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
@app.get("/")
def index():
    return render_template("upload.html", current_year=datetime.now().year)

@app.get("/download/bc_pay_template.csv")
def download_template_csv():
    csv_text = (
        "EmployeeID,Gender,Hourly,Overtime_Pay,Bonus,Received_Bonus\n"
        "1001,Woman,32.50,120.00,500,1\n"
        "1002,Man,29.75,0,0,0\n"
        "1003,Non-binary,35.00,210.50,1200,1\n"
        "1004,Prefer not to answer,27.00,0,0,0\n"
    )
    return Response(
        csv_text,
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=bc_pay_template.csv"}
    )

@app.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == "GET":
        return redirect(url_for("index"))

    form_meta = {
        k: (request.form.get(k) or "").strip()
        for k in ("legal_name","trade_name","naics","employees","contact","year",
                  "start_date","end_date","reference_gender","mailing_address","bc_headcount_range")
    }

    if not form_meta.get("bc_headcount_range"):
        derived = derive_bc_headcount_range(form_meta.get("employees"))
        if derived:
            form_meta["bc_headcount_range"] = derived
            flash(f"Derived B.C. headcount range: {derived} from reported count {form_meta.get('employees')}. You can override on the form.","info")

    try:
        year = int(form_meta.get("year") or datetime.now().year)
    except Exception:
        year = datetime.now().year

    mode = (request.form.get("mode") or "totals").lower()
    f = request.files.get("file")
    if not f or not f.filename:
        flash("Please choose a CSV or Excel file.","error")
        return redirect(url_for("index"))

    raw_path = os.path.join(tempfile.gettempdir(), f"{uuid4()}_{f.filename}")
    f.save(raw_path)

    ext = os.path.splitext(raw_path)[1].lower()
    if ext in (".xlsx",".xls"):
        try:
            df_sample = pd.read_excel(raw_path)
            csv_path = raw_path.replace(ext, ".csv")
            df_sample.to_csv(csv_path, index=False)
            raw_path = csv_path
        except Exception as e:
            flash(f"Could not read Excel file: {e}","error")
            return redirect(url_for("index"))

    try:
        df_head = pd.read_csv(raw_path, nrows=200)
    except Exception as e:
        flash(f"Could not read CSV: {e}","error")
        return redirect(url_for("index"))

    if mode == "totals" and _infer_mode_from_df(df_head) == "ledger":
        mode = "ledger"

    if mode == "ledger":
        out_path = os.path.join(tempfile.gettempdir(), f"totals_{uuid4()}.csv")
        try:
            transform_ledger_to_totals(ledger_csv=raw_path, output_csv=out_path, year=year, province_filter="BC")
        except Exception as e:
            flash(f"Could not convert your payroll ledger: {e}","error")
            return redirect(url_for("index"))
        try:
            _probe = pd.read_csv(out_path)
        except Exception:
            flash("Converted file couldn’t be read after transformation.","error")
            return redirect(url_for("index"))
        if _probe.empty:
            flash("After filtering by year/province there were no rows. Check that your ledger has B.C. rows and the selected year.","error")
            return redirect(url_for("index"))
        flash("Converted your payroll ledger to annual totals.","info")
        raw_path = out_path

    try:
        df_full = pd.read_csv(raw_path)
    except Exception as e:
        flash(f"Could not read file after conversion: {e}","error")
        return redirect(url_for("index"))

    colmap = auto_map_columns(df_full)
    has_gender = ("gender" in colmap) or ("Gender" in df_full.columns)
    has_any_pay = any(k in colmap for k in ("hourly","bonus","overtime_pay"))

    if has_gender and has_any_pay:
        base = apply_auto_mapping(df_full, colmap)

        class _F(dict):
            def get(self, k, default=None):
                return dict.__getitem__(self, k) if k in self else default

        preserved = _F({
            "legal_name":         form_meta.get("legal_name",""),
            "trade_name":         form_meta.get("trade_name",""),
            "naics":              form_meta.get("naics",""),
            "employees":          form_meta.get("employees",""),
            "reference_gender":   form_meta.get("reference_gender") or "Man",
            "start_date":         form_meta.get("start_date",""),
            "end_date":           form_meta.get("end_date",""),
            "posted":             "None",
            "contact":            form_meta.get("contact",""),
            "year":               form_meta.get("year",""),
            "mailing_address":    form_meta.get("mailing_address",""),
            "bc_headcount_range": form_meta.get("bc_headcount_range",""),
        })
        return _stage_and_redirect_to_review(base, preserved)

    csv_text = df_full.to_csv(index=False)
    guess = {
        "gender": colmap.get("gender",""),
        "hourly": colmap.get("hourly",""),
        "overtime_pay": colmap.get("overtime_pay",""),
        "bonus": colmap.get("bonus",""),
        "received_bonus": colmap.get("received_bonus",""),
    }
    return render_template(
        "mapping_confirm.html",
        columns=list(df_full.columns),
        guess=guess,
        raw_text=csv_text,
        meta={
            "legal_name":         form_meta.get("legal_name",""),
            "trade_name":         form_meta.get("trade_name",""),
            "naics":              form_meta.get("naics",""),
            "employees":          form_meta.get("employees",""),
            "reference_gender":   form_meta.get("reference_gender") or "Man",
            "start_date":         form_meta.get("start_date",""),
            "end_date":           form_meta.get("end_date",""),
            "posted":             "None",
            "contact":            form_meta.get("contact",""),
            "year":               form_meta.get("year",""),
            "mailing_address":    form_meta.get("mailing_address",""),
            "bc_headcount_range": form_meta.get("bc_headcount_range",""),
        },
    )

@app.post("/confirm_mapping")
def confirm_mapping():
    raw_text = request.form.get("raw_text","")
    if not raw_text.strip():
        flash("Missing data to map. Please upload again.","error")
        return redirect(url_for("index"))
    try:
        df = pd.read_csv(StringIO(raw_text))
    except Exception as e:
        flash(f"Could not read data: {e}","error")
        return redirect(url_for("index"))

    chosen = {
        "gender": request.form.get("map_gender","").strip(),
        "hourly": request.form.get("map_hourly","").strip(),
        "overtime_pay": request.form.get("map_overtime_pay","").strip(),
        "bonus": request.form.get("map_bonus","").strip(),
        "received_bonus": request.form.get("map_received_bonus","").strip(),
    }
    mapping = {k:v for k,v in chosen.items() if v}

    has_gender = ("gender" in mapping) or ("Gender" in df.columns)
    has_any_pay = any(k in mapping for k in ("hourly","bonus","overtime_pay"))
    if not (has_gender and has_any_pay):
        flash("Please choose a Gender column and at least one of Hourly, Bonus, or Overtime Pay.","error")
        return render_template(
            "mapping_confirm.html",
            columns=list(df.columns),
            guess={k: mapping.get(k,"") for k in ("gender","hourly","overtime_pay","bonus","received_bonus")},
            raw_text=raw_text,
            meta={k: request.form.get(k,"") for k in ("legal_name","trade_name","naics","employees","reference_gender",
                   "start_date","end_date","posted","contact","year","mailing_address","bc_headcount_range")}
        )

    base = apply_auto_mapping(df, mapping)

    class _F(dict):
        def get(self, k, default=None):
            return dict.__getitem__(self, k) if k in self else default

    preserved = _F({
        "legal_name":         request.form.get("legal_name",""),
        "trade_name":         request.form.get("trade_name",""),
        "naics":              request.form.get("naics",""),
        "employees":          request.form.get("employees",""),
        "reference_gender":   request.form.get("reference_gender","Man"),
        "start_date":         request.form.get("start_date",""),
        "end_date":           request.form.get("end_date",""),
        "posted":             request.form.get("posted","None"),
        "contact":            request.form.get("contact",""),
        "year":               request.form.get("year",""),
        "mailing_address":    request.form.get("mailing_address",""),
        "bc_headcount_range": request.form.get("bc_headcount_range",""),
    })
    return _stage_and_redirect_to_review(base, preserved)

@app.get("/review/<int:report_id>")
def review(report_id: int):
    item = REPORTS.get(report_id)
    if not item: abort(404)
    base_df = item["raw"]["df"]
    gender_summary = summarize_gender_counts(base_df)
    raw_frames = {
        "gender": item["raw"]["gender"], "hourly": item["raw"]["hourly"], "overtime": item["raw"]["overtime"],
        "bonus": item["raw"]["bonus"], "participation": item["raw"]["participation"], "quartiles": item["raw"]["quartiles"],
    }
    table_summary = summarize_tables(raw_frames)
    all_pay_tables_suppressed = all(
        s.get("available") and s.get("published_rows",0) == 0 and s.get("suppressed_rows",0) > 0
        for k,s in table_summary.items() if k in ("hourly","overtime","bonus")
    )
    return render_template("review.html",
        report=item["meta"], gender_summary=gender_summary, table_summary=table_summary,
        all_pay_tables_suppressed=all_pay_tables_suppressed, report_id=report_id)

@app.get("/report/<int:report_id>")
def report(report_id: int):
    item = REPORTS.get(report_id)
    if not item: abort(404)
    mode = (request.args.get("mode") or "clean").lower()
    collapse = (mode == "clean")
    is_admin = (request.args.get("admin") == "1")
    raw = item["raw"]
    tables = build_report_tables(
        {"gender":raw["gender"],"hourly":raw["hourly"],"overtime":raw["overtime"],"bonus":raw["bonus"],
         "participation":raw["participation"],"quartiles":raw["quartiles"]},
        ref_gender=item["meta"]["reference_gender"],
        collapse_suppressed=collapse,
    )
    item["tables"] = tables
    return render_template("report_bc.html", report=item["meta"], mode=mode, is_admin=is_admin, **tables)

@app.get("/download_pdf")
def download_pdf():
    report_id = request.args.get("report_id", type=int)
    item = REPORTS.get(report_id)
    if not item: abort(404)
    mode = (request.args.get("mode") or "clean").lower()
    collapse = (mode == "clean")
    raw = item["raw"]
    tables = build_report_tables(
        {"gender":raw["gender"],"hourly":raw["hourly"],"overtime":raw["overtime"],"bonus":raw["bonus"],
         "participation":raw["participation"],"quartiles":raw["quartiles"]},
        ref_gender=item["meta"]["reference_gender"],
        collapse_suppressed=collapse,
    )
    html = render_template("report_bc.html", report=item["meta"], mode=mode, **tables)
    html = _inject_print_css(html)

    if sync_playwright is not None:
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch()
                page = browser.new_page()
                page.emulate_media(media="print")
                page.set_content(html, wait_until="load")
                pdf_bytes = page.pdf(
                    format="A4",
                    landscape=True,
                    print_background=True,
                    prefer_css_page_size=True,
                    margin={"top":"12mm","right":"12mm","bottom":"12mm","left":"12mm"},
                )
                browser.close()
            return send_file(BytesIO(pdf_bytes), mimetype="application/pdf", as_attachment=True,
                             download_name=f"pay_transparency_report_{report_id}.pdf")
        except Exception:
            pass

    flash("PDF engine not available. Use the on-page Print / Save as PDF button.","info")
    return redirect(url_for("report", report_id=report_id, mode=mode))

@app.post("/publish")
def publish():
    rid = request.form.get("report_id", type=int)
    item = REPORTS.get(rid)
    if not item: abort(404)

    raw = item["raw"]
    tables = build_report_tables(
        {"gender":raw["gender"],"hourly":raw["hourly"],"overtime":raw["overtime"],"bonus":raw["bonus"],
         "participation":raw["participation"],"quartiles":raw["quartiles"]},
        ref_gender=item["meta"]["reference_gender"],
        collapse_suppressed=True,
    )
    html = render_template("report_bc.html", report=item["meta"], mode="clean", is_admin=False, **tables)
    html = _inject_print_css(html)

    out_dir = os.path.join(app.static_folder, "reports", str(rid))
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "index.html"), "w", encoding="utf-8") as f:
        f.write(html)

    if sync_playwright is not None:
        try:
            from pathlib import Path
            with sync_playwright() as p:
                browser = p.chromium.launch()
                page = browser.new_page()
                page.emulate_media(media="print")
                page.set_content(html, wait_until="load")
                page.pdf(
                    path=str(Path(os.path.join(out_dir, f"pay_transparency_report_{rid}.pdf"))),
                    format="A4",
                    landscape=True,
                    print_background=True,
                    prefer_css_page_size=True,
                    margin={"top":"12mm","right":"12mm","bottom":"12mm","left":"12mm"},
                )
                browser.close()
        except Exception:
            pass

    item["meta"]["posted"] = datetime.now().strftime("%Y-%m-%d")
    item["meta"]["public_url"] = url_for("static", filename=f"reports/{rid}/index.html")
    flash("Report published! Your public link is ready.","info")
    return redirect(url_for("report", report_id=rid, mode="clean"))

@app.get("/dl/<path:filename>")
def download_static(filename):
    return app.send_static_file(filename)

if __name__ == "__main__":
    app.config["TEMPLATES_AUTO_RELOAD"] = True
    # If 5000 is busy on macOS, switch to an open port
    port = int(os.environ.get("PORT", "5000"))
    try:
        app.run(debug=True, port=port)
    except OSError:
        app.run(debug=True, port=port+1)

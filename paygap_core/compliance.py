# paygap_core/compliance.py
from __future__ import annotations
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Dict, List, Tuple, Literal
import math, numpy as np, pandas as pd

G_WOMAN = "Woman"
G_MAN   = "Man"
G_NB    = "Non-binary, Two-Spirit, gender diverse"
G_PNA   = "Prefer not to answer / Unknown"
GENDER_ORDER = [G_WOMAN, G_MAN, G_NB, G_PNA]

NUM_COLS = ["hours_worked","ordinary_pay","special_salary","overtime_hours","overtime_pay","bonus_pay"]
REQ_COLS = ["employee_id","gender","province","reporting_period_start","reporting_period_end", *NUM_COLS]
ALLOWED_GENDERS = set(GENDER_ORDER)

@dataclass
class ValidationResult:
    ok: bool
    errors: List[str]
    warnings: List[str]
    cleaned: pd.DataFrame

def _fmt_money(x): 
    return "—" if x is None or (isinstance(x,float) and (math.isnan(x) or math.isinf(x))) else f"${x:,.2f}"

def _fmt_num(x):
    return "—" if x is None or (isinstance(x,float) and (math.isnan(x) or math.isinf(x))) else f"{x:,.2f}"

def _pct_gap(cat, ref):
    if ref in (None, 0) or cat in (None,):
        return "—"
    if isinstance(ref,float) and math.isnan(ref): return "—"
    if isinstance(cat,float) and math.isnan(cat): return "—"
    return f"{(ref-cat)/ref*100:.1f}%"

def validate_df(df: pd.DataFrame) -> ValidationResult:
    errors, warnings = [], []

    # Required columns
    missing = [c for c in REQ_COLS if c not in df.columns]
    if missing:
        errors.append(f"Missing columns: {', '.join(missing)}")
        return ValidationResult(False, errors, warnings, df)

    df = df.copy()

    # ---- Normalize text columns safely (use .str.* on Series) ----
    text_cols = ["employee_id","gender","province","reporting_period_start","reporting_period_end"]
    for c in text_cols:
        # make sure column exists and coerce to string first
        df[c] = df[c].astype(str).str.strip()

    # ---- Coerce numerics ----
    for c in NUM_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # ---- Province filter (BC only) ----
    non_bc = df[~df["province"].str.upper().eq("BC")].shape[0]
    if non_bc:
        warnings.append(f"{non_bc} row(s) are not BC; they will be ignored.")
    df = df[df["province"].str.upper().eq("BC")].copy()

    # ---- Gender codes ----
    bad_gender = df[~df["gender"].isin(ALLOWED_GENDERS)]
    if not bad_gender.empty:
        idxs = ", ".join(map(str, bad_gender.index.tolist()[:10]))
        errors.append(f"Invalid gender values in rows: {idxs} (allowed: {list(ALLOWED_GENDERS)})")

    # ---- Numeric sanity ----
    neg_rows = []
    for c in NUM_COLS:
        neg_rows += df[df[c] < 0].index.tolist()
    if neg_rows:
        errors.append(f"Negative numeric values at rows: {sorted(set(neg_rows))}")

    zero_hours_pay = df[(df["hours_worked"] == 0) & (df["ordinary_pay"] > 0)]
    if not zero_hours_pay.empty:
        errors.append(f"{len(zero_hours_pay)} row(s) have ordinary_pay > 0 but hours_worked = 0.")

    nan_hours_pay = df[df["hours_worked"].isna() & (df["ordinary_pay"] > 0)]
    if not nan_hours_pay.empty:
        errors.append(f"{len(nan_hours_pay)} row(s) have ordinary_pay > 0 but hours_worked is blank.")

    both = df[(df["special_salary"] > 0) & (df["ordinary_pay"] > 0)]
    if not both.empty:
        warnings.append(f"{len(both)} row(s) include both special_salary and ordinary_pay; review usage.")

    return ValidationResult(len(errors) == 0, errors, warnings, df)

def _agg(s: pd.Series) -> Tuple[float|None,float|None]:
    s = s.dropna()
    return (None, None) if s.empty else (float(round(s.mean(),2)), float(round(s.median(),2)))

def compute_aggregates(df: pd.DataFrame, suppression_threshold: int = 10,
                       reference: Literal["auto","Man","Woman","NB","PNA"]="auto") -> Dict:
    df = df.copy()
    df["hourly_pay"] = np.where(df["hours_worked"]>0, df["ordinary_pay"]/df["hours_worked"], np.nan)

    counts = df["gender"].value_counts().reindex(GENDER_ORDER).fillna(0).astype(int).to_dict()

    # choose reference gender
    def _ok(g): return counts.get(g,0) >= suppression_threshold
    if   reference=="Man"   and _ok(G_MAN):  ref = G_MAN
    elif reference=="Woman" and _ok(G_WOMAN):ref = G_WOMAN
    elif reference=="NB"    and _ok(G_NB):   ref = G_NB
    elif reference=="PNA"   and _ok(G_PNA):  ref = G_PNA
    else:
        ref = G_MAN if _ok(G_MAN) else (max({g:counts.get(g,0) for g in [G_WOMAN,G_MAN,G_NB]}.items(), key=lambda x: x[1])[0] if any(counts.get(g,0)>=suppression_threshold for g in [G_WOMAN,G_MAN,G_NB]) else G_PNA)

    out = {
        "gender_distribution": {"counts": counts, "publish": {g: counts[g]>=suppression_threshold for g in GENDER_ORDER}},
        "reference_gender": ref, "hourly": {}, "overtime": {}, "bonus": {}
    }

    for g in GENDER_ORDER:
        sub = df[df["gender"]==g]
        hp_mean, hp_med = _agg(sub["hourly_pay"])
        ot = sub[sub["overtime_pay"]>0];  ot_h_mean, ot_h_med = _agg(ot["overtime_hours"]);  ot_pay_mean, ot_pay_med = _agg(ot["overtime_pay"])
        b  = sub[sub["bonus_pay"]>0];     b_mean, b_med = _agg(b["bonus_pay"])
        out["hourly"][g]   = {"mean": hp_mean, "median": hp_med}
        out["overtime"][g] = {"hours_mean": ot_h_mean, "hours_median": ot_h_med, "pay_mean": ot_pay_mean, "pay_median": ot_pay_med}
        out["bonus"][g]    = {"mean": b_mean, "median": b_med}

    # build formatted tables with suppression + gaps
    ref_hp_mean = out["hourly"][ref]["mean"];  ref_hp_med = out["hourly"][ref]["median"]
    ref_ot_mean = out["overtime"][ref]["pay_mean"];  ref_b_mean = out["bonus"][ref]["mean"]

    tables = {"hourly":[], "overtime":[], "bonus":[]}
    for g in GENDER_ORDER:
        sup = counts[g] < suppression_threshold
        hp = out["hourly"][g]; ot = out["overtime"][g]; b = out["bonus"][g]
        tables["hourly"].append({
            "gender": g, "mean": "Suppressed" if sup else _fmt_money(hp["mean"]),
            "median":"Suppressed" if sup else _fmt_money(hp["median"]),
            "gap_mean": "Reference" if g==ref else ("Suppressed" if sup else _pct_gap(hp["mean"], ref_hp_mean)),
            "gap_median":"Reference" if g==ref else ("Suppressed" if sup else _pct_gap(hp["median"], ref_hp_med)),
        })
        tables["overtime"].append({
            "gender": g,
            "hours_mean":"Suppressed" if sup else _fmt_num(ot["hours_mean"]),
            "hours_median":"Suppressed" if sup else _fmt_num(ot["hours_median"]),
            "pay_mean":"Suppressed" if sup else _fmt_money(ot["pay_mean"]),
            "pay_median":"Suppressed" if sup else _fmt_money(ot["pay_median"]),
            "gap_pay_mean":"Reference" if g==ref else ("Suppressed" if sup else _pct_gap(ot["pay_mean"], ref_ot_mean)),
        })
        tables["bonus"].append({
            "gender": g, "mean":"Suppressed" if sup else _fmt_money(b["mean"]),
            "median":"Suppressed" if sup else _fmt_money(b["median"]),
            "gap_mean":"Reference" if g==ref else ("Suppressed" if sup else _pct_gap(b["mean"], ref_b_mean)),
        })
    out["tables"] = tables

        # --- Availability flags: if all published rows are zeros, mark section unavailable ---
    pub = out["gender_distribution"]["publish"]

    def _all_published_zero_hourly() -> bool:
        pubs = [g for g in GENDER_ORDER if pub.get(g)]
        if not pubs:
            return False
        return all(
            (out["hourly"][g]["mean"] in (None, 0.0)) and (out["hourly"][g]["median"] in (None, 0.0))
            for g in pubs
        )

    def _all_published_zero_otpay() -> bool:
        pubs = [g for g in GENDER_ORDER if pub.get(g)]
        if not pubs:
            return False
        return all(
            (out["overtime"][g]["pay_mean"] in (None, 0.0)) and (out["overtime"][g]["pay_median"] in (None, 0.0))
            for g in pubs
        )

    def _all_published_zero_bonus() -> bool:
        pubs = [g for g in GENDER_ORDER if pub.get(g)]
        if not pubs:
            return False
        return all(
            (out["bonus"][g]["mean"] in (None, 0.0)) and (out["bonus"][g]["median"] in (None, 0.0))
            for g in pubs
        )

    out["availability"] = {
        "hourly_all_zero":  _all_published_zero_hourly(),
        "overtime_all_zero": _all_published_zero_otpay(),
        "bonus_all_zero":    _all_published_zero_bonus(),
    }

    # (Optional, belt-and-suspenders) turn 0/0 gaps into "—" even if anything upstream changes
    if ref_hp_mean in (0, None):
        for r in out["tables"]["hourly"]:
            if r["gap_mean"] not in ("Reference", "Suppressed"):
                r["gap_mean"] = "—"
            if r["gap_median"] not in ("Reference", "Suppressed"):
                r["gap_median"] = "—"
    if ref_ot_mean in (0, None):
        for r in out["tables"]["overtime"]:
            if r["gap_pay_mean"] not in ("Reference", "Suppressed"):
                r["gap_pay_mean"] = "—"
    if ref_b_mean in (0, None):
        for r in out["tables"]["bonus"]:
            if r["gap_mean"] not in ("Reference", "Suppressed"):
                r["gap_mean"] = "—"

    return out

def fill_html_template(aggregates: Dict, meta: Dict, template_path: str|Path, out_path: str|Path) -> Path:
    html = Path(template_path).read_text(encoding="utf-8")
    tokens = {
        "[[ORG_NAME]]": meta.get("org_name","Your Company"),
        "[[ORG_LEGAL_NAME]]": meta.get("org_legal_name", meta.get("org_name","Your Company")),
        "[[ORG_TRADE_NAME]]": meta.get("org_trade_name", meta.get("org_name","Your Company")),
        "[[REPORT_YEAR]]": str(meta.get("report_year", date.today().year)),
        "[[PERIOD_START]]": str(meta.get("period_start","")), "[[PERIOD_END]]": str(meta.get("period_end","")),
        "[[POSTED_DATE]]": str(meta.get("posted_date", date.today())), "[[CONTACT_EMAIL]]": meta.get("contact_email","info@example.com"),
        "[[NAICS]]": meta.get("naics",""), "[[BC_HEADCOUNT]]": str(meta.get("bc_headcount","")),
        "[[REFERENCE_GENDER]]": aggregates["reference_gender"],
        "[[NARRATIVE_SUMMARY]]": meta.get("narrative",""),
        "[[ACTION_1]]": meta.get("action_1",""), "[[ACTION_2]]": meta.get("action_2",""), "[[ACTION_3]]": meta.get("action_3",""),
        "[[PLAN_1]]": meta.get("plan_1",""), "[[PLAN_2]]": meta.get("plan_2",""), "[[PLAN_3]]": meta.get("plan_3",""),
    }
    counts = aggregates["gender_distribution"]["counts"]; pub = aggregates["gender_distribution"]["publish"]
    tokens.update({
        "[[COUNT_WOMAN]]": str(counts.get(G_WOMAN,0)), "[[COUNT_MAN]]": str(counts.get(G_MAN,0)),
        "[[COUNT_NB]]": str(counts.get(G_NB,0)), "[[COUNT_PNA]]": str(counts.get(G_PNA,0)),
        "[[PUBLISH_WOMAN]]": "Published" if pub.get(G_WOMAN) else "Suppressed (n<10)",
        "[[PUBLISH_MAN]]": "Published" if pub.get(G_MAN) else "Suppressed (n<10)",
        "[[PUBLISH_NB]]": "Published" if pub.get(G_NB) else "Suppressed (n<10)",
        "[[PUBLISH_PNA]]": "Published" if pub.get(G_PNA) else "Suppressed (n<10)",
    })
    def _grab(tbl, g, key): return next(r for r in aggregates["tables"][tbl] if r["gender"]==g)[key]
    # hourly
    for g, tag in [(G_WOMAN,"W"),(G_MAN,"M"),(G_NB,"NB"),(G_PNA,"PNA")]:
        tokens.update({
            f"[[HP_MEAN_{tag}]]": _grab("hourly",g,"mean"), f"[[HP_MED_{tag}]]": _grab("hourly",g,"median"),
            f"[[HP_GAP_MEAN_{tag}]]": _grab("hourly",g,"gap_mean"), f"[[HP_GAP_MED_{tag}]]": _grab("hourly",g,"gap_median")
        })
    # OT
    for g, tag in [(G_WOMAN,"W"),(G_MAN,"M"),(G_NB,"NB"),(G_PNA,"PNA")]:
        tokens.update({
            f"[[OT_H_MEAN_{tag}]]": _grab("overtime",g,"hours_mean"),
            f"[[OT_H_MED_{tag}]]": _grab("overtime",g,"hours_median"),
            f"[[OT_PAY_MEAN_{tag}]]": _grab("overtime",g,"pay_mean"),
            f"[[OT_PAY_MED_{tag}]]": _grab("overtime",g,"pay_median"),
            f"[[OT_GAP_MEAN_{tag}]]": _grab("overtime",g,"gap_pay_mean"),
        })
    # Bonus
    for g, tag in [(G_WOMAN,"W"),(G_MAN,"M"),(G_NB,"NB"),(G_PNA,"PNA")]:
        tokens.update({
            f"[[B_MEAN_{tag}]]": _grab("bonus",g,"mean"),
            f"[[B_MED_{tag}]]": _grab("bonus",g,"median"),
            f"[[B_GAP_MEAN_{tag}]]": _grab("bonus",g,"gap_mean"),
        })
    for k,v in tokens.items(): html = html.replace(k, str(v))
    out_path = Path(out_path); out_path.write_text(html, encoding="utf-8"); return out_path

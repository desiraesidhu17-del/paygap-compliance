from __future__ import annotations

import math

import pandas as pd
from flask import Blueprint, abort, render_template, request

from paygap_compliance.services.metrics import compute_all_metrics
from paygap_compliance.services.schema import coerce_df

report_bp = Blueprint("report", __name__)


@report_bp.post("/report")
def preview_report():
    uploaded = request.files.get("file")
    if uploaded is None or uploaded.filename == "":
        abort(400, "CSV file required")

    try:
        df = pd.read_csv(uploaded)
    except Exception as exc:  # pragma: no cover - pandas should be deterministic here
        abort(400, f"Could not read CSV: {exc}")

    coerced = coerce_df(df)
    metrics = compute_all_metrics(coerced)
    hourly_rows = metrics.get("hourly", [])

    display_rows = []
    for row in hourly_rows:
        hourly_mean = row.get("HourlyMean")
        if isinstance(hourly_mean, (int, float)):
            if math.isnan(hourly_mean):
                hourly_display = ""
            else:
                hourly_display = f"{hourly_mean:.2f}"
        else:
            hourly_display = hourly_mean or ""

        display_rows.append(
            {
                "Gender": row.get("Gender", ""),
                "Count": row.get("Count", 0),
                "HourlyMean": hourly_display,
                "Suppressed": row.get("Suppressed", False),
            }
        )

    return render_template("preview.html", rows=display_rows)

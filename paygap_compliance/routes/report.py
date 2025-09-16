from __future__ import annotations

import asyncio
import math
import numbers

import pandas as pd
from flask import Blueprint, Response, abort, render_template, request, session

from paygap_compliance.services.metrics import compute_all_metrics
from paygap_compliance.services.pdf import html_to_pdf
from paygap_compliance.services.schema import coerce_df

report_bp = Blueprint("report", __name__)


@report_bp.get("/")
def upload_form():
    return render_template("upload.html")


@report_bp.post("/report")
def preview_report():
    rows = _rows_from_upload(request.files.get("file"))
    return render_template("preview.html", rows=rows)


@report_bp.post("/report/pdf")
def preview_report_pdf():
    if not session.get("paid"):
        return {"pay_url": "/fake-checkout"}, 402
    rows = _rows_from_upload(request.files.get("file"))
    html = render_template("preview.html", rows=rows)
    try:
        pdf_bytes = asyncio.run(html_to_pdf(html))
    except RuntimeError as exc:
        abort(500, str(exc))
    return Response(  # noqa: B950
        pdf_bytes,
        mimetype="application/pdf",
        headers={"Content-Disposition": "inline; filename=preview.pdf"},
    )


def _rows_from_upload(uploaded) -> list[dict]:
    if uploaded is None or uploaded.filename == "":
        abort(400, "CSV file required")

    try:
        df = pd.read_csv(uploaded)
    except Exception as exc:  # pragma: no cover - pandas should be deterministic here
        abort(400, f"Could not read CSV: {exc}")

    coerced = coerce_df(df)
    metrics = compute_all_metrics(coerced)
    hourly_rows = metrics.get("hourly", [])
    return [_format_row(row) for row in hourly_rows]


def _format_row(row: dict) -> dict:
    hourly_mean = row.get("HourlyMean")
    if isinstance(hourly_mean, numbers.Number):
        if math.isnan(hourly_mean):
            hourly_display = ""
        else:
            hourly_display = f"{hourly_mean:.2f}"
    else:
        hourly_display = hourly_mean or ""

    return {
        "Gender": row.get("Gender", ""),
        "Count": row.get("Count", 0),
        "HourlyMean": hourly_display,
        "Suppressed": row.get("Suppressed", False),
    }

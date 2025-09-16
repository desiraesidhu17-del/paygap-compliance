import io

import pytest

from app import app


@pytest.fixture
def client():
    with app.test_client() as client:
        yield client


def _sample_data():
    csv_content = (
        "Gender,OrdinaryPay,SpecialSalary,HoursWorked,OTHours,OTPay,Bonus\n"
        "Man,52000,0,2080,0,0,0\n"
        "Woman,50000,0,2080,0,0,0\n"
    )
    return {"file": (io.BytesIO(csv_content.encode("utf-8")), "sample.csv")}


def test_report_pdf_requires_payment(client):
    response = client.post("/report/pdf", data=_sample_data(), content_type="multipart/form-data")
    assert response.status_code == 402
    assert response.get_json() == {"pay_url": "/fake-checkout"}


def test_report_pdf_returns_bytes_when_paid(client):
    pytest.importorskip("playwright.async_api", reason="Playwright required for PDF export tests")

    with client.session_transaction() as sess:
        sess["paid"] = True

    response = client.post("/report/pdf", data=_sample_data(), content_type="multipart/form-data")

    assert response.status_code == 200
    assert response.headers.get("Content-Type") == "application/pdf"
    assert len(response.data) > 1024

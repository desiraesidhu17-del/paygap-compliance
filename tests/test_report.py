import io

import pytest

from app import app


@pytest.fixture
def client():
    with app.test_client() as client:
        yield client


def test_report_preview_returns_table(client):
    csv_content = (
        "Gender,OrdinaryPay,SpecialSalary,HoursWorked,OTHours,OTPay,Bonus\n"
        "Man,52000,0,2080,0,0,0\n"
        "Woman,50000,0,2080,0,0,0\n"
    )
    data = {"file": (io.BytesIO(csv_content.encode("utf-8")), "sample.csv")}

    response = client.post("/report", data=data, content_type="multipart/form-data")

    assert response.status_code == 200
    body = response.get_data(as_text=True)
    assert "Gender" in body
    assert "Hourly Metrics Preview" in body

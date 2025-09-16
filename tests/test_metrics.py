import pandas as pd
from services.metrics import compute_all_metrics

def df_base():
    return pd.DataFrame([
        {"Gender":"Man","OrdinaryPay":52000,"SpecialSalary":0,"HoursWorked":2080,"OTHours":0,"OTPay":0,"Bonus":1000},
        {"Gender":"Woman","OrdinaryPay":50000,"SpecialSalary":0,"HoursWorked":2080,"OTHours":0,"OTPay":0,"Bonus":0},
    ] * 10)  # 20 rows total => each group >=10

def test_reference_is_men_when_available():
    res = compute_all_metrics(df_base(), threshold=10)
    assert res["reference"] == "M"

def test_suppression_kicks_in_under_10():
    small = pd.DataFrame([
        {"Gender":"Man","OrdinaryPay":52000,"SpecialSalary":0,"HoursWorked":2080,"OTHours":0,"OTPay":0,"Bonus":0},
        {"Gender":"Woman","OrdinaryPay":50000,"SpecialSalary":0,"HoursWorked":2080,"OTHours":0,"OTPay":0,"Bonus":0},
    ])  # each group n=1
    res = compute_all_metrics(small, threshold=10)
    hourly = pd.DataFrame(res["hourly"])
    assert hourly["Suppressed"].all()


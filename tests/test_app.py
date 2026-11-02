import pytest

from app import app

CUSTOMER = {
    "tenure": 2, "MonthlyCharges": 90, "TotalCharges": 180, "gender": "Female", "SeniorCitizen": 0,
    "Partner": "No", "Dependents": "No", "PhoneService": "Yes", "MultipleLines": "No",
    "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
    "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No", "StreamingMovies": "No",
    "Contract": "Month-to-month", "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check",
}


@pytest.fixture
def client():
    return app.test_client()


def test_home_page(client):
    assert client.get("/").status_code == 200


def test_risky_customer_gets_high_score(client):
    res = client.post("/predict", json=CUSTOMER)
    body = res.get_json()
    assert res.status_code == 200
    assert 0 <= body["churn_probability"] <= 1
    assert body["risk"] == "High"


def test_loyal_customer_gets_low_score(client):
    loyal = dict(CUSTOMER, tenure=60, MonthlyCharges=55, TotalCharges=3300, Contract="Two year",
                 InternetService="DSL", PaymentMethod="Credit card (automatic)",
                 OnlineSecurity="Yes", TechSupport="Yes")
    body = client.post("/predict", json=loyal).get_json()
    assert body["risk"] == "Low"


def test_missing_fields(client):
    res = client.post("/predict", json={"tenure": 5})
    assert res.status_code == 400
    assert "missing" in res.get_json()["error"]


def test_negative_number(client):
    res = client.post("/predict", json=dict(CUSTOMER, tenure=-3))
    assert res.status_code == 400

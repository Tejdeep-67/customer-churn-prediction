"""
Checks that the maths in docs/app.js (scaled numbers x coef + one-hot weights, then sigmoid)
gives the same probabilities as the trained sklearn pipeline. The formula is copied here in
Python so it can run with pytest.
"""
import json
import math
import os

import joblib
import pytest

from src.train import load_data

BASE = os.path.join(os.path.dirname(__file__), "..")
MODEL_JSON = os.path.join(BASE, "docs", "model.json")
MODEL_PKL = os.path.join(BASE, "models", "churn_model.joblib")


def web_predict(m, row):
    z = m["intercept"]
    for col, p in m["numeric"].items():
        z += (row[col] - p["mean"]) / p["scale"] * p["coef"]
    for col, cats in m["categorical"].items():
        z += cats.get(str(row[col]), 0)
    return 1 / (1 + math.exp(-z))


@pytest.mark.skipif(not os.path.exists(MODEL_PKL), reason="run python src/train.py first")
def test_browser_formula_matches_sklearn():
    with open(MODEL_JSON) as f:
        m = json.load(f)
    pipe = joblib.load(MODEL_PKL)
    X, _ = load_data()
    sample = X.sample(300, random_state=1)
    expected = pipe.predict_proba(sample)[:, 1]
    for (_, row), p in zip(sample.iterrows(), expected):
        assert abs(web_predict(m, row) - p) < 1e-6


def test_average_contributions_cover_every_column():
    with open(MODEL_JSON) as f:
        m = json.load(f)
    assert set(m["average_contribution"]) == set(m["categorical"])
    q = m["test_percentiles"]
    assert len(q) == 101 and q == sorted(q)

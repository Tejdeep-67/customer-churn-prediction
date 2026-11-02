"""
Small Flask app for the churn model.

  GET  /          simple form
  POST /predict   JSON in, churn probability out

Run:  python src/train.py   (once, to create the model)
      python app.py
"""
import os

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

from src.train import CAT_COLS, NUM_COLS

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "churn_model.joblib")

app = Flask(__name__)
model = None


def get_model():
    global model
    if model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError("model not found, run: python src/train.py")
        model = joblib.load(MODEL_PATH)
    return model


def risk_level(p):
    if p >= 0.7:
        return "High"
    if p >= 0.4:
        return "Medium"
    return "Low"


def top_reasons(pipe, row, n=3):
    """features that pushed this customer's score up the most (coef * value)"""
    prep = pipe.named_steps["prep"]
    coef = pipe.named_steps["model"].coef_[0]
    values = prep.transform(row)
    values = values.toarray()[0] if hasattr(values, "toarray") else np.asarray(values)[0]
    contrib = pd.Series(values * coef, index=prep.get_feature_names_out())
    contrib = contrib[contrib > 0].sort_values(ascending=False).head(n)
    return [name.split("__", 1)[1] for name in contrib.index]


def validate(data):
    missing = [c for c in NUM_COLS + CAT_COLS if c not in data]
    if missing:
        return f"missing fields: {', '.join(missing)}"
    for c in NUM_COLS:
        try:
            if float(data[c]) < 0:
                return f"{c} can't be negative"
        except (TypeError, ValueError):
            return f"{c} must be a number"
    return None


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "send the customer as JSON"}), 400

    err = validate(data)
    if err:
        return jsonify({"error": err}), 400

    row = {c: float(data[c]) for c in NUM_COLS}
    row.update({c: data[c] for c in CAT_COLS})
    row["SeniorCitizen"] = int(row["SeniorCitizen"])  # 0/1 in the training data
    row = pd.DataFrame([row])

    pipe = get_model()
    p = float(pipe.predict_proba(row)[0, 1])
    return jsonify({
        "churn_probability": round(p, 3),
        "risk": risk_level(p),
        "top_reasons": top_reasons(pipe, row),
    })


if __name__ == "__main__":
    app.run(debug=True)

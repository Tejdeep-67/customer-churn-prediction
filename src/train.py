"""
Trains the final churn model and saves:
  models/churn_model.joblib   -> used by the Flask app
  docs/model.json             -> coefficients for the web demo (runs in the browser)
  models/metrics.json         -> test set scores

Run from the project root:  python src/train.py
"""
import json
import os

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE = os.path.join(os.path.dirname(__file__), "..")
DATA = os.path.join(BASE, "data", "Telco-Customer-Churn.csv")

NUM_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]
CAT_COLS = ["gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService", "MultipleLines",
            "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
            "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod"]


def load_data():
    df = pd.read_csv(DATA)
    # 11 new customers (tenure 0) have a blank TotalCharges
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
    X = df[NUM_COLS + CAT_COLS]
    y = (df["Churn"] == "Yes").astype(int)
    return X, y


def build_pipeline():
    preprocess = ColumnTransformer([
        ("num", StandardScaler(), NUM_COLS),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
    ])
    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    return Pipeline([("prep", preprocess), ("model", model)])


def export_for_web(pipe, path, X_train, test_proba):
    """save scaler + one-hot + coefficients so the same maths can run in javascript"""
    prep = pipe.named_steps["prep"]
    scaler = prep.named_transformers_["num"]
    ohe = prep.named_transformers_["cat"]
    coef = pipe.named_steps["model"].coef_[0]

    out = {
        "intercept": float(pipe.named_steps["model"].intercept_[0]),
        "numeric": {},
        "categorical": {},
    }
    i = 0
    for j, col in enumerate(NUM_COLS):
        out["numeric"][col] = {"mean": float(scaler.mean_[j]), "scale": float(scaler.scale_[j]), "coef": float(coef[i])}
        i += 1
    for col, cats in zip(CAT_COLS, ohe.categories_):
        out["categorical"][col] = {}
        for c in cats:
            out["categorical"][col][str(c)] = float(coef[i])
            i += 1

    # for the "why" chart in the demo: the average contribution of each feature over the
    # training customers. A customer's bar = their contribution minus this average, so the
    # bars show how they differ from a typical customer (numeric columns average to 0 already).
    out["average_contribution"] = {}
    for col in CAT_COLS:
        freq = X_train[col].astype(str).value_counts(normalize=True)
        out["average_contribution"][col] = float(sum(out["categorical"][col].get(k, 0) * v for k, v in freq.items()))

    # percentiles of the test set scores, so the demo can say "riskier than x% of customers"
    out["test_percentiles"] = [round(float(q), 4) for q in pd.Series(test_proba).quantile([i / 100 for i in range(101)])]

    with open(path, "w") as f:
        json.dump(out, f, indent=1)


def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    pipe = build_pipeline()
    pipe.fit(X_train, y_train)

    proba = pipe.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    metrics = {
        "roc_auc": round(roc_auc_score(y_test, proba), 3),
        "accuracy": round(accuracy_score(y_test, pred), 3),
        "precision": round(precision_score(y_test, pred), 3),
        "recall": round(recall_score(y_test, pred), 3),
        "f1": round(f1_score(y_test, pred), 3),
        "test_size": int(len(y_test)),
    }
    print(metrics)

    os.makedirs(os.path.join(BASE, "models"), exist_ok=True)
    os.makedirs(os.path.join(BASE, "docs"), exist_ok=True)
    joblib.dump(pipe, os.path.join(BASE, "models", "churn_model.joblib"))
    with open(os.path.join(BASE, "models", "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    export_for_web(pipe, os.path.join(BASE, "docs", "model.json"), X_train, proba)
    print("saved model to models/ and docs/model.json")


if __name__ == "__main__":
    main()

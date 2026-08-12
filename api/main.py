"""
Phase 10: Model Deployment
FastAPI real-time scoring service for credit risk and fraud detection.

Run with:  uvicorn api.main:app --reload --port 8000
Docs at:   http://127.0.0.1:8000/docs
"""
import sys
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
sys.path.insert(0, str(ROOT / "src"))

app = FastAPI(
    title="Credit Risk & Fraud Detection API",
    description="Real-time credit scoring and fraud detection endpoints.",
    version="1.0.0",
)


class CreditApplication(BaseModel):
    age: int
    income: float
    employment_length: int
    loan_amount: float
    credit_score: int
    existing_debt: float
    num_credit_accounts: int
    credit_history_years: int
    num_late_payments: int
    home_ownership: str
    loan_purpose: str


class Transaction(BaseModel):
    user_id: int
    amount: float
    timestamp: str
    merchant_category: str
    location: str
    device_type: str
    hour_of_day: int
    is_weekend: int
    distance_from_home: float


def _load(name: str):
    path = MODELS_DIR / name
    if not path.exists():
        raise HTTPException(status_code=503, detail=f"Model artifact '{name}' not found. Run pipeline.py first.")
    return joblib.load(path)


@app.get("/")
def root():
    return {"status": "ok", "endpoints": ["/score/credit", "/score/fraud"]}


@app.post("/score/credit")
def score_credit(application: CreditApplication):
    import feature_engineering as fe

    model = _load("credit_best_model.pkl")
    scaler = _load("credit_scaler.pkl")
    encoders = _load("credit_encoders.pkl")
    feature_cols = _load("credit_feature_cols.pkl")

    df = pd.DataFrame([application.dict()])
    df = fe.engineer_credit_features(df)
    for col, le in encoders.items():
        df[col] = le.transform(df[col].astype(str))

    X = df[feature_cols]
    num_cols = X.select_dtypes(include="number").columns.tolist()
    X[num_cols] = scaler.transform(X[num_cols])

    pd_score = float(model.predict_proba(X)[:, 1][0])
    expected_loss = pd_score * 0.45 * application.loan_amount
    bins = [0, 0.02, 0.05, 0.10, 0.20, 1.0]
    labels = ["A", "B", "C", "D", "E"]
    rating = str(pd.cut([pd_score], bins=bins, labels=labels)[0])

    return {
        "probability_of_default": round(pd_score, 4),
        "expected_loss": round(expected_loss, 2),
        "credit_rating": rating,
        "recommendation": "approve" if pd_score < 0.05 else ("review" if pd_score < 0.15 else "decline"),
    }


@app.post("/score/fraud")
def score_fraud(transaction: Transaction):
    model = _load("fraud_best_model.pkl")
    scaler = _load("fraud_scaler.pkl")
    encoders = _load("fraud_encoders.pkl")
    feature_cols = _load("fraud_feature_cols.pkl")

    df = pd.DataFrame([transaction.dict()])
    df["is_fraud"] = 0  # placeholder needed for feature engineering merchant-risk calc

    import feature_engineering as fe
    df = fe.engineer_fraud_features(df)

    for col, le in encoders.items():
        val = str(df[col].iloc[0])
        df[col] = le.transform([val]) if val in le.classes_ else [-1]

    X = df[feature_cols]
    num_cols = X.select_dtypes(include="number").columns.tolist()
    X[num_cols] = scaler.transform(X[num_cols])

    proba = float(model.predict_proba(X)[:, 1][0])
    risk_level = "HIGH" if proba >= 0.7 else ("MEDIUM" if proba >= 0.3 else "LOW")

    return {
        "fraud_probability": round(proba, 4),
        "risk_level": risk_level,
    }

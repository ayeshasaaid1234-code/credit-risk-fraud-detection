"""
Phase 4: Feature Engineering
Credit-risk features and fraud-detection features.
"""
import pandas as pd
import numpy as np


# ---------------------------------------------------------------- CREDIT ----

def engineer_credit_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Credit utilization ratio: existing debt relative to income
    df["credit_utilization_ratio"] = (df["existing_debt"] / (df["income"] + 1)).round(4)

    # Debt-to-income ratio
    df["debt_to_income_ratio"] = (
        (df["existing_debt"] + df["loan_amount"]) / (df["income"] + 1)
    ).round(4)

    # Payment history score (fewer late payments = higher score)
    df["payment_history_score"] = (1 / (1 + df["num_late_payments"])).round(4)

    # Credit age feature: history years relative to applicant age
    df["credit_age_ratio"] = (df["credit_history_years"] / (df["age"] + 1)).round(4)

    # Number-of-accounts feature: accounts opened per year of history
    df["account_density"] = (
        df["num_credit_accounts"] / (df["credit_history_years"] + 1)
    ).round(4)

    # Loan-to-income ratio
    df["loan_to_income_ratio"] = (df["loan_amount"] / (df["income"] + 1)).round(4)

    # Interaction: income stability proxy
    df["income_stability"] = df["employment_length"] * np.log1p(df["income"])

    return df


# ----------------------------------------------------------------- FRAUD ----

def engineer_fraud_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values(["user_id", "timestamp"])

    # Transaction velocity: transactions per user in trailing 24h
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp")
    velocity = (
        df.groupby("user_id")["amount"]
        .rolling("24h").count()
        .reset_index(level=0, drop=True)
    )
    df["transaction_velocity_24h"] = velocity.values
    df = df.reset_index()

    # Location anomaly flag
    df["is_international"] = (df["location"] == "international").astype(int)
    df["location_anomaly"] = ((df["distance_from_home"] > 100) | (df["is_international"] == 1)).astype(int)

    # Time-based features
    df["is_night_transaction"] = ((df["hour_of_day"] < 6) | (df["hour_of_day"] > 22)).astype(int)

    # Merchant category risk (encode by historical fraud propensity in this sample)
    merchant_risk = df.groupby("merchant_category")["is_fraud"].transform("mean")
    df["merchant_category_risk"] = merchant_risk.round(4)

    # User behavior pattern: deviation from user's typical transaction amount
    user_avg = df.groupby("user_id")["amount"].transform("mean")
    user_std = df.groupby("user_id")["amount"].transform("std").fillna(1)
    df["amount_zscore_vs_user"] = ((df["amount"] - user_avg) / (user_std + 1e-6)).round(4)

    # High velocity + odd hour + large amount combo flag
    df["composite_risk_flag"] = (
        (df["transaction_velocity_24h"] > df["transaction_velocity_24h"].quantile(0.95)).astype(int)
        + df["is_night_transaction"]
        + df["location_anomaly"]
    )

    return df

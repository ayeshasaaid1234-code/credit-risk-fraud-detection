"""
Phase 1: Data Collection & Preparation
Generates realistic SYNTHETIC credit-applicant and transaction datasets.

Replace this module's output with real datasets (same schema) when available —
everything downstream only depends on the column names defined here.
"""
import numpy as np
import pandas as pd
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

RNG = np.random.default_rng(42)


def generate_credit_data(n_samples: int = 10000) -> pd.DataFrame:
    """Generate synthetic credit applicant data with a realistic default rate (~8%)."""
    age = RNG.integers(21, 70, n_samples)
    income = np.round(RNG.lognormal(mean=10.5, sigma=0.5, size=n_samples), 2)
    employment_length = RNG.integers(0, 40, n_samples)
    loan_amount = np.round(RNG.lognormal(mean=9.2, sigma=0.6, size=n_samples), 2)
    credit_score = RNG.normal(650, 80, n_samples).clip(300, 850).astype(int)
    existing_debt = np.round(RNG.lognormal(mean=8.5, sigma=0.8, size=n_samples), 2)
    num_credit_accounts = RNG.integers(0, 15, n_samples)
    credit_history_years = RNG.integers(0, 30, n_samples)
    num_late_payments = RNG.poisson(1.2, n_samples)
    home_ownership = RNG.choice(["OWN", "MORTGAGE", "RENT"], n_samples, p=[0.25, 0.4, 0.35])
    loan_purpose = RNG.choice(
        ["debt_consolidation", "credit_card", "home_improvement", "car", "business", "other"],
        n_samples, p=[0.3, 0.25, 0.15, 0.15, 0.1, 0.05]
    )

    # Latent default-risk score drives the target (stronger signal, less noise
    # than a real-world dataset would have, so the demo models train well)
    z_credit = (credit_score - 650) / 80
    z_income = (income - income.mean()) / income.std()
    z_debt = (existing_debt - existing_debt.mean()) / existing_debt.std()
    z_hist = (credit_history_years - credit_history_years.mean()) / credit_history_years.std()
    z_emp = (employment_length - employment_length.mean()) / employment_length.std()

    risk_score = (
        -1.8 * z_credit
        + 0.9 * num_late_payments
        - 0.5 * z_emp
        + 0.7 * z_debt
        - 0.4 * z_income
        - 0.6 * z_hist
        + RNG.normal(0, 0.6, n_samples)  # residual noise (keeps it realistic, not deterministic)
    )
    default_prob = 1 / (1 + np.exp(-(risk_score - 1.0)))
    default = RNG.binomial(1, np.clip(default_prob, 0.01, 0.97))

    df = pd.DataFrame({
        "applicant_id": [f"APP{100000+i}" for i in range(n_samples)],
        "age": age,
        "income": income,
        "employment_length": employment_length,
        "loan_amount": loan_amount,
        "credit_score": credit_score,
        "existing_debt": existing_debt,
        "num_credit_accounts": num_credit_accounts,
        "credit_history_years": credit_history_years,
        "num_late_payments": num_late_payments,
        "home_ownership": home_ownership,
        "loan_purpose": loan_purpose,
        "default": default,
    })

    # Inject some missing values to simulate real-world data quality issues
    for col in ["income", "employment_length", "credit_score"]:
        mask = RNG.random(n_samples) < 0.02
        df.loc[mask, col] = np.nan

    return df


def generate_transaction_data(n_samples: int = 50000) -> pd.DataFrame:
    """Generate synthetic transaction data with a realistic fraud rate (~1.5%)."""
    user_ids = RNG.integers(1, 3000, n_samples)
    amount = np.round(RNG.lognormal(mean=3.5, sigma=1.2, size=n_samples), 2)
    hour_of_day = RNG.integers(0, 24, n_samples)
    is_weekend = RNG.integers(0, 2, n_samples)
    merchant_category = RNG.choice(
        ["grocery", "electronics", "travel", "restaurant", "online_retail",
         "gas_station", "entertainment", "utilities"],
        n_samples
    )
    location = RNG.choice(["domestic", "international"], n_samples, p=[0.92, 0.08])
    device_type = RNG.choice(["mobile", "web", "pos"], n_samples, p=[0.5, 0.3, 0.2])
    distance_from_home = np.round(np.abs(RNG.normal(15, 40, n_samples)), 2)

    timestamps = pd.to_datetime("2025-01-01") + pd.to_timedelta(
        RNG.integers(0, 180 * 24 * 60, n_samples), unit="m"
    )

    z_amount = (amount - amount.mean()) / amount.std()
    z_dist = (distance_from_home - distance_from_home.mean()) / distance_from_home.std()

    fraud_score = (
        1.1 * z_amount
        + 2.2 * (location == "international")
        + 1.3 * z_dist
        + 1.8 * ((hour_of_day < 5) | (hour_of_day > 23))
        + 0.9 * (device_type == "web")
        + RNG.normal(0, 0.7, n_samples)  # residual noise
    )
    fraud_prob = 1 / (1 + np.exp(-(fraud_score - 4.2)))
    is_fraud = RNG.binomial(1, np.clip(fraud_prob, 0.001, 0.97))

    df = pd.DataFrame({
        "transaction_id": [f"TXN{1000000+i}" for i in range(n_samples)],
        "user_id": user_ids,
        "amount": amount,
        "timestamp": timestamps,
        "merchant_category": merchant_category,
        "location": location,
        "device_type": device_type,
        "hour_of_day": hour_of_day,
        "is_weekend": is_weekend,
        "distance_from_home": distance_from_home,
        "is_fraud": is_fraud,
    })

    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def main():
    credit_df = generate_credit_data()
    txn_df = generate_transaction_data()

    credit_df.to_csv(RAW_DIR / "credit_data.csv", index=False)
    txn_df.to_csv(RAW_DIR / "transactions.csv", index=False)

    print(f"✅ credit_data.csv  -> {credit_df.shape}  default rate: {credit_df['default'].mean():.2%}")
    print(f"✅ transactions.csv -> {txn_df.shape}  fraud rate: {txn_df['is_fraud'].mean():.2%}")


if __name__ == "__main__":
    main()

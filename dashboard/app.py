"""
Phase 11: Dashboard Development
Streamlit dashboard for credit risk scoring and fraud monitoring.

Run with:  streamlit run dashboard/app.py
"""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
DATA_RAW = ROOT / "data" / "raw"
REPORTS_DIR = ROOT / "reports"

st.set_page_config(page_title="Credit Risk & Fraud Detection", layout="wide", page_icon="💳")

st.title("💳 Credit Risk Assessment & Fraud Detection Dashboard")


@st.cache_resource
def load_credit_artifacts():
    model = joblib.load(MODELS_DIR / "credit_best_model.pkl")
    scaler = joblib.load(MODELS_DIR / "credit_scaler.pkl")
    encoders = joblib.load(MODELS_DIR / "credit_encoders.pkl")
    feature_cols = joblib.load(MODELS_DIR / "credit_feature_cols.pkl")
    return model, scaler, encoders, feature_cols


@st.cache_resource
def load_fraud_artifacts():
    model = joblib.load(MODELS_DIR / "fraud_best_model.pkl")
    scaler = joblib.load(MODELS_DIR / "fraud_scaler.pkl")
    encoders = joblib.load(MODELS_DIR / "fraud_encoders.pkl")
    feature_cols = joblib.load(MODELS_DIR / "fraud_feature_cols.pkl")
    return model, scaler, encoders, feature_cols


tab1, tab2, tab3 = st.tabs(["🏦 Credit Risk Scoring", "🚨 Fraud Monitoring", "📊 Model Reports"])

# ---------------------------------------------------------------- TAB 1 ----
with tab1:
    st.subheader("Score a new credit applicant")
    try:
        model, scaler, encoders, feature_cols = load_credit_artifacts()
        artifacts_ready = True
    except FileNotFoundError:
        artifacts_ready = False
        st.warning("⚠️ No trained model found yet. Run `python src/pipeline.py` first.")

    if artifacts_ready:
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.number_input("Age", 18, 90, 35)
            income = st.number_input("Annual Income ($)", 0, 1_000_000, 55000)
            employment_length = st.number_input("Employment Length (years)", 0, 50, 5)
        with col2:
            loan_amount = st.number_input("Loan Amount ($)", 0, 500_000, 15000)
            credit_score = st.number_input("Credit Score", 300, 850, 680)
            existing_debt = st.number_input("Existing Debt ($)", 0, 500_000, 8000)
        with col3:
            num_credit_accounts = st.number_input("Number of Credit Accounts", 0, 30, 4)
            credit_history_years = st.number_input("Credit History (years)", 0, 40, 8)
            num_late_payments = st.number_input("Number of Late Payments", 0, 20, 1)

        col4, col5 = st.columns(2)
        with col4:
            home_ownership = st.selectbox("Home Ownership", ["OWN", "MORTGAGE", "RENT"])
        with col5:
            loan_purpose = st.selectbox(
                "Loan Purpose",
                ["debt_consolidation", "credit_card", "home_improvement", "car", "business", "other"]
            )

        if st.button("Assess Credit Risk", type="primary"):
            sys.path.insert(0, str(ROOT / "src"))
            import feature_engineering as fe

            raw = pd.DataFrame([{
                "age": age, "income": income, "employment_length": employment_length,
                "loan_amount": loan_amount, "credit_score": credit_score,
                "existing_debt": existing_debt, "num_credit_accounts": num_credit_accounts,
                "credit_history_years": credit_history_years, "num_late_payments": num_late_payments,
                "home_ownership": home_ownership, "loan_purpose": loan_purpose,
            }])
            raw = fe.engineer_credit_features(raw)
            for col, le in encoders.items():
                raw[col] = le.transform(raw[col].astype(str))

            X = raw[feature_cols]
            num_cols = X.select_dtypes(include="number").columns.tolist()
            X_scaled = X.copy()
            X_scaled[num_cols] = scaler.transform(X[num_cols])

            pd_score = model.predict_proba(X_scaled)[:, 1][0]
            expected_loss = pd_score * 0.45 * loan_amount
            rating_bins = [0, 0.02, 0.05, 0.10, 0.20, 1.0]
            rating_labels = ["A", "B", "C", "D", "E"]
            rating = pd.cut([pd_score], bins=rating_bins, labels=rating_labels)[0]

            r1, r2, r3 = st.columns(3)
            r1.metric("Probability of Default", f"{pd_score:.1%}")
            r2.metric("Expected Loss", f"${expected_loss:,.2f}")
            r3.metric("Credit Rating", str(rating))

            if pd_score < 0.05:
                st.success("✅ Low risk — recommend approval")
            elif pd_score < 0.15:
                st.warning("⚠️ Moderate risk — consider adjusted terms")
            else:
                st.error("🚫 High risk — recommend decline or additional review")

# ---------------------------------------------------------------- TAB 2 ----
with tab2:
    st.subheader("Live Fraud Monitoring Feed")
    try:
        model, scaler, encoders, feature_cols = load_fraud_artifacts()
        txn_df = pd.read_csv(DATA_RAW / "transactions.csv")
        artifacts_ready = True
    except FileNotFoundError:
        artifacts_ready = False
        st.warning("⚠️ No trained model or data found yet. Run `python src/pipeline.py` first.")

    if artifacts_ready:
        n_show = st.slider("Number of recent transactions to score", 10, 200, 50)
        sample = txn_df.tail(n_show).copy()

        sys.path.insert(0, str(ROOT / "src"))
        import feature_engineering as fe
        scored = fe.engineer_fraud_features(sample)

        for col, le in encoders.items():
            scored[col] = scored[col].astype(str).map(
                lambda v, le=le: le.transform([v])[0] if v in le.classes_ else -1
            )

        X = scored[feature_cols]
        num_cols = X.select_dtypes(include="number").columns.tolist()
        X_scaled = X.copy()
        X_scaled[num_cols] = scaler.transform(X[num_cols])

        proba = model.predict_proba(X_scaled)[:, 1]
        sample["fraud_probability"] = proba
        sample["risk_level"] = np.where(proba >= 0.7, "HIGH",
                                 np.where(proba >= 0.3, "MEDIUM", "LOW"))

        c1, c2, c3 = st.columns(3)
        c1.metric("Transactions Scored", len(sample))
        c2.metric("High-Risk Alerts", int((sample["risk_level"] == "HIGH").sum()))
        c3.metric("Avg Fraud Probability", f"{proba.mean():.1%}")

        st.markdown("#### 🚨 High-Risk Alerts")
        high_risk = sample[sample["risk_level"] == "HIGH"].sort_values("fraud_probability", ascending=False)
        st.dataframe(
            high_risk[["transaction_id", "user_id", "amount", "merchant_category",
                       "location", "fraud_probability", "risk_level"]],
            use_container_width=True
        )

        st.markdown("#### All Recent Transactions")
        st.dataframe(
            sample[["transaction_id", "user_id", "amount", "timestamp", "merchant_category",
                    "location", "fraud_probability", "risk_level"]].sort_values("timestamp", ascending=False),
            use_container_width=True
        )

        fig = px.histogram(sample, x="fraud_probability", nbins=30,
                            title="Fraud Probability Distribution")
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------- TAB 3 ----
with tab3:
    st.subheader("Model Performance Reports")
    for label, fname in [
        ("Credit model comparison", "credit_model_comparison.csv"),
        ("Fraud model comparison", "fraud_model_comparison.csv"),
        ("Credit ensemble comparison", "credit_ensemble_comparison.csv"),
    ]:
        f = REPORTS_DIR / fname
        if f.exists():
            st.markdown(f"**{label}**")
            st.dataframe(pd.read_csv(f), use_container_width=True)

    img_col1, img_col2 = st.columns(2)
    for label, fname, col in [
        ("Credit ROC Curves", "credit_roc_curves.png", img_col1),
        ("Fraud ROC Curves", "fraud_roc_curves.png", img_col2),
        ("Credit Feature Importance", "credit_feature_importance.png", img_col1),
        ("Fraud Feature Importance", "fraud_feature_importance.png", img_col2),
    ]:
        f = REPORTS_DIR / fname
        if f.exists():
            col.image(str(f), caption=label, use_container_width=True)

    if not REPORTS_DIR.exists() or not any(REPORTS_DIR.iterdir()):
        st.info("Run `python src/pipeline.py` to generate reports.")

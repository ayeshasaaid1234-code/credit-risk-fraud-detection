"""
Phase 6 / Requirement 3 & 7: Fraud Detection models
Supervised classifiers + unsupervised anomaly detection + real-time scoring.
"""
import time
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


def get_fraud_models(class_weight: dict | str = "balanced") -> dict:
    """Return a dict of untrained model instances for fraud classification."""
    return {
        "LogisticRegression": LogisticRegression(
            max_iter=1000, class_weight=class_weight, random_state=42
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, max_depth=10, class_weight=class_weight,
            random_state=42, n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            eval_metric="logloss", random_state=42, n_jobs=-1
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            class_weight=class_weight, random_state=42, n_jobs=-1, verbose=-1
        ),
    }


def train_fraud_models(models: dict, X_train, y_train) -> dict:
    trained = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained[name] = model
    return trained


def train_anomaly_detector(X_train, contamination: float = 0.02) -> IsolationForest:
    """Unsupervised anomaly detection to flag transactions unlike the training set."""
    iso = IsolationForest(
        n_estimators=200, contamination=contamination, random_state=42, n_jobs=-1
    )
    iso.fit(X_train)
    return iso


def anomaly_scores(iso_model: IsolationForest, X) -> np.ndarray:
    """Higher = more anomalous (we flip sklearn's sign convention for intuitiveness)."""
    return -iso_model.score_samples(X)


def real_time_fraud_score(model, transaction_features: pd.DataFrame) -> dict:
    """
    Score a single transaction (or small batch) and report latency —
    demonstrates the < 100ms real-time requirement.
    """
    start = time.perf_counter()
    proba = model.predict_proba(transaction_features)[:, 1]
    latency_ms = (time.perf_counter() - start) * 1000

    risk_level = np.where(proba >= 0.7, "HIGH",
                  np.where(proba >= 0.3, "MEDIUM", "LOW"))

    return {
        "fraud_probability": np.round(proba, 4).tolist(),
        "risk_level": risk_level.tolist(),
        "latency_ms": round(latency_ms, 3),
    }

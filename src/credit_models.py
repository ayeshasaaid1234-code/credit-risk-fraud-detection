"""
Phase 6 / Requirement 2 & 7: Credit Risk Assessment models
Probability of Default (PD), Expected Loss (EL), and credit rating system.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


def get_credit_models(class_weight: dict | str = "balanced") -> dict:
    """Return a dict of untrained model instances for credit risk (PD) prediction."""
    return {
        "LogisticRegression": LogisticRegression(
            max_iter=1000, class_weight=class_weight, random_state=42
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, max_depth=8, class_weight=class_weight,
            random_state=42, n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            eval_metric="logloss", random_state=42, n_jobs=-1
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            class_weight=class_weight, random_state=42, n_jobs=-1, verbose=-1
        ),
    }


def train_credit_models(models: dict, X_train, y_train) -> dict:
    trained = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained[name] = model
    return trained


def predict_pd(model, X) -> np.ndarray:
    """Predict Probability of Default."""
    return model.predict_proba(X)[:, 1]


def calculate_expected_loss(pd_scores: np.ndarray, loan_amounts: np.ndarray,
                             lgd: float = 0.45) -> np.ndarray:
    """
    Expected Loss = PD x LGD x EAD
    - PD: Probability of Default (model output)
    - LGD: Loss Given Default (assumed 45% industry-standard default)
    - EAD: Exposure at Default (approximated by loan amount)
    """
    return pd_scores * lgd * loan_amounts


def assign_credit_rating(pd_scores: np.ndarray) -> np.ndarray:
    """Bucket PD scores into A (best) - E (worst) credit ratings."""
    bins = [0, 0.02, 0.05, 0.10, 0.20, 1.0]
    labels = ["A", "B", "C", "D", "E"]
    return pd.cut(pd_scores, bins=bins, labels=labels, include_lowest=True)


def build_credit_scorecard(X, pd_scores, loan_amounts, lgd: float = 0.45) -> pd.DataFrame:
    """Produce a decision-support table: PD, Expected Loss, and Rating per applicant."""
    return pd.DataFrame({
        "probability_of_default": np.round(pd_scores, 4),
        "expected_loss": np.round(calculate_expected_loss(pd_scores, loan_amounts, lgd), 2),
        "credit_rating": assign_credit_rating(pd_scores),
    }, index=X.index if hasattr(X, "index") else None)

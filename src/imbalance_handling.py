"""
Phase 5 / Requirement 6: Handling Imbalanced Data
SMOTE, ADASYN, random under-sampling, and class-weight helpers.
"""
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from sklearn.utils.class_weight import compute_class_weight


def apply_smote(X, y, random_state: int = 42):
    sm = SMOTE(random_state=random_state)
    return sm.fit_resample(X, y)


def apply_adasyn(X, y, random_state: int = 42):
    ada = ADASYN(random_state=random_state)
    return ada.fit_resample(X, y)


def apply_undersampling(X, y, random_state: int = 42):
    rus = RandomUnderSampler(random_state=random_state)
    return rus.fit_resample(X, y)


def get_class_weights(y) -> dict:
    classes = np.unique(y)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    return dict(zip(classes, weights))


def compare_resampling_techniques(X, y) -> pd.DataFrame:
    """Report class balance before/after each resampling technique."""
    results = {"original": pd.Series(y).value_counts().to_dict()}

    X_sm, y_sm = apply_smote(X, y)
    results["SMOTE"] = pd.Series(y_sm).value_counts().to_dict()

    X_ada, y_ada = apply_adasyn(X, y)
    results["ADASYN"] = pd.Series(y_ada).value_counts().to_dict()

    X_us, y_us = apply_undersampling(X, y)
    results["undersampling"] = pd.Series(y_us).value_counts().to_dict()

    return pd.DataFrame(results).T

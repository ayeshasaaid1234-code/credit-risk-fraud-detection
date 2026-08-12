"""
Phase 2: Data Preprocessing
Missing values, outlier treatment, encoding, scaling, train/test split.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


def assess_data_quality(df: pd.DataFrame, name: str = "dataset") -> pd.DataFrame:
    """Quick data-quality report: missing values, dtypes, unique counts."""
    report = pd.DataFrame({
        "dtype": df.dtypes,
        "missing_count": df.isna().sum(),
        "missing_pct": (df.isna().mean() * 100).round(2),
        "n_unique": df.nunique(),
    })
    print(f"\n📋 Data quality report — {name}")
    print(report)
    return report


def handle_missing_values(df: pd.DataFrame, strategy: str = "median") -> pd.DataFrame:
    """Impute numeric columns (median/mean) and categorical columns (mode)."""
    df = df.copy()
    num_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols = df.select_dtypes(include=["object"]).columns

    for col in num_cols:
        if df[col].isna().any():
            fill_value = df[col].median() if strategy == "median" else df[col].mean()
            df[col] = df[col].fillna(fill_value)

    for col in cat_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].mode()[0])

    return df


def treat_outliers(df: pd.DataFrame, columns: list, method: str = "iqr_clip") -> pd.DataFrame:
    """Cap outliers using the IQR method to reduce their influence on models."""
    df = df.copy()
    for col in columns:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        df[col] = df[col].clip(lower, upper)
    return df


def encode_categoricals(df: pd.DataFrame, columns: list) -> tuple[pd.DataFrame, dict]:
    """Label-encode categorical columns; return encoders for inverse-transform / API use."""
    df = df.copy()
    encoders = {}
    for col in columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
    return df, encoders


def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame, columns: list):
    """Standard-scale numeric features, fit on train only."""
    scaler = StandardScaler()
    X_train = X_train.copy()
    X_test = X_test.copy()
    X_train[columns] = scaler.fit_transform(X_train[columns])
    X_test[columns] = scaler.transform(X_test[columns])
    return X_train, X_test, scaler


def split_data(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, stratify: bool = True):
    return train_test_split(
        X, y, test_size=test_size, random_state=42,
        stratify=y if stratify else None
    )

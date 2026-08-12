"""
Phase 9: Model Interpretability & Explainability
Feature importance, SHAP, LIME, and simple business-rule extraction.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.tree import DecisionTreeClassifier, export_text


def get_feature_importance(model, feature_names: list) -> pd.DataFrame:
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    else:
        raise ValueError("Model does not expose feature importances or coefficients.")

    return pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values("importance", ascending=False).reset_index(drop=True)


def plot_feature_importance(importance_df: pd.DataFrame, top_n: int = 15, save_path: str = None):
    top = importance_df.head(top_n)
    plt.figure(figsize=(7, 5))
    plt.barh(top["feature"][::-1], top["importance"][::-1], color="#4C72B0")
    plt.title(f"Top {top_n} Feature Importances")
    plt.xlabel("Importance")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close()


def shap_summary(model, X_sample: pd.DataFrame, save_path: str = None):
    """SHAP TreeExplainer summary plot (works for RF/XGBoost/LightGBM)."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    # Binary classifiers may return a list [class0, class1]
    values = shap_values[1] if isinstance(shap_values, list) else shap_values

    plt.figure()
    shap.summary_plot(values, X_sample, show=False)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return explainer, shap_values


def lime_explain_instance(model, X_train: pd.DataFrame, instance: pd.Series,
                           class_names=("no_default_no_fraud", "default_or_fraud")):
    from lime.lime_tabular import LimeTabularExplainer
    explainer = LimeTabularExplainer(
        X_train.values,
        feature_names=X_train.columns.tolist(),
        class_names=list(class_names),
        mode="classification",
        random_state=42,
    )
    explanation = explainer.explain_instance(
        instance.values, model.predict_proba, num_features=10
    )
    return explanation


def extract_business_rules(X_train: pd.DataFrame, y_train, max_depth: int = 4) -> str:
    """
    Train a small, fully interpretable decision tree purely to extract
    human-readable IF/THEN business rules for compliance/risk teams.
    """
    tree = DecisionTreeClassifier(max_depth=max_depth, class_weight="balanced", random_state=42)
    tree.fit(X_train, y_train)
    rules = export_text(tree, feature_names=list(X_train.columns))
    return rules

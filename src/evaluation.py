"""
Phase 8: Model Evaluation
Accuracy/Precision/Recall/F1/ROC-AUC, confusion matrix, cost-sensitive
evaluation, and cross-validation stability testing.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
from sklearn.model_selection import cross_val_score, StratifiedKFold


def evaluate_model(model, X_test, y_test, model_name: str = "model") -> dict:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "model": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }
    return metrics


def compare_models(trained_models: dict, X_test, y_test) -> pd.DataFrame:
    rows = [evaluate_model(m, X_test, y_test, name) for name, m in trained_models.items()]
    return pd.DataFrame(rows).sort_values("roc_auc", ascending=False).reset_index(drop=True)


def plot_confusion_matrix(model, X_test, y_test, model_name: str, save_path: str = None):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"])
    plt.title(f"Confusion Matrix — {model_name}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close()


def plot_roc_curves(trained_models: dict, X_test, y_test, save_path: str = None):
    plt.figure(figsize=(6, 5))
    for name, model in trained_models.items():
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves — Model Comparison")
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close()


def cost_sensitive_evaluation(y_test, y_pred, cost_fp: float = 10, cost_fn: float = 100) -> dict:
    """
    Business-relevant cost evaluation, e.g. for fraud:
    - False Positive (flagging a legit txn): customer friction, ~$10 cost
    - False Negative (missing real fraud): direct loss, ~$100 cost
    """
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    total_cost = fp * cost_fp + fn * cost_fn
    return {
        "true_positives": int(tp), "false_positives": int(fp),
        "true_negatives": int(tn), "false_negatives": int(fn),
        "total_cost": float(total_cost),
        "avg_cost_per_transaction": float(total_cost / len(y_test)),
    }


def cross_validation_stability(model, X, y, cv: int = 5, scoring: str = "roc_auc") -> dict:
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=skf, scoring=scoring, n_jobs=-1)
    return {
        "cv_scores": scores.tolist(),
        "mean": float(scores.mean()),
        "std": float(scores.std()),
    }

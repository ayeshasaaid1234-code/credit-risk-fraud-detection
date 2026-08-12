"""
End-to-end pipeline: runs every phase of the project in order and writes
all reports/models/plots to disk.

    python src/pipeline.py
"""
import sys
import warnings
from pathlib import Path

import joblib
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import data_generation
import preprocessing as prep
import feature_engineering as fe
import imbalance_handling as imb
import credit_models as cm
import fraud_models as fm
import ensemble as ens
import evaluation as ev
import interpretability as interp

DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
for d in (DATA_PROCESSED, MODELS_DIR, REPORTS_DIR):
    d.mkdir(parents=True, exist_ok=True)


def run_credit_pipeline():
    print("\n" + "=" * 70)
    print("CREDIT RISK ASSESSMENT PIPELINE")
    print("=" * 70)

    df = pd.read_csv(DATA_RAW / "credit_data.csv")
    prep.assess_data_quality(df, "credit_data")

    df = prep.handle_missing_values(df)
    df = prep.treat_outliers(df, ["income", "loan_amount", "existing_debt"])
    df = fe.engineer_credit_features(df)

    cat_cols = ["home_ownership", "loan_purpose"]
    df_enc, encoders = prep.encode_categoricals(df, cat_cols)

    feature_cols = [c for c in df_enc.columns if c not in ("applicant_id", "default")]
    X, y = df_enc[feature_cols], df_enc["default"]

    X_train, X_test, y_train, y_test = prep.split_data(X, y)
    num_cols = X_train.select_dtypes(include="number").columns.tolist()
    X_train, X_test, scaler = prep.scale_features(X_train, X_test, num_cols)

    print(f"\nClass balance before resampling: {y_train.value_counts().to_dict()}")
    X_train_sm, y_train_sm = imb.apply_smote(X_train, y_train)
    print(f"Class balance after SMOTE: {pd.Series(y_train_sm).value_counts().to_dict()}")

    models = cm.get_credit_models()
    trained = cm.train_credit_models(models, X_train_sm, y_train_sm)

    comparison = ev.compare_models(trained, X_test, y_test)
    print("\nModel comparison:\n", comparison)
    comparison.to_csv(REPORTS_DIR / "credit_model_comparison.csv", index=False)

    ev.plot_roc_curves(trained, X_test, y_test, save_path=str(REPORTS_DIR / "credit_roc_curves.png"))
    best_name = comparison.iloc[0]["model"]
    best_model = trained[best_name]
    ev.plot_confusion_matrix(best_model, X_test, y_test, best_name,
                              save_path=str(REPORTS_DIR / "credit_confusion_matrix.png"))

    # Ensembles
    ensembles = ens.fit_all_ensembles(models, X_train_sm, y_train_sm)
    ens_comparison = ev.compare_models(ensembles, X_test, y_test)
    print("\nEnsemble comparison:\n", ens_comparison)
    ens_comparison.to_csv(REPORTS_DIR / "credit_ensemble_comparison.csv", index=False)

    # PD -> Expected Loss -> Credit Rating
    pd_scores = cm.predict_pd(best_model, X_test)
    scorecard = cm.build_credit_scorecard(X_test, pd_scores, df.loc[X_test.index, "loan_amount"].values)
    scorecard.to_csv(REPORTS_DIR / "credit_scorecard_sample.csv", index=False)
    print("\nCredit rating distribution:\n", scorecard["credit_rating"].value_counts())

    # Interpretability
    importance = interp.get_feature_importance(best_model, feature_cols)
    interp.plot_feature_importance(importance, save_path=str(REPORTS_DIR / "credit_feature_importance.png"))
    if best_name in ("RandomForest", "XGBoost", "LightGBM"):
        interp.shap_summary(best_model, X_test.sample(min(500, len(X_test)), random_state=42),
                             save_path=str(REPORTS_DIR / "credit_shap_summary.png"))
    rules = interp.extract_business_rules(X_train, y_train)
    (REPORTS_DIR / "credit_business_rules.txt").write_text(rules)

    joblib.dump(best_model, MODELS_DIR / "credit_best_model.pkl")
    joblib.dump(scaler, MODELS_DIR / "credit_scaler.pkl")
    joblib.dump(encoders, MODELS_DIR / "credit_encoders.pkl")
    joblib.dump(feature_cols, MODELS_DIR / "credit_feature_cols.pkl")
    print(f"\n✅ Best credit model: {best_name} — saved to models/credit_best_model.pkl")


def run_fraud_pipeline():
    print("\n" + "=" * 70)
    print("FRAUD DETECTION PIPELINE")
    print("=" * 70)

    df = pd.read_csv(DATA_RAW / "transactions.csv")
    prep.assess_data_quality(df, "transactions")

    df = prep.handle_missing_values(df)
    df = fe.engineer_fraud_features(df)

    cat_cols = ["merchant_category", "location", "device_type"]
    df_enc, encoders = prep.encode_categoricals(df, cat_cols)

    drop_cols = ["transaction_id", "user_id", "timestamp", "is_fraud"]
    feature_cols = [c for c in df_enc.columns if c not in drop_cols]
    X, y = df_enc[feature_cols], df_enc["is_fraud"]

    X_train, X_test, y_train, y_test = prep.split_data(X, y)
    num_cols = X_train.select_dtypes(include="number").columns.tolist()
    X_train, X_test, scaler = prep.scale_features(X_train, X_test, num_cols)

    print(f"\nClass balance before resampling: {y_train.value_counts().to_dict()}")
    X_train_sm, y_train_sm = imb.apply_smote(X_train, y_train)
    print(f"Class balance after SMOTE: {pd.Series(y_train_sm).value_counts().to_dict()}")

    models = fm.get_fraud_models()
    trained = fm.train_fraud_models(models, X_train_sm, y_train_sm)

    comparison = ev.compare_models(trained, X_test, y_test)
    print("\nModel comparison:\n", comparison)
    comparison.to_csv(REPORTS_DIR / "fraud_model_comparison.csv", index=False)

    ev.plot_roc_curves(trained, X_test, y_test, save_path=str(REPORTS_DIR / "fraud_roc_curves.png"))
    best_name = comparison.iloc[0]["model"]
    best_model = trained[best_name]
    ev.plot_confusion_matrix(best_model, X_test, y_test, best_name,
                              save_path=str(REPORTS_DIR / "fraud_confusion_matrix.png"))

    y_pred = best_model.predict(X_test)
    cost_eval = ev.cost_sensitive_evaluation(y_test, y_pred)
    print("\nCost-sensitive evaluation:", cost_eval)

    stability = ev.cross_validation_stability(best_model, X_train_sm, y_train_sm)
    print("Cross-validation stability:", stability)

    # Unsupervised anomaly detection as an additional signal
    iso = fm.train_anomaly_detector(X_train)
    anomaly_scores = fm.anomaly_scores(iso, X_test)
    print(f"\nAnomaly score range: {anomaly_scores.min():.3f} - {anomaly_scores.max():.3f}")

    # Real-time scoring demo
    sample_txn = X_test.iloc[:5]
    rt_result = fm.real_time_fraud_score(best_model, sample_txn)
    print("\nReal-time scoring demo:", rt_result)

    # Interpretability
    importance = interp.get_feature_importance(best_model, feature_cols)
    interp.plot_feature_importance(importance, save_path=str(REPORTS_DIR / "fraud_feature_importance.png"))
    if best_name in ("RandomForest", "XGBoost", "LightGBM"):
        interp.shap_summary(best_model, X_test.sample(min(500, len(X_test)), random_state=42),
                             save_path=str(REPORTS_DIR / "fraud_shap_summary.png"))
    rules = interp.extract_business_rules(X_train, y_train)
    (REPORTS_DIR / "fraud_business_rules.txt").write_text(rules)

    joblib.dump(best_model, MODELS_DIR / "fraud_best_model.pkl")
    joblib.dump(iso, MODELS_DIR / "fraud_anomaly_detector.pkl")
    joblib.dump(scaler, MODELS_DIR / "fraud_scaler.pkl")
    joblib.dump(encoders, MODELS_DIR / "fraud_encoders.pkl")
    joblib.dump(feature_cols, MODELS_DIR / "fraud_feature_cols.pkl")
    print(f"\n✅ Best fraud model: {best_name} — saved to models/fraud_best_model.pkl")


def main():
    print("Generating synthetic datasets (replace with real data in data/raw/ if available)...")
    data_generation.main()

    run_credit_pipeline()
    run_fraud_pipeline()

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE — see reports/ and models/ folders")
    print("=" * 70)


if __name__ == "__main__":
    main()

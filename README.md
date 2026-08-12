# Credit Risk Assessment and Fraud Detection System

Machine Learning system for **credit risk scoring** (probability of default) and
**transaction fraud detection**, built for financial services use-cases.

## 📁 Project Structure

```
credit_risk_fraud_project/
│
├── data/
│   ├── raw/                    # Raw generated datasets
│   └── processed/              # Cleaned / feature-engineered datasets
│
├── src/
│   ├── data_generation.py      # Creates synthetic credit + transaction data
│   ├── preprocessing.py        # Cleaning, encoding, scaling, train/test split
│   ├── feature_engineering.py  # Credit + fraud domain features
│   ├── imbalance_handling.py   # SMOTE / ADASYN / undersampling / class weights
│   ├── credit_models.py        # Credit risk (PD) models + Expected Loss + rating
│   ├── fraud_models.py         # Fraud detection models + anomaly detection
│   ├── ensemble.py             # Voting / Stacking / Bagging ensembles
│   ├── evaluation.py           # Metrics, ROC-AUC, confusion matrix, CV
│   ├── interpretability.py     # Feature importance, SHAP, LIME, business rules
│   └── pipeline.py             # Orchestrates the full end-to-end run
│
├── models/                     # Saved trained models (.pkl)
├── reports/                    # Generated evaluation reports & plots
├── dashboard/
│   └── app.py                  # Streamlit dashboard (risk + fraud monitoring)
├── api/
│   └── main.py                 # FastAPI real-time scoring service
├── notebooks/
│   └── 01_end_to_end_walkthrough.ipynb
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the full pipeline (generates data, trains & evaluates all models)
python src/pipeline.py

# 3. Launch the interactive dashboard
streamlit run dashboard/app.py

# 4. (Optional) Launch the real-time scoring API
uvicorn api.main:app --reload --port 8000
```

## 🧩 What's Included

| Phase | Module | Description |
|-------|--------|-------------|
| 1 | `data_generation.py` | Synthetic but realistic credit-applicant & transaction data (since no real dataset was supplied — swap in your own CSVs by matching the same schema). |
| 2 | `preprocessing.py` | Missing-value handling, outlier treatment, encoding, scaling, splitting |
| 3 | `feature_engineering.py` | Credit utilization ratio, DTI, credit age, transaction velocity, location anomalies, time-based features |
| 4 | `imbalance_handling.py` | SMOTE, ADASYN, random under-sampling, class weighting |
| 5 | `credit_models.py` | Logistic Regression, Random Forest, XGBoost, LightGBM for PD; Expected Loss = PD × LGD × EAD; A–E credit rating buckets |
| 6 | `fraud_models.py` | Same model family + Isolation Forest anomaly detection; real-time-style scoring function |
| 7 | `ensemble.py` | Voting Classifier, Stacking Classifier, Bagging |
| 8 | `evaluation.py` | Accuracy, Precision, Recall, F1, ROC-AUC, confusion matrix, cost-sensitive evaluation, stability via cross-validation |
| 9 | `interpretability.py` | Feature importance, SHAP summary/force plots, LIME local explanations, extracted business rules |
| 10 | `dashboard/app.py` | Streamlit app: credit scoring form, fraud monitoring feed, alerts, charts |
| 11 | `api/main.py` | FastAPI `/score/credit` and `/score/fraud` endpoints for real-time inference |

## 🔁 Using Your Own Data

Replace the generated CSVs in `data/raw/` with your real datasets, keeping these columns
(or edit `src/preprocessing.py` to match your schema):

**Credit data** (`credit_data.csv`): `age, income, employment_length, loan_amount, credit_score,
existing_debt, num_credit_accounts, credit_history_years, num_late_payments, home_ownership,
loan_purpose, default` (target)

**Transaction data** (`transactions.csv`): `transaction_id, user_id, amount, timestamp,
merchant_category, location, device_type, hour_of_day, is_weekend, distance_from_home,
is_fraud` (target)

## 📊 Outputs

Running `pipeline.py` produces, under `reports/`:
- `credit_model_comparison.csv`, `fraud_model_comparison.csv`
- ROC curves, confusion matrices, SHAP summary plots (PNG)
- `business_rules.txt` — human-readable rules extracted from the best tree model
- Trained models saved to `models/*.pkl`

## ⚠️ Note on Synthetic Data

Because the assignment did not provide a real dataset, `data_generation.py` builds a
statistically realistic synthetic dataset so the whole pipeline is runnable end-to-end.
The code is written so switching to a real dataset only requires updating the data-loading
step — everything downstream (feature engineering, modeling, evaluation, dashboard) works
the same way.

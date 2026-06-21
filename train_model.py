"""
UC01 - Wellness-Linked Lapse Prediction
Model training: ColumnTransformer -> SMOTE -> XGBoost, with SHAP explainability.
Gender is deliberately excluded from the feature set (fairness-by-design).
"""
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, recall_score, precision_score,
                              f1_score, accuracy_score, brier_score_loss, confusion_matrix)
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import shap

from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
DATA_PATH = APP_DIR / "data" / "momentum_lapse_train.csv"
ARTIFACT_DIR = APP_DIR / "artifacts"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "lapsed"
DROP_COLS = ["policy_id", "client_first_name", "preferred_language", "preferred_channel", "gender"]

NUMERIC_FEATURES = [
    "num_dependants", "policy_tenure_months", "num_addon_riders",
    "monthly_premium_zar", "total_premiums_paid_zar",
    "active_dayz_3m_avg", "recharge_dayz_avg_weekly", "gym_visits_monthly",
    "lifestyle_quiz_score", "lifestyle_quiz_score_delta_6m",
    "health_assessment_days_since", "partner_reward_redemptions_qtr",
    "wellness_app_logins_monthly", "multiply_points_balance",
    "days_since_last_reward_earned", "engagement_velocity_30d",
]
CATEGORICAL_FEATURES = [
    "senior_citizen", "has_spouse_on_policy", "premium_payment_term",
    "digital_communication_opt_in", "payment_method", "product_type",
    "market", "multiply_tier", "multiply_tier_change_direction",
]


def load_data():
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=[TARGET] + DROP_COLS)
    y = (df[TARGET] == "Yes").astype(int)
    return X, y


def build_pipeline():
    pre = ColumnTransformer(
        transformers=[
            ("num", "passthrough", NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
    model = XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.85, colsample_bytree=0.85,
        eval_metric="auc", random_state=42, n_jobs=-1,
    )
    pipe = ImbPipeline(steps=[
        ("preprocess", pre),
        ("smote", SMOTE(random_state=42)),
        ("model", model),
    ])
    return pipe


def get_feature_names(pre: ColumnTransformer):
    cat_encoder = pre.named_transformers_["cat"]
    cat_names = list(cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES))
    return NUMERIC_FEATURES + cat_names


def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipe = build_pipeline()
    pipe.fit(X_train, y_train)

    proba = pipe.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    metrics = {
        "auc": round(float(roc_auc_score(y_test, proba)), 4),
        "recall": round(float(recall_score(y_test, pred)), 4),
        "precision": round(float(precision_score(y_test, pred)), 4),
        "f1": round(float(f1_score(y_test, pred)), 4),
        "accuracy": round(float(accuracy_score(y_test, pred)), 4),
        "brier_calibration": round(float(brier_score_loss(y_test, proba)), 4),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "base_lapse_rate": round(float(y.mean()), 4),
    }
    cm = confusion_matrix(y_test, pred).tolist()
    metrics["confusion_matrix"] = cm  # [[TN, FP], [FN, TP]]

    print("=== MODEL CARD ===")
    print(json.dumps(metrics, indent=2))

    # Persist pipeline
    joblib.dump(pipe, ARTIFACT_DIR / "lapse_model_pipeline.joblib")

    # Persist SHAP explainer built on the fitted XGBoost step + transformed background
    pre = pipe.named_steps["preprocess"]
    model = pipe.named_steps["model"]
    feature_names = get_feature_names(pre)

    X_test_transformed = pre.transform(X_test)
    if hasattr(X_test_transformed, "toarray"):
        X_test_transformed = X_test_transformed.toarray()

    explainer = shap.TreeExplainer(model)
    joblib.dump(explainer, ARTIFACT_DIR / "shap_explainer.joblib")
    joblib.dump(feature_names, ARTIFACT_DIR / "feature_names.joblib")

    with open(ARTIFACT_DIR / "model_card.json", "w") as f:
        json.dump(metrics, f, indent=2)

    with open(ARTIFACT_DIR / "feature_config.json", "w") as f:
        json.dump({
            "numeric": NUMERIC_FEATURES,
            "categorical": CATEGORICAL_FEATURES,
            "drop": DROP_COLS,
            "target": TARGET,
        }, f, indent=2)

    print("\nArtifacts saved to", ARTIFACT_DIR)


if __name__ == "__main__":
    main()

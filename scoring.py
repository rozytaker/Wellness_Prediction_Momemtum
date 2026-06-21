"""
Shared scoring utilities used by the Streamlit dashboard.
Loads the trained pipeline + SHAP explainer once, scores any compatible
dataframe, and returns risk scores / tiers / per-row SHAP contributions.
"""
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = APP_DIR / "artifacts"

REQUIRED_COLUMNS = [
    "num_dependants", "policy_tenure_months", "num_addon_riders",
    "monthly_premium_zar", "total_premiums_paid_zar",
    "active_dayz_3m_avg", "recharge_dayz_avg_weekly", "gym_visits_monthly",
    "lifestyle_quiz_score", "lifestyle_quiz_score_delta_6m",
    "health_assessment_days_since", "partner_reward_redemptions_qtr",
    "wellness_app_logins_monthly", "multiply_points_balance",
    "days_since_last_reward_earned", "engagement_velocity_30d",
    "senior_citizen", "has_spouse_on_policy", "premium_payment_term",
    "digital_communication_opt_in", "payment_method", "product_type",
    "market", "multiply_tier", "multiply_tier_change_direction",
]


def load_artifacts():
    pipe = joblib.load(ARTIFACT_DIR / "lapse_model_pipeline.joblib")
    explainer = joblib.load(ARTIFACT_DIR / "shap_explainer.joblib")
    feature_names = joblib.load(ARTIFACT_DIR / "feature_names.joblib")
    with open(ARTIFACT_DIR / "model_card.json") as f:
        model_card = json.load(f)
    return pipe, explainer, feature_names, model_card


def validate_columns(df: pd.DataFrame):
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    return missing


def risk_tier(score: float) -> str:
    if score >= 71:
        return "Red"
    elif score >= 41:
        return "Amber"
    return "Green"


def score_dataframe(df: pd.DataFrame, pipe, explainer, feature_names):
    X = df[REQUIRED_COLUMNS].copy()

    pre = pipe.named_steps["preprocess"]
    model = pipe.named_steps["model"]

    Xt = pre.transform(X)
    if hasattr(Xt, "toarray"):
        Xt = Xt.toarray()

    proba = model.predict_proba(Xt)[:, 1]
    risk_score = np.round(proba * 100, 1)
    tiers = [risk_tier(s) for s in risk_score]

    shap_values = explainer.shap_values(Xt)

    result = df.copy()
    result["risk_score"] = risk_score
    result["risk_tier"] = tiers

    return result, shap_values, Xt, feature_names


def top_drivers_for_row(shap_values_row, feature_names, k=6):
    idx = np.argsort(np.abs(shap_values_row))[::-1][:k]
    return [(feature_names[i], float(shap_values_row[i])) for i in idx]

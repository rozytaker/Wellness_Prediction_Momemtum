"""
UC01 - Wellness-Linked Lapse Prediction & GenAI Retention Outreach
Data generation module.

Builds two datasets from a single source:
  1. Momentum policy dataset  <- relabelled from IBM Telco Customer Churn (open-source proxy)
  2. Multiply wellness dataset <- simulated, driven by a single latent "engagement" variable
     so every wellness signal (Active Dayz, gym visits, tier, lifestyle score, etc.) moves
     consistently with the real lapse label instead of being independently random.

Output:
  data/momentum_lapse_train.csv          (full labelled set, used to train the model)
  data/momentum_lapse_test_for_dashboard.csv  (holdout sample, target hidden, used in the Streamlit demo)
"""
import numpy as np
import pandas as pd
from pathlib import Path

RAW_PATH = "/mnt/user-data/uploads/IBM_telco-customer-churn.csv"
APP_DIR = Path(__file__).resolve().parent
OUT_DIR = APP_DIR / "data"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
rng = np.random.default_rng(SEED)

# ---------------------------------------------------------------------------
# Reference data for realism (grounded in Momentum Multiply's real mechanics:
# Active Dayz, Recharge Dayz, Lifestyle Quiz, status tiers Blue->Diamond, etc.)
# ---------------------------------------------------------------------------
MARKETS = ["South Africa", "Lesotho", "Botswana", "Namibia"]
MARKET_WEIGHTS = [0.60, 0.16, 0.14, 0.10]

MARKET_LANGUAGES = {
    "South Africa": ["English", "Afrikaans", "isiZulu", "Sesotho"],
    "Lesotho": ["Sesotho", "English"],
    "Botswana": ["Setswana", "English"],
    "Namibia": ["English", "Afrikaans", "Oshiwambo"],
}
MARKET_LANG_WEIGHTS = {
    "South Africa": [0.55, 0.20, 0.15, 0.10],
    "Lesotho": [0.75, 0.25],
    "Botswana": [0.70, 0.30],
    "Namibia": [0.45, 0.35, 0.20],
}

FIRST_NAMES = {
    "South Africa": ["Thabo", "Palesa", "Sipho", "Nomvula", "Pieter", "Annelie", "Lerato", "Bongani",
                      "Zanele", "Johan", "Karabo", "Refilwe", "Mandla", "Esther", "Trevor", "Naledi"],
    "Lesotho": ["Palesa", "Thabang", "Lineo", "Mpho", "Teboho", "Mamello", "Relebohile", "Khotso"],
    "Botswana": ["Kagiso", "Boitumelo", "Tshepo", "Lesego", "Goitseone", "Mpho", "Kebafetotse", "Onneile"],
    "Namibia": ["Frieda", "Johannes", "Ndapewa", "Helvi", "Festus", "Selma", "Tangeni", "Petrus"],
}

PRODUCT_MAP = {
    "DSL": "Momentum Life Cover",
    "Fiber optic": "Momentum Family Funeral Plan",
    "No": "Momentum Basic Funeral Plan",
}

PAYMENT_MAP = {
    "Electronic check": "EFT",
    "Mailed check": "Retail / Cash Payment",
    "Bank transfer (automatic)": "Debit Order",
    "Credit card (automatic)": "Card (Auto)",
}

CONTRACT_MAP = {
    "Month-to-month": "Monthly",
    "One year": "Annual",
    "Two year": "Biennial",
}

CHANNELS = ["WhatsApp", "SMS", "Email"]
CHANNEL_WEIGHTS = [0.62, 0.23, 0.15]


def load_and_relabel_insurance() -> pd.DataFrame:
    df = pd.read_csv(RAW_PATH)
    n = len(df)

    # TotalCharges has 11 blank strings for brand-new customers -> 0
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)

    out = pd.DataFrame()
    out["policy_id"] = "MOM-" + df["customerID"].str.replace("-", "")
    out["gender"] = df["gender"]
    out["senior_citizen"] = df["SeniorCitizen"].map({1: "65+", 0: "Under 65"})
    out["has_spouse_on_policy"] = df["Partner"]

    has_dep = df["Dependents"].eq("Yes")
    out["num_dependants"] = np.where(has_dep, rng.integers(1, 5, size=n), 0)

    out["policy_tenure_months"] = df["tenure"]
    out["premium_payment_term"] = df["Contract"].map(CONTRACT_MAP)
    out["digital_communication_opt_in"] = df["PaperlessBilling"]
    out["payment_method"] = df["PaymentMethod"].map(PAYMENT_MAP)
    out["product_type"] = df["InternetService"].map(PRODUCT_MAP)

    addon_cols = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
                  "TechSupport", "StreamingTV", "StreamingMovies"]
    out["num_addon_riders"] = (df[addon_cols] == "Yes").sum(axis=1)

    # Rescale USD telco charges -> ZAR-realistic monthly insurance premium
    mc = df["MonthlyCharges"]
    out["monthly_premium_zar"] = np.round(
        350 + (mc - mc.min()) / (mc.max() - mc.min()) * (3200 - 350), 2
    )
    out["total_premiums_paid_zar"] = np.round(
        out["monthly_premium_zar"] * out["policy_tenure_months"] *
        (0.92 + rng.normal(0, 0.04, n)).clip(0.8, 1.05), 2
    )
    out["total_premiums_paid_zar"] = out["total_premiums_paid_zar"].clip(lower=0)

    out["lapsed"] = df["Churn"]  # target: Yes/No

    # Market assignment (independent geo flavor, not a churn driver by construction)
    out["market"] = rng.choice(MARKETS, size=n, p=MARKET_WEIGHTS)

    # Name + language + channel, conditioned on market for realism
    first_names, languages, channels = [], [], []
    for m in out["market"]:
        first_names.append(rng.choice(FIRST_NAMES[m]))
        languages.append(rng.choice(MARKET_LANGUAGES[m], p=MARKET_LANG_WEIGHTS[m]))
    out["client_first_name"] = first_names
    out["preferred_language"] = languages
    out["preferred_channel"] = rng.choice(CHANNELS, size=n, p=CHANNEL_WEIGHTS)

    return out


def simulate_multiply_layer(insurance_df: pd.DataFrame) -> pd.DataFrame:
    """
    All Multiply wellness signals are derived from one latent engagement variable
    so they move together and stay consistent with the real `lapsed` label —
    instead of being simulated independently (which could contradict the label).
    """
    n = len(insurance_df)
    lapsed_flag = (insurance_df["lapsed"] == "Yes").to_numpy()
    tenure = insurance_df["policy_tenure_months"].to_numpy()
    tenure_z = (tenure - tenure.mean()) / (tenure.std() + 1e-6)

    lapsed_shift = np.where(lapsed_flag, -1.15, 0.80)
    noise = rng.normal(0, 1.0, n)
    latent = 0.55 * lapsed_shift + 0.20 * tenure_z + 0.55 * noise

    # rank -> percentile in [0,1], smooths latent into a usable engagement score
    engagement_pct = pd.Series(latent).rank(pct=True).to_numpy()

    def clip_round(x, lo, hi, decimals=0):
        return np.round(np.clip(x, lo, hi), decimals)

    active_dayz = clip_round(2 + engagement_pct * 27 + rng.normal(0, 2.5, n), 0, 30)
    recharge_dayz = clip_round(1 + engagement_pct * 5 + rng.normal(0, 0.9, n), 0, 7)
    gym_visits = clip_round(engagement_pct * 16 + rng.normal(0, 2.2, n), 0, 20)
    lifestyle_score = clip_round(18 + engagement_pct * 72 + rng.normal(0, 6, n), 0, 100)
    lifestyle_delta_6m = clip_round((engagement_pct - 0.5) * 40 + rng.normal(0, 8, n), -30, 30)
    health_assess_days = clip_round((1 - engagement_pct) * 480 + 20 + rng.normal(0, 55, n), 7, 720)
    redemptions_qtr = rng.poisson(lam=np.clip(engagement_pct * 8 + 0.3, 0.1, None))
    app_logins = clip_round(5 + engagement_pct * 45 + rng.normal(0, 6, n), 0, 60)
    points_balance = clip_round(500 + engagement_pct * 15000 + rng.normal(0, 1400, n), 0, None)
    days_since_reward = clip_round((1 - engagement_pct) * 88 + 2 + rng.normal(0, 14, n), 1, 180)

    # Tier: quantile-bin a noisy version of engagement so it's "sticky" but not a hard cutoff
    tier_score = engagement_pct * 0.8 + rng.uniform(0, 0.2, n)
    tier_labels = ["Blue", "Bronze", "Silver", "Gold", "Diamond"]
    tier = pd.qcut(tier_score, q=5, labels=tier_labels)

    # Tier change direction, biased by engagement tertile
    tertile = pd.qcut(engagement_pct, q=3, labels=["low", "mid", "high"])
    tier_change = []
    probs = {
        "low": [0.55, 0.35, 0.10],
        "mid": [0.25, 0.50, 0.25],
        "high": [0.10, 0.35, 0.55],
    }
    for t in tertile:
        tier_change.append(rng.choice(["Downgraded", "Stable", "Upgraded"], p=probs[t]))
    tier_change = np.array(tier_change)

    # 30-day engagement velocity, consistent with tier_change direction
    delta = np.where(
        tier_change == "Upgraded", rng.normal(4, 2, n),
        np.where(tier_change == "Downgraded", rng.normal(-4, 2, n), rng.normal(0, 1, n))
    )
    engagement_velocity_30d = np.round(delta, 1)

    multiply = pd.DataFrame({
        "policy_id": insurance_df["policy_id"].values,
        "multiply_tier": tier.astype(str),
        "multiply_tier_change_direction": tier_change,
        "active_dayz_3m_avg": active_dayz.astype(int),
        "recharge_dayz_avg_weekly": recharge_dayz.astype(int),
        "gym_visits_monthly": gym_visits.astype(int),
        "lifestyle_quiz_score": lifestyle_score.astype(int),
        "lifestyle_quiz_score_delta_6m": lifestyle_delta_6m.astype(int),
        "health_assessment_days_since": health_assess_days.astype(int),
        "partner_reward_redemptions_qtr": redemptions_qtr.astype(int),
        "wellness_app_logins_monthly": app_logins.astype(int),
        "multiply_points_balance": points_balance.astype(int),
        "days_since_last_reward_earned": days_since_reward.astype(int),
        "engagement_velocity_30d": engagement_velocity_30d,
    })
    return multiply


def build_datasets():
    insurance_df = load_and_relabel_insurance()
    multiply_df = simulate_multiply_layer(insurance_df)

    # Explicit merge on policy_id (the real-world join key between policy admin & Multiply systems)
    full = pd.merge(insurance_df, multiply_df, on="policy_id", how="inner", validate="1:1")

    train_path = OUT_DIR / "momentum_lapse_train.csv"
    full.to_csv(train_path, index=False)

    # Dashboard test set: stratified sample, target hidden (renamed + held out) to mimic unseen data
    yes_sample = full[full["lapsed"] == "Yes"].sample(frac=0.045, random_state=SEED)
    no_sample = full[full["lapsed"] == "No"].sample(frac=0.045, random_state=SEED)
    test_sample = pd.concat([yes_sample, no_sample]).sample(frac=1, random_state=SEED).reset_index(drop=True)

    dashboard_test = test_sample.drop(columns=["lapsed"]).copy()
    dashboard_test["actual_outcome_DEMO_ONLY"] = test_sample["lapsed"].values  # optional, for live validation

    test_path = OUT_DIR / "momentum_lapse_test_for_dashboard.csv"
    dashboard_test.to_csv(test_path, index=False)

    return full, dashboard_test, train_path, test_path


if __name__ == "__main__":
    full, test_df, train_path, test_path = build_datasets()
    print("TRAIN:", full.shape, "->", train_path)
    print("TEST :", test_df.shape, "->", test_path)
    print("\nLapse rate:", (full['lapsed'] == 'Yes').mean().round(4))
    print("\nColumns:", list(full.columns))

"""
Momentum Africa — Wellness-Linked Lapse Prediction & GenAI Retention Outreach
Streamlit demo dashboard.

Run with:  streamlit run app.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from scoring import load_artifacts, score_dataframe, top_drivers_for_row
from genai_layer import get_chroma_collection, generate_retention_message, ollama_available

APP_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Brand tokens — taken directly from the Momentum Africa deck
# ---------------------------------------------------------------------------
NAVY = "#0B1F3D"
NAVY_LIGHT = "#142B52"
TERRACOTTA = "#C2502F"
TEAL = "#1B7A93"
PURPLE = "#6B3FA0"
RED = "#C0392B"
ORANGE = "#D9822B"
GREEN = "#1E8F5F"
BG = "#F4F6F9"
CARD = "#FFFFFF"
INK = "#1A2233"
MUTED = "#6B7686"

TIER_COLOR = {"Green": GREEN, "Amber": ORANGE, "Red": RED}

st.set_page_config(
    page_title="Momentum Africa | Lapse Prediction & Retention Outreach",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
st.markdown(f"""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Sora:wght@500;600;700;800&family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">

<style>
html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
    color: {INK};
}}
h1, h2, h3, h4, .display {{
    font-family: 'Sora', sans-serif !important;
    letter-spacing: -0.01em;
}}
.stApp {{
    background: {BG};
}}
.block-container {{
    padding-top: 1rem !important;
}}
footer {{visibility: hidden;}}

.stTabs [data-baseweb="tab-list"] {{
    gap: 0px;
}}
.stTabs [data-baseweb="tab"] {{
    flex: 1;
    justify-content: center;
    padding: 10px 16px;
    font-size: 13px;
    font-weight: 600;
}}

.hero {{
    background: linear-gradient(120deg, {NAVY} 0%, {NAVY_LIGHT} 100%);
    border-radius: 18px;
    padding: 34px 40px;
    margin-bottom: 22px;
    position: relative;
    overflow: hidden;
}}
.hero::after {{
    content: "";
    position: absolute;
    right: -60px; top: -60px;
    width: 220px; height: 220px;
    border-radius: 50%;
    background: {TERRACOTTA}33;
}}
.hero-eyebrow {{
    color: {TERRACOTTA};
    font-weight: 700;
    letter-spacing: 0.14em;
    font-size: 12.5px;
    text-transform: uppercase;
    margin-bottom: 6px;
}}
.hero-title {{
    color: white;
    font-family: 'Sora', sans-serif;
    font-weight: 800;
    font-size: 30px;
    line-height: 1.15;
    margin: 0 0 8px 0;
}}
.hero-sub {{
    color: #B9C3D6;
    font-size: 14.5px;
    max-width: 700px;
}}
.hero-byline {{
    color: #7E8BAA;
    font-size: 12.5px;
    margin-top: 14px;
}}

.kpi-card {{
    background: {CARD};
    border-radius: 14px;
    padding: 18px 20px;
    border-left: 5px solid var(--accent, {TEAL});
    box-shadow: 0 1px 3px rgba(11,31,61,0.06);
    height: 100%;
}}
.kpi-label {{
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: {MUTED};
    font-weight: 600;
    margin-bottom: 6px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}}
.kpi-value {{
    font-family: 'Sora', sans-serif;
    font-size: 22px;
    font-weight: 700;
    color: {NAVY};
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}}
.kpi-sub {{
    font-size: 12px;
    color: {MUTED};
    margin-top: 2px;
}}

.section-card {{
    background: {CARD};
    border-radius: 14px;
    padding: 22px 24px;
    box-shadow: 0 1px 3px rgba(11,31,61,0.06);
    margin-bottom: 18px;
}}
.section-title {{
    font-family: 'Sora', sans-serif;
    font-weight: 700;
    font-size: 16px;
    color: {NAVY};
    margin-bottom: 4px;
}}
.section-caption {{
    color: {MUTED};
    font-size: 12.5px;
    margin-bottom: 14px;
}}

.badge {{
    display: inline-block;
    padding: 4px 12px;
    border-radius: 999px;
    font-size: 11.5px;
    font-weight: 700;
    letter-spacing: 0.03em;
}}
.badge-green {{ background: {GREEN}1A; color: {GREEN}; }}
.badge-amber {{ background: {ORANGE}1A; color: {ORANGE}; }}
.badge-red {{ background: {RED}1A; color: {RED}; }}
.badge-navy {{ background: {NAVY}14; color: {NAVY}; }}
.badge-teal {{ background: {TEAL}1A; color: {TEAL}; }}

.fact-chip {{
    background: #F0F3F8;
    border: 1px solid #E1E7F0;
    border-radius: 10px;
    padding: 9px 13px;
    font-size: 12.5px;
    color: {INK};
    margin-bottom: 8px;
}}

.phone-frame {{
    background: #ECE5DD;
    border-radius: 22px;
    padding: 18px;
    max-width: 360px;
    margin: 0 auto;
    box-shadow: 0 6px 24px rgba(11,31,61,0.18);
}}
.phone-header {{
    background: {GREEN};
    color: white;
    border-radius: 12px 12px 0 0;
    padding: 10px 14px;
    font-weight: 700;
    font-size: 13px;
    margin: -18px -18px 10px -18px;
}}
.chat-bubble {{
    background: white;
    border-radius: 10px;
    padding: 12px 14px;
    font-size: 13.5px;
    line-height: 1.5;
    color: #111;
    box-shadow: 0 1px 1px rgba(0,0,0,0.08);
    position: relative;
}}
.chat-meta {{
    text-align: right;
    font-size: 10.5px;
    color: #9aa0a6;
    margin-top: 6px;
}}

.driver-row {{
    display:flex; justify-content:space-between; align-items:center;
    padding: 7px 0; border-bottom: 1px solid #EEF1F6; font-size: 13px;
}}

footer.brand-footer {{
    background: {NAVY};
    color: #AEB9D0;
    border-radius: 14px;
    padding: 16px 24px;
    font-size: 12px;
    margin-top: 24px;
    display:flex; justify-content:space-between; align-items:center;
}}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def _load_artifacts():
    return load_artifacts()

@st.cache_resource(show_spinner=False)
def _load_kb_collection():
    return get_chroma_collection()

@st.cache_data(show_spinner=False)
def _load_default_test_set():
    return pd.read_csv(APP_DIR / "data" / "momentum_lapse_test_for_dashboard.csv")


pipe, explainer, feature_names, model_card = _load_artifacts()
kb_collection = _load_kb_collection()

# ---------------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------------
st.markdown(f"""
<div class="hero">
  <div class="hero-eyebrow">USE CASE 01 · MOMENTUM + MULTIPLY · RETENTION</div>
  <div class="hero-title">Wellness-Linked Lapse Prediction &amp;<br/>GenAI Retention Outreach</div>
  <div class="hero-sub">Predicts which policyholders are at risk of lapsing using Multiply wellness
  signals, then drafts a grounded, personalised, multilingual retention message — never inventing
  a benefit the client doesn't actually have.</div>
  <div class="hero-byline">Rohit Nair · Lead Data Scientist · Momentum Africa Data Science</div>
</div>
""", unsafe_allow_html=True)

ollama_up = ollama_available()

# ---------------------------------------------------------------------------
# Upload & generate gate
# ---------------------------------------------------------------------------
if "raw_df" not in st.session_state:
    st.session_state["raw_df"] = _load_default_test_set()
if "source_label" not in st.session_state:
    st.session_state["source_label"] = "Sample portfolio (317 clients, held out)"

raw_df = st.session_state["raw_df"]
source_label = st.session_state["source_label"]
is_uploaded = source_label != "Sample portfolio (317 clients, held out)"

# ---------------------------------------------------------------------------
# Score (runs on whatever is currently active — uploaded file or sample)
# ---------------------------------------------------------------------------
scored_df, shap_values, Xt, fn = score_dataframe(raw_df, pipe, explainer, feature_names)

n_clients = len(scored_df)
n_red = (scored_df["risk_tier"] == "Red").sum()
n_amber = (scored_df["risk_tier"] == "Amber").sum()
premium_at_risk = scored_df.loc[scored_df["risk_tier"].isin(["Red", "Amber"]), "monthly_premium_zar"].sum() * 12

if is_uploaded:
    st.markdown(f"""
    <div style="background:{GREEN}14; border:1px solid {GREEN}55; border-radius:10px;
                padding:10px 16px; margin-bottom:14px; font-size:13px; color:{NAVY};">
    </div>
    """, unsafe_allow_html=True)



# ---------------------------------------------------------------------------
# KPI strip
# ---------------------------------------------------------------------------
k1, k2, k3, k4 = st.columns(4)
kpis = [
    (k1, TEAL, "Clients scored", f"{n_clients:,}", source_label),
    (k2, RED, "Red-tier (high risk)", f"{n_red:,}", f"{n_red/n_clients*100:.1f}% of portfolio"),
    (k3, ORANGE, "Annual premium at risk", f"R{premium_at_risk:,.0f}", "Red + Amber tiers"),
    (k4, GREEN, "Model AUC", f"{model_card['auc']:.3f}", f"Recall {model_card['recall']*100:.0f}% · Precision {model_card['precision']*100:.0f}%"),
]
for col, color, label, value, sub in kpis:
    with col:
        st.markdown(f"""
        <div class="kpi-card" style="--accent:{color};">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-sub">{sub}</div>
        </div>
        """, unsafe_allow_html=True)

st.write("")

tab1, tab2, tab3 = st.tabs([
    "📊  Portfolio Overview", "🔍  Client Risk Explorer", "💬  GenAI Outreach"
])

# ---------------------------------------------------------------------------
# TAB 1 — Portfolio overview
# ---------------------------------------------------------------------------
with tab1:
    c1, c2 = st.columns([1, 1.3])

    with c1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Risk tier distribution</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-caption">Green 0–40 · Amber 41–70 · Red 71–100</div>', unsafe_allow_html=True)
        tier_counts = scored_df["risk_tier"].value_counts().reindex(["Green", "Amber", "Red"]).fillna(0)
        fig = go.Figure(data=[go.Pie(
            labels=tier_counts.index, values=tier_counts.values, hole=0.62,
            marker=dict(colors=[TIER_COLOR[t] for t in tier_counts.index]),
            textinfo="label+percent", textfont=dict(size=12, family="Inter"),
        )])
        fig.update_layout(
            showlegend=False, height=300, margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            annotations=[dict(text=f"{n_clients}<br>clients", x=0.5, y=0.5, font_size=14, showarrow=False)]
        )
        st.plotly_chart(fig, width='stretch')
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Risk tier by market</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-caption">Where retention effort should concentrate first</div>', unsafe_allow_html=True)
        mk = scored_df.groupby(["market", "risk_tier"]).size().reset_index(name="count")
        fig2 = px.bar(
            mk, x="market", y="count", color="risk_tier",
            color_discrete_map=TIER_COLOR, barmode="stack",
            category_orders={"risk_tier": ["Green", "Amber", "Red"]},
        )
        fig2.update_layout(
            height=300, margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, title=None),
            xaxis_title=None, yaxis_title="Clients", font=dict(family="Inter", size=12),
        )
        st.plotly_chart(fig2, width='stretch')
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Top portfolio-level lapse drivers</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-caption">Mean |SHAP| impact across the scored portfolio — note how many are Multiply wellness signals Momentum already owns</div>', unsafe_allow_html=True)
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    order = np.argsort(mean_abs_shap)[::-1][:12]
    driver_df = pd.DataFrame({
        "feature": [fn[i] for i in order],
        "impact": [mean_abs_shap[i] for i in order],
    })
    multiply_kw = ["dayz", "gym", "lifestyle", "tier", "points", "redemption", "reward",
                   "app_logins", "engagement", "recharge", "health_assessment"]
    driver_df["group"] = driver_df["feature"].apply(
        lambda f: "Multiply wellness signal" if any(k in f.lower() for k in multiply_kw) else "Policy / transactional signal"
    )
    fig3 = px.bar(
        driver_df.sort_values("impact"), x="impact", y="feature", color="group", orientation="h",
        color_discrete_map={"Multiply wellness signal": TEAL, "Policy / transactional signal": NAVY},
    )
    fig3.update_layout(
        height=400, margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, title=None),
        xaxis_title="Mean |SHAP| impact on risk score", yaxis_title=None,
        font=dict(family="Inter", size=12),
    )
    st.plotly_chart(fig3, width='stretch')
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# TAB 2 — Client risk explorer
# ---------------------------------------------------------------------------
with tab2:
    # --- Client selector ---
    options_all = scored_df.sort_values("risk_score", ascending=False)["policy_id"].tolist()
    st.markdown(f"""
    <div style="background:linear-gradient(90deg,{NAVY} 0%,{NAVY_LIGHT} 100%);
                border-radius:14px; padding:18px 24px; margin:0 0 6px 0;
                display:flex; align-items:center; gap:14px;">
      <span style="font-size:20px;">🔍</span>
      <div>
        <div style="color:{TERRACOTTA}; font-size:11px; font-weight:700;
                    text-transform:uppercase; letter-spacing:0.1em; margin-bottom:2px;">
          Client Inspector
        </div>
        <div style="color:white; font-family:'Sora',sans-serif; font-size:15px; font-weight:700;">
          Select a client to inspect
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)
    _default_inspect = "MOM-5835BEQEU" if "MOM-5835BEQEU" in options_all else options_all[0]
    selected_policy = st.selectbox(
        "client_inspect",
        options_all,
        index=options_all.index(_default_inspect),
        format_func=lambda pid: (
            f"{pid} · {scored_df.set_index('policy_id').loc[pid, 'client_first_name']} "
            f"· Risk {scored_df.set_index('policy_id').loc[pid, 'risk_score']:.0f} "
            f"· {scored_df.set_index('policy_id').loc[pid, 'risk_tier']}"
        ),
        label_visibility="collapsed",
    )
    if selected_policy:
        st.session_state["selected_policy"] = selected_policy

    st.write("")

    # --- Client detail panel ---
    if not selected_policy:
        st.info("No clients match the current filters.")
    else:
        row_idx = scored_df.index[scored_df["policy_id"] == selected_policy][0]
        row = scored_df.loc[row_idx]
        tier = row["risk_tier"]

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown(f"""
        <div style="display:flex; justify-content:space-between; align-items:center;">
          <div>
            <div class="section-title">{row['client_first_name']} · {row['policy_id']}</div>
            <div class="section-caption">{row['market']} · {row['product_type']}</div>
          </div>
          <span class="badge badge-{tier.lower()}">RISK {row['risk_score']:.0f} · {tier.upper()}</span>
        </div>
        """, unsafe_allow_html=True)

        stats = [
            ("Multiply Tier",      row["multiply_tier"]),
            ("Active Dayz (3m)",   f"{int(row['active_dayz_3m_avg'])}/30"),
            ("Gym visits/mo",      int(row["gym_visits_monthly"])),
            ("Lifestyle Score",    int(row["lifestyle_quiz_score"])),
            ("Tenure (mo)",        int(row["policy_tenure_months"])),
            ("Monthly Premium",    f"R{row['monthly_premium_zar']:.0f}"),
            ("Tier Trend",         row["multiply_tier_change_direction"]),
            ("Points Balance",     f"{int(row['multiply_points_balance']):,}"),
        ]
        cols = st.columns(4)
        for i, (label, val) in enumerate(stats):
            cols[i % 4].markdown(f"""
            <div style="background:#F0F3F8; border-radius:10px; padding:10px 14px; margin-bottom:8px;">
              <div style="font-size:11px; color:{MUTED}; font-weight:600; text-transform:uppercase;
                          letter-spacing:0.06em; margin-bottom:4px;">{label}</div>
              <div style="font-size:15px; font-weight:700; color:{NAVY}; white-space:nowrap;
                          overflow:hidden; text-overflow:ellipsis;">{val}</div>
            </div>
            """, unsafe_allow_html=True)

        st.write("")
        st.markdown('<div class="section-title" style="font-size:14px;">Why this client is flagged (SHAP)</div>', unsafe_allow_html=True)
        drivers = top_drivers_for_row(shap_values[row_idx], fn, k=8)
        ddf = pd.DataFrame(drivers, columns=["feature", "shap"])
        ddf["direction"] = np.where(ddf["shap"] > 0, "Increases risk", "Decreases risk")
        ddf = ddf.sort_values("shap")
        fig4 = px.bar(
            ddf, x="shap", y="feature", orientation="h", color="direction",
            color_discrete_map={"Increases risk": RED, "Decreases risk": GREEN},
        )
        fig4.update_layout(
            height=320, margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, title=None),
            xaxis_title="SHAP contribution", yaxis_title=None, font=dict(family="Inter", size=11),
        )
        st.plotly_chart(fig4, width='stretch')
        st.markdown('</div>', unsafe_allow_html=True)

    st.write("")

    # --- Filters + table ---
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Scored client book</div>', unsafe_allow_html=True)

    f1, f2 = st.columns(2)
    tier_filter = f1.multiselect("Risk tier", ["Red", "Amber", "Green"], default=["Red", "Amber"])
    market_filter = f2.multiselect("Market", sorted(scored_df["market"].unique()), default=list(scored_df["market"].unique()))

    view = scored_df[scored_df["risk_tier"].isin(tier_filter) & scored_df["market"].isin(market_filter)]
    view = view.sort_values("risk_score", ascending=False)
    options = view["policy_id"].tolist()

    st.dataframe(
        view[["policy_id", "client_first_name", "market", "multiply_tier", "risk_score", "risk_tier",
              "monthly_premium_zar", "active_dayz_3m_avg"]].rename(columns={
            "policy_id": "Policy", "client_first_name": "Name", "market": "Market",
            "multiply_tier": "Multiply Tier", "risk_score": "Risk Score", "risk_tier": "Tier",
            "monthly_premium_zar": "Premium (ZAR)", "active_dayz_3m_avg": "Active Dayz (3m avg)",
        }),
        height=320, width='stretch', hide_index=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# TAB 3 — GenAI Outreach
# ---------------------------------------------------------------------------
with tab3:
    col_a, col_b = st.columns([1, 1.2])

    with col_a:
        st.markdown('<div class="section-title">1 · Select a flagged client</div>', unsafe_allow_html=True)
        red_amber = scored_df[scored_df["risk_tier"].isin(["Red", "Amber"])].sort_values("risk_score", ascending=False)
        if red_amber.empty:
            st.success("No clients currently flagged Red or Amber in this portfolio.")
            st.stop()

        default_policy = st.session_state.get("selected_policy", "MOM-5835BEQEU")
        if default_policy not in red_amber["policy_id"].values:
            default_policy = "MOM-5835BEQEU" if "MOM-5835BEQEU" in red_amber["policy_id"].values else red_amber.iloc[0]["policy_id"]

        chosen = st.selectbox(
            "Client", red_amber["policy_id"].tolist(),
            index=red_amber["policy_id"].tolist().index(default_policy),
            format_func=lambda pid: f"{pid} · {red_amber.set_index('policy_id').loc[pid, 'client_first_name']} "
                                     f"· Risk {red_amber.set_index('policy_id').loc[pid, 'risk_score']:.0f}",
        )
        row = scored_df[scored_df["policy_id"] == chosen].iloc[0].to_dict()

        st.write("")
        generate = st.button("✨ Generate personalised retention message", width='stretch', type="primary")

        st.write("")
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">2 · Client data payload</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-caption">Only these facts are passed to the LLM</div>', unsafe_allow_html=True)
        st.json({
            "first_name": row["client_first_name"], "language": row["preferred_language"],
            "channel": row["preferred_channel"], "product_type": row["product_type"],
            "market": row["market"], "multiply_tier": row["multiply_tier"],
            "active_dayz_3m_avg": int(row["active_dayz_3m_avg"]),
            "tier_change_direction": row["multiply_tier_change_direction"],
            "monthly_premium_zar": row["monthly_premium_zar"],
            "risk_tier": row["risk_tier"], "risk_score": row["risk_score"],
        }, expanded=False)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_b:
        if generate:
            with st.spinner("Retrieving approved facts and drafting message..."):
                out = generate_retention_message(row, kb_collection)
            st.session_state["last_genai_output"] = out
            st.session_state["last_genai_client"] = chosen

        out = st.session_state.get("last_genai_output")
        shown_for = st.session_state.get("last_genai_client")

        st.markdown('<div class="section-title">3 · Outreach preview</div>', unsafe_allow_html=True)
        if out and shown_for == chosen:
            src_badge = {
                "ollama_qwen2.5": ("badge-green", "🟢 Generated live by Ollama · Qwen2.5"),
                "template_fallback": ("badge-amber", "🟡 Grounded template fallback (Ollama unreachable)"),
                "template_fallback_after_guardrail": ("badge-amber", "🟡 Guardrail intercepted LLM output → grounded fallback used"),
            }[out["source"]]
            check = out["guardrail_check"]
            guard_badge = ("badge-green", "✅ Factual-consistency check passed") if check["passed"] \
                else ("badge-red", f"⚠️ Ungrounded claims blocked: {check['ungrounded_claims']}")

            st.markdown(f'<span class="badge {src_badge[0]}">{src_badge[1]}</span> '
                        f'<span class="badge {guard_badge[0]}">{guard_badge[1]}</span>', unsafe_allow_html=True)
            st.write("")

            channel = row["preferred_channel"]
            st.markdown(f"""
            <div class="phone-frame">
              <div class="phone-header">📱 {channel} · {row['client_first_name']}</div>
              <div class="chat-bubble">{out['message']}</div>
              <div class="chat-meta">Delivered via {channel} · {row['preferred_language']}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Click **Generate personalised retention message** to produce a grounded, "
                    "channel-ready message for this client.")


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown(f"""
<footer class="brand-footer">
  <div>MOMENTUM GROUP · METROPOLITAN · GUARDRISK · MULTIPLY</div>
  <div>UC01 · Wellness-Linked Lapse Prediction & GenAI Retention Outreach — Rohit Nair</div>
</footer>
""", unsafe_allow_html=True)
"""
UC01 - GenAI Layer: Personalised Retention Outreach (page 7 architecture)

Pipeline:
  1. Client scored Red/Amber/Green by the ML model (upstream)
  2. Client data payload assembled (product, Multiply milestone, language, offer, policy)
  3. ChromaDB RAG retrieves ONLY approved facts relevant to this client's tier/market
     (benefits, partner rewards, premium-relief options) -> the LLM is grounded on these,
     it does not "know" benefits on its own.
  4. Ollama (Qwen2.5) generates the message from client payload + retrieved facts only
  5. Automated factual-consistency check blocks numbers/claims not present in the grounding
  6. If Ollama is unreachable, a template-based generator produces an equivalent grounded
     message so the demo never breaks.
"""
import re
import json
import requests
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
OLLAMA_HOST = "http://localhost:11434"
OLLAMA_MODEL = "qwen2.5:latest"
OLLAMA_TIMEOUT = 12

# ---------------------------------------------------------------------------
# 1. Approved knowledge base — the ONLY source of truth the LLM may draw on.
#    In production this would sync from Momentum's product/benefits systems.
# ---------------------------------------------------------------------------
KB_DOCS = [
    {"id": "tier_blue", "tier": "Blue",
     "text": "Multiply Blue status members earn base cashback rates and access Active Dayz tracking. "
             "No partner travel or gym discounts unlocked yet."},
    {"id": "tier_bronze", "tier": "Bronze",
     "text": "Multiply Bronze status unlocks Weekly Wins vouchers and up to 25% cashback at selected "
             "retail partners including Dis-Chem."},
    {"id": "tier_silver", "tier": "Silver",
     "text": "Multiply Silver status unlocks up to 40% gym membership discounts at Virgin Active and "
             "Planet Fitness, plus increased Weekly Wins voucher value."},
    {"id": "tier_gold", "tier": "Gold",
     "text": "Multiply Gold status unlocks up to 60% gym discounts, flight and car rental partner "
             "discounts, and higher HealthReturns cashback."},
    {"id": "tier_diamond", "tier": "Diamond",
     "text": "Multiply Diamond status unlocks the full partner suite: up to 75% gym discounts, premium "
             "travel partner rates, and invitations to MultiplyCONNECT wellness retreats."},
    {"id": "premium_pause", "tier": "ANY",
     "text": "Clients showing financial strain (irregular payment history) may apply for a once-off "
             "1-month premium pause on Momentum Life and Funeral cover without losing continuous cover, "
             "to be confirmed by an advisor."},
    {"id": "active_dayz_reward", "tier": "ANY",
     "text": "Active Dayz are counted from linked fitness devices or the Momentum App; reaching the "
             "monthly Active Dayz goal earns that month's Multiply reward and protects current status."},
    {"id": "advisor_review", "tier": "ANY",
     "text": "Clients who have not had a policy review in over 12 months are entitled to a free annual "
             "cover review with their Momentum financial adviser."},
    {"id": "wellness_checkin", "tier": "ANY",
     "text": "Members who complete the Lifestyle Quiz or a Health Assessment within 90 days earn bonus "
             "Multiply points and a refreshed Lifestyle Score."},
    {"id": "lesotho_market", "tier": "ANY", "market": "Lesotho",
     "text": "Lesotho-market clients can reach Momentum advisers via the regional WhatsApp line and "
             "receive communication in Sesotho or English."},
    {"id": "botswana_market", "tier": "ANY", "market": "Botswana",
     "text": "Botswana-market clients can manage their policy and Multiply status via the Momentum "
             "Botswana app, with support in Setswana or English."},
    {"id": "namibia_market", "tier": "ANY", "market": "Namibia",
     "text": "Namibia-market clients receive Multiply partner benefits aligned to local retail and fuel "
             "partners, with support in English, Afrikaans or Oshiwambo."},
]


class LocalTfidfEmbedding(EmbeddingFunction):
    """
    Offline embedding function for ChromaDB (no external model download required).
    Fit once on the fixed knowledge base corpus at startup.
    """
    def __init__(self, corpus: list[str]):
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.vectorizer.fit(corpus)

    def __call__(self, input: Documents) -> Embeddings:
        mat = self.vectorizer.transform(input)
        return mat.toarray().tolist()


def get_chroma_collection(persist_dir=None):
    if persist_dir is None:
        persist_dir = str(APP_DIR / "knowledge_base")
    client = chromadb.PersistentClient(path=persist_dir)
    corpus = [d["text"] for d in KB_DOCS]
    embed_fn = LocalTfidfEmbedding(corpus)

    try:
        client.delete_collection("momentum_multiply_kb")
    except Exception:
        pass
    collection = client.create_collection("momentum_multiply_kb", embedding_function=embed_fn)
    collection.add(
        ids=[d["id"] for d in KB_DOCS],
        documents=[d["text"] for d in KB_DOCS],
        metadatas=[{"tier": d.get("tier", "ANY"), "market": d.get("market", "ANY")} for d in KB_DOCS],
    )
    return collection


def retrieve_facts(collection, client_row: dict, k: int = 4) -> list[str]:
    query = (f"Multiply {client_row['multiply_tier']} status benefits offers for a client in "
             f"{client_row['market']} with risk tier {client_row['risk_tier']} and product "
             f"{client_row['product_type']}")
    res = collection.query(query_texts=[query], n_results=k)
    docs = res["documents"][0]
    metas = res["metadatas"][0]

    # Always include the client's own tier benefit doc + market doc if available (deterministic grounding)
    forced = [d["text"] for d in KB_DOCS if d.get("tier") == client_row["multiply_tier"]]
    forced += [d["text"] for d in KB_DOCS if d.get("market") == client_row["market"]]

    facts = list(dict.fromkeys(forced + docs))  # de-dupe, preserve order
    return facts[:k + 2]


# ---------------------------------------------------------------------------
# Client payload assembly (page 7, step 2)
# ---------------------------------------------------------------------------
def build_client_payload(row: dict) -> dict:
    active_dayz = int(row.get("active_dayz_3m_avg", 0))
    goal = 21
    days_to_goal = max(goal - active_dayz, 0)

    payload = {
        "first_name": row.get("client_first_name", "Valued Client"),
        "language": row.get("preferred_language", "English"),
        "channel": row.get("preferred_channel", "WhatsApp"),
        "product_type": row.get("product_type", "Momentum Cover"),
        "market": row.get("market", "South Africa"),
        "multiply_tier": row.get("multiply_tier", "Blue"),
        "active_dayz_3m_avg": active_dayz,
        "active_dayz_goal_gap": days_to_goal,
        "lifestyle_quiz_score": row.get("lifestyle_quiz_score", None),
        "tier_change_direction": row.get("multiply_tier_change_direction", "Stable"),
        "monthly_premium_zar": row.get("monthly_premium_zar", None),
        "risk_tier": row.get("risk_tier", "Amber"),
        "risk_score": row.get("risk_score", 50),
        "num_dependants": row.get("num_dependants", 0),
    }
    return payload


# ---------------------------------------------------------------------------
# LLM generation (page 7, step 3) via Ollama / Qwen2.5
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are Momentum's retention outreach assistant. Write a short, warm, human
retention message to a policyholder who has been flagged as at risk of lapsing.

STRICT GROUNDING RULES:
- Use ONLY facts given to you in CLIENT DATA and APPROVED FACTS below.
- Never invent a benefit, discount, percentage, or offer that is not explicitly stated.
- Never mention the words "risk score", "model", "AI", "algorithm", or "flagged".
- Reference the client's actual Active Dayz progress and Multiply tier naturally.
- If APPROVED FACTS mention a premium pause or advisor review and the client's situation
  fits, you may offer it; otherwise do not mention it.
- Write in {language} where natural; keep it understandable even if mixed with English.
- Keep it under 80 words, suitable for {channel}.
- Sign off as "— Your Momentum team".
"""


def ollama_available() -> bool:
    try:
        r = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def generate_with_ollama(payload: dict, facts: list[str]) -> str | None:
    system = SYSTEM_PROMPT.format(language=payload["language"], channel=payload["channel"])
    user_msg = (
        "CLIENT DATA:\n" + json.dumps(payload, indent=2) +
        "\n\nAPPROVED FACTS (use only these for any claim/benefit/offer):\n- " +
        "\n- ".join(facts)
    )
    try:
        resp = requests.post(
            f"{OLLAMA_HOST}/api/chat",
            json={
                "model": OLLAMA_MODEL,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_msg},
                ],
                "stream": False,
                "options": {"temperature": 0.6},
            },
            timeout=OLLAMA_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"].strip()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Template fallback (used if Ollama is unreachable)
# ---------------------------------------------------------------------------
def generate_with_template(payload: dict, facts: list[str]) -> str:
    name = payload["first_name"]
    gap = payload["active_dayz_goal_gap"]
    tier = payload["multiply_tier"]
    product = payload["product_type"]

    activity_line = (
        f"you're only {gap} Active Dayz away from your next Multiply reward"
        if gap > 0 else
        f"you've already hit your Active Dayz goal this month — great work staying on {tier} status"
    )

    offer_line = ""
    if any("premium pause" in f.lower() for f in facts) and payload["risk_tier"] == "Red":
        offer_line = " If things feel tight this month, you can apply for a 1-month premium pause and keep your cover active."

    msg = (
        f"Hi {name}, just checking in — {activity_line}. We noticed your {product} hasn't been "
        f"reviewed in a while.{offer_line} Tap here to sort it in a minute. — Your Momentum team"
    )
    return msg


# ---------------------------------------------------------------------------
# Step 4: factual-consistency guardrail
# ---------------------------------------------------------------------------
def factual_consistency_check(message: str, facts: list[str], payload: dict) -> dict:
    """
    Lightweight grounding check: every ZAR amount / percentage in the message
    must also appear in the approved facts or the client payload.
    """
    facts_blob = " ".join(facts) + " " + json.dumps(payload)
    numbers_in_msg = set(re.findall(r"\d+%|\bR\s?\d+[\d,]*\b", message))
    numbers_in_grounding = set(re.findall(r"\d+%|\bR\s?\d+[\d,]*\b", facts_blob))
    ungrounded = numbers_in_msg - numbers_in_grounding
    return {"passed": len(ungrounded) == 0, "ungrounded_claims": list(ungrounded)}


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def generate_retention_message(row: dict, collection) -> dict:
    payload = build_client_payload(row)
    facts = retrieve_facts(collection, row)

    used_ollama = ollama_available()
    message = generate_with_ollama(payload, facts) if used_ollama else None
    source = "ollama_qwen2.5"

    if message is None:
        message = generate_with_template(payload, facts)
        source = "template_fallback"

    check = factual_consistency_check(message, facts, payload)
    if not check["passed"] and source == "ollama_qwen2.5":
        # Guardrail tripped on the LLM output -> fall back to the deterministic, grounded template
        message = generate_with_template(payload, facts)
        source = "template_fallback_after_guardrail"
        check = factual_consistency_check(message, facts, payload)

    return {
        "message": message,
        "source": source,
        "facts_used": facts,
        "client_payload": payload,
        "guardrail_check": check,
    }

import os
import sys
import types
import io
import calendar
from datetime import datetime
from collections import defaultdict

import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
import joblib
import torch
import huggingface_hub
from huggingface_hub import hf_hub_download
from sklearn.metrics.pairwise import cosine_similarity
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

try:
    from ofxparse import OfxParser
except ImportError:  # pragma: no cover - handled gracefully when dependency missing
    OfxParser = None

# Backward compatibility for saved SentenceTransformer artifacts
if not hasattr(huggingface_hub, "cached_download"):
    huggingface_hub.cached_download = hf_hub_download

if "sentence_transformers.model_card" not in sys.modules:
    model_card_module = types.ModuleType("sentence_transformers.model_card")

    class SentenceTransformerModelCardData(dict):
        """Minimal data holder for legacy SentenceTransformer model cards."""

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.__dict__ = self

        def to_json(self):
            return dict(self)

    class SentenceTransformerModelCard(dict):
        """Stub model card keeping parity with early sentence-transformers."""

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.__dict__ = self

        def to_json(self):
            return dict(self)

    def generate_model_card(*_, **__):
        """Compatibility shim for newer sentence-transformers."""
        return {}

    model_card_module.SentenceTransformerModelCardData = SentenceTransformerModelCardData
    model_card_module.SentenceTransformerModelCard = SentenceTransformerModelCard
    model_card_module.generate_model_card = generate_model_card
    sys.modules["sentence_transformers.model_card"] = model_card_module

from sentence_transformers import SentenceTransformer

SPEND_CATEGORY_KEYWORDS = {
    "dining": ["restaurant", "dining", "food", "cafe", "eat", "swiggy", "zomato", "ubereats", "coffee"],
    "groceries": ["grocery", "grocer", "supermarket", "mart", "bigbasket", "instamart"],
    "online_shopping": ["amazon", "flipkart", "myntra", "ajio", "nykaa", "shopping", "ecommerce"],
    "travel": ["air", "flight", "airlines", "hotel", "travel", "ola", "uber", "make my trip", "agoda"],
    "fuel": ["fuel", "petrol", "diesel", "hpcl", "bpcl", "ioc", "shell"],
}

DISPLAY_SPEND_CATEGORIES = [
    ("dining", "Dining & Food Delivery"),
    ("groceries", "Groceries"),
    ("online_shopping", "Online Shopping"),
    ("travel", "Travel"),
    ("fuel", "Fuel"),
]

MAX_PINNED_CARDS = 3


def ensure_session_defaults() -> None:
    """Initialize frequently used session keys."""
    if "pinned_cards" not in st.session_state:
        st.session_state["pinned_cards"] = []
    st.session_state.setdefault("spend_profile_values", None)
    st.session_state.setdefault("spend_profile_table", None)
    st.session_state.setdefault("seasonal_summary", "")
    st.session_state.setdefault("latest_recommendations", None)


def _infer_column(columns, keyword_candidates):
    for keyword in keyword_candidates:
        for column in columns:
            if keyword in column.lower():
                return column
    return None


def load_transactions(uploaded_file: io.BytesIO) -> pd.DataFrame:
    suffix = os.path.splitext(uploaded_file.name)[1].lower()
    if suffix == ".csv":
        df = pd.read_csv(uploaded_file)
        return df
    if suffix == ".ofx":
        if not OfxParser:
            raise ValueError("OFX support requires the 'ofxparse' package.")
        ofx = OfxParser.parse(uploaded_file)
        rows = []
        for account in ofx.accounts:
            for transaction in account.statement.transactions:
                rows.append(
                    {
                        "date": transaction.date,
                        "description": transaction.payee or transaction.memo or transaction.type,
                        "amount": transaction.amount,
                        "category_hint": transaction.type,
                    }
                )
        if not rows:
            raise ValueError("No transactions detected in the OFX file.")
        return pd.DataFrame(rows)
    raise ValueError("Unsupported file type. Please upload a CSV or OFX statement.")


def categorize_description(description: str, category_hint: str | None = None) -> str:
    if category_hint:
        normalized_hint = category_hint.lower()
        for slug in SPEND_CATEGORY_KEYWORDS:
            if slug in normalized_hint:
                return slug

    desc = (description or "").lower()
    for slug, keywords in SPEND_CATEGORY_KEYWORDS.items():
        if any(keyword in desc for keyword in keywords):
            return slug
    return "other"


def build_spend_profile(df: pd.DataFrame, months_window: int = 6):
    columns = [col for col in df.columns]
    date_col = _infer_column(columns, ["date"])
    desc_col = _infer_column(columns, ["description", "details", "narration", "merchant", "memo"])
    amount_col = _infer_column(columns, ["amount", "amt", "debit", "withdrawal", "value"])

    if not all([date_col, desc_col, amount_col]):
        raise ValueError("Unable to detect date/description/amount columns in the uploaded statement.")

    optional_cols = []
    for candidate in ["category", "category_hint"]:
        if candidate in df.columns:
            optional_cols.append(candidate)
    working_df = df[[date_col, desc_col, amount_col] + optional_cols].copy()
    working_df.columns = ["date", "description", "amount"] + optional_cols
    working_df["date"] = pd.to_datetime(working_df["date"], errors="coerce")
    working_df["amount"] = pd.to_numeric(working_df["amount"], errors="coerce")
    working_df = working_df.dropna(subset=["date", "amount"])

    if working_df.empty:
        raise ValueError("No valid transactions detected after parsing.")

    negative_share = (working_df["amount"] < 0).mean()
    if negative_share >= 0.5:
        working_df["spend_amount"] = working_df["amount"].clip(upper=0).abs()
    else:
        working_df["spend_amount"] = working_df["amount"].clip(lower=0)

    working_df = working_df[working_df["spend_amount"] > 0]
    if working_df.empty:
        raise ValueError("No expenses detected in the uploaded statement.")

    if "category" in working_df.columns:
        working_df["category_hint"] = working_df["category"]
    elif "category_hint" not in working_df.columns:
        working_df["category_hint"] = None

    working_df["category"] = working_df.apply(
        lambda row: categorize_description(row["description"], row.get("category_hint")), axis=1
    )

    end_date = working_df["date"].max()
    cutoff_date = end_date - pd.DateOffset(months=months_window - 1)
    window_df = working_df[working_df["date"] >= cutoff_date].copy()
    if window_df.empty:
        raise ValueError("No spend data available within the selected time window.")

    period_range = pd.period_range(cutoff_date.to_period("M"), end_date.to_period("M"), freq="M")
    months_count = max(1, len(period_range))

    profile_values = {slug: 0.0 for slug, _ in DISPLAY_SPEND_CATEGORIES}
    for slug in profile_values:
        total = window_df.loc[window_df["category"] == slug, "spend_amount"].sum()
        profile_values[slug] = float(total / months_count)

    summary_rows = [
        {
            "Category": label,
            "Avg Monthly Spend (₹)": round(profile_values.get(slug, 0.0), 2),
        }
        for slug, label in DISPLAY_SPEND_CATEGORIES
    ]
    summary_df = pd.DataFrame(summary_rows)

    seasonal_bits = []
    window_df["month"] = window_df["date"].dt.month
    for slug, label in DISPLAY_SPEND_CATEGORIES:
        category_monthly = window_df[window_df["category"] == slug]
        if category_monthly.empty:
            continue
        monthly_totals = category_monthly.groupby("month")["spend_amount"].sum()
        if monthly_totals.empty:
            continue
        peak_month = monthly_totals.idxmax()
        peak_value = monthly_totals.max()
        if peak_value >= category_monthly["spend_amount"].sum() * 0.35:
            seasonal_bits.append(f"{label} peaks in {calendar.month_abbr[int(peak_month)]}")

    seasonal_summary = "; ".join(seasonal_bits)
    return profile_values, summary_df, seasonal_summary


def format_spend_profile_text(profile: dict[str, float]) -> str:
    parts = []
    for slug, label in DISPLAY_SPEND_CATEGORIES:
        value = profile.get(slug, 0.0)
        parts.append(f"{label}: ₹{value:,.0f}")
    return ", ".join(parts)


def handle_statement_upload(uploaded_file, months_window: int):
    try:
        df = load_transactions(uploaded_file)
        profile, table, seasonal = build_spend_profile(df, months_window)
        st.session_state["spend_profile_values"] = profile
        st.session_state["spend_profile_table"] = table
        st.session_state["seasonal_summary"] = seasonal
        st.success("Transactions processed. Using data-driven spend profile.", icon="✅")
    except Exception as exc:  # pylint: disable=broad-except
        st.session_state["spend_profile_values"] = None
        st.session_state["spend_profile_table"] = None
        st.session_state["seasonal_summary"] = ""
        st.error(f"Unable to process statement: {exc}")


def clear_uploaded_profile():
    st.session_state["spend_profile_values"] = None
    st.session_state["spend_profile_table"] = None
    st.session_state["seasonal_summary"] = ""


def add_card_to_pins(card_payload: dict):
    pins = st.session_state.get("pinned_cards", [])
    if any(pin["Card Name"] == card_payload["Card Name"] for pin in pins):
        return
    if len(pins) >= MAX_PINNED_CARDS:
        pins.pop(0)
    pins.append(card_payload)
    st.session_state["pinned_cards"] = pins


def remove_card_from_pins(card_name: str):
    pins = st.session_state.get("pinned_cards", [])
    st.session_state["pinned_cards"] = [pin for pin in pins if pin["Card Name"] != card_name]


def render_comparison_section():
    st.subheader("📊 Comparison Board")
    pins = st.session_state.get("pinned_cards")
    if not pins:
        st.info("Pin cards from the recommendations to compare them side-by-side.")
        return
    comparison_columns = [
        "Card Name",
        "Issuer",
        "Annual Fee",
        "Interest Rate (p.m.)",
        "Reward Description",
        "Key Features",
    ]
    comp_df = pd.DataFrame(pins)[comparison_columns]
    st.dataframe(comp_df, use_container_width=True)
    if st.button("Clear comparison board"):
        st.session_state["pinned_cards"] = []


def render_recommendations(payload: dict):
    cards = payload.get("cards") or []
    if not cards:
        st.info("Provide your preferences and click “Find My Best Cards” to see recommendations.")
        return

    cards_df = pd.DataFrame(cards)
    spend_source = payload.get("spend_profile_source", "your inputs")
    scenario_desc = payload.get("scenario_label", "standard")
    st.subheader("🌟 Top Recommendations For You")
    st.info(f"Based on your {spend_source} – {scenario_desc} scenario – and selected preferences", icon="ℹ️")

    insight_cache = payload.setdefault("insights", {})
    for i, (_, row) in enumerate(cards_df.iterrows()):
        with st.expander(f"#{i+1}: {row['Card Name']} ({row['Issuer']})", expanded=True):
            col1, col2 = st.columns([1, 3])
            with col1:
                st.metric("Annual Fee", f"₹{row['Annual Fee']}")
                st.metric("Interest Rate", f"{row['Interest Rate (p.m.)']}% p.m.")
                st.metric("Net Annual Value", f"₹{row.get('Net Annual Value (₹)', 0):,.0f}")
                st.metric("Est. Rewards", f"₹{row.get('Estimated Annual Rewards (₹)', 0):,.0f}")
            with col2:
                st.write(f"**Rewards:** {row.get('Reward Description', '')}")
                st.write(f"**Key Features:** {row['Key Features']}")

            card_payload = row.to_dict()
            pinned_names = [pin["Card Name"] for pin in st.session_state.get("pinned_cards", [])]
            if row["Card Name"] in pinned_names:
                st.success("Pinned for comparison")
                if st.button("Remove from comparison", key=f"remove_{row['Card Name']}_{i}"):
                    remove_card_from_pins(row["Card Name"])
            else:
                if st.button("📌 Pin for comparison", key=f"pin_{row['Card Name']}_{i}", help="Pin up to three cards"):
                    add_card_to_pins(card_payload)

            llm_profile = payload.get("llm_profile", "")
            if row["Card Name"] in insight_cache:
                insight = insight_cache[row["Card Name"]]
            else:
                with st.spinner("Generating personalized insights..."):
                    insight = generate_insight(llm_profile, row)
                insight_cache[row["Card Name"]] = insight
            st.info(f"💡 **Why this card?**\n{insight}")
        st.divider()

    net_rows = payload.get("net_value_rows")
    if net_rows:
        st.subheader("💰 Net Annual Value Projection")
        net_df = pd.DataFrame(net_rows)
        chart = (
            alt.Chart(net_df)
            .mark_bar()
            .encode(
                x=alt.X("Net Annual Value (₹):Q", title="Net annual value (₹)"),
                y=alt.Y("Card:N", sort="-x", title="Card"),
                tooltip=["Card", "Net Annual Value (₹)", "Estimated Annual Rewards (₹)"],
            )
            .properties(height=200)
        )
        st.altair_chart(chart, use_container_width=True)

    render_comparison_section()


# Load environment variables
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

os.environ['STREAMLIT_SERVER_FILE_WATCHER_TYPE'] = 'none'
st.set_page_config(
    page_title="Cardly – Smart Credit Card Advisor",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource(show_spinner=False)
def load_resources():
    try:
        embedder = joblib.load('model/credit_card_embedder.joblib')
    except Exception as exc:
        st.warning(
            "Fallback to all-MiniLM-L6-v2 embedder "
            f"(joblib artifact missing or incompatible: {exc})"
        )
        embedder = SentenceTransformer("all-MiniLM-L6-v2")

    # Ensure older pickled models have the expected device attribute
    if not hasattr(embedder, "_target_device"):
        target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        embedder._target_device = target_device
        embedder.to(target_device)

    # Patch missing config attributes for legacy transformer configs
    transformer_module = None
    try:
        transformer_module = embedder[0]
    except Exception:
        transformer_module = None

    if transformer_module and hasattr(transformer_module, "auto_model"):
        config = transformer_module.auto_model.config
        for attr, default in {
            "output_attentions": False,
            "output_hidden_states": False,
            "return_dict": False,
        }.items():
            if not hasattr(config, attr):
                setattr(config, attr, default)
            private_attr = f"_{attr}"
            if not hasattr(config, private_attr):
                setattr(config, private_attr, default)

    try:
        scaler = joblib.load('model/credit_card_scaler.joblib')
    except Exception as exc:
        st.error(f"Failed to load scaler: {exc}")
        st.stop()

    try:
        card_vectors = np.load('model/credit_card_hybrid_embeddings.npy')
    except Exception as exc:
        st.error(f"Failed to load card embeddings: {exc}")
        st.stop()

    try:
        df = pd.read_csv('model/credit_card_data_final.csv')
    except Exception as exc:
        st.error(f"Failed to load card dataset: {exc}")
        st.stop()

    return embedder, scaler, card_vectors, df

embedder, scaler, card_vectors, df = load_resources()
ensure_session_defaults()

# LLM setup
chat_llm = ChatHuggingFace(
    llm=HuggingFaceEndpoint(
        repo_id="HuggingFaceH4/zephyr-7b-beta",
        task="conversational",
        temperature=0.3,
        max_new_tokens=256,
        huggingfacehub_api_token=HF_TOKEN
    )
)

def generate_insight(user_profile, card_row):
    prompt = f"""
    You are a financial advisor helping a user choose credit cards. 
    The user profile: {user_profile}
    Credit Card Details:
    - Card: {card_row['Card Name']}
    - Issuer: {card_row['Issuer']}
    - Annual Fee: ₹{card_row['Annual Fee']}
    - Reward Type: {card_row.get('Reward Description', '')}
    - Key Features: {card_row['Key Features']}
    
    Explain why this card is a good fit for the user in 2-3 sentences. 
    Highlight specific benefits that match the user's needs and estimate potential savings.
    """
    response = chat_llm.invoke([HumanMessage(content=prompt)])
    return response.content if hasattr(response, "content") else response

st.title("💳 Cardly")
st.caption("Smart credit card advisor for personalized recommendations based on your spending habits and preferences")

with st.sidebar:
    st.subheader("🔑 Authentication")
    if not HF_TOKEN:
        st.warning("Hugging Face token not found. Please set HF_TOKEN in environment variables.")
    else:
        st.success("Hugging Face token loaded successfully")
    
    st.divider()
    st.subheader("📂 Upload Spend Data (optional)")
    months_window = st.slider("Lookback window (months)", min_value=3, max_value=6, value=6, help="Average your transactions over the selected number of months.")
    uploaded_statement = st.file_uploader("Upload CSV or OFX statement", type=["csv", "ofx"], help="Only stored in memory for profiling – never uploaded.")
    if uploaded_statement is not None:
        handle_statement_upload(uploaded_statement, months_window)
    if st.session_state.get("spend_profile_table") is not None:
        st.dataframe(
            st.session_state["spend_profile_table"],
            use_container_width=True,
            hide_index=True,
        )
        if st.button("Clear uploaded data"):
            clear_uploaded_profile()

    st.subheader("⚙️ Your Preferences")
    monthly_income = st.number_input("Monthly Income (₹)", min_value=10000, value=50000, step=5000)
    st.caption("Minimum income requirement for most cards: ₹20,000-₹50,000")
    
    st.subheader("💰 Monthly Spending")
    dining = st.slider("Dining & Food Delivery", 0, 30000, 5000)
    groceries = st.slider("Groceries", 0, 30000, 6000)
    online_shopping = st.slider("Online Shopping", 0, 50000, 8000)
    travel = st.slider("Travel", 0, 50000, 3000)
    fuel = st.slider("Fuel", 0, 20000, 4000)
    if st.session_state.get("spend_profile_values"):
        st.success("Using spend insights from uploaded statements.")
    else:
        st.caption("Using manual slider values for spend profile.")
    
    st.subheader("⭐ Preferred Features")
    preferred_features = st.multiselect(
        "Select features important to you",
        ["Cashback", "Reward Points", "Lounge Access", "Travel Benefits", 
         "Fuel Savings", "No Annual Fee", "Movie Offers", "Airport Services"]
    )
    
    st.subheader("⚖️ Financial Preferences")
    joining_fee = st.number_input("Max Joining Fee (₹)", value=0)
    annual_fee = st.number_input("Max Annual Fee (₹)", value=1000)
    eligibility = monthly_income
    reward_rate = st.slider("Expected Reward Rate (%)", 0.0, 10.0, 3.0, 0.1)
    interest_rate = st.number_input("Max Interest Rate (% p.m.)", value=3.5)

manual_spend_profile = {
    "dining": dining,
    "groceries": groceries,
    "online_shopping": online_shopping,
    "travel": travel,
    "fuel": fuel,
}
active_spend_profile = st.session_state.get("spend_profile_values") or manual_spend_profile
seasonal_summary = st.session_state.get("seasonal_summary") or ""
spend_profile_source = "uploaded transactions" if st.session_state.get("spend_profile_values") else "manual sliders"
spend_summary_text = format_spend_profile_text(active_spend_profile)

st.markdown(f"**Current spend profile ({spend_profile_source})**: {spend_summary_text}")
if seasonal_summary:
    st.caption(f"Seasonal trends detected: {seasonal_summary}")

st.subheader("🧪 What-if scenario")
scenario_enabled = st.checkbox("Simulate travel surge (double travel budget)")
travel_multiplier = st.slider(
    "Travel multiplier", 1.0, 3.0, 2.0, 0.1, disabled=not scenario_enabled, help="Applied only when the scenario toggle is enabled."
)
scenario_profile = active_spend_profile.copy()
scenario_label = "standard"
if scenario_enabled:
    scenario_profile["travel"] = scenario_profile.get("travel", 0.0) * travel_multiplier
    scenario_label = f"travel x{travel_multiplier:.1f} scenario"
scenario_summary_text = format_spend_profile_text(scenario_profile)
st.caption(f"Scenario profile ({scenario_label}): {scenario_summary_text}")

user_query = st.text_area(
    "Describe your credit card needs:",
    "I want a card with good rewards for my regular spending. "
    "I frequently order food online and shop on e-commerce sites.",
    height=100
)

latest_payload = st.session_state.get("latest_recommendations")

if st.button("🔍 Find My Best Cards", use_container_width=True):
    with st.spinner("Analyzing your profile and finding the best cards..."):
        feature_text = ", ".join(preferred_features) if preferred_features else "General rewards"
        scenario_source_desc = f"{spend_profile_source} ({scenario_label})"
        user_text = (
            f"{user_query} | Features: {feature_text} | "
            f"Scenario spend profile ({scenario_source_desc}): {scenario_summary_text}"
        )
        if seasonal_summary:
            user_text += f" | Seasonal behavior: {seasonal_summary}"
        user_text_vec = embedder.encode([user_text])
        user_num_vec = scaler.transform([[joining_fee, annual_fee, eligibility, reward_rate, interest_rate]])

        # Combine vectors and handle NaN values
        user_vector = np.hstack([user_text_vec, user_num_vec]).astype('float32')
        user_vector = np.nan_to_num(user_vector)  # Replace NaNs with zeros

        # Find top 5 recommendations (ensure card_vectors has no NaNs)
        if np.isnan(card_vectors).any():
            card_vectors = np.nan_to_num(card_vectors)
        
        # Find top 5 recommendations
        scores = cosine_similarity(user_vector.reshape(1, -1), card_vectors)[0]
        top_indices = np.argsort(scores)[-5:][::-1]
        recommendations = df.iloc[top_indices].copy()

        annual_spend = sum(scenario_profile.values()) * 12
        cards_records = recommendations.to_dict("records")
        net_rows = []
        for card in cards_records:
            reward_rate_card = float(card.get("Reward Rate (%)") or 0.0)
            annual_fee_card = float(card.get("Annual Fee") or 0.0)
            est_rewards = (reward_rate_card / 100.0) * annual_spend
            net_value = est_rewards - annual_fee_card
            card["Estimated Annual Rewards (₹)"] = round(est_rewards, 2)
            card["Net Annual Value (₹)"] = round(net_value, 2)
            net_rows.append(
                {
                    "Card": card["Card Name"],
                    "Net Annual Value (₹)": round(net_value, 2),
                    "Estimated Annual Rewards (₹)": round(est_rewards, 2),
                }
            )

        llm_profile = (
            f"Income: ₹{monthly_income}/month | Spend profile: {scenario_summary_text} "
            f"({scenario_source_desc}) | Preferred features: {feature_text}"
        )
        if seasonal_summary:
            llm_profile += f" | Seasonal trends: {seasonal_summary}"

        latest_payload = {
            "cards": cards_records,
            "spend_profile_source": spend_profile_source,
            "scenario_label": scenario_label,
            "spend_summary_text": scenario_summary_text,
            "seasonal_summary": seasonal_summary,
            "feature_text": feature_text,
            "monthly_income": monthly_income,
            "llm_profile": llm_profile,
            "net_value_rows": net_rows,
            "annual_spend": annual_spend,
            "insights": {},
        }
        st.session_state["latest_recommendations"] = latest_payload

if latest_payload:
    render_recommendations(latest_payload)

st.divider()
st.caption("""
    **How it works:**
    - We analyze your spending habits and preferences
    - Match against 50+ credit cards using AI
    - Generate personalized explanations for each recommendation
    - All calculations done locally - your data never leaves your device
""")

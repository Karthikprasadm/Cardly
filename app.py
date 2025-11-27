import os
import sys
import types
import io
import json
import base64
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
from fpdf import FPDF, HTMLMixin

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

LANGUAGE_STRINGS = {
    "English": {
        "app_title": "Cardly",
        "app_subtitle": "Smart credit card advisor for personalized recommendations based on your spending habits and preferences",
        "experience_settings": "🛠 Experience settings",
        "language": "Language",
        "currency": "Currency",
        "contrast": "High contrast mode",
        "scenario_header": "🧪 Scenario planner",
        "scenario_caption": "Stack multiple what-if scenarios and use sliders to stress-test your recommendations.",
        "scenario_none": "standard",
        "scenario_active": "Active scenarios",
        "break_even": "Break-even spend",
        "upload_section": "📂 Upload Spend Data (optional)",
        "preferences_section": "⚙️ Your Preferences",
        "spend_section": "💰 Monthly Spending",
        "features_section": "⭐ Preferred Features",
        "finance_section": "⚖️ Financial Preferences",
        "language_hint": "Choose your preferred UI language",
        "currency_hint": "Select the currency for displaying values",
        "auth_section": "🔑 Authentication",
        "search_button": "🔍 Find My Best Cards",
    },
    "हिन्दी": {
        "app_title": "Cardly",
        "app_subtitle": "आपकी खर्च करने की आदतों पर आधारित स्मार्ट क्रेडिट कार्ड सलाहकार",
        "experience_settings": "🛠 अनुभव सेटिंग्स",
        "language": "भाषा",
        "currency": "मुद्रा",
        "contrast": "उच्च कंट्रास्ट मोड",
        "scenario_header": "🧪 परिदृश्य योजना",
        "scenario_caption": "एक से अधिक परिदृश्यों को आज़माएँ और स्लाइडर से परिणाम देखें।",
        "scenario_none": "मानक",
        "scenario_active": "सक्रिय परिदृश्य",
        "break_even": "ब्रेक-ईवन खर्च",
        "upload_section": "📂 खर्च डेटा अपलोड करें (वैकल्पिक)",
        "preferences_section": "⚙️ आपकी प्राथमिकताएँ",
        "spend_section": "💰 मासिक खर्च",
        "features_section": "⭐ पसंदीदा सुविधाएँ",
        "finance_section": "⚖️ वित्तीय प्राथमिकताएँ",
        "language_hint": "अपनी पसंदीदा भाषा चुनें",
        "currency_hint": "मुद्रा चुनें जिसमें मान दिखाए जाएँ",
        "auth_section": "🔑 प्रमाणीकरण",
        "search_button": "🔍 मेरे लिए सर्वश्रेष्ठ कार्ड खोजें",
    },
}

CURRENCY_CONFIG = {
    "INR": {"symbol": "₹", "rate_to_inr": 1.0},
    "USD": {"symbol": "$", "rate_to_inr": 83.0},
    "EUR": {"symbol": "€", "rate_to_inr": 90.0},
}

SCENARIO_CONFIGS = [
    {
        "key": "travel_surge",
        "label": "Travel surge",
        "description": "Simulate a travel-heavy period by multiplying your travel budget.",
        "adjustments": [
            {
                "category": "travel",
                "type": "multiplier",
                "label": "Travel multiplier",
                "min": 1.0,
                "max": 3.0,
                "default": 2.0,
                "step": 0.1,
            }
        ],
    },
    {
        "key": "festive_shopping",
        "label": "Festive shopping month",
        "description": "Boost online shopping and grocery spends for seasonal spikes.",
        "adjustments": [
            {
                "category": "online_shopping",
                "type": "multiplier",
                "label": "Online shopping multiplier",
                "min": 1.0,
                "max": 4.0,
                "default": 2.0,
                "step": 0.1,
            },
            {
                "category": "groceries",
                "type": "multiplier",
                "label": "Groceries multiplier",
                "min": 1.0,
                "max": 3.0,
                "default": 1.5,
                "step": 0.1,
            },
        ],
    },
    {
        "key": "fuel_spike",
        "label": "Fuel price increase",
        "description": "Account for higher fuel prices or longer commutes.",
        "adjustments": [
            {
                "category": "fuel",
                "type": "multiplier",
                "label": "Fuel multiplier",
                "min": 1.0,
                "max": 2.5,
                "default": 1.5,
                "step": 0.1,
            }
        ],
    },
    {
        "key": "new_subscription",
        "label": "New subscription",
        "description": "Add a new recurring subscription to your online spending.",
        "adjustments": [
            {
                "category": "online_shopping",
                "type": "addition",
                "label": "Monthly subscription amount",
                "min_inr": 0,
                "max_inr": 5000,
                "default_inr": 1500,
                "step_inr": 250,
            }
        ],
    },
]

SCENARIO_LABEL_LOOKUP = {cfg["key"]: cfg["label"] for cfg in SCENARIO_CONFIGS}

CARD_THEME_KEYWORDS = {
    "travel": ["travel", "flight", "airline", "lounge", "hotel", "miles", "vistara", "airport"],
    "fuel": ["fuel", "petrol", "diesel", "bpcl", "hpcl", "iocl", "fuel surcharge"],
    "cashback": ["cashback", "cash back", "cash-back", "cash back", "5%", "reward points"],
    "premium": ["premium", "luxury", "black", "infinia", "reserve", "metal", "infinite"],
    "online": ["online", "ecommerce", "amazon", "flipkart", "myntra", "swiggy", "instamart"],
    "groceries": ["grocery", "groceries", "supermarket", "daily needs", "departmental"],
}

PORTFOLIO_BUNDLES = [
    {
        "name": "Travel + Fuel Saver",
        "description": "Pair a travel-focused card with a fuel optimizer to cover commutes and getaways.",
        "themes": ["travel", "fuel"],
    },
    {
        "name": "Premium Lifestyle Duo",
        "description": "Mix a premium/luxury card with a cashback daily spender to balance perks and value.",
        "themes": ["premium", "cashback"],
    },
    {
        "name": "E-commerce + Essentials",
        "description": "Blend an online-shopping specialist with a groceries/dining card for daily life.",
        "themes": ["online", "groceries"],
    },
]


def translate_text(language: str, key: str) -> str:
    fallback = LANGUAGE_STRINGS["English"]
    lang_pack = LANGUAGE_STRINGS.get(language, fallback)
    return lang_pack.get(key, fallback.get(key, key))


def convert_from_inr(amount_in_inr: float, currency: str) -> float:
    rate = CURRENCY_CONFIG.get(currency, CURRENCY_CONFIG["INR"])["rate_to_inr"]
    return amount_in_inr / rate if rate else amount_in_inr


def convert_to_inr(amount: float, currency: str) -> float:
    rate = CURRENCY_CONFIG.get(currency, CURRENCY_CONFIG["INR"])["rate_to_inr"]
    return amount * rate


def format_currency(amount_in_inr: float, currency: str, decimals: int = 0) -> str:
    config = CURRENCY_CONFIG.get(currency, CURRENCY_CONFIG["INR"])
    if amount_in_inr is None:
        return f"{config['symbol']}0"
    converted = convert_from_inr(amount_in_inr, currency)
    return f"{config['symbol']}{converted:,.{decimals}f}"


def get_category_label(slug: str) -> str:
    return dict(DISPLAY_SPEND_CATEGORIES).get(slug, slug.replace("_", " ").title())


def currency_number_input(
    label: str,
    value_in_inr: float,
    step_in_inr: float,
    currency: str,
    min_in_inr: float = 0,
    help_text: str | None = None,
):
    min_display = int(convert_from_inr(min_in_inr, currency))
    value_display = int(convert_from_inr(value_in_inr, currency))
    step_display = max(1, int(convert_from_inr(step_in_inr, currency)))
    display_value = st.number_input(
        label,
        min_value=min_display,
        value=value_display,
        step=step_display,
        help=help_text,
    )
    return float(convert_to_inr(display_value, currency))


def currency_slider(
    label: str,
    min_in_inr: float,
    max_in_inr: float,
    default_in_inr: float,
    step_in_inr: float,
    currency: str,
    key: str | None = None,
):
    min_display = int(convert_from_inr(min_in_inr, currency))
    max_display = int(convert_from_inr(max_in_inr, currency))
    default_display = int(convert_from_inr(default_in_inr, currency))
    step_display = max(1, int(convert_from_inr(step_in_inr, currency)))
    selected_display = st.slider(
        label,
        min_value=min_display,
        max_value=max_display,
        value=default_display,
        step=step_display,
        key=key,
    )
    return float(convert_to_inr(selected_display, currency))


def infer_card_themes(card_row: dict) -> set[str]:
    text_parts = [
        card_row.get("Card Name", ""),
        card_row.get("Reward Description", ""),
        card_row.get("Key Features", ""),
    ]
    normalized = " ".join(str(part) for part in text_parts if part).lower()
    matched = set()
    for theme, keywords in CARD_THEME_KEYWORDS.items():
        if any(keyword in normalized for keyword in keywords):
            matched.add(theme)
    return matched or {"general"}


def build_portfolio_bundles(cards: list[dict], net_rows: list[dict], net_value_label: str, currency_symbol: str):
    card_theme_map = {}
    net_value_lookup = {}
    for card, net in zip(cards, net_rows):
        card_theme_map[card["Card Name"]] = infer_card_themes(card)
        net_value_lookup[card["Card Name"]] = net

    bundles_output = []
    for bundle in PORTFOLIO_BUNDLES:
        selected_cards = []
        used_names = set()
        for theme in bundle["themes"]:
            candidate = next(
                (
                    card
                    for card in cards
                    if theme in card_theme_map.get(card["Card Name"], set()) and card["Card Name"] not in used_names
                ),
                None,
            )
            if not candidate:
                break
            selected_cards.append(candidate)
            used_names.add(candidate["Card Name"])
        if len(selected_cards) == len(bundle["themes"]):
            coverage = ", ".join(sorted({theme for card in selected_cards for theme in card_theme_map[card["Card Name"]]}))
            total_net = sum(
                net_value_lookup.get(card["Card Name"], {}).get(net_value_label, 0)
                for card in selected_cards
            )
            bundles_output.append(
                {
                    "name": bundle["name"],
                    "description": bundle["description"],
                    "cards": selected_cards,
                    "coverage": coverage,
                    "total_net_value": total_net,
                    "currency_symbol": currency_symbol,
                }
            )
    return bundles_output


class TablePDF(FPDF, HTMLMixin):
    pass


def dataframe_to_pdf_bytes(df: pd.DataFrame) -> bytes:
    pdf = TablePDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Cardly Comparison", ln=True)
    pdf.ln(4)
    pdf.set_font("Arial", size=10)

    table_html = df.to_html(index=False, escape=True)
    table_html = table_html.replace(
        '<table border="1" class="dataframe">',
        '<table border="1" width="100%" cellspacing="0" cellpadding="4">',
    )
    pdf.write_html(table_html)
    output = pdf.output(dest="S")
    if isinstance(output, bytes):
        return output
    if isinstance(output, bytearray):
        return bytes(output)
    return str(output).encode("latin-1")


def serialize_comparison_state(df: pd.DataFrame) -> str:
    payload = df.to_dict(orient="records")
    raw = json.dumps(payload).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("utf-8")


def load_comparison_from_code(code: str) -> pd.DataFrame | None:
    try:
        raw = base64.urlsafe_b64decode(code.encode("utf-8"))
        data = json.loads(raw.decode("utf-8"))
        return pd.DataFrame(data)
    except Exception:
        return None


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


def format_spend_profile_text(profile: dict[str, float], currency: str) -> str:
    parts = []
    for slug, label in DISPLAY_SPEND_CATEGORIES:
        value = profile.get(slug, 0.0)
        parts.append(f"{label}: {format_currency(value, currency)}")
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
    csv_bytes = comp_df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download comparison (CSV)", csv_bytes, file_name="cardly-comparison.csv", use_container_width=True)
    try:
        pdf_bytes = dataframe_to_pdf_bytes(comp_df)
        st.download_button(
            "⬇️ Download comparison (PDF)",
            data=pdf_bytes,
            file_name="cardly-comparison.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
    except Exception as exc:  # pragma: no cover
        st.warning(f"Unable to generate PDF export: {exc}")
    share_code = serialize_comparison_state(comp_df)
    st.text_area(
        "Share this code with advisors (paste into Cardly to preload pins)",
        share_code,
        height=80,
    )
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
    currency = payload.get("currency", "INR")
    language = payload.get("language", "English")
    currency_symbol = CURRENCY_CONFIG.get(currency, CURRENCY_CONFIG["INR"])["symbol"]
    net_value_label = payload.get("net_value_label", f"Net Annual Value ({currency_symbol})")
    rewards_label = payload.get("rewards_label", f"Estimated Annual Rewards ({currency_symbol})")
    horizon_label = payload.get("horizon_label", f"{payload.get('time_horizon_months', 12)}-month Value ({currency_symbol})")
    time_horizon_months = payload.get("time_horizon_months", 12)
    st.subheader("🌟 Top Recommendations For You")
    st.info(f"Based on your {spend_source} – {scenario_desc} scenario – and selected preferences", icon="ℹ️")

    insight_cache = payload.setdefault("insights", {})
    for i, (_, row) in enumerate(cards_df.iterrows()):
        with st.expander(f"#{i+1}: {row['Card Name']} ({row['Issuer']})", expanded=True):
            col1, col2 = st.columns([1, 3])
            with col1:
                annual_fee_value = float(row.get("Annual Fee") or 0.0)
                st.metric("Annual Fee", format_currency(annual_fee_value, currency))
                st.metric("Interest Rate", f"{row['Interest Rate (p.m.)']}% p.m.")
                st.metric(
                    net_value_label,
                    format_currency(row.get("Net Annual Value (₹)", 0.0), currency),
                )
                st.metric(
                    rewards_label,
                    format_currency(row.get("Estimated Annual Rewards (₹)", 0.0), currency),
                )
                st.metric(
                    horizon_label,
                    format_currency(row.get("Projected Horizon Value (₹)", 0.0), currency),
                )
                reward_rate_card = float(row.get("Reward Rate (%)") or 0.0)
                break_even_spend = None
                if reward_rate_card > 0:
                    break_even_spend = annual_fee_value / (reward_rate_card / 100.0)
                st.metric(
                    translate_text(language, "break_even"),
                    format_currency(break_even_spend, currency) if break_even_spend is not None else "N/A",
                )
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
        st.caption("Calculated totals per recommendation (based on your current scenario).")
        st.dataframe(
            net_df[["Card", rewards_label, net_value_label, horizon_label]],
            use_container_width=True,
            hide_index=True,
        )
        x_axis_title = f"Net annual value ({currency_symbol})"
        chart = (
            alt.Chart(net_df)
            .mark_bar()
            .encode(
                x=alt.X(f"{net_value_label}:Q", title=x_axis_title),
                y=alt.Y("Card:N", sort="-x", title="Card"),
                tooltip=["Card", net_value_label, rewards_label],
            )
            .properties(height=200)
        )
        st.altair_chart(chart, use_container_width=True)
        bundles = build_portfolio_bundles(cards, net_rows, net_value_label, currency_symbol)
        if bundles:
            st.subheader("🧩 Portfolio Bundles")
            for bundle in bundles:
                with st.expander(bundle["name"], expanded=False):
                    st.caption(bundle["description"])
                    bundle_cards = pd.DataFrame(bundle["cards"])[["Card Name", "Issuer", "Reward Description"]]
                    st.dataframe(bundle_cards, hide_index=True, use_container_width=True)
                    st.markdown(f"**Combined coverage:** {bundle['coverage']}")
                    st.markdown(
                        f"**Total projected {time_horizon_months}-month value:** "
                        f"{bundle['currency_symbol']}{bundle['total_net_value']:,.2f}"
                    )

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

language_choice = st.session_state.get("preferred_language", "English")
currency_choice = st.session_state.get("preferred_currency", "INR")
high_contrast_pref = st.session_state.get("high_contrast", False)

with st.sidebar:
    st.subheader(translate_text(language_choice, "experience_settings"))
    language_options = list(LANGUAGE_STRINGS.keys())
    language_choice = st.selectbox(
        translate_text(language_choice, "language"),
        language_options,
        index=language_options.index(language_choice),
        help=translate_text(language_choice, "language_hint"),
    )
    st.session_state["preferred_language"] = language_choice

    currency_options = list(CURRENCY_CONFIG.keys())
    currency_choice = st.selectbox(
        translate_text(language_choice, "currency"),
        currency_options,
        index=currency_options.index(currency_choice),
        help=translate_text(language_choice, "currency_hint"),
    )
    st.session_state["preferred_currency"] = currency_choice

    high_contrast = st.checkbox(
        translate_text(language_choice, "contrast"),
        value=high_contrast_pref,
    )
    st.session_state["high_contrast"] = high_contrast

    shared_code_input = st.text_input("Paste a shared comparison code")
    if st.button("Load shared comparison", disabled=not shared_code_input):
        shared_df = load_comparison_from_code(shared_code_input)
        if shared_df is not None:
            st.session_state["pinned_cards"] = shared_df.to_dict(orient="records")
            st.success("Loaded shared comparison into the board.")
        else:
            st.error("Invalid share code. Please verify and try again.")

    st.divider()
    st.subheader(translate_text(language_choice, "auth_section"))
    if not HF_TOKEN:
        st.warning("Hugging Face token not found. Please set HF_TOKEN in environment variables.")
    else:
        st.success("Hugging Face token loaded successfully")

    st.divider()
    st.subheader(translate_text(language_choice, "upload_section"))
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

    currency_symbol = CURRENCY_CONFIG[currency_choice]["symbol"]

    st.subheader(translate_text(language_choice, "preferences_section"))
    monthly_income = currency_number_input(
        f"Monthly Income ({currency_symbol})",
        value_in_inr=50000,
        step_in_inr=5000,
        currency=currency_choice,
        min_in_inr=10000,
        help_text="Minimum income requirement for most cards: ₹20,000-₹50,000",
    )

    st.subheader(translate_text(language_choice, "spend_section"))
    dining = currency_slider(f"Dining & Food Delivery ({currency_symbol})", 0, 30000, 5000, 500, currency_choice, key="spend_dining")
    groceries = currency_slider(f"Groceries ({currency_symbol})", 0, 30000, 6000, 500, currency_choice, key="spend_groceries")
    online_shopping = currency_slider(f"Online Shopping ({currency_symbol})", 0, 50000, 8000, 500, currency_choice, key="spend_online")
    travel = currency_slider(f"Travel ({currency_symbol})", 0, 50000, 3000, 500, currency_choice, key="spend_travel")
    fuel = currency_slider(f"Fuel ({currency_symbol})", 0, 20000, 4000, 250, currency_choice, key="spend_fuel")
    if st.session_state.get("spend_profile_values"):
        st.success("Using spend insights from uploaded statements.")
    else:
        st.caption("Using manual slider values for spend profile.")

    st.subheader(translate_text(language_choice, "features_section"))
    preferred_features = st.multiselect(
        "Select features important to you",
        ["Cashback", "Reward Points", "Lounge Access", "Travel Benefits", "Fuel Savings", "No Annual Fee", "Movie Offers", "Airport Services"],
    )

    st.subheader(translate_text(language_choice, "finance_section"))
    joining_fee = currency_number_input(f"Max Joining Fee ({currency_symbol})", value_in_inr=0, step_in_inr=500, currency=currency_choice)
    annual_fee = currency_number_input(f"Max Annual Fee ({currency_symbol})", value_in_inr=1000, step_in_inr=500, currency=currency_choice)
    eligibility = monthly_income
    reward_rate = st.slider("Expected Reward Rate (%)", 0.0, 10.0, 3.0, 0.1)
    interest_rate = st.number_input("Max Interest Rate (% p.m.)", value=3.5)

    st.subheader("📈 Bonus & fee forecasting")
    time_horizon_months = st.selectbox(
        "Projection horizon",
        options=[12, 24, 36],
        format_func=lambda m: f"{m} months",
        index=0,
    )
    welcome_bonus_inr = currency_number_input(
        f"Expected welcome bonus ({currency_symbol})",
        value_in_inr=10000,
        step_in_inr=1000,
        currency=currency_choice,
    )
    statement_credit_inr = currency_number_input(
        f"Expected statement credits ({currency_symbol})",
        value_in_inr=5000,
        step_in_inr=500,
        currency=currency_choice,
    )
    milestone_reward_inr = currency_number_input(
        f"Milestone rewards ({currency_symbol})",
        value_in_inr=3000,
        step_in_inr=500,
        currency=currency_choice,
    )
    anticipate_fee_waiver = st.checkbox("Expect annual fee waiver after meeting spend threshold?")
    fee_waiver_threshold_inr = 0.0
    if anticipate_fee_waiver:
        fee_waiver_threshold_inr = currency_number_input(
            f"Annual spend needed for waiver ({currency_symbol})",
            value_in_inr=240000,
            step_in_inr=10000,
            currency=currency_choice,
            help_text="If your projected annual spend meets this threshold, we treat the annual fee as waived.",
        )

if st.session_state.get("high_contrast"):
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #0b0c0f;
            color: #f5f5f5;
        }
        .stApp [data-testid="stHeader"] {
            background: transparent;
        }
        .stButton>button {
            background-color: #f5f5f5;
            color: #0b0c0f;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

st.title(f"💳 {translate_text(language_choice, 'app_title')}")
st.caption(translate_text(language_choice, "app_subtitle"))

with st.expander("🧭 Guided onboarding", expanded=False):
    steps = [
        {
            "label": "Upload statements or set sliders",
            "done": bool(st.session_state.get("spend_profile_values")),
            "hint": "Import CSV/OFX files or tune manual spends.",
        },
        {
            "label": "Configure scenarios & preferences",
            "done": bool(st.session_state.get("scenario_active_flags")),
            "hint": "Use the new planner and forecasting inputs.",
        },
        {
            "label": "Run recommendations & review results",
            "done": bool(st.session_state.get("latest_recommendations")),
            "hint": "Click the search button and explore charts/tables.",
        },
    ]
    for idx, step in enumerate(steps, start=1):
        status_icon = "✅" if step["done"] else "⬜️"
        st.markdown(f"{status_icon} **Step {idx}: {step['label']}** — {step['hint']}")

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
spend_summary_text = format_spend_profile_text(active_spend_profile, currency_choice)

st.markdown(f"**Current spend profile ({spend_profile_source})**: {spend_summary_text}")
if seasonal_summary:
    st.caption(f"Seasonal trends detected: {seasonal_summary}")

st.subheader(translate_text(language_choice, "scenario_header"))
st.caption(translate_text(language_choice, "scenario_caption"))
scenario_profile = active_spend_profile.copy()
scenario_selection = st.multiselect(
    "Select scenarios",
    options=[cfg["key"] for cfg in SCENARIO_CONFIGS],
    format_func=lambda key: SCENARIO_LABEL_LOOKUP.get(key, key),
)
scenario_notes = []
for cfg in SCENARIO_CONFIGS:
    if cfg["key"] not in scenario_selection:
        continue
    with st.expander(cfg["label"], expanded=False):
        st.caption(cfg["description"])
        for adj in cfg["adjustments"]:
            slider_key = f"{cfg['key']}_{adj['category']}_{adj['label']}"
            if adj["type"] == "multiplier":
                slider_value = st.slider(
                    adj["label"],
                    min_value=float(adj["min"]),
                    max_value=float(adj["max"]),
                    value=float(adj["default"]),
                    step=float(adj["step"]),
                    key=slider_key,
                )
                scenario_profile[adj["category"]] = scenario_profile.get(adj["category"], 0.0) * slider_value
                scenario_notes.append(f"{cfg['label']} – {get_category_label(adj['category'])} x{slider_value:.1f}")
            else:
                addition_value = currency_slider(
                    f"{adj['label']} ({currency_symbol})",
                    adj["min_inr"],
                    adj["max_inr"],
                    adj["default_inr"],
                    adj["step_inr"],
                    currency_choice,
                    key=slider_key,
                )
                scenario_profile[adj["category"]] = scenario_profile.get(adj["category"], 0.0) + addition_value
                scenario_notes.append(
                    f"{cfg['label']} +{format_currency(addition_value, currency_choice)} to {get_category_label(adj['category'])}"
                )

scenario_label = translate_text(language_choice, "scenario_none")
if scenario_selection:
    scenario_label = ", ".join(SCENARIO_LABEL_LOOKUP.get(key, key) for key in scenario_selection)
scenario_summary_text = format_spend_profile_text(scenario_profile, currency_choice)
if scenario_notes:
    st.caption(f"{translate_text(language_choice, 'scenario_active')}: {'; '.join(scenario_notes)}")
st.caption(f"Scenario profile ({scenario_label}): {scenario_summary_text}")
st.session_state["scenario_active_flags"] = bool(scenario_selection)

user_query = st.text_area(
    "Describe your credit card needs:",
    "I want a card with good rewards for my regular spending. "
    "I frequently order food online and shop on e-commerce sites.",
    height=100
)

latest_payload = st.session_state.get("latest_recommendations")

if st.button(translate_text(language_choice, "search_button"), use_container_width=True):
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
        net_value_label = f"Net Annual Value ({currency_symbol})"
        rewards_label = f"Estimated Annual Rewards ({currency_symbol})"
        horizon_label = f"{time_horizon_months}-month Value ({currency_symbol})"
        horizon_years = time_horizon_months / 12.0
        for card in cards_records:
            reward_rate_card = float(card.get("Reward Rate (%)") or 0.0)
            annual_fee_card = float(card.get("Annual Fee") or 0.0)
            est_rewards = (reward_rate_card / 100.0) * annual_spend
            net_value = est_rewards - annual_fee_card
            card["Estimated Annual Rewards (₹)"] = round(est_rewards, 2)
            card["Net Annual Value (₹)"] = round(net_value, 2)
            fee_waived = anticipate_fee_waiver and annual_spend >= fee_waiver_threshold_inr
            effective_fee = 0.0 if fee_waived else annual_fee_card
            projected_rewards = est_rewards * horizon_years
            total_value_horizon = (
                projected_rewards
                + welcome_bonus_inr
                + statement_credit_inr
                + milestone_reward_inr
                - effective_fee * horizon_years
            )
            card["Projected Horizon Value (₹)"] = round(total_value_horizon, 2)
            net_rows.append(
                {
                    "Card": card["Card Name"],
                    rewards_label: round(convert_from_inr(est_rewards, currency_choice), 2),
                    net_value_label: round(convert_from_inr(net_value, currency_choice), 2),
                    horizon_label: round(convert_from_inr(total_value_horizon, currency_choice), 2),
                }
            )

        llm_profile = (
            f"Income: {format_currency(monthly_income, currency_choice)}/month | "
            f"Spend profile: {scenario_summary_text} ({scenario_source_desc}) | Preferred features: {feature_text}"
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
            "currency": currency_choice,
            "language": language_choice,
            "net_value_label": net_value_label,
            "rewards_label": rewards_label,
            "horizon_label": horizon_label,
            "time_horizon_months": time_horizon_months,
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

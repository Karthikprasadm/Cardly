import os
import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity
from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
os.environ['STREAMLIT_SERVER_FILE_WATCHER_TYPE'] = 'none'
st.set_page_config(
    page_title="Smart Credit Card Advisor",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource(show_spinner=False)
def load_resources():
    embedder = joblib.load('model/credit_card_embedder.joblib')
    scaler = joblib.load('model/credit_card_scaler.joblib')
    card_vectors = np.load('model/credit_card_hybrid_embeddings.npy')
    df = pd.read_csv('model/credit_card_data_final.csv')
    return embedder, scaler, card_vectors, df

embedder, scaler, card_vectors, df = load_resources()

# LLM setup
llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    temperature=0.3,
    max_new_tokens=256,
    huggingfacehub_api_token=HF_TOKEN
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
    return llm.invoke(prompt)

st.title("💳 Smart Credit Card Advisor")
st.caption("Get personalized credit card recommendations based on your spending habits and preferences")

with st.sidebar:
    st.subheader("🔑 Authentication")
    if not HF_TOKEN:
        st.warning("Hugging Face token not found. Please set HF_TOKEN in environment variables.")
    else:
        st.success("Hugging Face token loaded successfully")
    
    st.divider()
    st.subheader("⚙️ Your Preferences")
    monthly_income = st.number_input("Monthly Income (₹)", min_value=10000, value=50000, step=5000)
    st.caption("Minimum income requirement for most cards: ₹20,000-₹50,000")
    
    st.subheader("💰 Monthly Spending")
    dining = st.slider("Dining & Food Delivery", 0, 30000, 5000)
    groceries = st.slider("Groceries", 0, 30000, 6000)
    online_shopping = st.slider("Online Shopping", 0, 50000, 8000)
    travel = st.slider("Travel", 0, 50000, 3000)
    fuel = st.slider("Fuel", 0, 20000, 4000)
    
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

user_query = st.text_area(
    "Describe your credit card needs:",
    "I want a card with good rewards for my regular spending. "
    "I frequently order food online and shop on e-commerce sites.",
    height=100
)

if st.button("🔍 Find My Best Cards", use_container_width=True):
    with st.spinner("Analyzing your profile and finding the best cards..."):
        # Prepare user vector (order: Joining Fee, Annual Fee, Eligibility, Reward Rate, Interest Rate)
        # Prepare user vector
        user_text = f"{user_query} | Features: {', '.join(preferred_features)}"
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
        recommendations = df.iloc[top_indices]
        
        # Display recommendations
        st.subheader("🌟 Top Recommendations For You")
        st.info("Based on your spending habits and preferences", icon="ℹ️")
        
        for i, (_, row) in enumerate(recommendations.iterrows()):
            with st.expander(f"#{i+1}: {row['Card Name']} ({row['Issuer']})", expanded=True):
                col1, col2 = st.columns([1, 3])
                with col1:
                    st.metric("Annual Fee", f"₹{row['Annual Fee']}")
                    st.metric("Interest Rate", f"{row['Interest Rate (p.m.)']}% p.m.")
                with col2:
                    st.write(f"**Rewards:** {row.get('Reward Description', '')}")
                    st.write(f"**Key Features:** {row['Key Features']}")
                with st.spinner("Generating personalized insights..."):
                    insight = generate_insight(
                        f"Income: ₹{monthly_income}/month | "
                        f"Dining: ₹{dining} | Groceries: ₹{groceries} | "
                        f"Online: ₹{online_shopping} | Travel: ₹{travel} | "
                        f"Fuel: ₹{fuel} | Features: {', '.join(preferred_features)}",
                        row
                    )
                    st.info(f"💡 **Why this card?**\n{insight}")
                st.divider()

st.divider()
st.caption("""
    **How it works:**
    - We analyze your spending habits and preferences
    - Match against 50+ credit cards using AI
    - Generate personalized explanations for each recommendation
    - All calculations done locally - your data never leaves your device
""")

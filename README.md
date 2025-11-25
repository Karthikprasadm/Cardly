
---

# Cardly – Smart Credit Card Advisor

Cardly is a web-based, AI-powered credit card recommendation system that combines traditional machine learning with large language models (LLMs) to provide personalized credit card suggestions based on a user’s financial profile and stated preferences.

---

## Project Overview

The system uses a hybrid embedding approach (text + numeric features) to find the best-matching cards for any user profile. A Streamlit frontend handles the user interaction layer, while a Hugging Face Zephyr LLM produces concise explanations for every recommendation.

---

## Features

- Personalized credit card recommendations based on user input
- Hybrid embeddings combining card descriptions and numeric features
- LLM-powered explanations for each recommended card
- Interactive Streamlit UI for easy user interaction
- Deployed on Streamlit Cloud with Hugging Face integration
- Transaction upload (CSV/OFX) to auto-build spend profiles with seasonal weighting
- Card comparison board to pin up to three cards and review them side-by-side
- Net annual value projection with what-if travel scenarios

---

## Project Structure

```
creditCardRecommendationSystem/
 ┣ model/
 ┃ ┣ credit_card_data_cleaned.csv
 ┃ ┣ credit_card_data_final.csv
 ┃ ┣ credit_card_embedder.joblib
 ┃ ┣ credit_card_hybrid_embeddings.npy
 ┃ ┗ credit_card_scaler.joblib
 ┣ app.py
 ┣ README.md
 ┣ requirements.txt
 ┗ sodapdf-converted.pdf
```

---

## Live Demo & Repository

- **Video link:** [https://vimeo.com/1094748093/63f14e9da2?share=copy](https://vimeo.com/1094748093/63f14e9da2?share=copy)
- **Live App:** [https://creditcardrecommendationsystem-project.streamlit.app/](https://creditcardrecommendationsystem-project.streamlit.app/)
- **GitHub (current repo):** [https://github.com/Karthikprasadm/Cardly](https://github.com/Karthikprasadm/Cardly)
<div align="center">
  <img src="Untitled video - Made with Clipchamp (7).gif" height="500" />
</div>
---

## Quick Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/Karthikprasadm/Cardly.git
   cd Cardly
   ```

2. **(Recommended) Create and activate a virtual environment**
   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # macOS/Linux
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure secrets**
   - Create a `.env` file (not committed to git) containing:
     ```
     HF_TOKEN=your_huggingface_token_here
     ```
   - Never push the `.env` file or your token/credentials to GitHub. Use `git status` to verify it’s untracked.

5. **Run the Streamlit app**
   ```bash
   streamlit run app.py
   ```

6. **Open** `http://localhost:8501` **and start exploring recommendations.**

For a detailed walkthrough (including screenshots, troubleshooting tips, and how to interpret the outputs) see [`how_to_run&use.md`](how_to_run&use.md).

---

## Usage

- Enter your financial profile and preferences in the UI.
- Get personalized credit card recommendations with detailed explanations.
- Upload recent statements to let Cardly auto-categorize your spending (the sliders become fallbacks).
- Pin recommended cards to compare annual fees, rewards, and perks.
- Use the “What-if scenario” toggle to simulate travel-heavy months and review the net annual value bar chart to see how each card stacks up.

---

## How It Works

- **Data Layer:** Contains cleaned credit card data and precomputed embeddings.
- **Embedding Models:** Sentence Transformer model for text embeddings and scaler for numeric features.
- **Recommendation Engine:** Combines text and numeric embeddings to find best matches.
- **LLM Integration:** Uses Hugging Face Zephyr-7B-Beta model for personalized insights.
- **Frontend:** Streamlit app for user interaction.

---

## Data Processing

- Extracted credit card data from PDF using pdfplumber.
- Parsed and cleaned data with regex to handle various formats.
- Converted monetary values (lakh, crore) to numeric.
- Saved cleaned data as CSV for model training.

---

## Model Training and Embeddings

- Used 'all-MiniLM-L6-v2' sentence transformer for text embeddings.
- Normalized numeric features with MinMaxScaler.
- Combined text and numeric embeddings into hybrid vectors.
- Saved embeddings and models for inference.

---

## Recommendation Logic

- User inputs text preferences and numeric constraints.
- User input is embedded and normalized.
- Cosine similarity is computed against card embeddings.
- Top 5 cards are selected and passed to LLM for explanation.

---

## Deployment

- Deployed on Streamlit Cloud with Hugging Face integration.
- Hugging Face API token managed securely via environment variables.

---

## Future Work

- Add more cards and update dataset regularly.
- Improve LLM prompts for better explanations.
- Add user authentication and history tracking.

---

## Contributing

Contributions are welcome! Please open issues or pull requests for improvements.

---

## License

This project is licensed under the MIT License.

---

## Contact

For questions or contributions, please open an issue or pull request on the [GitHub repository](https://github.com/alpha2lucifer/creditCardRecommendationSystem).

---

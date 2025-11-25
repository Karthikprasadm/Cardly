
---

# Cardly – Smart Credit Card Advisor

Cardly is a web-based, AI-powered credit card recommendation system that combines traditional machine learning with large language models (LLMs) to provide personalized credit card suggestions based on a user’s financial profile and stated preferences.

---

## Project Overview

The system uses a hybrid embedding approach (text + numeric features) to find the best-matching cards for any user profile. A Streamlit frontend handles the user interaction layer, while a Hugging Face Zephyr LLM produces concise explanations for every recommendation.

---

## Features at a Glance

- **AI-assisted matching** – text + numeric embeddings pick the best cards for the user’s narrative and financial constraints.
- **LLM explanations** – every recommendation includes a short “Why this card?” summary generated with the Zephyr-7B model.
- **Statement-driven spend modeling** – upload CSV/OFX files to auto-categorize 3–6 months of spend and detect seasonal spikes; sliders stay available as a fallback.
- **Scenario planner** – flip a toggle to simulate travel-heavy months and see how the ranking changes.
- **Net annual value analytics** – projected rewards minus fees are displayed per card and as a bar chart so users instantly see the best ROI.
- **Pin & compare** – select up to three cards, then view their fees, rewards, perks, and insights side-by-side.
- **Ready for deployment** – light Streamlit frontend, Hugging Face backend, simple `.env` configuration.

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

## Quick Start (Detailed)

1. **Check prerequisites**
   - Python 3.10 (tested on 3.10.11)
   - `pip` and `git`
   - Hugging Face account + access token (fine-grained, read scope is enough)
   - Optional: CSV/OFX statements for upload

2. **Clone the repository**
   ```bash
   git clone https://github.com/Karthikprasadm/Cardly.git
   cd Cardly
   ```

3. **Create and activate a virtual environment (recommended)**
   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # macOS/Linux
   source .venv/bin/activate
   ```
   Your shell prompt should now include `(.venv)`.

4. **Install dependencies**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

5. **Configure environment variables**
   - Create a `.env` file in the project root:
     ```
     HF_TOKEN=hf_your_token_here
     ```
   - Keep `.env` out of version control (`.gitignore` is already configured).

6. **Launch the app**
   ```bash
   streamlit run app.py
   ```
   Streamlit prints a local URL (default `http://localhost:8501`). Open it in a browser.

Need screenshots and step-by-step guidance? Check the companion guide: [`how_to_run&use.md`](how_to_run&use.md).

---

## UI Walkthrough

1. **Authentication panel** – instantly shows whether the Hugging Face token was detected.
2. **Upload Spend Data** – drag & drop CSV/OFX statements. Cardly auto-maps columns, filters expenses, averages the last 3–6 months, and flags seasonal peaks. If no file is uploaded, the manual sliders stay in control.
3. **Preferences & sliders** – set monthly income, category-level spend, preferred perks, and fee thresholds.
4. **Scenario toggle** – simulate a travel bump (or any multiplier you choose) before running the recommender.
5. **Run search** – hit “Find My Best Cards.” A hybrid embedding compares your profile with the stored embeddings and surfaces the top five matches.
6. **Results section**
   - Each expander shows fees, interest rate, rewards, key features, estimated annual rewards, and the net value (rewards minus fees).
   - Zephyr-7B generates a short “Why this card?” explanation using your profile and scenario context.
   - Pin cards with the 📌 button; pinned cards appear in the comparison table below.
7. **Net value chart & comparison board** – visualize the ranking via a bar chart, then inspect pinned cards side-by-side.

---

## How It Works (Under the Hood)

| Layer | Description |
| --- | --- |
| Data layer | `model/credit_card_data_final.csv` holds curated card features. Additional artifacts (`.joblib`, `.npy`) store the embedder, scaler, and precomputed hybrids. |
| Embeddings | Text features use `all-MiniLM-L6-v2`; numeric features are normalized via `MinMaxScaler`. The resulting hybrid vector sits in `credit_card_hybrid_embeddings.npy`. |
| Spend modeling | Uploads are parsed with pandas/`ofxparse`, categorized via keyword mapping, and averaged across months. Seasonal spikes are noted for LLM context. |
| Recommendation engine | User text + numeric preferences are embedded, then cosine similarity locates the five best cards. |
| LLM integration | LangChain’s `ChatHuggingFace` wrapper calls Zephyr-7B-Beta to explain the match. |
| Frontend | Streamlit surfaces the whole experience with session-state persistence so pinning/comparison feels native. |

---

## Data Processing Highlights

1. Source PDF (`sodapdf-converted.pdf`) extracted with `pdfplumber`.
2. Regex-based parsing normalized monetary units (lakhs/crores) and flattened multi-line descriptions.
3. Cleaned data saved to `credit_card_data_cleaned.csv`, then curated columns exported to `credit_card_data_final.csv`.
4. Embeddings computed once and stored under `model/` so local inference stays fast.

---

## Recommendation Logic Recap

1. Build user vector  
   `user_text = narrative + preferred features + scenario summary`  
   `user_numeric = [joining_fee, annual_fee, eligibility, reward_rate, interest_rate]`
2. Encode text, scale numerics, and concatenate.
3. Replace NaNs, compare against stored card vectors via cosine similarity.
4. Sort scores, select top 5, compute projected rewards & net annual value.
5. Generate Zephyr insights and render the UI (metrics, chart, comparison board).

---

## Deployment Notes

- Works locally, on Streamlit Community Cloud, or any VM. Just set `HF_TOKEN` as an environment variable.
- If you rehost elsewhere, copy the `model/` directory or regenerate embeddings.

---

## Future Ideas

- Multi-user profiles and saved histories
- Rewards projections that factor welcome bonuses or fuel surcharge waivers
- Automated dataset refresh pipeline + GitHub Actions
- Integrations with Plaid/SaltEdge for live transaction pulls

---

## Contributing

Contributions are welcome! Please open issues or pull requests for improvements.

---

## License

This project is licensed under the MIT License.

---

## Contact

Open an issue or pull request on [GitHub](https://github.com/Karthikprasadm/Cardly) if you’d like to collaborate, report a bug, or request a feature.

---

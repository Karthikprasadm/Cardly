# How to Run & Use Cardly

This guide walks through everything you need to run Cardly (the Smart Credit Card Advisor) locally and get the most out of the UI.

---

## 1. Prerequisites

- **Python** 3.10 (the project was tested on 3.10.11)
- **pip** and **git**
- **Hugging Face account** with an access token (read or inference scope)
- Recommended: **virtual environment** tooling (`venv`, `conda`, etc.)
- Optional: **Bank statements** in CSV or OFX format (OFX support is powered by the bundled `ofxparse` dependency)

---

## 2. Clone the Repository

```bash
git clone https://github.com/Karthikprasadm/Cardly.git
cd Cardly
```

> Tip: the repo includes a pre-trained embedder (~87 MB). Let the download finish; it only happens once.

---

## 3. Create & Activate a Virtual Environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

When your shell prompt is prefixed with `(.venv)` you’re ready to continue.

---

## 4. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This pulls in Streamlit, scikit-learn, sentence-transformers, Hugging Face client libraries, pdfplumber, and the rest of the stack.

---

## 5. Configure Secrets (.env)

1. Create a Hugging Face access token (https://huggingface.co/settings/tokens). Fine-grained tokens with at least **read** scope work best.
2. Create a `.env` file in the project root (same folder as `app.py`):
   ```
   HF_TOKEN=hf_xxx_your_secret_token
   ```
3. Confirm `.env` is **ignored by git** (already handled in `.gitignore`). Never commit secrets to the repository.

---

## 6. Run the App

```bash
streamlit run app.py
```

- Streamlit will display the local URL (default `http://localhost:8501`).  
- The first run may take longer because sentence-transformer models and tokenizer weights are loaded.

---

## 7. Using the UI

1. **Authentication (sidebar)**  
   - Immediate status of `HF_TOKEN`. Red warning = fix `.env` or restart.

2. **Upload Spend Data** *(optional but powerful)*  
   - Accepts CSV or OFX. Cardly looks for three fields:
     | Needed data | Sample headers |
     | --- | --- |
     | Transaction date | `Date`, `Txn Date`, `Post Date`, `Value Date` |
     | Narrative / merchant | `Description`, `Narration`, `Merchant`, `Details` |
     | Amount | `Amount`, `Debit`, `Txn Amount`, `Value` |
   - Expenses are auto-categorized (dining, groceries, online shopping, travel, fuel, other).  
   - A 3–6 month window averages the spend, and seasonal spikes are noted.  
   - If nothing is uploaded, the original sliders remain the source of truth (“manual sliders” label appears).

3. **Preferences & sliders**  
   - Adjust monthly income, joining/annual fee ceiling, expected reward rate, interest tolerance, and preferred features.  
   - Sliders mirror the upload-derived values but can be manually tweaked at any time.

4. **Scenario planner**  
   - Toggle “simulate travel surge” to stress-test the ranking.  
   - Choose a multiplier (1–3×) for travel spend—the scenario summary updates instantly and feeds into the embedding.

5. **Describe your needs**  
   - Use plain language (“Need cashback on groceries, minimal fee, occasional lounge access”).  
   - This narrative becomes part of the hybrid embedding.

6. **Run the recommender**  
   - Click **Find My Best Cards**. Streamlit shows a spinner while embeddings and cosine similarity execute.

7. **Review the recommendations**  
   - Each expander includes issuer, fees, rates, rewards, key features, estimated annual rewards, and **net annual value**.  
   - Zephyr-7B generates a succinct “Why this card?” explanation referencing your spend profile + scenario.  
   - Pin favorite cards with the 📌 button (max three); unpin from the same place.

8. **Analyze**  
   - A bar chart ranks cards by net annual value (rewards − fees) for the current scenario.  
   - The comparison board shows pinned cards in a table—perfect for screenshots or stakeholder updates.

9. **Iterate**  
   - Upload another statement, change the lookback window, or tweak the travel multiplier to answer “what-if” questions on the fly.

---

## 8. Troubleshooting

| Symptom | Fix |
| --- | --- |
| `ModuleNotFoundError: sentence_transformers.model_card` | Ensure you are using the bundled virtual environment or reinstall `sentence-transformers==2.2.2`. The app also includes backward-compatible shims, so reinstalling dependencies (`pip install -r requirements.txt`) usually resolves it. |
| `ValueError: Model ... not supported for task text-generation` | The app already configures the Zephyr endpoint with `task="conversational"`. If you change models/tasks, verify they support text generation in the Hugging Face docs. |
| HTTP 401 from Hugging Face | Token missing or invalid. Regenerate on the Hugging Face dashboard and update `.env`. Restart the Streamlit session after changes. |
| Warning about `credit_card_embedder.joblib` size when pushing to GitHub | The artifact is ~87 MB. Use Git LFS or host the model elsewhere before publishing the repo. |

If you hit an issue not covered above, capture the terminal stack trace and open an issue in the repo.

---

## 9. Next Steps

- **Add new cards** – edit `model/credit_card_data_final.csv`, rerun embedding generation, and redeploy.  
- **Adjust messaging** – tweak `generate_insight()` to change tone, add CTAs, or experiment with longer LLM responses.  
- **Deploy** – Streamlit Cloud, Hugging Face Spaces, or any VM works; just set `HF_TOKEN` as an environment variable.  
- **Automate** – consider GitHub Actions for linting/tests and for refreshing embeddings when data changes.

Enjoy building with Cardly! 🎉


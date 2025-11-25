# How to Run & Use Cardly

This guide walks through everything you need to run the Smart Credit Card Advisor locally and get the most out of the UI.

---

## 1. Prerequisites

- **Python** 3.10 (the project was tested on 3.10.11)
- **pip** and **git**
- **Hugging Face account** with an access token (read or inference scope)
- Recommended: **virtual environment** tooling (`venv`, `conda`, etc.)

---

## 2. Clone the Repository

```bash
git clone https://github.com/Karthikprasadm/Cardly.git
cd Cardly
```

> Tip: The repo contains a pre-trained embedder (~87 MB). Make sure you have a stable connection for the initial clone.

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

1. Create a Hugging Face access token (https://huggingface.co/settings/tokens). “Fine grained” tokens with at least **read** scope work best.
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

1. **Authentication panel (sidebar)**  
   - Shows whether your Hugging Face token was detected.  
   - If you see a warning, double-check the `.env` file or restart the session.

2. **Preferences & Spend Inputs**  
   - Monthly income, spend sliders (dining, groceries, shopping, travel, fuel), preferred features, and financial constraints feed the hybrid embedding pipeline.

3. **User Query Box**  
   - Describe your goals in natural language (“I travel twice a year, want lounge access, avoid high annual fees”).

4. **Find My Best Cards**  
   - Click the CTA button; Cardly embeds your text + numeric profile, compares it with the stored embeddings, and ranks the top 5 cards.

5. **Results Section**  
   - Each expander shows issuer, annual fee, interest rate, reward descriptions, and key features.  
   - The app calls the Zephyr-7B LLM (via Hugging Face Inference) to craft personalized “Why this card?” insights.

---

## 8. Troubleshooting

| Symptom | Fix |
| --- | --- |
| `ModuleNotFoundError: sentence_transformers.model_card` | Ensure you are using the bundled virtual environment or reinstall `sentence-transformers==2.2.2`. The app also includes backward-compatible shims, so reinstalling dependencies (`pip install -r requirements.txt`) usually resolves it. |
| `ValueError: Model ... not supported for task text-generation` | The app already configures the Zephyr endpoint with `task="conversational"`. If you change models/tasks, verify they support text generation in the Hugging Face docs. |
| HTTP 401 from Hugging Face | Token missing or invalid. Regenerate on the Hugging Face dashboard and update `.env`. Restart the Streamlit session after changes. |
| Warning about `credit_card_embedder.joblib` size when pushing to GitHub | Track the file with Git LFS or host it externally before sharing the repo publicly. |

If you hit an issue not covered above, capture the terminal stack trace and open an issue in the repo.

---

## 9. Next Steps

- Customize the dataset (`model/credit_card_data_final.csv`) with new cards.  
- Fine-tune the prompts inside `generate_insight()` to change the voice or format of recommendations.  
- Deploy to Streamlit Cloud or Hugging Face Spaces by setting environment variables (`HF_TOKEN`) in the hosting platform.

Enjoy building with Cardly! 🎉


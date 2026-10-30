# Personalized Product Recommendation Engine

## Quick start (Windows / VS Code)

1. Extract this ZIP.
2. Open the extracted `personalized_product_recommendation_engine` folder in VS Code.
3. Open Terminal → New Terminal.
4. Run:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

If PowerShell blocks activation, run without activation:

```powershell
py -m pip install -r requirements.txt
py -m streamlit run app.py
```

Open the local URL shown in the terminal (normally http://localhost:8501).

## Features
- Product search and category filters
- Preference and budget controls
- Personalized recommendations
- TF-IDF and cosine similarity ranking
- Match score and explanation
- Built-in demo product catalog

## Note
This is a runnable demonstration using sample products, not a production store or a model trained on real user purchase history.

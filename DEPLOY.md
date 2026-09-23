# Deploying this demo to Streamlit Community Cloud

This folder is self-contained: it does not need the full `DSE_Data.csv`, and it
does not retrain anything at startup. It loads a model that was already trained
and saved (`model/lgbm_model.txt`), and serves a small bundled data slice
(`data/demo_history.csv`, ~500 KB) covering 15 securities.

## What's in here

```
streamlit_app/
├── app.py                    # the Streamlit app
├── requirements.txt          # app-specific dependencies
├── model/
│   ├── lgbm_model.txt         # trained LightGBM model (text format, ~210 KB)
│   ├── feature_columns.json   # the exact 48 feature names, in order
│   └── headline_metrics.json  # the real test-set metrics from the notebook
├── data/
│   └── demo_history.csv       # last 700 cleaned rows per demo security (~500 KB)
└── src/                       # vendored copy of the project's data_prep,
                                # features and modeling modules
```

The 15 securities were chosen from the notebook's real per-security results: the
5 with the lowest test-period error, the 5 with the highest (including
RECKITTBEN, the single biggest contributor to the model's overall shortfall),
and 5 of the most-traded names for recognisability.

## Steps

1. **Put this folder in its own GitHub repository** (or a subfolder of the main
   project repo — either works, since `requirements.txt` and `app.py` are both
   self-contained here).

   ```bash
   git init
   git add .
   git commit -m "DSE next-day close predictor — Streamlit demo"
   git branch -M main
   git remote add origin https://github.com/<your-username>/<repo-name>.git
   git push -u origin main
   ```

2. Go to **share.streamlit.io** and sign in with GitHub.

3. Click **"New app"**, pick the repository and branch, and set:
   - **Main file path**: `app.py` (or `streamlit_app/app.py` if you kept it as a
     subfolder of the main repo — either is fine, just be consistent)

4. Click **Deploy**. First build takes 1-3 minutes (installing LightGBM is the
   slow part). After that, the app is live at a public
   `https://<something>.streamlit.app` URL.

## Before you share the link

- **Update `REPO_URL`** near the top of `app.py` to point at your actual GitHub
  repo, so the "read the full write-up" link in the app works.
- The app states plainly, on load, that the model does not beat the naive
  baseline. That's intentional — keep it. It's the same honest framing as the
  README and notebook, and removing it would misrepresent the project.
- This is a historical demo, not a live market feed. The data stops in April
  2025. If anyone asks whether this predicts today's DSE prices: no.

## If you want more than 15 securities later

The app is built to make that a data change, not a code change: extend
`data/demo_history.csv` with more tickers (same columns, same "last ~700 rows"
shape) and the security list in the app updates itself. The model itself
already covers the full panel — it was trained on all ~450 securities, not
just these 15 — so no retraining is needed either.

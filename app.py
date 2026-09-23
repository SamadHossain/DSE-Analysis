"""
DSE Next-Day Close Predictor — Streamlit demo

This app serves the exact LightGBM model trained and evaluated in the project
notebook (see the GitHub repo's README and notebooks/dse_stock_analysis.ipynb),
restricted to a small set of securities bundled with the app so it can run
without the full ~75 MB dataset.

Nothing here is a live market feed. All prices are from the historical dataset
(Sunny, MD Abu Sayed, 2025, "Dhaka Stock Exchange Historical Data (1999-2025)",
Harvard Dataverse) and stop in April 2025. "Next trading day" means the day
after the last date in that file, not today.
"""

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import sys
sys.path.insert(0, str(Path(__file__).parent))
from src import data_prep, features, modeling  # noqa: E402

APP_DIR = Path(__file__).parent
MODEL_PATH = APP_DIR / "model" / "lgbm_model.txt"
FEATURE_COLS_PATH = APP_DIR / "model" / "feature_columns.json"
HEADLINE_METRICS_PATH = APP_DIR / "model" / "headline_metrics.json"
DATA_PATH = APP_DIR / "data" / "demo_history.csv"

REPO_URL = "https://github.com/<your-username>/dse-stock-price-prediction"

TICKER_NOTES = {
    "SQURPHARMA": "lowest test-period error in the full evaluation",
    "BATASHOE": "low test-period error",
    "GP": "low test-period error",
    "JAMUNAOIL": "low test-period error",
    "APEXFOOT": "low test-period error",
    "RECKITTBEN": "highest test-period error — the single biggest contributor "
                  "to the model's overall shortfall (see Key Findings in the README)",
    "PLFSL": "very high test-period error",
    "FAMILYTEX": "very high test-period error",
    "ICBIBANK": "very high test-period error",
    "TUNGHAI": "very high test-period error",
}


@st.cache_resource
def load_model():
    booster = lgb.Booster(model_file=str(MODEL_PATH))
    with open(FEATURE_COLS_PATH) as f:
        cols = json.load(f)
    return booster, cols


@st.cache_data
def load_headline_metrics():
    with open(HEADLINE_METRICS_PATH) as f:
        return json.load(f)


@st.cache_data
def load_demo_features():
    """Clean the bundled per-security slices and build both feature tables.

    ``backtest`` keeps only rows with a known next-day close (for the actual-vs-
    predicted chart). ``latest`` additionally keeps each security's final row,
    whose target is not yet known — that row is what a genuine forward
    prediction runs on.
    """
    raw = pd.read_csv(DATA_PATH, parse_dates=["Date"])
    clean, _ = data_prep.clean(raw, drop_non_equity=False)
    clean = data_prep.add_daily_return(clean)
    backtest = features.build_features(clean)
    latest = features.build_features(clean, keep_latest_incomplete=True)
    return backtest, latest


def predict(booster, cols, df):
    return booster.predict(df[cols])


def main():
    st.set_page_config(page_title="DSE Next-Day Close — Demo", layout="wide")

    st.title("Dhaka Stock Exchange — Next-Day Close Predictor (Demo)")
    st.caption(
        "A live demo of the model from a university data-science project. "
        f"Full write-up, notebook and code: {REPO_URL}"
    )

    booster, cols = load_model()
    headline = load_headline_metrics()
    backtest, latest = load_demo_features()

    with st.container(border=True):
        st.subheader("⚠️ Read this first")
        st.markdown(
            "Evaluated on the full test set (2024-01-01 to 2025-04-07, 104,848 rows "
            "across ~450 securities), **this model does not beat a naive "
            "\"tomorrow equals today\" baseline** — it does clearly worse on every "
            "metric, including the direction of movement:"
        )
        m1, m2, m3 = st.columns(3)
        naive = headline["Naive (previous close)"]
        model_m = headline["LightGBM"]
        m1.metric("RMSE", f"{model_m['RMSE']:.2f}", f"naive: {naive['RMSE']:.2f}",
                   delta_color="inverse")
        m2.metric("MAE", f"{model_m['MAE']:.2f}", f"naive: {naive['MAE']:.2f}",
                   delta_color="inverse")
        m3.metric("Directional accuracy",
                   f"{model_m['Directional accuracy (%)']:.1f}%",
                   f"naive: {naive['Directional accuracy (%)']:.1f}%",
                   delta_color="inverse")
        st.markdown(
            "This app exists to show the model honestly, including where it "
            "underperforms — not to claim it can be traded on. Full findings and "
            "the likely cause are in the project README."
        )

    st.divider()

    tickers = sorted(backtest["Trading_Code"].unique())
    labels = {t: (f"{t} — {TICKER_NOTES[t]}" if t in TICKER_NOTES else t) for t in tickers}
    ticker = st.selectbox("Choose a security", tickers, format_func=lambda t: labels[t])

    tb = backtest[backtest["Trading_Code"] == ticker].sort_values("Date").copy()
    tl = latest[latest["Trading_Code"] == ticker].sort_values("Date")

    if tb.empty:
        st.warning("Not enough history for this security in the bundled demo slice.")
        return

    tb["Predicted_Close"] = predict(booster, cols, tb)
    tb["Naive_Close"] = tb["Close"]  # today's close predicts tomorrow's close

    col_chart, col_stats = st.columns([3, 1])

    with col_chart:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=tb["Date"], y=tb[features.TARGET],
                                  name="Actual next-day close", mode="lines"))
        fig.add_trace(go.Scatter(x=tb["Date"], y=tb["Predicted_Close"],
                                  name="LightGBM prediction", mode="lines",
                                  line=dict(dash="dash")))
        fig.add_trace(go.Scatter(x=tb["Date"], y=tb["Naive_Close"],
                                  name="Naive baseline (today's close)", mode="lines",
                                  line=dict(dash="dot")))
        fig.update_layout(
            title=f"{ticker} — actual vs. predicted next-day close",
            xaxis_title="Date", yaxis_title="Close (BDT)",
            height=450, legend=dict(orientation="h", y=1.08),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_stats:
        st.markdown("**This security, backtest window**")
        res = {
            "Naive": modeling.regression_metrics(tb[features.TARGET], tb["Naive_Close"]),
            "LightGBM": modeling.regression_metrics(tb[features.TARGET], tb["Predicted_Close"]),
        }
        comp = modeling.compare_models(res)[["RMSE", "MAE", "MAPE (%)"]]
        st.dataframe(comp, use_container_width=True)
        st.caption(f"{len(tb)} days in this security's bundled demo window.")

    st.divider()

    st.subheader("Next trading day (forward prediction, not a backtest)")
    if tl.empty:
        st.info("No row with complete inputs available for a forward prediction.")
    else:
        last_row = tl.iloc[[-1]]
        next_pred = predict(booster, cols, last_row)[0]
        last_date = last_row["Date"].iloc[0]
        last_close = last_row["Close"].iloc[0]
        c1, c2, c3 = st.columns(3)
        c1.metric("Last known date", f"{last_date:%Y-%m-%d}")
        c2.metric("Last known close", f"{last_close:.2f} BDT")
        c3.metric("Predicted next-day close", f"{next_pred:.2f} BDT",
                   f"{next_pred - last_close:+.2f} vs naive")
        st.caption(
            "\"Next trading day\" means the day after the dataset's last recorded "
            "date for this security, not a real-time forecast for today."
        )

    st.divider()
    st.caption(
        "Model: LightGBM, 74 trees, trained on 1,114,141 rows (1999-2022), validated "
        "on 2023, tested on 2024-2025. Data: Sunny, MD Abu Sayed (2025), \"Dhaka Stock "
        "Exchange Historical Data (1999-2025)\", Harvard Dataverse, "
        "doi.org/10.7910/DVN/XIFYT1. This is a coursework project; nothing here is "
        "financial advice."
    )


if __name__ == "__main__":
    main()

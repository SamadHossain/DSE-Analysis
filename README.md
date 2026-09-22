# Stock Market Behavior Analysis and Price Movement Prediction on Dhaka Stock Exchange: A Machine Learning Approach

A supervised regression study of daily trading data from the Dhaka Stock Exchange (DSE),
predicting the next trading day's closing price of a security from information available
at the close of the current day.

Coursework project, Post Graduate Diploma in Data Science, United International
University (UIU), Bangladesh.

---

## Overview

The DSE is a frontier market with characteristics that make it a useful subject for
this kind of study: thinner liquidity than developed exchanges, a wide dispersion of
price levels across listed securities, and long price histories for the
longest-listed names. Publicly analysed machine-learning work on it is comparatively
sparse.

The project treats next-day closing price as a **supervised regression** problem. For
each security and each trading day, a feature vector is built from that day's and
earlier days' open, high, low, close and volume; the target is the following trading
day's close. A single LightGBM regressor is fitted across the pooled panel of all
securities.

### A caution about what the metrics mean

Consecutive daily closing prices are very highly autocorrelated. Any model predicting
a price *level* will report a high R² even if it has learned nothing beyond "tomorrow
resembles today". The evaluation therefore compares the model against a **naive
previous-close baseline** and reports **directional accuracy** alongside the standard
error metrics. Those comparisons, not the headline R², are what indicate whether the
model carries information. This is a deliberate revision to the original analysis,
which reported R² alone.

## Objectives

1. Assess and document the quality of the raw DSE daily price data, and clean it to a
   state suitable for modelling.
2. Explore price and volume behaviour, distributions, and anomalies across the panel.
3. Engineer lag, trend, volatility, liquidity and calendar features within each security.
4. Fit a model predicting the next trading day's closing price, using a strictly
   chronological train/validation/test split.
5. Evaluate it against a naive baseline and report where it does and does not improve
   on one.

## Dataset

| | |
| --- | --- |
| File | `DSE_Data.csv` (daily OHLCV, one row per security per trading day) |
| Source | Sunny, MD Abu Sayed (2025). *Dhaka Stock Exchange Historical Data (1999–2025).* Harvard Dataverse. [doi.org/10.7910/DVN/XIFYT1](https://doi.org/10.7910/DVN/XIFYT1) |
| Raw records (as loaded in this project) | 1,523,921 |
| Raw records (per dataset description) | 1,684,249 |
| Date range | 1999-01-02 to 2025-04-08 |
| Columns | `Trading_Code`, `Date`, `Open`, `High`, `Low`, `Close`, `Volume` |
| Securities | 700+ listed companies per the dataset description; ~500 distinct trading codes seen in the file used here, including indices, bonds and mutual funds before filtering |

Per the Harvard Dataverse listing, the data was collected primarily from the official
DSE website and supplemented with other publicly available sources, and is intended
for informational and research purposes only; the publisher notes that inconsistencies
or errors may exist and recommends independently verifying any critical figures before
use.

**Row-count discrepancy.** The dataset description states 1,684,249 rows, while the
file loaded in this project (`df.info()`) reports 1,523,921. The difference (about
160,000 rows, roughly 9%) was not investigated as part of this project. Possible
explanations include a newer snapshot on Harvard Dataverse than the copy the group
downloaded, or an export/versioning difference — this has not been confirmed either
way. Anyone re-running this project from a freshly downloaded copy of the dataset
should expect the exact row counts throughout this README (and the cleaning log in the
notebook) to shift accordingly.

The raw file is **not committed to this repository**, both because of its size (over
1.5 million rows) and because the Dataverse listing does not state a redistribution
licence. To run the project, download the CSV from the DOI above and place it at
`data/raw/DSE_Data.csv`.

### Cleaning

Applied in order, with the row count logged at each step:

1. rows with any missing value (2 rows had a missing `Volume` in the original run)
2. exact duplicate rows (63,914 in the original run)
3. repeated `(Trading_Code, Date)` pairs
4. rows where any OHLCV field is zero (25,554 in the original run)
5. indices (DSEX, DS30, DSES), corporate and perpetual bonds, and closed-end mutual
   funds — 62 codes in total, whose price behaviour differs from ordinary shares

In the original run, steps 1–4 and 5 together reduced the panel to 1,307,591 rows.
Step 3 is new, so the cleaned counts from a fresh run may differ slightly.

### Features

All features are computed **within** each security, from information available at or
before the close of day *t*. The target is the close of day *t+1*.

| Group | Features |
| --- | --- |
| Lags | Open, High, Low, Close, Volume at lags 1, 2, 3, 5, 10 |
| Trend | 5/10/20-day simple and exponential moving averages of Close |
| Volatility | rolling standard deviation of daily return over 5/10/20 days |
| Liquidity | volume relative to its own 20-day mean |
| Calendar | day of week, month, day of year, week of year, day of month |
| Intraday | High − Low, Open − Close |

Rolling volatility and the volume ratio are additions to the original feature set.
`Year` is deliberately excluded from the model inputs: under a chronological split the
test years never appear in training.

## Methodology

```text
Data loading
      ↓
Data quality assessment (before)
      ↓
Cleaning  →  quality assessment (after)
      ↓
Daily returns
      ↓
Exploratory data analysis
      ↓
Feature engineering  (within-security lags, trend, volatility, calendar)
      ↓
Chronological train / validation / test split
      ↓
LightGBM training  (early stopping on validation)
      ↓
Evaluation against a naive previous-close baseline
      ↓
One-step-ahead prediction
```

## Model

**LightGBM regressor** — the only algorithm used in this project. It handles a large
tabular panel efficiently, requires no feature scaling, and copes with the strongly
skewed price and volume distributions without transformation. The number of trees is
selected by early stopping on a validation block that sits chronologically between the
training and test periods.

**Naive previous-close baseline** — predicts tomorrow's close as today's close. Not a
fitted model; it is the reference the regressor has to beat.

No other algorithm is implemented here. If your course proposal also committed to
classification models or clustering, that work is not part of this notebook and is not
claimed anywhere in this repository.

## Evaluation

Metrics reported: **RMSE**, **MAE**, **MAPE**, **R²**, and **directional accuracy**
(the share of days where the predicted direction of movement was correct).

MAPE is included because pooled RMSE is dominated by a handful of high-priced
securities — an error of 30 BDT means something very different on a 2,000 BDT share
than on a 15 BDT one. That gap was visible in the original run, where RMSE was 36.60
against an MAE of 2.68.

### Results from the original notebook

These are the figures the original Colab run reported, preserved for reference:

| Metric | Value |
| --- | --- |
| MSE | 1339.3508 |
| RMSE | 36.5971 |
| MAE | 2.6777 |
| R² | 0.9886 |

Train: 1,193,615 rows (1999-02-06 to 2023-12-28). Test: 104,858 rows (2024-01-01 to
2025-04-07). No baseline, validation set or directional metric was computed.

### Results from the revised pipeline

**Not yet run.** The revised pipeline changes the split (a validation block is carved
out for early stopping), corrects the daily-return calculation, adds features and
excludes `Year`, so its numbers will not match the table above and must be produced by
actually running the notebook against the dataset. Fill in the table below afterwards,
and do not copy the original figures into it.

| Model | RMSE | MAE | MAPE (%) | R² | Directional accuracy (%) |
| --- | --- | --- | --- | --- | --- |
| Naive (previous close) | | | | | |
| LightGBM | | | | | |

## Key findings

To be completed once the revised notebook has been run end to end. The findings should
address how LightGBM compares with the naive baseline, whether directional accuracy is
meaningfully above chance, and how widely per-security error varies — including the
cases where the model does worse.

What can be stated from the original run: the model tracked the test-period price
series closely for the sampled securities, and the large gap between RMSE and MAE
indicates that errors are concentrated in a small number of high-priced or volatile
securities rather than spread evenly across the panel.

## Limitations

- **Source data quality is not independently verified.** The publisher's own listing
  states the data may contain inconsistencies and recommends verifying critical
  figures independently; this project has not done so beyond the cleaning steps in
  section 4. The unresolved row-count discrepancy noted above is one open question.
- **Prices are not adjusted** for dividends, bonus issues, rights issues or splits.
  Corporate actions appear as large one-day jumps and are indistinguishable from real
  moves in this dataset; some extreme observed returns are almost certainly this.
- **Price-level prediction flatters the metrics.** R² on a highly autocorrelated series
  is close to uninformative on its own.
- **Uneven coverage and survivorship.** Securities delisted mid-period, and those with
  very short histories, are treated identically to continuously listed ones.
- **A single pooled model** treats a 15 BDT share and a 2,000 BDT share as draws from
  the same process.
- **No transaction costs, liquidity constraints or microstructure** are modelled.
  Nothing here supports a claim about tradeable profit.
- **One algorithm only** — no comparison against linear models, ARIMA, or other
  gradient-boosting implementations.
- Short-horizon price prediction is inherently difficult; prices respond to information
  well outside daily OHLCV data.

## Future work

Not implemented in this project:

- compare against linear, ARIMA and other gradient-boosting baselines
- reframe the task as directional classification, where the metrics are harder to
  inflate than price-level regression
- adjust prices for corporate actions, if an adjusted series can be sourced (the
  Harvard Dataverse dataset used here does not appear to be adjusted)
- multi-horizon forecasting with recursive prediction and explicit uncertainty
- rolling-origin (walk-forward) cross-validation instead of a single fixed split
- per-sector or per-liquidity-tier models rather than one pooled model
- hyperparameter tuning, which was not performed here

## Project structure

```text
dse-stock-price-prediction/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/              # place DSE_Data.csv here (not committed)
│   └── processed/        # cleaned output (not committed)
├── notebooks/
│   └── dse_stock_analysis.ipynb
├── src/
│   ├── config.py         # paths, constants, excluded non-equity codes
│   ├── data_prep.py      # loading, quality report, cleaning, returns
│   ├── features.py       # feature construction
│   ├── modeling.py       # split, training, baseline, metrics
│   └── plots.py          # styling and shared figures
├── figures/              # figures written by the notebook
└── docs/
    └── CHANGES.md        # what changed from the original notebook, and why
```

The analysis lives in one notebook, with the reusable logic factored into `src/`. The
notebook stays readable as a narrative, while the parts worth testing or reusing —
cleaning rules, feature construction, metrics — sit in plain Python modules. The
64-code exclusion list and the sort-before-groupby rules in particular are easier to
review as code than buried in a cell.

## Installation

```bash
git clone https://github.com/<your-username>/dse-stock-price-prediction.git
cd dse-stock-price-prediction
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Running the project

1. Place `DSE_Data.csv` at `data/raw/DSE_Data.csv`.
2. Launch Jupyter and open the notebook:

```bash
jupyter notebook notebooks/dse_stock_analysis.ipynb
```

3. Run all cells in order. Figures are written to `figures/`; the cleaned dataset can
   be written to `data/processed/`.

Training on the full panel is memory-hungry — expect to need several GB of RAM.

## Technologies used

Python, pandas, NumPy, Matplotlib, seaborn, scikit-learn, LightGBM, Jupyter Notebook.

## Authors

- Mohammad Samad Hossain — 905253013
- Mohammad Ali Rubel — 905253032
- Md. Mahmud Ullah — 905253018

## Academic context

**Post Graduate Diploma in Data Science**

**United International University (UIU), Bangladesh**

**Course Instructor: Ahmed Imran Kabir**

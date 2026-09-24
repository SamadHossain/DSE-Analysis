"""Splitting, model training, baselines and evaluation."""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .config import RANDOM_STATE, TRAIN_END, VALID_END


def chronological_split(df: pd.DataFrame, train_end=TRAIN_END, valid_end=VALID_END):
    """Split by date into train / validation / test.

    A random split would let the model see future observations of the same
    security while predicting its past, so the split is strictly by calendar
    date. The validation block sits between train and test and is used only
    for early stopping, which keeps the test period untouched during fitting.
    """
    train_end, valid_end = pd.Timestamp(train_end), pd.Timestamp(valid_end)
    train = df[df["Date"] <= train_end]
    valid = df[(df["Date"] > train_end) & (df["Date"] <= valid_end)]
    test = df[df["Date"] > valid_end]
    return train, valid, test


def train_lightgbm(X_train, y_train, X_valid, y_valid, **overrides):
    """Fit a LightGBM regressor, stopping when validation error plateaus."""
    params = dict(
        objective="regression",
        num_leaves=31,
        learning_rate=0.05,
        n_estimators=2000,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    params.update(overrides)

    model = lgb.LGBMRegressor(**params)
    common = dict(
        eval_metric="rmse",
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )

    # LightGBM 4.7 renamed the validation arguments; fall back for older versions.
    try:
        model.fit(X_train, y_train, eval_X=X_valid, eval_y=y_valid, **common)
    except TypeError:
        model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], **common)
    return model


def regression_metrics(y_true, y_pred) -> dict:
    """RMSE, MAE, MAPE, R^2 and directional accuracy is handled separately.

    MAPE is included because the pooled RMSE is dominated by a handful of
    high-priced securities: an error of 30 taka means something very
    different on a 2,000 taka share than on a 15 taka one.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MAPE (%)": float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100),
        "R2": float(r2_score(y_true, y_pred)),
    }


def directional_accuracy(last_close, y_true, y_pred) -> float:
    """Share of days where the predicted direction of change is correct.

    Ties (a flat actual move) are counted as incorrect, which is the
    conservative choice on an exchange with frequent unchanged closes.
    """
    actual_up = np.asarray(y_true) > np.asarray(last_close)
    predicted_up = np.asarray(y_pred) > np.asarray(last_close)
    return float((actual_up == predicted_up).mean() * 100)


def naive_baseline(last_close) -> np.ndarray:
    """Predict tomorrow's close as today's close.

    This is the reference every price-level model has to beat. Because
    consecutive closes are nearly identical, it scores extremely well on
    RMSE and R^2 without containing any information at all. Reporting the
    model's R^2 without it next to it would overstate what the model learned.
    """
    return np.asarray(last_close, dtype=float)


def compare_models(results: dict[str, dict]) -> pd.DataFrame:
    """Turn {name: metrics} into a comparison table."""
    return pd.DataFrame(results).T.round(4)

"""
forecast/evaluator.py
─────────────────────
Forecast accuracy evaluation.
Computes MAPE, RMSE, MAE, Bias per SKU × Warehouse.
Uses walk-forward (time-series cross-validation) on the last N days.

Public API
──────────
  evaluate_forecast(sales_df, test_days, model) → accuracy_df
  compute_summary_accuracy(accuracy_df)         → dict
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from forecast.models import run_forecast, ModelType


# ════════════════════════════════════════════════════════════════════════════
# METRIC HELPERS
# ════════════════════════════════════════════════════════════════════════════

def _mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean Absolute Percentage Error — skips zero-actual rows."""
    mask = actual > 0
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)


def _rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def _mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - predicted)))


def _bias(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean forecast bias — positive = over-forecast, negative = under-forecast."""
    return float(np.mean(predicted - actual))


def _fa(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Forecast Accuracy = 1 - MAPE/100, clipped to [0, 1]."""
    m = _mape(actual, predicted)
    return float(np.clip(1 - m / 100, 0, 1)) if not np.isnan(m) else np.nan


# ════════════════════════════════════════════════════════════════════════════
# WALK-FORWARD EVALUATION
# ════════════════════════════════════════════════════════════════════════════

def evaluate_forecast(
    sales_df: pd.DataFrame,
    test_days: int = 30,
    model: ModelType = "xgboost",
) -> pd.DataFrame:
    """
    Walk-forward evaluation: train on [start → cutoff], test on [cutoff+1 → end].

    Parameters
    ----------
    sales_df   : Full sales history DataFrame
    test_days  : How many tail days to hold out as test set
    model      : Forecasting model to evaluate

    Returns
    -------
    DataFrame with per-SKU × Warehouse accuracy metrics:
      sku | warehouse | mape | rmse | mae | bias | forecast_accuracy | n_test_days
    """
    sales_df = sales_df.copy()
    sales_df["date"] = pd.to_datetime(sales_df["date"])

    cutoff_date = sales_df["date"].max() - pd.Timedelta(days=test_days)

    train_df = sales_df[sales_df["date"] <= cutoff_date]
    test_df  = sales_df[sales_df["date"] >  cutoff_date]

    if train_df.empty or test_df.empty:
        raise ValueError("Not enough data for train/test split.")

    print(f"\n  📊  Evaluating [{model.upper()}]")
    print(f"      Train: up to {cutoff_date.date()} | Test: {test_days} days")

    # Run forecast on training data
    forecast_df = run_forecast(
        sales_df=train_df,
        horizon_days=test_days,
        model=model,
    )

    forecast_df["date"] = pd.to_datetime(forecast_df["date"])

    # Aggregate actuals to daily per SKU × WH
    actual_daily = (
        test_df.groupby(["sku", "warehouse", "date"])["demand_qty"]
        .sum()
        .reset_index()
        .rename(columns={"demand_qty": "actual_qty"})
    )

    # Merge forecast vs actual
    merged = forecast_df.merge(actual_daily, on=["sku", "warehouse", "date"], how="inner")

    if merged.empty:
        print("  ⚠️  No overlapping dates between forecast and actuals.")
        return pd.DataFrame()

    rows = []
    for (sku, wh), grp in merged.groupby(["sku", "warehouse"]):
        actual    = grp["actual_qty"].values.astype(float)
        predicted = grp["forecast_qty"].values.astype(float)
        n         = len(grp)

        rows.append({
            "sku":               sku,
            "warehouse":         wh,
            "mape":              round(_mape(actual, predicted), 2),
            "rmse":              round(_rmse(actual, predicted), 2),
            "mae":               round(_mae(actual, predicted),  2),
            "bias":              round(_bias(actual, predicted),  2),
            "forecast_accuracy": round(_fa(actual, predicted),   4),
            "n_test_days":       n,
            "model":             model,
        })

    return pd.DataFrame(rows)


# ════════════════════════════════════════════════════════════════════════════
# SUMMARY ACCURACY KPIs
# ════════════════════════════════════════════════════════════════════════════

def compute_summary_accuracy(accuracy_df: pd.DataFrame) -> dict:
    """
    Aggregate accuracy metrics into system-level KPIs.

    Returns
    -------
    dict with keys:
      mean_mape | median_mape | mean_forecast_accuracy
      pct_below_20_mape | pct_below_30_mape | mean_bias
    """
    if accuracy_df.empty:
        return {}

    df = accuracy_df.dropna(subset=["mape", "forecast_accuracy"])

    return {
        "mean_mape":             round(df["mape"].mean(), 2),
        "median_mape":           round(df["mape"].median(), 2),
        "mean_forecast_accuracy":round(df["forecast_accuracy"].mean() * 100, 2),
        "pct_below_20_mape":     round((df["mape"] < 20).mean() * 100, 1),
        "pct_below_30_mape":     round((df["mape"] < 30).mean() * 100, 1),
        "mean_bias":             round(df["bias"].mean(), 2),
        "n_combinations":        len(df),
    }
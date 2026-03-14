"""
forecast/models.py
──────────────────
Demand Forecasting Layer
Supports three model families per SKU × Warehouse:
  1. Moving Average (baseline)
  2. Exponential Smoothing (trend + seasonality via statsmodels)
  3. XGBoost with lag / calendar features (ML)

Public API
──────────
  run_forecast(sales_df, horizon_days, model, promo_df) → forecast_df
"""
from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from typing import Literal

warnings.filterwarnings("ignore")

# ── optional heavy deps ──────────────────────────────────────────────────────
try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    _HAS_STATSMODELS = True
except ImportError:
    _HAS_STATSMODELS = False

try:
    from xgboost import XGBRegressor as _XGBRegressor
    _HAS_XGB = True
except ImportError:
    # Fallback: use sklearn GradientBoostingRegressor (same API surface)
    from sklearn.ensemble import GradientBoostingRegressor as _XGBRegressor  # type: ignore
    _HAS_XGB = True  # always available via sklearn

ModelType = Literal["moving_average", "exponential_smoothing", "xgboost"]

# ════════════════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════════════════

def _make_future_dates(last_date: pd.Timestamp, horizon: int) -> pd.DatetimeIndex:
    return pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon, freq="D")


def _add_promo_uplift(
    forecast: pd.Series,
    future_dates: pd.DatetimeIndex,
    sku: str,
    warehouse: str,
    promo_df: pd.DataFrame | None,
) -> pd.Series:
    """Multiply forecast by promo uplift on promotion days."""
    if promo_df is None or promo_df.empty:
        return forecast

    mask = (
        (promo_df["sku"] == sku) &
        (promo_df["warehouse"] == warehouse)
    )
    relevant = promo_df[mask]

    uplift = np.ones(len(future_dates))
    for _, row in relevant.iterrows():
        promo_mask = (future_dates >= row["promo_start"]) & (future_dates <= row["promo_end"])
        uplift[promo_mask] *= (1 + row["uplift_pct"] / 100)

    return forecast * uplift


def _build_daily_series(group_df: pd.DataFrame) -> pd.Series:
    """Aggregate demand_qty to a daily time-series with no gaps."""
    ts = (
        group_df.groupby("date")["demand_qty"]
        .sum()
        .asfreq("D")
        .fillna(0)
    )
    return ts


# ════════════════════════════════════════════════════════════════════════════
# MODEL 1 – MOVING AVERAGE
# ════════════════════════════════════════════════════════════════════════════

def _forecast_moving_average(
    ts: pd.Series,
    horizon: int,
    window: int = 28,
) -> np.ndarray:
    """
    Simple rolling mean forecast.
    Uses last `window` days as the flat forward projection.
    """
    avg = ts.tail(window).mean()
    avg = max(avg, 0.0)
    return np.full(horizon, avg)


# ════════════════════════════════════════════════════════════════════════════
# MODEL 2 – EXPONENTIAL SMOOTHING (Holt-Winters)
# ════════════════════════════════════════════════════════════════════════════

def _forecast_exp_smoothing(
    ts: pd.Series,
    horizon: int,
    seasonal_periods: int = 7,
) -> np.ndarray:
    """
    Holt-Winters additive model with weekly seasonality.
    Falls back to moving average when statsmodels is absent
    or the series is too short.
    """
    if not _HAS_STATSMODELS or len(ts) < seasonal_periods * 2:
        return _forecast_moving_average(ts, horizon)

    try:
        model = ExponentialSmoothing(
            ts,
            trend="add",
            seasonal="add",
            seasonal_periods=seasonal_periods,
            initialization_method="estimated",
        ).fit(optimized=True, use_brute=False)
        preds = model.forecast(horizon)
        return np.clip(preds.values, 0, None)
    except Exception:
        return _forecast_moving_average(ts, horizon)


# ════════════════════════════════════════════════════════════════════════════
# MODEL 3 – XGBOOST
# ════════════════════════════════════════════════════════════════════════════

def _make_features(ts: pd.Series, lags: list[int], horizon_offset: int = 0) -> pd.DataFrame:
    """
    Build a feature matrix from a time series.
    Features: lag values + calendar signals.
    """
    df = ts.to_frame(name="y").copy()
    df["ds"] = df.index

    df["dayofweek"] = df["ds"].dt.dayofweek
    df["dayofmonth"] = df["ds"].dt.day
    df["month"]      = df["ds"].dt.month
    df["weekofyear"] = df["ds"].dt.isocalendar().week.astype(int)

    for lag in lags:
        df[f"lag_{lag}"] = df["y"].shift(lag)

    df["rolling_7"]  = df["y"].shift(1).rolling(7).mean()
    df["rolling_28"] = df["y"].shift(1).rolling(28).mean()

    return df.dropna()


def _forecast_xgboost(
    ts: pd.Series,
    horizon: int,
    lags: list[int] | None = None,
) -> np.ndarray:
    """
    XGBoost regressor with lag + calendar features.
    Recursive multi-step forecasting.
    """
    if not _HAS_XGB or len(ts) < 60:
        return _forecast_exp_smoothing(ts, horizon)

    if lags is None:
        lags = [1, 7, 14, 28]

    feat_df = _make_features(ts, lags)
    feature_cols = [c for c in feat_df.columns if c not in ("y", "ds")]

    X = feat_df[feature_cols].values
    y = feat_df["y"].values

    model = _XGBRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42,
    )
    model.fit(X, y)

    # Recursive forecasting
    history = ts.copy()
    preds   = []

    for _ in range(horizon):
        window = history.tail(max(lags) + 28 + 5)
        feat   = _make_features(window, lags)
        if feat.empty:
            preds.append(history.mean())
            history = pd.concat([history, pd.Series([history.mean()], index=[history.index[-1] + pd.Timedelta(days=1)])])
            continue

        last_feat = feat[feature_cols].iloc[[-1]].values
        pred = float(model.predict(last_feat)[0])
        pred = max(pred, 0.0)
        preds.append(pred)

        next_date = history.index[-1] + pd.Timedelta(days=1)
        history = pd.concat([history, pd.Series([pred], index=[next_date])])

    return np.array(preds)


# ════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ════════════════════════════════════════════════════════════════════════════

def run_forecast(
    sales_df: pd.DataFrame,
    horizon_days: int = 90,
    model: ModelType = "xgboost",
    promo_df: pd.DataFrame | None = None,
    min_history_days: int = 30,
) -> pd.DataFrame:
    """
    Run demand forecast for every active SKU × Warehouse combination.

    Parameters
    ----------
    sales_df        : DataFrame with columns [date, sku, warehouse, demand_qty]
    horizon_days    : Number of days to forecast forward
    model           : 'moving_average' | 'exponential_smoothing' | 'xgboost'
    promo_df        : Optional promotion calendar DataFrame
    min_history_days: Skip combinations with fewer history days

    Returns
    -------
    DataFrame with columns:
      sku | warehouse | date | forecast_qty | model | lower_bound | upper_bound
    """
    sales_df = sales_df.copy()
    sales_df["date"] = pd.to_datetime(sales_df["date"])

    results = []
    groups  = sales_df.groupby(["sku", "warehouse"])
    total   = len(groups)

    print(f"  🔮  Running [{model.upper()}] forecast — {total} SKU×WH combinations, horizon={horizon_days}d")

    for i, ((sku, warehouse), grp) in enumerate(groups, 1):
        if i % 50 == 0 or i == total:
            print(f"      {i}/{total} …")

        grp = grp.sort_values("date")

        if len(grp) < min_history_days:
            continue

        ts           = _build_daily_series(grp)
        last_date    = ts.index[-1]
        future_dates = _make_future_dates(last_date, horizon_days)

        # ── Dispatch to model ──────────────────────────────────────────────
        if model == "moving_average":
            raw_preds = _forecast_moving_average(ts, horizon_days)
        elif model == "exponential_smoothing":
            raw_preds = _forecast_exp_smoothing(ts, horizon_days)
        else:
            raw_preds = _forecast_xgboost(ts, horizon_days)

        # ── Promo uplift ───────────────────────────────────────────────────
        preds_series = pd.Series(raw_preds, index=future_dates)
        preds_series = _add_promo_uplift(preds_series, future_dates, sku, warehouse, promo_df)

        # ── Uncertainty bounds (±10% MA, ±15% ES, ±20% XGB as proxy) ─────
        uncertainty = {"moving_average": 0.10, "exponential_smoothing": 0.15, "xgboost": 0.20}
        u = uncertainty[model]

        for date, qty in zip(future_dates, preds_series.values):
            results.append({
                "sku":          sku,
                "warehouse":    warehouse,
                "date":         date,
                "forecast_qty": round(float(qty), 1),
                "lower_bound":  round(float(qty) * (1 - u), 1),
                "upper_bound":  round(float(qty) * (1 + u), 1),
                "model":        model,
            })

    return pd.DataFrame(results)
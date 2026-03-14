"""
forecast/runner.py
──────────────────
Orchestrates the full Demand Forecasting Layer.

Modes
─────
  1. AUTO (default) — runs all 3 models per SKU×WH, picks lowest MAPE
  2. SINGLE         — runs one specific model

Usage
─────
  python -m forecast.runner                         # auto best model
  python -m forecast.runner --model moving_average  # single model
  python -m forecast.runner --model xgboost
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.data_loader  import load_sales_history, load_promo_calendar
from forecast.models    import run_forecast, ModelType
from forecast.evaluator import evaluate_forecast, compute_summary_accuracy

OUTPUT_DIR = ROOT / "forecast" / "outputs"

ALL_MODELS: list[ModelType] = ["moving_average", "exponential_smoothing", "xgboost"]


# ════════════════════════════════════════════════════════════════════════════
# AUTO BEST-MODEL SELECTOR
# ════════════════════════════════════════════════════════════════════════════

def _run_all_models(
    sales_df:     pd.DataFrame,
    promo_df:     pd.DataFrame,
    horizon_days: int,
    test_days:    int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Run all 3 models, evaluate each, then for every SKU×WH combination
    keep only the forecast from the model with the lowest MAPE.

    Returns
    -------
    best_forecast_df   — forecast rows using best model per SKU×WH
    full_accuracy_df   — accuracy rows for ALL models (for reporting)
    model_selection_df — which model was selected per SKU×WH
    """
    all_accuracy:  list[pd.DataFrame] = []
    all_forecasts: dict[str, pd.DataFrame] = {}

    for model in ALL_MODELS:
        print(f"\n  ▶  Running model: {model.upper()}")

        fc_df = run_forecast(
            sales_df     = sales_df,
            horizon_days = horizon_days,
            model        = model,
            promo_df     = promo_df,
        )
        acc_df = evaluate_forecast(
            sales_df  = sales_df,
            test_days = test_days,
            model     = model,
        )

        all_forecasts[model] = fc_df
        all_accuracy.append(acc_df)

        mean_mape = acc_df["mape"].mean() if not acc_df.empty else float("nan")
        print(f"     ✅  {model.upper():28s} → Mean MAPE: {mean_mape:.2f}%")

    full_accuracy_df = pd.concat(all_accuracy, ignore_index=True)

    # ── Per SKU×WH: pick model with lowest MAPE ───────────────────────────
    print("\n  🏆  Selecting best model per SKU×WH …")

    model_selection_df = (
        full_accuracy_df
        .sort_values("mape")
        .groupby(["sku", "warehouse"])
        .first()
        .reset_index()
        [["sku", "warehouse", "model", "mape", "forecast_accuracy"]]
        .rename(columns={"model": "best_model", "mape": "best_mape",
                         "forecast_accuracy": "best_forecast_accuracy"})
    )

    # ── Build final forecast using best model per SKU×WH ─────────────────
    best_rows: list[pd.DataFrame] = []
    for _, row in model_selection_df.iterrows():
        fc = all_forecasts[row["best_model"]]
        subset = fc[
            (fc["sku"] == row["sku"]) &
            (fc["warehouse"] == row["warehouse"])
        ].copy()
        subset["model"] = row["best_model"]
        best_rows.append(subset)

    best_forecast_df = pd.concat(best_rows, ignore_index=True)

    # ── Print selection summary ───────────────────────────────────────────
    print("\n" + "─" * 60)
    print("  BEST MODEL SELECTION SUMMARY")
    print("─" * 60)
    model_counts = model_selection_df["best_model"].value_counts()
    for m, count in model_counts.items():
        pct      = count / len(model_selection_df) * 100
        avg_mape = model_selection_df[model_selection_df["best_model"] == m]["best_mape"].mean()
        bar      = "█" * int(pct / 3)
        print(f"  {m.upper():28s} {bar:<20} {count:3d} SKU×WH ({pct:4.1f}%)  MAPE: {avg_mape:.2f}%")

    overall_mape = model_selection_df["best_mape"].mean()
    print(f"  {'─'*56}")
    print(f"  Overall best avg MAPE: {overall_mape:.2f}%  across {len(model_selection_df)} SKU×WH")
    print("─" * 60)

    return best_forecast_df, full_accuracy_df, model_selection_df


# ════════════════════════════════════════════════════════════════════════════
# MAIN RUN FUNCTION
# ════════════════════════════════════════════════════════════════════════════

def run(
    model:        ModelType | None = None,
    horizon_days: int  = 90,
    test_days:    int  = 30,
    save:         bool = True,
) -> dict[str, pd.DataFrame]:
    """
    Full forecast pipeline.

    Parameters
    ----------
    model : None → auto-select best model per SKU×WH
            str  → run a single specific model

    Returns
    -------
    dict with keys: 'forecast', 'accuracy', 'summary', 'model_selection'
    """
    print("\n" + "═" * 60)
    print("  DEMAND FORECASTING LAYER")
    print("═" * 60)

    auto_mode  = (model is None)
    mode_label = "AUTO — best model per SKU×WH" if auto_mode else model.upper()
    print(f"  Mode      : {mode_label}")
    print(f"  Horizon   : {horizon_days} days")
    print(f"  Test days : {test_days} days")

    # ── 1. Load inputs ────────────────────────────────────────────────────
    print("\n📂  Loading data …")
    sales_df = load_sales_history()
    promo_df = load_promo_calendar()

    print(f"    Sales history : {len(sales_df):,} rows  "
          f"({sales_df['date'].min().date()} → {sales_df['date'].max().date()})")
    print(f"    Promos        : {len(promo_df)} planned promotions")
    print(f"    SKUs          : {sales_df['sku'].nunique()}")
    print(f"    Warehouses    : {sales_df['warehouse'].nunique()}")

    model_selection_df = pd.DataFrame()

    # ── 2. Run forecast ───────────────────────────────────────────────────
    if auto_mode:
        print(f"\n🔮  AUTO MODE — running all {len(ALL_MODELS)} models, selecting best per SKU×WH …")
        forecast_df, accuracy_df, model_selection_df = _run_all_models(
            sales_df     = sales_df,
            promo_df     = promo_df,
            horizon_days = horizon_days,
            test_days    = test_days,
        )
    else:
        print(f"\n🔮  SINGLE MODE — forecasting with [{model.upper()}] …")
        forecast_df = run_forecast(
            sales_df     = sales_df,
            horizon_days = horizon_days,
            model        = model,
            promo_df     = promo_df,
        )
        print(f"    ✅  Forecast rows: {len(forecast_df):,}")

        print(f"\n📊  Evaluating accuracy (test window = {test_days} days) …")
        accuracy_df = evaluate_forecast(
            sales_df  = sales_df,
            test_days = test_days,
            model     = model,
        )

    # ── 3. Summary KPIs ───────────────────────────────────────────────────
    if auto_mode and not model_selection_df.empty:
        # Build accuracy df from best-model rows only for summary
        best_acc = (
            accuracy_df
            .sort_values("mape")
            .groupby(["sku", "warehouse"])
            .first()
            .reset_index()
        )
        summary = compute_summary_accuracy(best_acc)
    else:
        summary = compute_summary_accuracy(accuracy_df)

    # ── 4. Print summary ──────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("  FORECAST ACCURACY SUMMARY")
    print("─" * 60)
    if summary:
        print(f"  Mode                  : {mode_label}")
        print(f"  SKU × WH combinations : {summary['n_combinations']}")
        print(f"  Mean MAPE             : {summary['mean_mape']:.1f}%")
        print(f"  Median MAPE           : {summary['median_mape']:.1f}%")
        print(f"  Forecast Accuracy     : {summary['mean_forecast_accuracy']:.1f}%")
        print(f"  MAPE < 20%            : {summary['pct_below_20_mape']:.1f}% of SKUs")
        print(f"  MAPE < 30%            : {summary['pct_below_30_mape']:.1f}% of SKUs")
        print(f"  Mean Bias             : {summary['mean_bias']:+.1f} units/day")
    print("─" * 60)

    # ── 5. Save outputs ───────────────────────────────────────────────────
    if save:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        forecast_path  = OUTPUT_DIR / "forecast_qty.csv"
        accuracy_path  = OUTPUT_DIR / "forecast_accuracy.csv"
        summary_path   = OUTPUT_DIR / "forecast_summary.csv"
        selection_path = OUTPUT_DIR / "model_selection.csv"

        forecast_df.to_csv(forecast_path, index=False)
        accuracy_df.to_csv(accuracy_path, index=False)
        pd.DataFrame([summary]).to_csv(summary_path, index=False)

        print(f"\n💾  Saved outputs:")
        print(f"    → {forecast_path.relative_to(ROOT)}")
        print(f"    → {accuracy_path.relative_to(ROOT)}")
        print(f"    → {summary_path.relative_to(ROOT)}")

        if not model_selection_df.empty:
            model_selection_df.to_csv(selection_path, index=False)
            print(f"    → {selection_path.relative_to(ROOT)}")

    # ── 6. Sample previews ────────────────────────────────────────────────
    print("\n📋  Forecast sample (first 5 rows):")
    preview_cols = ["sku", "warehouse", "date", "forecast_qty", "lower_bound", "upper_bound", "model"]
    available    = [c for c in preview_cols if c in forecast_df.columns]
    print(forecast_df[available].head(5).to_string(index=False))

    if not accuracy_df.empty:
        print("\n📋  Top 5 SKU×WH by lowest MAPE:")
        best_cols = ["sku", "warehouse", "model", "mape", "forecast_accuracy"]
        avail     = [c for c in best_cols if c in accuracy_df.columns]
        print(accuracy_df.sort_values("mape")[avail].head(5).to_string(index=False))

    if not model_selection_df.empty:
        print("\n📋  Model selection (first 10 SKU×WH):")
        print(model_selection_df.head(10).to_string(index=False))

    return {
        "forecast":        forecast_df,
        "accuracy":        accuracy_df,
        "summary":         pd.DataFrame([summary]),
        "model_selection": model_selection_df,
    }


# ════════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Demand Forecasting Layer")
    parser.add_argument(
        "--model",
        choices=["moving_average", "exponential_smoothing", "xgboost"],
        default=None,
        help="Specific model to use. Omit this flag to auto-select best model per SKU×WH.",
    )
    parser.add_argument("--horizon",   type=int, default=90, help="Forecast horizon in days")
    parser.add_argument("--test-days", type=int, default=30, help="Evaluation test window")
    args = parser.parse_args()

    run(model=args.model, horizon_days=args.horizon, test_days=args.test_days)
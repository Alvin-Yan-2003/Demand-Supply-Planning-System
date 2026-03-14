"""
main.py
───────
End-to-end Supply Chain Planning Pipeline orchestrator.

Runs all three planning layers in sequence:
  Layer 1 → Demand Forecasting
  Layer 2 → Inventory Planning
  Layer 3 → Supply & Replenishment

Usage
─────
  # Full pipeline (default: xgboost forecast)
  python main.py

  # Choose forecast model
  python main.py --forecast-model moving_average
  python main.py --forecast-model exponential_smoothing

  # Skip forecast re-run (use existing output)
  python main.py --skip-forecast
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent


def _banner(title: str) -> None:
    print("\n" + "█" * 60)
    print(f"  {title}")
    print("█" * 60)


def run_pipeline(
    forecast_model: str | None = None,   # None = auto best model per SKU×WH
    horizon_days:   int  = 90,
    test_days:      int  = 30,
    skip_forecast:  bool = False,
    save:           bool = True,
) -> dict:

    t_start = time.time()

    mode_label = "AUTO (best model per SKU×WH)" if forecast_model is None else forecast_model.upper()

    _banner("SUPPLY CHAIN PLANNING SYSTEM  –  FULL PIPELINE")
    print(f"  Forecast mode  : {mode_label}")
    print(f"  Horizon        : {horizon_days} days")
    print(f"  Skip forecast  : {skip_forecast}")

    results = {}

    # ══════════════════════════════════════════════════════════════════════
    # LAYER 1 – DEMAND FORECASTING
    # ══════════════════════════════════════════════════════════════════════
    if not skip_forecast:
        import sys; sys.path.insert(0, str(ROOT))
        from forecast.runner import run as run_forecast

        t1 = time.time()
        results["forecast"] = run_forecast(
            model        = forecast_model,
            horizon_days = horizon_days,
            test_days    = test_days,
            save         = save,
        )
        print(f"\n  ⏱  Forecast layer done in {time.time()-t1:.1f}s")
    else:
        print("\n  ⏩  Skipping forecast (using cached output)")
        results["forecast"] = None

    # ══════════════════════════════════════════════════════════════════════
    # LAYER 2 – INVENTORY PLANNING
    # ══════════════════════════════════════════════════════════════════════
    from inventory.runner import run as run_inventory

    t2 = time.time()
    results["inventory"] = run_inventory(save=save)
    print(f"\n  ⏱  Inventory layer done in {time.time()-t2:.1f}s")

    # ══════════════════════════════════════════════════════════════════════
    # LAYER 3 – SUPPLY & REPLENISHMENT
    # ══════════════════════════════════════════════════════════════════════
    from supply.runner import run as run_supply

    t3 = time.time()
    results["supply"] = run_supply(save=save)
    print(f"\n  ⏱  Supply layer done in {time.time()-t3:.1f}s")

    # ══════════════════════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ══════════════════════════════════════════════════════════════════════
    elapsed = time.time() - t_start

    _banner("PIPELINE COMPLETE")

    # Collect key numbers
    inv_kpis  = results["inventory"]["kpis"].iloc[0].to_dict() if results["inventory"] else {}
    sup_kpis  = results["supply"]["kpis"].iloc[0].to_dict()    if results["supply"]    else {}

    print(f"\n  ✅  All 3 planning layers completed in {elapsed:.1f}s")
    print()
    print(f"  {'─'*50}")
    print(f"  INVENTORY HEALTH")
    print(f"  {'─'*50}")
    print(f"    Avg Days of Supply      : {inv_kpis.get('avg_dos', '–')} days")
    print(f"    Total Inventory Value   : {inv_kpis.get('total_inventory_value_b_vnd', '–')} B VND")
    print(f"    CRITICAL items          : {inv_kpis.get('pct_critical', '–')}%")
    print(f"    REORDER items           : {inv_kpis.get('pct_reorder', '–')}%")
    print(f"    Avg Stockout Risk       : {inv_kpis.get('avg_stockout_risk_pct', '–')}%")
    print()
    print(f"  {'─'*50}")
    print(f"  REPLENISHMENT PLAN")
    print(f"  {'─'*50}")
    print(f"    Orders to Place         : {sup_kpis.get('total_orders', '–')}")
    print(f"    Total Units to Order    : {sup_kpis.get('total_units_ordered', 0):,}")
    print(f"    Total Order Value       : {sup_kpis.get('total_order_value_b_vnd', '–')} B VND")
    print(f"    Expected Arrivals       : {sup_kpis.get('earliest_arrival', '–')} → {sup_kpis.get('latest_arrival', '–')}")
    print()
    print(f"  {'─'*50}")
    print(f"  OUTPUT FILES")
    print(f"  {'─'*50}")
    output_files = [
        "forecast/outputs/forecast_qty.csv",
        "forecast/outputs/forecast_accuracy.csv",
        "forecast/outputs/forecast_summary.csv",
        "inventory/outputs/inventory_policy.csv",
        "inventory/outputs/inventory_kpis.csv",
        "supply/outputs/replenishment_plan.csv",
        "supply/outputs/supply_kpis.csv",
    ]
    for f in output_files:
        path = ROOT / f
        exists = "✅" if path.exists() else "❌"
        size   = f"{path.stat().st_size/1024:.1f}KB" if path.exists() else ""
        print(f"    {exists}  {f:<45} {size}")

    print()
    return results


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(ROOT))

    parser = argparse.ArgumentParser(description="Supply Chain Planning Pipeline")
    parser.add_argument(
        "--forecast-model",
        choices=["moving_average", "exponential_smoothing", "xgboost"],
        default=None,   # None = auto best model per SKU×WH
        help="Forecast model to use. Omit for auto best-model selection per SKU×WH.",
    )
    parser.add_argument("--horizon",        type=int,  default=90)
    parser.add_argument("--test-days",      type=int,  default=30)
    parser.add_argument("--skip-forecast",  action="store_true")
    args = parser.parse_args()

    run_pipeline(
        forecast_model = args.forecast_model,
        horizon_days   = args.horizon,
        test_days      = args.test_days,
        skip_forecast  = args.skip_forecast,
    )
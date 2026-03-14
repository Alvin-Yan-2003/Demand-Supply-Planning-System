"""
supply/runner.py
────────────────
Orchestrates the full Supply & Replenishment Planning Layer:
  1. Load policy (from inventory layer output)
  2. Load forecast (from forecast layer output)
  3. Generate replenishment plan
  4. Print summary + priority report
  5. Save outputs to supply/outputs/

Usage
─────
  python -m supply.runner
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.data_loader import (
    load_inventory,
    load_lead_time,
    load_supplier_master,
    load_warehouse_master,
    load_product_master,
)
from supply.replenishment import generate_replenishment_plan, compute_supply_kpis

OUTPUT_DIR    = ROOT / "supply"    / "outputs"
INV_POLICY    = ROOT / "inventory" / "outputs" / "inventory_policy.csv"
FORECAST_FILE = ROOT / "forecast"  / "outputs" / "forecast_qty.csv"


# ════════════════════════════════════════════════════════════════════════════

def run(
    plan_date: pd.Timestamp | None = None,
    save: bool = True,
) -> dict[str, pd.DataFrame]:
    """
    Full supply planning pipeline.

    Returns
    -------
    dict with keys: 'plan', 'kpis'
    """
    print("\n" + "═" * 60)
    print("  SUPPLY & REPLENISHMENT PLANNING LAYER")
    print("═" * 60)

    # ── 1. Load inputs ────────────────────────────────────────────────────
    print("\n📂  Loading data …")

    # Policy from inventory layer (required)
    if not INV_POLICY.exists():
        raise FileNotFoundError(
            f"Inventory policy not found at {INV_POLICY}\n"
            "Run `python -m inventory.runner` first."
        )
    policy_df = pd.read_csv(INV_POLICY)
    print(f"    Inventory policy : {len(policy_df):,} rows loaded")

    # Forecast from forecast layer (optional — enriches demand_during_lt)
    forecast_df = None
    if FORECAST_FILE.exists():
        forecast_df = pd.read_csv(FORECAST_FILE, parse_dates=["date"])
        print(f"    Forecast         : {len(forecast_df):,} rows loaded")
    else:
        print("    Forecast         : not found — using avg demand as fallback")

    inventory_df  = load_inventory()
    lead_time_df  = load_lead_time()
    supplier_df   = load_supplier_master()
    warehouse_df  = load_warehouse_master()
    product_df    = load_product_master()

    n_critical = (policy_df["policy_status"] == "CRITICAL").sum()
    n_reorder  = (policy_df["policy_status"] == "REORDER").sum()
    print(f"\n    Items needing replenishment:")
    print(f"      🔴 CRITICAL : {n_critical}")
    print(f"      🟡 REORDER  : {n_reorder}")

    # ── 2. Generate plan ─────────────────────────────────────────────────
    if plan_date is None:
        plan_date = pd.Timestamp("2024-06-29")  # day after last history date

    print(f"\n🚚  Generating replenishment plan (plan date: {plan_date.date()}) …")
    plan_df = generate_replenishment_plan(
        policy_df    = policy_df,
        inventory_df = inventory_df,
        lead_time_df = lead_time_df,
        supplier_df  = supplier_df,
        warehouse_df = warehouse_df,
        product_df   = product_df,
        forecast_df  = forecast_df,
        plan_date    = plan_date,
    )

    # ── 3. KPI summary ────────────────────────────────────────────────────
    kpis = compute_supply_kpis(plan_df)

    print("\n" + "─" * 60)
    print("  REPLENISHMENT PLAN SUMMARY")
    print("─" * 60)
    print(f"  Total Orders Generated       : {kpis['total_orders']}")
    print(f"    🔴 Critical orders         : {kpis['n_critical_orders']}")
    print(f"    🟡 Reorder orders          : {kpis['n_reorder_orders']}")
    print(f"  Total Units to Order         : {kpis['total_units_ordered']:,}")
    print(f"  Total Order Value            : {kpis['total_order_value_b_vnd']:.2f} B VND")
    print(f"  Avg Order Value              : {kpis['avg_order_value_m_vnd']:.2f} M VND")
    print(f"  Avg Lead Time                : {kpis['avg_lead_time_days']:.1f} days")
    print(f"  Expected Arrivals            : {kpis['earliest_arrival']} → {kpis['latest_arrival']}")
    print(f"  Suppliers Involved           : {kpis['n_suppliers_involved']}")
    print(f"  Warehouses Receiving         : {kpis['n_warehouses_receiving']}")
    print(f"  Capacity Compliance          : {kpis['pct_capacity_ok']:.1f}%")
    print("─" * 60)

    # ── 4. Breakdown by warehouse ─────────────────────────────────────────
    print("\n📋  Plan by Warehouse:")
    wh_summary = (
        plan_df
        .groupby("warehouse")
        .agg(
            n_orders        = ("order_id", "count"),
            total_units     = ("adjusted_qty", "sum"),
            order_value_b   = ("order_value_vnd", lambda x: round(x.sum() / 1e9, 2)),
            cbm_incoming    = ("cbm_required", "sum"),
            n_critical      = ("policy_status", lambda x: (x == "CRITICAL").sum()),
        )
        .sort_values("n_critical", ascending=False)
    )
    print(wh_summary.to_string())

    # ── 5. Top 10 priority orders ─────────────────────────────────────────
    print("\n🔝  Top 10 Priority Orders:")
    top_cols = [
        "order_id", "sku", "warehouse", "abc_class", "policy_status",
        "adjusted_qty", "order_value_vnd", "expected_arrival",
        "stockout_risk", "dos_before_order",
    ]
    print(plan_df.head(10)[top_cols].to_string(index=False))

    # ── 6. Save outputs ───────────────────────────────────────────────────
    if save:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        plan_path = OUTPUT_DIR / "replenishment_plan.csv"
        kpi_path  = OUTPUT_DIR / "supply_kpis.csv"

        plan_df.to_csv(plan_path, index=False)
        pd.DataFrame([kpis]).to_csv(kpi_path, index=False)

        print(f"\n💾  Saved outputs:")
        print(f"    → {plan_path.relative_to(ROOT)}")
        print(f"    → {kpi_path.relative_to(ROOT)}")

    return {"plan": plan_df, "kpis": pd.DataFrame([kpis])}


if __name__ == "__main__":
    run()
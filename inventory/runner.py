"""
inventory/runner.py
───────────────────
Orchestrates the full Inventory Planning Layer:
  1. Load all inputs
  2. Compute inventory policy (SS, ROP, EOQ, DOS, risk)
  3. Print KPI summary
  4. Save outputs to inventory/outputs/

Usage
─────
  python -m inventory.runner
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.data_loader import (
    load_sales_history,
    load_inventory,
    load_lead_time,
    load_cost_parameters,
    load_product_master,
)
from inventory.policy import compute_inventory_policy, compute_inventory_kpis

OUTPUT_DIR = ROOT / "inventory" / "outputs"


# ════════════════════════════════════════════════════════════════════════════

def run(save: bool = True) -> dict[str, pd.DataFrame]:
    """
    Full inventory planning pipeline.

    Returns
    -------
    dict with keys: 'policy', 'kpis'
    """
    print("\n" + "═" * 60)
    print("  INVENTORY PLANNING LAYER")
    print("═" * 60)

    # ── 1. Load inputs ────────────────────────────────────────────────────
    print("\n📂  Loading data …")
    sales_df    = load_sales_history()
    inventory_df= load_inventory()
    lead_time_df= load_lead_time()
    cost_df     = load_cost_parameters()
    product_df  = load_product_master()

    print(f"    Active SKUs      : {product_df[product_df['is_active']]['sku'].nunique()}")
    print(f"    Warehouses       : {sales_df['warehouse'].nunique()}")
    print(f"    Inventory rows   : {len(inventory_df):,}")
    print(f"    Lead time rows   : {len(lead_time_df)}")

    # ── 2. Compute policy ─────────────────────────────────────────────────
    print("\n⚙️   Computing inventory policy …")
    policy_df = compute_inventory_policy(
        inventory_df = inventory_df,
        sales_df     = sales_df,
        lead_time_df = lead_time_df,
        cost_df      = cost_df,
        product_df   = product_df,
    )
    print(f"    ✅  Policy rows computed: {len(policy_df):,}")

    # ── 3. KPI summary ────────────────────────────────────────────────────
    kpis = compute_inventory_kpis(policy_df)

    print("\n" + "─" * 60)
    print("  INVENTORY POLICY KPI SUMMARY")
    print("─" * 60)
    print(f"  Total SKU × WH combinations : {kpis['total_skus']:,}")
    print(f"  Total Inventory Value       : {kpis['total_inventory_value_b_vnd']:.2f} B VND")
    print(f"  Avg Days of Supply          : {kpis['avg_dos']} days")
    print(f"  Median Days of Supply       : {kpis['median_dos']} days")
    print(f"  Avg Stockout Risk           : {kpis['avg_stockout_risk_pct']:.2f}%")
    print(f"  Replenishment Needed        : {kpis['n_replenishment_needed']} SKU×WH")
    print()
    print(f"  Policy Status Breakdown:")
    print(f"    🔴 CRITICAL   : {kpis['pct_critical']:.1f}%")
    print(f"    🟡 REORDER    : {kpis['pct_reorder']:.1f}%")
    print(f"    🟢 OK         : {kpis['pct_ok']:.1f}%")
    print(f"    🔵 OVERSTOCK  : {kpis['pct_overstock']:.1f}%")
    print(f"  Avg Safety Stock            : {kpis['avg_safety_stock']:.0f} units")
    print(f"  Avg EOQ                     : {kpis['avg_eoq']:.0f} units")
    print("─" * 60)

    # ── 4. Policy breakdown by ABC class ──────────────────────────────────
    print("\n📋  Policy by ABC Class:")
    abc_summary = (
        policy_df
        .groupby("abc_class")
        .agg(
            n_skus              = ("sku", "count"),
            avg_safety_stock    = ("safety_stock", "mean"),
            avg_rop             = ("reorder_point", "mean"),
            avg_eoq             = ("eoq", "mean"),
            avg_dos             = ("dos", lambda x: x.replace(999, float("nan")).mean()),
            pct_needs_reorder   = ("replenishment_needed", "mean"),
            avg_stockout_risk   = ("stockout_risk", "mean"),
        )
        .round(1)
    )
    print(abc_summary.to_string())

    # ── 5. Top 10 critical items ──────────────────────────────────────────
    critical = policy_df[policy_df["policy_status"].isin(["CRITICAL", "REORDER"])]
    if not critical.empty:
        print(f"\n⚠️   Top 10 items needing immediate attention:")
        top10_cols = ["sku", "warehouse", "abc_class", "policy_status",
                      "on_hand_units", "reorder_point", "safety_stock",
                      "dos", "stockout_risk"]
        print(
            critical
            .sort_values(["policy_status", "stockout_risk"], ascending=[True, False])
            .head(10)[top10_cols]
            .to_string(index=False)
        )

    # ── 6. Save ───────────────────────────────────────────────────────────
    if save:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        policy_path = OUTPUT_DIR / "inventory_policy.csv"
        kpi_path    = OUTPUT_DIR / "inventory_kpis.csv"

        policy_df.to_csv(policy_path, index=False)
        pd.DataFrame([kpis]).to_csv(kpi_path, index=False)

        print(f"\n💾  Saved outputs:")
        print(f"    → {policy_path.relative_to(ROOT)}")
        print(f"    → {kpi_path.relative_to(ROOT)}")

    return {"policy": policy_df, "kpis": pd.DataFrame([kpis])}


if __name__ == "__main__":
    run()
"""
supply/replenishment.py
───────────────────────
Supply & Replenishment Planning Layer
Generates replenishment order recommendations per SKU × Warehouse.

Planning logic
──────────────
  1. Identify which SKU×WH need replenishment (on_hand ≤ ROP)
  2. Compute replenishment_qty = max_stock − on_hand  (up to EOQ)
  3. Apply supplier constraints: MOQ, available supply
  4. Apply warehouse capacity constraints (CBM check)
  5. Assign order_date (today) and expected_arrival (+ lead time)
  6. Prioritize by: CRITICAL > REORDER, then ABC class, then stockout risk

Outputs
───────
  replenishment_plan.csv  — order-level recommendations
  supply_summary.csv      — aggregate KPIs

Public API
──────────
  generate_replenishment_plan(policy_df, forecast_df, inventory_df,
                               lead_time_df, supplier_df,
                               warehouse_df, plan_date) → plan_df
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PRIORITY_ORDER = {"CRITICAL": 0, "REORDER": 1, "OK": 2, "OVERSTOCK": 3}
ABC_PRIORITY   = {"A": 0, "B": 1, "C": 2}


# ════════════════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════════════════

def _pick_supplier(sku: str, lead_time_df: pd.DataFrame) -> dict:
    """Select primary supplier for a SKU (shortest lead time, highest reliability proxy)."""
    lt = lead_time_df[lead_time_df["sku"] == sku]
    if lt.empty:
        return {"supplier_id": "UNKNOWN", "lead_time_avg": 14, "lead_time_std": 2}
    best = lt.sort_values("lead_time_days").iloc[0]
    return {
        "supplier_id":    best["supplier_id"],
        "lead_time_avg":  best["lead_time_days"],
        "lead_time_std":  best["lead_time_std_days"],
    }


def _apply_moq(qty: float, moq: float) -> float:
    """Round up order quantity to nearest MOQ multiple."""
    if moq <= 0:
        return max(qty, 1)
    return float(np.ceil(qty / moq) * moq)


def _cbm_needed(qty: float, volume_cbm_per_unit: float) -> float:
    return qty * volume_cbm_per_unit


def _available_capacity_cbm(
    warehouse: str,
    warehouse_df: pd.DataFrame,
    inventory_df: pd.DataFrame,
) -> float:
    """Remaining free CBM in a warehouse."""
    cap_row = warehouse_df[warehouse_df["warehouse"] == warehouse]
    if cap_row.empty:
        return 99_999.0  # unknown → no constraint

    capacity_cbm = float(cap_row.iloc[0]["capacity_cbm"])

    used_cbm = inventory_df[inventory_df["warehouse"] == warehouse]["used_cbm"].sum()

    return max(0.0, capacity_cbm - used_cbm)


def _expected_arrival(order_date: pd.Timestamp, lead_time_days: float) -> pd.Timestamp:
    return order_date + pd.Timedelta(days=int(np.ceil(lead_time_days)))


# ════════════════════════════════════════════════════════════════════════════
# FORECAST-BASED DEMAND FORWARD FILL
# ════════════════════════════════════════════════════════════════════════════

def _forecast_demand_during_lt(
    sku: str,
    warehouse: str,
    lead_time_days: float,
    plan_date: pd.Timestamp,
    forecast_df: pd.DataFrame | None,
    fallback_daily_demand: float,
) -> float:
    """
    Sum of forecast demand from plan_date for lead_time_days.
    Falls back to avg × LT if no forecast available.
    """
    if forecast_df is None or forecast_df.empty:
        return fallback_daily_demand * lead_time_days

    mask = (
        (forecast_df["sku"] == sku) &
        (forecast_df["warehouse"] == warehouse) &
        (forecast_df["date"] >= plan_date) &
        (forecast_df["date"] < plan_date + pd.Timedelta(days=int(lead_time_days)))
    )
    demand_lt = forecast_df[mask]["forecast_qty"].sum()

    return demand_lt if demand_lt > 0 else fallback_daily_demand * lead_time_days


# ════════════════════════════════════════════════════════════════════════════
# MAIN ENGINE
# ════════════════════════════════════════════════════════════════════════════

def generate_replenishment_plan(
    policy_df:     pd.DataFrame,
    inventory_df:  pd.DataFrame,
    lead_time_df:  pd.DataFrame,
    supplier_df:   pd.DataFrame,
    warehouse_df:  pd.DataFrame,
    product_df:    pd.DataFrame,
    forecast_df:   pd.DataFrame | None = None,
    plan_date:     pd.Timestamp | None = None,
) -> pd.DataFrame:
    """
    Generate replenishment order recommendations.

    Parameters
    ----------
    policy_df    : Output of inventory.policy.compute_inventory_policy()
    inventory_df : Current inventory snapshot
    lead_time_df : Lead time per SKU × Supplier
    supplier_df  : Supplier master (MOQ, reliability)
    warehouse_df : Warehouse master (capacity)
    product_df   : Product master (volume CBM per unit)
    forecast_df  : Optional demand forecast (from forecast layer)
    plan_date    : Planning date (default: today / last snapshot date)

    Returns
    -------
    DataFrame with columns:
      order_id | sku | warehouse | supplier_id
      | on_hand_units | reorder_point | max_stock
      | replenishment_qty | adjusted_qty | moq
      | order_date | expected_arrival | lead_time_days
      | demand_during_lt | demand_avg
      | cbm_required | capacity_available_cbm | capacity_ok
      | order_value_vnd | priority | abc_class | policy_status
      | replenishment_flag
    """
    if plan_date is None:
        plan_date = pd.Timestamp.today().normalize()

    # ── Pre-process ────────────────────────────────────────────────────────
    prod_vol = product_df.set_index("sku")[["volume_cbm_per_unit", "unit_cost_vnd"]]
    sup_moq  = supplier_df.set_index("supplier_id")["min_order_qty"]

    # Work only on items that need replenishment
    needs_replenishment = policy_df[
        policy_df["policy_status"].isin(["CRITICAL", "REORDER"])
    ].copy()

    # Sort by priority: CRITICAL first, then ABC class, then stockout risk
    needs_replenishment["_priority_num"] = needs_replenishment["policy_status"].map(PRIORITY_ORDER)
    needs_replenishment["_abc_num"]      = needs_replenishment["abc_class"].map(ABC_PRIORITY).fillna(2)
    needs_replenishment = needs_replenishment.sort_values(
        ["_priority_num", "_abc_num", "stockout_risk"],
        ascending=[True, True, False],
    ).reset_index(drop=True)

    # Track remaining warehouse capacity as orders are placed
    wh_remaining_cbm: dict[str, float] = {}
    for wh in warehouse_df["warehouse"].unique():
        wh_remaining_cbm[wh] = _available_capacity_cbm(wh, warehouse_df, inventory_df)

    rows    = []
    skipped = 0

    for idx, row in needs_replenishment.iterrows():
        sku       = row["sku"]
        warehouse = row["warehouse"]

        # ── Supplier selection ─────────────────────────────────────────────
        sup_info   = _pick_supplier(sku, lead_time_df)
        supplier   = sup_info["supplier_id"]
        lt_days    = sup_info["lead_time_avg"]
        moq        = float(sup_moq.get(supplier, 500))

        # ── Replenishment quantity ─────────────────────────────────────────
        # Base qty: fill up to max_stock
        base_qty = max(0, row["max_stock"] - row["on_hand_units"])

        # Never order less than EOQ
        base_qty = max(base_qty, row["eoq"])

        # Apply MOQ rounding
        adj_qty = _apply_moq(base_qty, moq)

        # ── Demand during lead time ────────────────────────────────────────
        demand_lt = _forecast_demand_during_lt(
            sku                  = sku,
            warehouse            = warehouse,
            lead_time_days       = lt_days,
            plan_date            = plan_date,
            forecast_df          = forecast_df,
            fallback_daily_demand= row["demand_avg"],
        )

        # Ensure order covers demand during lead time + safety stock
        min_qty = max(demand_lt + row["safety_stock"] - row["on_hand_units"], 0)
        adj_qty = max(adj_qty, _apply_moq(min_qty, moq))

        # ── Capacity check ─────────────────────────────────────────────────
        vol_per_unit = float(prod_vol.at[sku, "volume_cbm_per_unit"]) if sku in prod_vol.index else 0.05
        unit_cost    = float(prod_vol.at[sku, "unit_cost_vnd"])        if sku in prod_vol.index else 10_000

        cbm_required  = _cbm_needed(adj_qty, vol_per_unit)
        cap_available = wh_remaining_cbm.get(warehouse, 99_999.0)
        capacity_ok   = cbm_required <= cap_available

        if not capacity_ok:
            # Scale down to what fits
            if vol_per_unit > 0 and cap_available > 0:
                max_by_cap = cap_available / vol_per_unit
                adj_qty    = _apply_moq(max(max_by_cap, moq), moq)
                cbm_required = _cbm_needed(adj_qty, vol_per_unit)
                capacity_ok  = True
            else:
                skipped += 1
                continue

        # Deduct from warehouse remaining capacity
        wh_remaining_cbm[warehouse] = max(0, wh_remaining_cbm.get(warehouse, 0) - cbm_required)

        # ── Build order record ─────────────────────────────────────────────
        arrival_date = _expected_arrival(plan_date, lt_days)

        rows.append({
            "order_id":               f"ORD-{len(rows)+1:04d}",
            "sku":                    sku,
            "warehouse":              warehouse,
            "supplier_id":            supplier,
            "abc_class":              row["abc_class"],
            "policy_status":          row["policy_status"],

            # Current state
            "on_hand_units":          int(row["on_hand_units"]),
            "reorder_point":          int(row["reorder_point"]),
            "safety_stock":           int(row["safety_stock"]),
            "max_stock":              int(row["max_stock"]),
            "demand_avg":             round(row["demand_avg"], 1),
            "demand_during_lt":       round(demand_lt, 0),

            # Order details
            "replenishment_qty":      int(base_qty),
            "adjusted_qty":           int(adj_qty),
            "moq":                    int(moq),
            "order_date":             plan_date,
            "expected_arrival":       arrival_date,
            "lead_time_days":         int(lt_days),

            # Capacity
            "cbm_required":           round(cbm_required, 2),
            "capacity_available_cbm": round(cap_available, 2),
            "capacity_ok":            capacity_ok,

            # Financials
            "order_value_vnd":        round(adj_qty * unit_cost, 0),

            # Risk
            "stockout_risk":          round(row["stockout_risk"], 4),
            "dos_before_order":       round(row["dos"], 1),
        })

    print(f"    ✅  Orders generated : {len(rows)}")
    if skipped:
        print(f"    ⚠️   Skipped (no capacity) : {skipped}")

    return pd.DataFrame(rows)


# ════════════════════════════════════════════════════════════════════════════
# SUMMARY KPIs
# ════════════════════════════════════════════════════════════════════════════

def compute_supply_kpis(plan_df: pd.DataFrame) -> dict:
    if plan_df.empty:
        return {}

    return {
        "total_orders":                 len(plan_df),
        "total_order_value_b_vnd":      round(plan_df["order_value_vnd"].sum() / 1e9, 2),
        "avg_order_value_m_vnd":        round(plan_df["order_value_vnd"].mean() / 1e6, 2),
        "total_units_ordered":          int(plan_df["adjusted_qty"].sum()),
        "n_critical_orders":            int((plan_df["policy_status"] == "CRITICAL").sum()),
        "n_reorder_orders":             int((plan_df["policy_status"] == "REORDER").sum()),
        "pct_capacity_ok":              round(plan_df["capacity_ok"].mean() * 100, 1),
        "avg_lead_time_days":           round(plan_df["lead_time_days"].mean(), 1),
        "earliest_arrival":             str(plan_df["expected_arrival"].min().date()),
        "latest_arrival":               str(plan_df["expected_arrival"].max().date()),
        "n_suppliers_involved":         plan_df["supplier_id"].nunique(),
        "n_warehouses_receiving":       plan_df["warehouse"].nunique(),
    }
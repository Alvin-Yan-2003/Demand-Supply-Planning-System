"""
inventory/policy.py
───────────────────
Inventory Planning Layer
Computes optimal inventory policies per SKU × Warehouse:

  - Safety Stock   : buffer against demand & lead time variability
  - Reorder Point  : when to trigger a replenishment order
  - EOQ            : optimal order quantity (Economic Order Quantity)
  - Max Stock      : upper inventory target
  - Days of Supply : how many days current stock will last
  - Stockout Risk  : probability of stockout before next replenishment

Policy formulas
───────────────
  σ_demand_LT = sqrt(LT × σ_d² + demand_avg² × σ_LT²)
  Safety Stock = Z(SL) × σ_demand_LT
  ROP          = demand_avg × LT + Safety Stock
  EOQ          = sqrt(2 × D × S / (H × C))
  Max Stock    = ROP + EOQ
  DOS          = on_hand / avg_daily_demand

Public API
──────────
  compute_inventory_policy(inventory_df, sales_df, lead_time_df,
                            cost_df, product_df) → policy_df
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


# ════════════════════════════════════════════════════════════════════════════
# Z-SCORE LOOKUP
# ════════════════════════════════════════════════════════════════════════════

def _z_score(service_level: float) -> float:
    """Map service level (0–1) → Z-score from standard normal distribution."""
    sl = np.clip(service_level, 0.50, 0.9999)
    return float(stats.norm.ppf(sl))


# ════════════════════════════════════════════════════════════════════════════
# DEMAND STATISTICS  (per SKU × Warehouse, from sales history)
# ════════════════════════════════════════════════════════════════════════════

def _compute_demand_stats(sales_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily demand statistics per SKU × Warehouse.

    Returns DataFrame with:
      sku | warehouse | demand_avg | demand_std | demand_cv | n_days
    """
    daily = (
        sales_df
        .groupby(["sku", "warehouse", "date"])["demand_qty"]
        .sum()
        .reset_index()
    )

    stats_df = (
        daily
        .groupby(["sku", "warehouse"])["demand_qty"]
        .agg(
            demand_avg="mean",
            demand_std="std",
            n_days="count",
        )
        .reset_index()
    )

    stats_df["demand_std"]  = stats_df["demand_std"].fillna(0)
    stats_df["demand_cv"]   = np.where(
        stats_df["demand_avg"] > 0,
        stats_df["demand_std"] / stats_df["demand_avg"],
        0,
    )

    return stats_df


# ════════════════════════════════════════════════════════════════════════════
# LEAD TIME STATS  (per SKU — pick best/primary supplier)
# ════════════════════════════════════════════════════════════════════════════

def _compute_lead_time_stats(lead_time_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate lead time to one row per SKU
    (use supplier with shortest average lead time as primary).

    Returns DataFrame with:
      sku | lead_time_avg | lead_time_std | review_period_days
    """
    lt = (
        lead_time_df
        .sort_values("lead_time_days")
        .groupby("sku")
        .first()
        .reset_index()
        [["sku", "lead_time_days", "lead_time_std_days", "review_period_days"]]
        .rename(columns={
            "lead_time_days":     "lead_time_avg",
            "lead_time_std_days": "lead_time_std",
        })
    )
    return lt


# ════════════════════════════════════════════════════════════════════════════
# CORE POLICY CALCULATIONS
# ════════════════════════════════════════════════════════════════════════════

def _safety_stock(
    demand_avg: float,
    demand_std: float,
    lead_time_avg: float,
    lead_time_std: float,
    z: float,
) -> float:
    """
    Safety Stock using combined demand & lead time uncertainty:
      σ_combined = sqrt(LT_avg × σ_d² + d_avg² × σ_LT²)
      SS = Z × σ_combined
    """
    if lead_time_avg <= 0:
        return 0.0

    sigma_combined = np.sqrt(
        lead_time_avg * (demand_std ** 2) +
        (demand_avg ** 2) * (lead_time_std ** 2)
    )
    return max(0.0, z * sigma_combined)


def _reorder_point(demand_avg: float, lead_time_avg: float, safety_stock: float) -> float:
    """ROP = avg demand during lead time + safety stock."""
    return max(0.0, demand_avg * lead_time_avg + safety_stock)


def _eoq(
    annual_demand: float,
    ordering_cost: float,
    unit_cost: float,
    holding_cost_pct: float,
) -> float:
    """
    Economic Order Quantity:
      EOQ = sqrt(2 × D × S / (H × C))
    """
    h = holding_cost_pct * unit_cost   # annual holding cost per unit
    if h <= 0 or annual_demand <= 0:
        return 0.0
    return float(np.sqrt(2 * annual_demand * ordering_cost / h))


def _days_of_supply(on_hand: float, demand_avg: float) -> float:
    """DOS = on_hand / avg_daily_demand"""
    if demand_avg <= 0:
        return 999.0
    return round(on_hand / demand_avg, 1)


def _stockout_risk(on_hand: float, rop: float, demand_std: float, lead_time_avg: float) -> float:
    """
    Approximate probability of stockout before replenishment arrives.
    P(stockout) = P(demand during LT > on_hand)
    """
    if demand_std <= 0 or lead_time_avg <= 0:
        return 0.0

    demand_during_lt_std = demand_std * np.sqrt(lead_time_avg)
    if demand_during_lt_std <= 0:
        return 0.0

    # Using on_hand as the "available" buffer
    z = (on_hand - rop) / demand_during_lt_std
    return float(np.clip(1 - stats.norm.cdf(z), 0, 1))


# ════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ════════════════════════════════════════════════════════════════════════════

def compute_inventory_policy(
    inventory_df:  pd.DataFrame,
    sales_df:      pd.DataFrame,
    lead_time_df:  pd.DataFrame,
    cost_df:       pd.DataFrame,
    product_df:    pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute full inventory policy for every SKU × Warehouse.

    Returns
    -------
    DataFrame with columns:
      sku | warehouse | demand_avg | demand_std | demand_cv
      | lead_time_avg | lead_time_std | z_score | safety_stock
      | reorder_point | eoq | max_stock | on_hand_units
      | dos | inventory_value_vnd | stockout_risk
      | replenishment_needed | abc_class | target_service_level
      | policy_status
    """
    print("  📐  Computing demand statistics …")
    demand_stats = _compute_demand_stats(sales_df)

    print("  📐  Computing lead time statistics …")
    lt_stats = _compute_lead_time_stats(lead_time_df)

    # ── Merge all inputs ──────────────────────────────────────────────────
    policy = demand_stats.merge(lt_stats, on="sku", how="left")

    # Cost params
    cost_cols = ["sku", "holding_cost_pct", "ordering_cost_vnd",
                 "stockout_cost_per_unit_vnd", "target_service_level"]
    policy = policy.merge(cost_df[cost_cols], on="sku", how="left")

    # Product master
    prod_cols = ["sku", "unit_cost_vnd", "abc_class"]
    policy = policy.merge(product_df[prod_cols], on="sku", how="left")

    # Current inventory snapshot (latest per SKU × WH)
    inv_snap = (
        inventory_df
        .sort_values("snapshot_date")
        .groupby(["sku", "warehouse"])
        .last()
        .reset_index()
        [["sku", "warehouse", "on_hand_units", "inventory_value_vnd"]]
    )
    policy = policy.merge(inv_snap, on=["sku", "warehouse"], how="left")
    policy["on_hand_units"]       = policy["on_hand_units"].fillna(0)
    policy["inventory_value_vnd"] = policy["inventory_value_vnd"].fillna(0)

    # ── Fill missing lead time defaults ───────────────────────────────────
    policy["lead_time_avg"]       = policy["lead_time_avg"].fillna(14)
    policy["lead_time_std"]       = policy["lead_time_std"].fillna(2)
    policy["review_period_days"]  = policy["review_period_days"].fillna(7)
    policy["target_service_level"]= policy["target_service_level"].fillna(0.95)
    policy["holding_cost_pct"]    = policy["holding_cost_pct"].fillna(0.20)
    policy["ordering_cost_vnd"]   = policy["ordering_cost_vnd"].fillna(1_000_000)
    policy["unit_cost_vnd"]       = policy["unit_cost_vnd"].fillna(10_000)

    print("  📐  Calculating Safety Stock, ROP, EOQ …")

    rows = []
    for _, row in policy.iterrows():

        z  = _z_score(float(row["target_service_level"]))
        ss = _safety_stock(
            demand_avg    = row["demand_avg"],
            demand_std    = row["demand_std"],
            lead_time_avg = row["lead_time_avg"],
            lead_time_std = row["lead_time_std"],
            z             = z,
        )
        rop = _reorder_point(row["demand_avg"], row["lead_time_avg"], ss)

        annual_demand = row["demand_avg"] * 365
        eoq = _eoq(
            annual_demand    = annual_demand,
            ordering_cost    = row["ordering_cost_vnd"],
            unit_cost        = row["unit_cost_vnd"],
            holding_cost_pct = row["holding_cost_pct"],
        )

        max_stock = rop + eoq
        dos       = _days_of_supply(row["on_hand_units"], row["demand_avg"])
        risk      = _stockout_risk(row["on_hand_units"], rop, row["demand_std"], row["lead_time_avg"])

        replenishment_needed = bool(row["on_hand_units"] <= rop)

        # Policy status label
        if row["on_hand_units"] <= ss:
            status = "CRITICAL"
        elif row["on_hand_units"] <= rop:
            status = "REORDER"
        elif row["on_hand_units"] > max_stock * 1.2:
            status = "OVERSTOCK"
        else:
            status = "OK"

        rows.append({
            "sku":                    row["sku"],
            "warehouse":              row["warehouse"],
            "abc_class":              row.get("abc_class", "C"),
            "target_service_level":   round(float(row["target_service_level"]), 3),
            "z_score":                round(z, 3),

            # Demand stats
            "demand_avg":             round(row["demand_avg"], 1),
            "demand_std":             round(row["demand_std"], 1),
            "demand_cv":              round(row["demand_cv"],  3),

            # Lead time
            "lead_time_avg":          round(row["lead_time_avg"], 1),
            "lead_time_std":          round(row["lead_time_std"], 1),
            "review_period_days":     int(row["review_period_days"]),

            # Policy outputs
            "safety_stock":           round(ss,        0),
            "reorder_point":          round(rop,       0),
            "eoq":                    round(eoq,       0),
            "max_stock":              round(max_stock, 0),

            # Current state
            "on_hand_units":          round(row["on_hand_units"], 0),
            "inventory_value_vnd":    round(row["inventory_value_vnd"], 0),
            "dos":                    dos,
            "stockout_risk":          round(risk, 4),
            "replenishment_needed":   replenishment_needed,
            "policy_status":          status,
        })

    return pd.DataFrame(rows)


# ════════════════════════════════════════════════════════════════════════════
# SUMMARY KPIs
# ════════════════════════════════════════════════════════════════════════════

def compute_inventory_kpis(policy_df: pd.DataFrame) -> dict:
    """
    Aggregate policy results into system-level KPIs.
    """
    total = len(policy_df)
    if total == 0:
        return {}

    return {
        # Stock coverage
        "avg_dos":                round(policy_df["dos"].replace(999, np.nan).mean(), 1),
        "median_dos":             round(policy_df["dos"].replace(999, np.nan).median(), 1),

        # Risk
        "pct_critical":           round((policy_df["policy_status"] == "CRITICAL").mean() * 100, 1),
        "pct_reorder":            round((policy_df["policy_status"] == "REORDER").mean() * 100, 1),
        "pct_ok":                 round((policy_df["policy_status"] == "OK").mean() * 100, 1),
        "pct_overstock":          round((policy_df["policy_status"] == "OVERSTOCK").mean() * 100, 1),
        "n_replenishment_needed": int(policy_df["replenishment_needed"].sum()),

        # Value
        "total_inventory_value_b_vnd": round(policy_df["inventory_value_vnd"].sum() / 1e9, 2),
        "avg_stockout_risk_pct":       round(policy_df["stockout_risk"].mean() * 100, 2),

        # Safety stock coverage
        "avg_safety_stock":       round(policy_df["safety_stock"].mean(), 1),
        "avg_eoq":                round(policy_df["eoq"].mean(), 1),

        "total_skus":             total,
    }
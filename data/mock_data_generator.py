"""
==============================================================
SUPPLY CHAIN PLANNING SYSTEM – MOCK DATA GENERATOR
==============================================================
Generates all input datasets required by the planning system:
  1. Product Master (SKU Master)
  2. Warehouse Master
  3. Sales History (180 days)
  4. Inventory Snapshot
  5. Lead Time Data
  6. Supplier Master
  7. Promotion Calendar
  8. Cost Parameters
==============================================================
FIX (Streamlit Cloud):
  output_dir default now uses Path(__file__) — absolute path,
  not relative to CWD.  Works on any machine / cloud env.
==============================================================
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path

# ──────────────────────────────────────────────
# BUSINESS CONSTANTS
# ──────────────────────────────────────────────

WAREHOUSES = [
    "WH-001",
    "WH-002",
    "WH-003",
    "WH-004",
    "WH-005",
    "WH-006",
    "WH-007",
]

REGIONS = {
    "WH-001": "South",
    "WH-002": "North",
    "WH-003": "Central",
    "WH-004": "South",
    "WH-005": "North",
    "WH-006": "Central",
    "WH-007": "South",
}

SKU_GROUPS = [
    "Group A",
    "Group B",
    "Group C",
    "Group D",
]

INV_TYPES = [
    "Finished Goods",
    "Third-Party Storage",
    "Raw Material",
]

SUPPLIERS = ["SUP-001", "SUP-002", "SUP-003", "SUP-004"]

START_DATE   = pd.Timestamp("2024-01-01")
HISTORY_DAYS = 180   # ~6 months of daily sales history
N_SKU        = 30

# ──────────────────────────────────────────────
# RNG HELPER
# ──────────────────────────────────────────────

def _rng(seed: int = 42) -> np.random.Generator:
    return np.random.default_rng(seed)


# ══════════════════════════════════════════════
# 1. PRODUCT MASTER
# ══════════════════════════════════════════════

def generate_product_master(n_sku: int = N_SKU, seed: int = 42) -> pd.DataFrame:
    """
    Output columns:
      sku | sku_group | inventory_type | unit_cost_vnd | volume_cbm_per_unit
      | weight_kg | shelf_life_days | abc_class | is_active
    """
    r = _rng(seed)
    rows = []

    for i in range(n_sku):
        grp      = r.choice(SKU_GROUPS)
        inv_type = r.choice(INV_TYPES)

        cost_range = {
            "Group A": (18_000, 25_000),
            "Group B": (9_000,  15_000),
            "Group C": (22_000, 32_000),
            "Group D": (7_000,  12_000),
        }[grp]

        unit_cost  = round(float(r.uniform(*cost_range)), 0)
        volume     = round(float(r.uniform(0.02, 0.08)), 4)
        weight     = round(float(r.uniform(1.0, 25.0)), 2)
        shelf_life = int(r.choice([90, 180, 365, 730]))

        if unit_cost >= 20_000:
            abc = "A"
        elif unit_cost >= 12_000:
            abc = "B"
        else:
            abc = "C"

        rows.append({
            "sku":                 f"SKU-{1000 + i}",
            "sku_group":           grp,
            "inventory_type":      inv_type,
            "unit_cost_vnd":       unit_cost,
            "volume_cbm_per_unit": volume,
            "weight_kg":           weight,
            "shelf_life_days":     shelf_life,
            "abc_class":           abc,
            "is_active":           bool(r.choice([True, True, True, False])),
        })

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════
# 2. WAREHOUSE MASTER
# ══════════════════════════════════════════════

def generate_warehouse_master(seed: int = 42) -> pd.DataFrame:
    """
    Output columns:
      warehouse | region | capacity_cbm | capacity_tons
      | rental_cost_per_month_vnd | is_active
    """
    r    = _rng(seed + 200)
    rows = []

    for wh in WAREHOUSES:
        cap_cbm  = int(r.integers(3_000, 10_000))
        cap_tons = round(cap_cbm * float(r.uniform(0.4, 0.6)), 1)
        rental   = round(float(r.uniform(50_000_000, 300_000_000)), 0)

        rows.append({
            "warehouse":                 wh,
            "region":                    REGIONS[wh],
            "capacity_cbm":              cap_cbm,
            "capacity_tons":             cap_tons,
            "rental_cost_per_month_vnd": rental,
            "is_active":                 True,
        })

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════
# 3. SUPPLIER MASTER
# ══════════════════════════════════════════════

def generate_supplier_master(seed: int = 42) -> pd.DataFrame:
    """
    Output columns:
      supplier_id | supplier_name | region | reliability_score
      | min_order_qty | payment_terms_days
    """
    r = _rng(seed + 500)

    data = {
        "supplier_id":        SUPPLIERS,
        "supplier_name":      ["Supplier Alpha", "Supplier Beta", "Supplier Gamma", "Supplier Delta"],
        "region":             ["South", "South", "Central", "North"],
        "reliability_score":  [round(float(r.uniform(0.80, 0.99)), 2) for _ in SUPPLIERS],
        "min_order_qty":      [int(r.integers(500, 3000))              for _ in SUPPLIERS],
        "payment_terms_days": [int(r.choice([30, 45, 60]))             for _ in SUPPLIERS],
    }

    return pd.DataFrame(data)


# ══════════════════════════════════════════════
# 4. LEAD TIME DATA  (SKU × Supplier)
# ══════════════════════════════════════════════

def generate_lead_time(
    product_master: pd.DataFrame,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Output columns:
      sku | supplier_id | lead_time_days | lead_time_std_days
      | transit_days | review_period_days
    """
    r    = _rng(seed + 600)
    rows = []

    for _, row in product_master.iterrows():
        n_sup = int(r.integers(1, 3))
        sups  = r.choice(SUPPLIERS, size=n_sup, replace=False)

        for sup in sups:
            lt      = int(r.integers(3, 21))
            lt_std  = round(float(r.uniform(0.5, 3.0)), 1)
            transit = int(r.integers(1, 5))

            rows.append({
                "sku":                row["sku"],
                "supplier_id":        sup,
                "lead_time_days":     lt,
                "lead_time_std_days": lt_std,
                "transit_days":       transit,
                "review_period_days": int(r.choice([7, 14, 30])),
            })

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════
# 5. SALES HISTORY
# ══════════════════════════════════════════════

def _demand_signal(
    day_idx: int,
    base: float,
    trend: float,
    season_amp: float,
    season_period: int,
    r: np.random.Generator,
    noise_std: float,
) -> float:
    """Combine trend + seasonality + noise into a daily demand signal."""
    trend_component  = base + trend * day_idx
    season_component = season_amp * np.sin(2 * np.pi * day_idx / season_period)
    noise            = float(r.normal(0, noise_std))
    return max(0.0, trend_component + season_component + noise)


def generate_sales_history(
    product_master: pd.DataFrame,
    history_days: int = HISTORY_DAYS,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Realistic daily sales per SKU × Warehouse.
    Output columns:
      date | sku | warehouse | sales_qty | demand_qty
      | sales_value_vnd | is_promotion_day | stockout_flag | month
    """
    r     = _rng(seed + 300)
    rows  = []
    dates = [START_DATE + pd.Timedelta(days=d) for d in range(history_days)]

    sku_params: dict[str, dict] = {}
    for _, row in product_master.iterrows():
        sku = row["sku"]
        sku_params[sku] = {
            "base":          float(r.uniform(50, 400)),
            "trend":         float(r.uniform(-0.5, 1.0)),
            "season_amp":    float(r.uniform(10, 80)),
            "season_period": int(r.choice([30, 90])),
            "noise_std":     float(r.uniform(10, 60)),
        }

    promo_days = set(
        r.choice(history_days, size=int(history_days * 0.15), replace=False).tolist()
    )

    for day_idx, date in enumerate(dates):
        is_promo         = day_idx in promo_days
        promo_multiplier = float(r.uniform(1.2, 1.8)) if is_promo else 1.0

        for wh in WAREHOUSES:
            wh_scale = float(r.uniform(0.5, 1.5))

            for _, row in product_master.iterrows():
                if not row["is_active"]:
                    continue

                sku = row["sku"]
                p   = sku_params[sku]

                demand_raw = _demand_signal(
                    day_idx       = day_idx,
                    base          = p["base"],
                    trend         = p["trend"],
                    season_amp    = p["season_amp"],
                    season_period = p["season_period"],
                    r             = r,
                    noise_std     = p["noise_std"],
                )

                demand = int(demand_raw * wh_scale * promo_multiplier)

                stockout  = bool(r.random() < 0.05)
                sales_qty = int(demand * float(r.uniform(0.0, 0.7))) if stockout else demand

                rows.append({
                    "date":             date,
                    "sku":              sku,
                    "warehouse":        wh,
                    "sales_qty":        sales_qty,
                    "demand_qty":       demand,
                    "sales_value_vnd":  sales_qty * row["unit_cost_vnd"],
                    "is_promotion_day": is_promo,
                    "stockout_flag":    stockout,
                })

    df         = pd.DataFrame(rows)
    df["month"] = df["date"].dt.to_period("M").astype(str)
    return df


# ══════════════════════════════════════════════
# 6. CURRENT INVENTORY SNAPSHOT
# ══════════════════════════════════════════════

def generate_inventory_snapshot(
    product_master: pd.DataFrame,
    snapshot_date: pd.Timestamp | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Output columns:
      snapshot_date | sku | warehouse | on_hand_units
      | unit_cost_vnd | inventory_value_vnd | used_cbm
      | inventory_type | lot_number | expiry_date
    """
    r         = _rng(seed + 100)
    snap_date = snapshot_date or (START_DATE + pd.Timedelta(days=HISTORY_DAYS - 1))
    rows      = []

    for wh in WAREHOUSES:
        for _, row in product_master.iterrows():
            if not row["is_active"]:
                continue

            qty    = int(r.integers(100, 3_000))
            lot    = f"LOT-{r.integers(10000, 99999)}"
            expiry = snap_date + pd.Timedelta(
                days=int(r.integers(30, row["shelf_life_days"]))
            )

            rows.append({
                "snapshot_date":       snap_date,
                "sku":                 row["sku"],
                "warehouse":           wh,
                "on_hand_units":       qty,
                "unit_cost_vnd":       row["unit_cost_vnd"],
                "inventory_value_vnd": qty * row["unit_cost_vnd"],
                "used_cbm":            round(qty * row["volume_cbm_per_unit"], 3),
                "inventory_type":      row["inventory_type"],
                "lot_number":          lot,
                "expiry_date":         expiry,
            })

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════
# 7. PROMOTION CALENDAR
# ══════════════════════════════════════════════

def generate_promotion_calendar(
    product_master: pd.DataFrame,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Output columns:
      promo_id | sku | warehouse | promo_start | promo_end
      | uplift_pct | promo_type
    """
    r           = _rng(seed + 700)
    rows        = []
    promo_types = ["Price Off", "Bundle", "Display", "Flash Sale"]
    active_skus = product_master[product_master["is_active"]]["sku"].tolist()

    for i in range(40):
        sku          = r.choice(active_skus)
        wh           = r.choice(WAREHOUSES)
        start_offset = int(r.integers(0, 90))
        duration     = int(r.integers(3, 14))
        start        = START_DATE + pd.Timedelta(days=HISTORY_DAYS + start_offset)
        end          = start + pd.Timedelta(days=duration)

        rows.append({
            "promo_id":    f"PROMO-{i+1:03d}",
            "sku":         sku,
            "warehouse":   wh,
            "promo_start": start,
            "promo_end":   end,
            "uplift_pct":  round(float(r.uniform(10, 60)), 1),
            "promo_type":  r.choice(promo_types),
        })

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════
# 8. COST PARAMETERS
# ══════════════════════════════════════════════

def generate_cost_parameters(
    product_master: pd.DataFrame,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Output columns:
      sku | holding_cost_pct | ordering_cost_vnd
      | stockout_cost_per_unit_vnd | target_service_level
    """
    r    = _rng(seed + 800)
    rows = []

    for _, row in product_master.iterrows():
        h_cost = round(float(r.uniform(0.15, 0.30)), 3)

        sl_target = {
            "A": round(float(r.uniform(0.97, 0.99)), 3),
            "B": round(float(r.uniform(0.93, 0.97)), 3),
            "C": round(float(r.uniform(0.88, 0.93)), 3),
        }[row["abc_class"]]

        rows.append({
            "sku":                        row["sku"],
            "holding_cost_pct":           h_cost,
            "ordering_cost_vnd":          round(float(r.uniform(500_000, 5_000_000)), 0),
            "stockout_cost_per_unit_vnd": round(float(r.uniform(2_000, 15_000)), 0),
            "target_service_level":       sl_target,
        })

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════
# MASTER RUNNER
# ══════════════════════════════════════════════

# FIX: default output_dir uses absolute path relative to THIS file
# so it works on Streamlit Cloud regardless of CWD
_DEFAULT_OUTPUT_DIR = str(Path(__file__).resolve().parent / "outputs")


def generate_all(
    seed: int = 42,
    save_csv: bool = False,
    output_dir: str = _DEFAULT_OUTPUT_DIR,   # ← FIXED (was "data/outputs")
) -> dict[str, pd.DataFrame]:
    """
    Generate every input dataset and return as a dictionary.
    Optionally save to CSV files.
    """
    print("⚙️  Generating Supply Chain mock data …")

    product_master   = generate_product_master(seed=seed)
    warehouse_master = generate_warehouse_master(seed=seed)
    supplier_master  = generate_supplier_master(seed=seed)
    lead_time        = generate_lead_time(product_master, seed=seed)
    sales_history    = generate_sales_history(product_master, seed=seed)
    inventory        = generate_inventory_snapshot(product_master, seed=seed)
    promo_calendar   = generate_promotion_calendar(product_master, seed=seed)
    cost_params      = generate_cost_parameters(product_master, seed=seed)

    datasets = {
        "product_master":   product_master,
        "warehouse_master": warehouse_master,
        "supplier_master":  supplier_master,
        "lead_time":        lead_time,
        "sales_history":    sales_history,
        "inventory":        inventory,
        "promo_calendar":   promo_calendar,
        "cost_parameters":  cost_params,
    }

    print(f"\n{'='*55}")
    print(f"{'Dataset':<25} {'Rows':>8} {'Cols':>6}")
    print(f"{'='*55}")
    for name, df in datasets.items():
        print(f"  {name:<23} {len(df):>8,} {len(df.columns):>6}")
    print(f"{'='*55}\n")

    if save_csv:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        for name, df in datasets.items():
            path = out / f"{name}.csv"
            df.to_csv(path, index=False)
            print(f"  ✅  Saved → {path}")

    return datasets


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

if __name__ == "__main__":
    # FIX: always save relative to THIS file, not shell CWD
    _out = str(Path(__file__).resolve().parent / "outputs")
    datasets = generate_all(seed=42, save_csv=True, output_dir=_out)

    print("\n📋  Product Master sample:")
    print(datasets["product_master"].head(5).to_string(index=False))

    print("\n📋  Sales History sample (last 3 rows):")
    print(datasets["sales_history"].tail(3).to_string(index=False))

    print("\n📋  Inventory Snapshot sample:")
    print(datasets["inventory"].head(5).to_string(index=False))

    print("\n📋  Lead Time sample:")
    print(datasets["lead_time"].head(5).to_string(index=False))

    print("\n📋  Cost Parameters sample:")
    print(datasets["cost_parameters"].head(5).to_string(index=False))
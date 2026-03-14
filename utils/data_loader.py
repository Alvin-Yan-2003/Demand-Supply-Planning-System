"""
utils/data_loader.py  —  FIXED for Streamlit Cloud deployment
──────────────────────────────────────────────────────────────
Changes vs original:
  1. DATA_DIR resolved via Path(__file__) — immune to CWD differences
  2. _ensure_data() auto-generates mock data on first boot if CSV missing
  3. _load() never raises unhandled exception — logs clearly instead
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# ── Always resolve from THIS file, not from CWD ──────────────────────────
ROOT     = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "outputs"


def _ensure_data() -> None:
    """Auto-generate mock data if data/outputs/ is missing (Streamlit Cloud)."""
    if (DATA_DIR / "sales_history.csv").exists():
        return

    print(f"[data_loader] data/outputs/ missing — auto-generating mock data …")
    print(f"[data_loader] Target dir: {DATA_DIR}")

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    try:
        from data.mock_data_generator import generate_all  # type: ignore
        generate_all(seed=42, save_csv=True, output_dir=str(DATA_DIR))
        print("[data_loader] ✅  Mock data generated successfully.")
    except Exception as exc:
        print(f"[data_loader] ⚠️  Auto-generate failed: {exc}")
        print("[data_loader]    Commit data/outputs/*.csv to your Git repo.")


def _load(filename: str, parse_dates: list[str] | None = None) -> pd.DataFrame:
    _ensure_data()
    path = DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"[data_loader] File not found: {path}\n"
            f"  DATA_DIR = {DATA_DIR}\n"
            "  Run `python data/mock_data_generator.py` then commit the CSVs."
        )
    return pd.read_csv(path, parse_dates=parse_dates)


def load_sales_history() -> pd.DataFrame:
    df = _load("sales_history.csv", parse_dates=["date"])
    df.columns = df.columns.str.lower()
    return df

def load_product_master() -> pd.DataFrame:
    df = _load("product_master.csv")
    df.columns = df.columns.str.lower()
    return df

def load_warehouse_master() -> pd.DataFrame:
    df = _load("warehouse_master.csv")
    df.columns = df.columns.str.lower()
    return df

def load_inventory() -> pd.DataFrame:
    df = _load("inventory.csv", parse_dates=["snapshot_date", "expiry_date"])
    df.columns = df.columns.str.lower()
    return df

def load_lead_time() -> pd.DataFrame:
    df = _load("lead_time.csv")
    df.columns = df.columns.str.lower()
    return df

def load_supplier_master() -> pd.DataFrame:
    df = _load("supplier_master.csv")
    df.columns = df.columns.str.lower()
    return df

def load_promo_calendar() -> pd.DataFrame:
    df = _load("promo_calendar.csv", parse_dates=["promo_start", "promo_end"])
    df.columns = df.columns.str.lower()
    return df

def load_cost_parameters() -> pd.DataFrame:
    df = _load("cost_parameters.csv")
    df.columns = df.columns.str.lower()
    return df

def load_all() -> dict[str, pd.DataFrame]:
    return {
        "sales_history":    load_sales_history(),
        "product_master":   load_product_master(),
        "warehouse_master": load_warehouse_master(),
        "inventory":        load_inventory(),
        "lead_time":        load_lead_time(),
        "supplier_master":  load_supplier_master(),
        "promo_calendar":   load_promo_calendar(),
        "cost_parameters":  load_cost_parameters(),
    }
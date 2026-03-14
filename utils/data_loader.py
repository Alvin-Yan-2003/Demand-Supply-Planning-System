"""
utils/data_loader.py
────────────────────
Central data loader – reads all CSV inputs from data/outputs/
and returns typed, normalized DataFrames.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "outputs"


def _load(filename: str, parse_dates: list[str] | None = None) -> pd.DataFrame:
    path = DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            "Run `python data/mock_data_generator.py` first."
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
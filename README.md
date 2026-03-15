# 📦 Demand–Supply Planning System

> **An end-to-end Supply Chain Planning prototype** built with Python — simulating the core planning engine of modern platforms like SAP IBP, Oracle SCM, and Kinaxis RapidResponse.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://demand-supply-planning.streamlit.app)
&nbsp;
![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32-FF4B4B?logo=streamlit&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-189AB4?logo=xgboost&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-5.18-3F4F75?logo=plotly&logoColor=white)

---

## 🎯 The Problem This System Solves

In real supply chain operations, planners face three interconnected decisions every day:

- **How much demand will we see next month?** — Inaccurate forecasts lead to either stockouts (lost revenue) or overstock (tied-up capital)
- **How much safety stock should we hold?** — Too little means stockouts; too much means waste
- **When and how much should we replenish?** — Late orders disrupt service; early orders inflate costs

This system addresses all three with a **data-driven, automated planning pipeline** — replacing manual spreadsheet-based planning with a unified, real-time decision engine.

---

## 🏗️ System Architecture

The system is structured as **three integrated planning layers**, mirroring the architecture of enterprise S&OP (Sales & Operations Planning) platforms:

```
┌─────────────────────────────────────────────────────────┐
│                  DATA INPUTS                            │
│  Sales History · Product Master · Warehouse Master      │
│  Inventory Snapshot · Lead Times · Supplier Master      │
└──────────────────────┬──────────────────────────────────┘
                       │
          ┌────────────▼────────────┐
          │   LAYER 1: FORECASTING  │
          │  Moving Average         │
          │  Exponential Smoothing  │
          │  XGBoost (ML)           │
          │  → Auto best per SKU×WH │
          └────────────┬────────────┘
                       │  forecast_qty · accuracy · MAPE
          ┌────────────▼────────────┐
          │  LAYER 2: INVENTORY     │
          │  Safety Stock (Z×σ×√LT) │
          │  Reorder Point (ROP)    │
          │  EOQ Optimization       │
          │  ABC Classification     │
          └────────────┬────────────┘
                       │  policy · DOS · stockout risk
          ┌────────────▼────────────┐
          │  LAYER 3: REPLENISHMENT │
          │  Order Quantity         │
          │  Supplier Selection     │
          │  Capacity Constraints   │
          │  Priority Scoring       │
          └────────────┬────────────┘
                       │
          ┌────────────▼────────────┐
          │   STREAMLIT DASHBOARD   │
          │  Overview · Forecast    │
          │  Inventory · Supply     │
          └─────────────────────────┘
```

---

## 📊 Planning Layers — Deep Dive

### Layer 1 — Demand Forecasting

**Objective:** Predict future product demand at SKU × Warehouse × Day granularity.

The system runs **3 models in parallel** and auto-selects the best per SKU×Warehouse based on MAPE:

| Model | Method | Best For |
|---|---|---|
| **Moving Average** | Rolling window mean | Stable, low-volatility SKUs |
| **Exponential Smoothing** | Weighted recent history | SKUs with trend or mild seasonality |
| **XGBoost** | Gradient boosting with lag features | Complex patterns, promotional uplift |

**Key outputs:** `forecast_qty`, `upper_bound`, `lower_bound` (confidence interval), MAPE per SKU×WH

**Accuracy evaluation:** Walk-forward cross-validation on 30-day holdout period

---

### Layer 2 — Inventory Planning

**Objective:** Calculate optimal inventory policies to maintain service level while minimizing holding cost.

**Safety Stock formula** (accounts for both demand and lead time variability):

```
SS = Z × √(LT × σ_demand² + demand_avg² × σ_LT²)
```

Where:
- `Z` = service level z-score (e.g. 95% SL → Z = 1.645)
- `LT` = average lead time (days)
- `σ_demand` = demand standard deviation
- `σ_LT` = lead time standard deviation

**Reorder Point:**
```
ROP = demand_avg × lead_time_avg + Safety Stock
```

**Economic Order Quantity (EOQ):**
```
EOQ = √(2 × D × S / (H × C))
```

Where `D` = annual demand, `S` = ordering cost, `H` = holding cost rate, `C` = unit cost

**Policy Status classification:**

| Status | Condition | Action |
|---|---|---|
| 🔴 CRITICAL | on_hand ≤ safety_stock | Immediate replenishment |
| 🟡 REORDER | safety_stock < on_hand ≤ ROP | Place order now |
| 🟢 OK | ROP < on_hand ≤ max_stock | Monitor |
| 🔵 OVERSTOCK | on_hand > max_stock × 1.2 | Review / defer orders |

---

### Layer 3 — Supply & Replenishment Planning

**Objective:** Generate prioritized replenishment orders with supplier and capacity constraints.

**Planning logic:**
1. Identify SKU×WH where `on_hand ≤ ROP`
2. Calculate `replenishment_qty = max_stock − on_hand`
3. Apply supplier **MOQ** (Minimum Order Quantity) rounding
4. Check **warehouse CBM capacity** — flag if insufficient
5. Assign `order_date` and `expected_arrival` (+ lead time days)
6. **Priority scoring:** CRITICAL > REORDER, then ABC class (A > B > C), then stockout risk

**Constraints handled:**
- Supplier lead time variability
- Warehouse storage capacity (CBM)
- Minimum order quantities (MOQ)
- Multi-supplier selection (shortest lead time)

---

## 📈 Dashboard Overview

The Streamlit dashboard provides **4 interactive planning views**:

| Tab | Key Features |
|---|---|
| **📊 Overview** | System KPIs · Policy distribution · Inventory value by warehouse · Critical alerts |
| **🔮 Forecast** | SKU-level demand chart · Model comparison · MAPE by model · Accuracy table |
| **📦 Inventory** | What-if analysis (lead time / service level) · DOS heatmap · EOQ scatter · Policy table |
| **🚚 Supply** | Order value by warehouse · Arrival schedule · Supplier breakdown · Full replenishment plan |

**Global filters:** Warehouse · ABC Class · Sales History Range

**What-if parameters:** Lead Time Override · Service Level Target · Safety Buffer % · Review Period

---

## 🗂️ Project Structure

```
Demand-Supply-Planning-System/
│
├── data/
│   ├── mock_data_generator.py    # Generates all simulation datasets
│   ├── DATA_schema.md            # Full data dictionary
│   └── outputs/                  # Generated CSVs
│
├── forecast/
│   ├── models.py                 # MA, Exp Smoothing, XGBoost
│   ├── evaluator.py              # MAPE, RMSE, MAE, Bias
│   ├── runner.py                 # Forecast pipeline orchestrator
│   └── outputs/                  # forecast_qty, accuracy, model_selection
│
├── inventory/
│   ├── policy.py                 # Safety Stock, ROP, EOQ calculations
│   ├── runner.py                 # Inventory pipeline orchestrator
│   └── outputs/                  # inventory_policy, inventory_kpis
│
├── supply/
│   ├── replenishment.py          # Order generation + constraints
│   ├── runner.py                 # Supply pipeline orchestrator
│   └── outputs/                  # replenishment_plan, supply_kpis
│
├── dashboard/
│   ├── app.py                    # Streamlit dashboard (4 tabs)
│   └── requirements.txt
│
├── utils/
│   └── data_loader.py            # Centralized data loading helpers
│
└── main.py                       # End-to-end pipeline runner
```

---

## 📦 Dataset

Simulated dataset representing a **Vietnamese FMCG distribution network**:

| Dataset | Rows | Description |
|---|---|---|
| Sales History | 30,240 | 180 days × 30 SKUs × 7 warehouses |
| Product Master | 30 | SKUs across 4 categories (Premium/Standard/Organic/Industrial) |
| Warehouse Master | 7 | Warehouses across North/Central/South regions |
| Supplier Master | 4 | Suppliers with reliability scores and MOQ |
| Lead Time | 46 | SKU × Supplier lead time with variability |
| Inventory Snapshot | 168 | Current on-hand by SKU × Warehouse |
| Promo Calendar | 40 | Forward-looking promotions with uplift % |

> Sales history includes `demand_qty` (true demand) vs `sales_qty` (after stockout) — enabling **unbiased forecast training**.

---

## ⚙️ How to Run

### Option 1 — Run via Streamlit Cloud
👉 **[Live Demo](https://demand-supply-planning.streamlit.app)**

### Option 2 — Run locally

```bash
# 1. Clone the repo
git clone https://github.com/Alvin-Yan-2003/Demand-Supply-Planning-System.git
cd Demand-Supply-Planning-System

# 2. Install dependencies
pip install -r dashboard/requirements.txt

# 3. Generate mock data
python data/mock_data_generator.py

# 4. Run the full planning pipeline
python main.py

# 5. Launch the dashboard
streamlit run dashboard/app.py
```

### Pipeline options

```bash
# Auto-select best forecast model per SKU×Warehouse
python main.py

# Use a specific forecast model
python main.py --forecast-model xgboost
python main.py --forecast-model moving_average
python main.py --forecast-model exponential_smoothing

# Skip forecast re-run (use cached output)
python main.py --skip-forecast
```

---

## 🛠️ Tech Stack

| Layer | Libraries |
|---|---|
| **Data Processing** | `pandas`, `numpy` |
| **Forecasting** | `scikit-learn`, `xgboost`, `statsmodels` |
| **Statistics** | `scipy` |
| **Visualization** | `plotly`, `streamlit` |
| **Pipeline** | Pure Python modular architecture |

---

## 📌 Key Design Decisions

**Why auto model selection?**
Different SKUs have different demand patterns. A one-size-fits-all model underperforms — auto-selection picks Moving Average for stable SKUs, XGBoost for promotional/volatile SKUs, resulting in lower overall MAPE.

**Why separate Safety Stock formula for lead time variability?**
Standard `SS = Z × σ × √LT` ignores lead time uncertainty. This system uses the full formula accounting for both demand and lead time std — more realistic for multi-supplier environments.

**Why ABC classification?**
Prioritizing replenishment by ABC ensures Class A items (high-value, high-velocity) are never stocked out, while Class C items can tolerate lower service levels — matching real-world inventory policy practice.

---

## 👤 Author

**Nim Hung Hoan** — Aspiring Supply Chain Analyst / Planner

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?logo=linkedin)](https://www.linkedin.com/in/hhoan1811/)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?logo=github)](https://github.com/Alvin-Yan-2003)

---

*This project is a portfolio simulation. All data is synthetically generated and does not represent any real company or supply chain.*
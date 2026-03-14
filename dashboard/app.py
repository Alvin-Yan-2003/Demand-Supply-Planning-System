"""
dashboard/app.py
────────────────
Supply Chain Planning System – Streamlit Dashboard

Sidebar  : Global filters + Forecast settings + Inventory/Supply parameters
Tab 1    : 📊 Overview      — System-wide KPI summary
Tab 2    : 🔮 Forecast      — Demand forecast & accuracy
Tab 3    : 📦 Inventory     — Policy table, DOS, risk
Tab 4    : 🚚 Supply        — Replenishment plan

FIX (Streamlit Cloud):
  - _ensure_pipeline() auto-generates data + runs pipeline on first boot
  - load_data() uses ROOT / Path(__file__) — absolute paths, immune to CWD

Run:
  streamlit run dashboard/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ══════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Supply Chain Planning",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════
# THEME & GLOBAL CSS
# ══════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=Syne:wght@400;600;700;800&display=swap');

:root {
    --bg:        #0a0d12;
    --bg-card:   #111620;
    --bg-hover:  #181e2a;
    --border:    #1e2736;
    --accent:    #00d4ff;
    --accent2:   #ff6b35;
    --accent3:   #7fff6b;
    --warn:      #ffb830;
    --danger:    #ff4444;
    --text:      #e2e8f0;
    --muted:     #64748b;
    --font-head: 'Syne', sans-serif;
    --font-mono: 'DM Mono', monospace;
}

html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: var(--font-mono) !important;
}
[data-testid="stSidebar"] {
    background: #0d1117 !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }

/* ── KPI cards ── */
.kpi-grid { display: flex; flex-wrap: wrap; gap: 12px; margin: 8px 0 20px; }
.kpi-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 14px 18px;
    min-width: 150px;
    flex: 1;
}
.kpi-card.accent  { border-top: 3px solid var(--accent); }
.kpi-card.accent2 { border-top: 3px solid var(--accent2); }
.kpi-card.accent3 { border-top: 3px solid var(--accent3); }
.kpi-card.warn    { border-top: 3px solid var(--warn); }
.kpi-card.danger  { border-top: 3px solid var(--danger); }
.kpi-label { font-size: 0.68rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.08em; }
.kpi-value { font-size: 1.55rem; font-weight: 700; font-family: var(--font-head); color: var(--text); line-height: 1.2; }
.kpi-sub   { font-size: 0.7rem; color: var(--muted); margin-top: 2px; }

/* ── Section titles ── */
.section-title {
    font-family: var(--font-head);
    font-size: 0.78rem;
    font-weight: 600;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin: 24px 0 10px;
    padding-bottom: 6px;
    border-bottom: 1px solid var(--border);
}

/* ── Sidebar ── */
.sidebar-brand { font-family: var(--font-head); font-size: 1.1rem; font-weight: 800; color: var(--accent); margin-bottom: 2px; }
.sidebar-sub   { font-size: 0.65rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 16px; }
.sidebar-section { font-size: 0.65rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.1em; margin: 14px 0 6px; }

/* ── Page title ── */
.page-title {
    font-family: var(--font-head);
    font-size: 2rem;
    font-weight: 800;
    background: linear-gradient(135deg, var(--accent) 0%, #4fc3f7 50%, var(--accent3) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0;
    line-height: 1.1;
}
.page-subtitle {
    font-size: 0.72rem;
    color: var(--muted);
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-top: 0.25rem;
    margin-bottom: 1.5rem;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# PLOTLY THEME
# ══════════════════════════════════════════════════════════════════════════

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Mono, monospace", color="#94a3b8", size=11),
    title_font=dict(family="Syne, sans-serif", color="#e2e8f0", size=13),
    xaxis=dict(gridcolor="#1e2736", linecolor="#1e2736", tickcolor="#1e2736"),
    yaxis=dict(gridcolor="#1e2736", linecolor="#1e2736", tickcolor="#1e2736"),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#1e2736"),
    margin=dict(l=40, r=20, t=40, b=40),
    colorway=["#00d4ff", "#ff6b35", "#7fff6b", "#ffb830", "#a78bfa", "#f472b6"],
)

STATUS_COLORS = {
    "CRITICAL":  "#ff4444",
    "REORDER":   "#ffb830",
    "OK":        "#7fff6b",
    "OVERSTOCK": "#00d4ff",
}
STATUS_FILL_COLORS = {
    "CRITICAL":  "rgba(255, 68,  68,  0.15)",
    "REORDER":   "rgba(255, 184, 48,  0.15)",
    "OK":        "rgba(127, 255, 107, 0.15)",
    "OVERSTOCK": "rgba(0,   212, 255, 0.15)",
}
MODEL_COLORS = {
    "moving_average":        "#ffb830",
    "exponential_smoothing": "#a78bfa",
    "xgboost":               "#00d4ff",
}


# ══════════════════════════════════════════════════════════════════════════
# AUTO-INIT  (FIX: generates data + pipeline on first Streamlit Cloud boot)
# ══════════════════════════════════════════════════════════════════════════

def _ensure_pipeline() -> None:
    """
    If CSV outputs are missing (fresh Streamlit Cloud deployment),
    auto-generate mock data then run the full planning pipeline.
    This makes the app fully self-contained — no manual pre-step needed.
    """
    data_csv   = ROOT / "data"      / "outputs" / "sales_history.csv"
    policy_csv = ROOT / "inventory" / "outputs" / "inventory_policy.csv"
    plan_csv   = ROOT / "supply"    / "outputs" / "replenishment_plan.csv"
    fc_csv     = ROOT / "forecast"  / "outputs" / "forecast_qty.csv"

    # Step 1 — raw data
    if not data_csv.exists():
        with st.spinner("⚙️  Generating mock data (first boot) …"):
            try:
                from data.mock_data_generator import generate_all  # type: ignore
                generate_all(
                    seed=42,
                    save_csv=True,
                    output_dir=str(ROOT / "data" / "outputs"),
                )
            except Exception as exc:
                st.error(f"❌ Data generation failed: {exc}")
                return

    # Step 2 — pipeline outputs
    if not all(p.exists() for p in [policy_csv, plan_csv, fc_csv]):
        with st.spinner("🔄  Running planning pipeline (~30 s, first boot only) …"):
            try:
                from main import run_pipeline  # type: ignore
                run_pipeline(forecast_model="moving_average", save=True)
                st.toast("✅ Pipeline ready!", icon="🚀")
            except Exception as exc:
                st.warning(
                    f"⚠️  Auto-pipeline failed: {exc}\n\n"
                    "**Quick fix:** run `python main.py` locally, "
                    "then commit all `*/outputs/*.csv` files to your repo and redeploy."
                )


# ══════════════════════════════════════════════════════════════════════════
# DATA LOADER  (FIX: absolute paths via ROOT / Path(__file__))
# ══════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner="Loading planning data …")
def load_data() -> dict[str, pd.DataFrame]:
    """Load all pipeline CSVs. Returns empty DataFrames on missing files."""
    out: dict[str, pd.DataFrame] = {}

    paths = {
        "policy":          ROOT / "inventory" / "outputs" / "inventory_policy.csv",
        "inv_kpis":        ROOT / "inventory" / "outputs" / "inventory_kpis.csv",
        "plan":            ROOT / "supply"    / "outputs" / "replenishment_plan.csv",
        "supply_kpis":     ROOT / "supply"    / "outputs" / "supply_kpis.csv",
        "forecast":        ROOT / "forecast"  / "outputs" / "forecast_qty.csv",
        "accuracy":        ROOT / "forecast"  / "outputs" / "forecast_accuracy.csv",
        "summary":         ROOT / "forecast"  / "outputs" / "forecast_summary.csv",
        "model_selection": ROOT / "forecast"  / "outputs" / "model_selection.csv",
        "sales":           ROOT / "data"      / "outputs" / "sales_history.csv",
    }
    for key, path in paths.items():
        if path.exists():
            df = pd.read_csv(path)
            for col in ["date", "order_date"]:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col])
            if "expected_arrival" in df.columns:
                df["expected_arrival"] = pd.to_datetime(df["expected_arrival"])
            out[key] = df
        else:
            out[key] = pd.DataFrame()
    return out


# ══════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════

def kpi_card(label: str, value: str, sub: str = "", color: str = "accent") -> str:
    return f"""<div class="kpi-card {color}">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-sub">{sub}</div>
    </div>"""

def section(title: str) -> None:
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)

def sb_section(title: str) -> None:
    st.markdown(f'<div class="sidebar-section">{title}</div>', unsafe_allow_html=True)

def safe_layout(extra: dict) -> dict:
    layout = {k: v for k, v in PLOTLY_LAYOUT.items() if k not in extra}
    layout.update(extra)
    return layout


# ══════════════════════════════════════════════════════════════════════════
# ── SIDEBAR ───────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown('<div class="sidebar-brand">⬡ SC Planning</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-sub">Demand-Supply System</div>', unsafe_allow_html=True)

    # FIX: run auto-init before loading data
    _ensure_pipeline()

    DATA       = load_data()
    policy_raw = DATA.get("policy", pd.DataFrame())
    plan_raw   = DATA.get("plan",   pd.DataFrame())
    sales_raw  = DATA.get("sales",  pd.DataFrame())

    # ── 📅 Global Data Filters ────────────────────────────────────────────
    sb_section("📅 Global Filters")

    wh_opts = (["All"] + sorted(policy_raw["warehouse"].unique().tolist())
               if not policy_raw.empty else ["All"])
    g_warehouse = st.selectbox("Warehouse", wh_opts, key="g_wh")

    g_abc = st.selectbox("ABC Class", ["All", "A", "B", "C"], key="g_abc")

    if not sales_raw.empty and "date" in sales_raw.columns:
        min_date = sales_raw["date"].min().date()
        max_date = sales_raw["date"].max().date()
        g_date_range = st.slider(
            "Sales History Range",
            min_value=min_date,
            max_value=max_date,
            value=(min_date, max_date),
            format="YYYY-MM-DD",
            key="g_date",
        )
    else:
        g_date_range = (None, None)

    # ── 🔮 Forecast Settings ──────────────────────────────────────────────
    sb_section("🔮 Forecast Settings")
    g_horizon = st.slider("Forecast Horizon (days)", 30, 180, 90, key="g_horizon")

    # ── 📦 Inventory Parameters ───────────────────────────────────────────
    sb_section("📦 Inventory Parameters")
    g_service_level = st.slider(
        "Service Level (%)", 80, 99, 95, key="g_sl",
        help="Target service level for safety stock calculation",
    )
    g_lead_time = st.slider(
        "Lead Time Override (days)", 1, 30, 14, key="g_lt",
        help="Override lead time for what-if analysis",
    )
    g_review_period = st.slider(
        "Review Period (days)", 1, 30, 7, key="g_rp",
        help="Replenishment review cycle",
    )

    # ── 🚚 Supply Parameters ──────────────────────────────────────────────
    sb_section("🚚 Supply Parameters")
    g_safety_buffer = st.slider(
        "Safety Buffer (%)", min_value=0, max_value=30, value=10, step=5,
        key="g_sb",
        help="Extra buffer added on top of EOQ order quantity",
    )
    g_moq_override = st.toggle(
        "Apply Min Order Qty Override",
        value=False,
        key="g_moq",
        help="Force all orders to respect supplier MOQ",
    )

    st.markdown("---")

    # ── Refresh ───────────────────────────────────────────────────────────
    if st.button("🔄  Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    # ── Pipeline Status ───────────────────────────────────────────────────
    sb_section("Pipeline Status")
    checks = {
        "Sales History":      not DATA["sales"].empty,
        "Forecast Output":    not DATA["forecast"].empty,
        "Inventory Policy":   not DATA["policy"].empty,
        "Replenishment Plan": not DATA["plan"].empty,
    }
    for lbl, ok in checks.items():
        st.markdown(f"{'🟢' if ok else '🔴'}  {lbl}")

    st.markdown("---")
    st.caption(
        f"ℹ️ Lead time: **{g_lead_time}d** · SL: **{g_service_level}%** · "
        f"Buffer: **{g_safety_buffer}%** · Review: **{g_review_period}d**"
    )


# ══════════════════════════════════════════════════════════════════════════
# GLOBAL FILTER HELPER
# ══════════════════════════════════════════════════════════════════════════

def apply_global(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    if g_warehouse != "All" and "warehouse" in df.columns:
        df = df[df["warehouse"] == g_warehouse]
    if g_abc != "All" and "abc_class" in df.columns:
        df = df[df["abc_class"] == g_abc]
    return df


# ══════════════════════════════════════════════════════════════════════════
# PAGE HEADER
# ══════════════════════════════════════════════════════════════════════════

st.markdown('<div class="page-title">Demand–Supply Planning</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="page-subtitle">Integrated Planning System · Real-time Analytics</div>',
    unsafe_allow_html=True,
)

# ══════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4 = st.tabs([
    "📊  Overview",
    "🔮  Forecast",
    "📦  Inventory",
    "🚚  Supply",
])


# ══════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════

with tab1:
    policy  = apply_global(DATA.get("policy",      pd.DataFrame()).copy())
    plan    = apply_global(DATA.get("plan",         pd.DataFrame()).copy())
    inv_kpi = DATA.get("inv_kpis",    pd.DataFrame())
    sup_kpi = DATA.get("supply_kpis", pd.DataFrame())
    summ    = DATA.get("summary",     pd.DataFrame())

    with st.expander("⚙️  Overview Options", expanded=False):
        top_n_alerts = st.slider("Top N Critical Alerts", 5, 30, 15, key="ov_topn")

    section("System KPIs")
    ik = inv_kpi.iloc[0].to_dict() if not inv_kpi.empty else {}
    sk = sup_kpi.iloc[0].to_dict() if not sup_kpi.empty else {}
    fk = summ.iloc[0].to_dict()    if not summ.empty    else {}

    n_critical = int(len(policy[policy["policy_status"] == "CRITICAL"]) if not policy.empty else 0)

    cards  = '<div class="kpi-grid">'
    cards += kpi_card("Total Inventory Value",  f"{ik.get('total_inventory_value_b_vnd',0):.1f}B", "VND", "accent")
    cards += kpi_card("Avg Days of Supply",     f"{ik.get('avg_dos',0):.1f}", "days remaining", "accent3")
    cards += kpi_card("Replenishment Orders",   str(sk.get('total_orders',0)), f"{sk.get('total_order_value_b_vnd',0):.1f}B VND", "accent2")
    cards += kpi_card("Forecast Accuracy",      f"{fk.get('mean_forecast_accuracy',0):.1f}%", f"Mean MAPE {fk.get('mean_mape',0):.1f}%", "warn")
    cards += kpi_card("Critical Alerts",        str(n_critical), "SKU×WH below safety stock", "danger")
    cards += kpi_card("Avg Stockout Risk",      f"{ik.get('avg_stockout_risk_pct',0):.1f}%", "probability before reorder", "danger")
    cards += '</div>'
    st.markdown(cards, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        section("Policy Status Distribution")
        if not policy.empty:
            sc = policy["policy_status"].value_counts().reset_index()
            sc.columns = ["Status", "Count"]
            fig = go.Figure(go.Pie(
                labels=sc["Status"], values=sc["Count"], hole=0.62,
                marker_colors=[STATUS_COLORS.get(s, "#64748b") for s in sc["Status"]],
                textinfo="label+percent",
                textfont=dict(family="DM Mono", size=10, color="#e2e8f0"),
            ))
            fig.add_annotation(
                text=f"<b>{len(policy)}</b><br><span style='font-size:10px'>Total</span>",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=18, color="#e2e8f0", family="Syne"),
            )
            fig.update_layout(**PLOTLY_LAYOUT, height=300, showlegend=True)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Run the pipeline to see inventory policy data.")

    with col2:
        section("Top Replenishment Orders by Value")
        if not plan.empty:
            top_plan = (plan.nlargest(15, "order_value_vnd")
                        .assign(label=lambda d: d["sku"] + " / " + d["warehouse"],
                                value_m=lambda d: d["order_value_vnd"] / 1e6))
            fig = go.Figure(go.Bar(
                x=top_plan["value_m"], y=top_plan["label"], orientation="h",
                marker_color="#ff6b35",
                text=top_plan["value_m"].map(lambda x: f"{x:.0f}M"),
                textposition="outside", textfont=dict(size=9),
            ))
            fig.update_layout(**PLOTLY_LAYOUT, height=300,
                              xaxis_title="Order Value (M VND)", yaxis_title="")
            st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        section("Inventory vs Safety Stock by ABC Class")
        if not policy.empty:
            abc_inv = (policy.groupby("abc_class")
                       .agg(avg_on_hand=("on_hand_units","mean"),
                            avg_ss=("safety_stock","mean"),
                            avg_rop=("reorder_point","mean"))
                       .reset_index())
            fig = go.Figure()
            fig.add_trace(go.Bar(name="Avg On Hand",      x=abc_inv["abc_class"], y=abc_inv["avg_on_hand"],  marker_color="#00d4ff"))
            fig.add_trace(go.Bar(name="Avg Safety Stock", x=abc_inv["abc_class"], y=abc_inv["avg_ss"],       marker_color="#ff6b35"))
            fig.add_trace(go.Bar(name="Avg ROP",          x=abc_inv["abc_class"], y=abc_inv["avg_rop"],      marker_color="#ffb830", opacity=0.6))
            fig.update_layout(**PLOTLY_LAYOUT, barmode="group", height=260,
                              xaxis_title="ABC Class", yaxis_title="Units (avg)")
            st.plotly_chart(fig, use_container_width=True)

    with col4:
        section("Stockout Risk Distribution")
        if not policy.empty:
            fig = go.Figure()
            for status, color in STATUS_COLORS.items():
                sub = policy[policy["policy_status"] == status]
                if not sub.empty:
                    fig.add_trace(go.Box(
                        y=sub["stockout_risk"] * 100, name=status,
                        marker_color=color, line_color=color,
                        fillcolor=STATUS_FILL_COLORS.get(status, "rgba(100,116,139,0.15)"),
                    ))
            fig.update_layout(**PLOTLY_LAYOUT, height=260,
                              yaxis_title="Stockout Risk (%)", showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    section(f"🔴  Critical Alerts — Top {top_n_alerts} by Stockout Risk")
    if not policy.empty:
        critical = (policy[policy["policy_status"] == "CRITICAL"]
                    .sort_values("stockout_risk", ascending=False)
                    .head(top_n_alerts))
        if not critical.empty:
            show = critical[["sku","warehouse","abc_class","on_hand_units",
                             "safety_stock","reorder_point","dos","stockout_risk","policy_status"]].copy()
            show["stockout_risk"] = (show["stockout_risk"]*100).map(lambda x: f"{x:.1f}%")
            show["dos"]           = show["dos"].map(lambda x: f"{x:.1f}d")
            st.dataframe(show, use_container_width=True, hide_index=True)
        else:
            st.success("✅ No critical alerts for current filters.")


# ══════════════════════════════════════════════════════════════════════════
# TAB 2 — FORECAST
# ══════════════════════════════════════════════════════════════════════════

with tab2:
    forecast_df  = apply_global(DATA.get("forecast",  pd.DataFrame()).copy())
    accuracy_df  = apply_global(DATA.get("accuracy",  pd.DataFrame()).copy())
    sales_df     = apply_global(DATA.get("sales",     pd.DataFrame()).copy())
    summ_df      = DATA.get("summary",         pd.DataFrame())
    model_sel_df = DATA.get("model_selection", pd.DataFrame())

    section("Forecast Accuracy KPIs")
    fk = summ_df.iloc[0].to_dict() if not summ_df.empty else {}

    cards  = '<div class="kpi-grid">'
    cards += kpi_card("Forecast Accuracy", f"{fk.get('mean_forecast_accuracy',0):.1f}%", "mean across all SKU×WH", "accent3")
    cards += kpi_card("Mean MAPE",         f"{fk.get('mean_mape',0):.1f}%",              "lower is better", "warn")
    cards += kpi_card("Median MAPE",       f"{fk.get('median_mape',0):.1f}%",            "", "warn")
    cards += kpi_card("MAPE < 20%",        f"{fk.get('pct_below_20_mape',0):.1f}%",      "of SKU×WH combos", "accent")
    cards += kpi_card("Mean Bias",         f"{fk.get('mean_bias',0):+.1f}",              "units/day (+over, -under)", "accent2")
    cards += kpi_card("Combinations",      str(int(fk.get('n_combinations',0))),         "SKU × Warehouse", "accent")
    cards += '</div>'
    st.markdown(cards, unsafe_allow_html=True)

    section("Demand Forecast — SKU View")

    sku_list = sorted(forecast_df["sku"].unique().tolist())       if not forecast_df.empty else []
    wh_list  = sorted(forecast_df["warehouse"].unique().tolist()) if not forecast_df.empty else []

    lf_col1, lf_col2, lf_col3, lf_col4 = st.columns([2, 2, 2, 1])
    with lf_col1:
        sel_sku = st.selectbox("SKU", sku_list, key="fc_sku")
    with lf_col2:
        sel_wh2 = st.selectbox("Warehouse", wh_list, key="fc_wh")
    with lf_col3:
        model_opts = (["All Models (Compare)"] + sorted(forecast_df["model"].unique().tolist())
                      if not forecast_df.empty else ["All Models (Compare)"])
        sel_model_view = st.selectbox("Model View", model_opts, key="fc_model_view")
    with lf_col4:
        show_ci = st.toggle("Confidence Band", value=True, key="fc_ci")

    if not forecast_df.empty and not sales_df.empty and sku_list:
        hist_sub = (sales_df[(sales_df["sku"] == sel_sku) & (sales_df["warehouse"] == sel_wh2)]
                    .groupby("date")["demand_qty"].sum().reset_index()
                    .sort_values("date").tail(90))

        fig = go.Figure()

        if not hist_sub.empty:
            fig.add_trace(go.Scatter(
                x=hist_sub["date"], y=hist_sub["demand_qty"],
                name="Historical Demand",
                line=dict(color="#7fff6b", width=1.5), mode="lines",
            ))

        models_to_plot = (
            forecast_df["model"].unique().tolist()
            if sel_model_view == "All Models (Compare)"
            else [sel_model_view]
        )

        for m in models_to_plot:
            fc_sub = (forecast_df[
                (forecast_df["sku"] == sel_sku) &
                (forecast_df["warehouse"] == sel_wh2) &
                (forecast_df["model"] == m)
            ].sort_values("date"))

            if fc_sub.empty:
                continue

            color = MODEL_COLORS.get(m, "#00d4ff")
            r_c, g_c, b_c = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
            show_band = show_ci and sel_model_view != "All Models (Compare)"

            fig.add_trace(go.Scatter(
                x=fc_sub["date"], y=fc_sub["forecast_qty"],
                name=m.replace("_", " ").title(),
                line=dict(color=color, width=2, dash="dash"), mode="lines",
            ))

            if show_band and "forecast_qty" in fc_sub.columns:
                sigma = fc_sub["forecast_qty"].std() * 0.3
                fig.add_trace(go.Scatter(
                    x=pd.concat([fc_sub["date"], fc_sub["date"].iloc[::-1]]),
                    y=pd.concat([fc_sub["forecast_qty"] + sigma,
                                 (fc_sub["forecast_qty"] - sigma).iloc[::-1]]),
                    fill="toself",
                    fillcolor=f"rgba({r_c},{g_c},{b_c},0.1)",
                    line=dict(color="rgba(0,0,0,0)"),
                    showlegend=False, name=f"{m} band",
                ))

        fig.update_layout(**PLOTLY_LAYOUT, height=340,
                          xaxis_title="Date", yaxis_title="Demand (units)",
                          title=f"Forecast — {sel_sku} @ {sel_wh2}")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Run the pipeline to generate forecast data.")

    if not accuracy_df.empty:
        section("Forecast Accuracy by SKU × Warehouse")
        c1, c2, c3 = st.columns(3)
        with c1:
            sort_by_acc = st.selectbox("Sort by", ["mape","forecast_accuracy","bias"], key="acc_sort")
        with c2:
            max_mape = st.slider("Max MAPE (%)", 10, 100, 50, key="acc_mape")
        with c3:
            acc_model_filter = st.selectbox(
                "Model", ["All"] + sorted(accuracy_df["model"].unique().tolist()),
                key="acc_model"
            )

        acc_show = accuracy_df[accuracy_df["mape"] <= max_mape].copy()
        if acc_model_filter != "All":
            acc_show = acc_show[acc_show["model"] == acc_model_filter]
        acc_show = acc_show.sort_values(sort_by_acc)
        acc_show["forecast_accuracy"] = (acc_show["forecast_accuracy"]*100).map(lambda x: f"{x:.1f}%")
        acc_show["mape"]  = acc_show["mape"].map(lambda x: f"{x:.1f}%")
        acc_show["bias"]  = acc_show["bias"].map(lambda x: f"{x:+.1f}")
        st.dataframe(acc_show.reset_index(drop=True), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════
# TAB 3 — INVENTORY
# ══════════════════════════════════════════════════════════════════════════

with tab3:
    policy_base = apply_global(DATA.get("policy", pd.DataFrame()).copy())

    section("Inventory Filters")
    lf1, lf2, lf3, lf4 = st.columns(4)

    with lf1:
        status_filter = st.selectbox(
            "Policy Status", ["All","CRITICAL","REORDER","OK","OVERSTOCK"], key="inv_status"
        )
    with lf2:
        dos_max = st.slider("Max DOS (days)", 1, 60, 60, key="inv_dos")
    with lf3:
        risk_min = st.slider("Min Stockout Risk (%)", 0, 100, 0, key="inv_risk")
    with lf4:
        sort_inv = st.selectbox(
            "Sort Table By", ["stockout_risk","dos","inventory_value_vnd","on_hand_units"],
            key="inv_sort"
        )

    policy_f = policy_base.copy()
    if status_filter != "All" and not policy_f.empty:
        policy_f = policy_f[policy_f["policy_status"] == status_filter]
    if not policy_f.empty and "dos" in policy_f.columns:
        policy_f = policy_f[policy_f["dos"].replace(999, 999) <= dos_max]
        policy_f = policy_f[policy_f["stockout_risk"] * 100 >= risk_min]

    # What-if recalculation
    from scipy import stats as scipy_stats

    def _recalc_ss(row, lt_override, sl_override):
        z     = float(scipy_stats.norm.ppf(np.clip(sl_override / 100, 0.5, 0.9999)))
        lt    = lt_override
        lt_std = row.get("lead_time_std", 2.0)
        d_avg  = row.get("demand_avg", 0)
        d_std  = row.get("demand_std", 0)
        sigma  = np.sqrt(lt * d_std**2 + d_avg**2 * lt_std**2)
        ss     = max(0.0, z * sigma)
        rop    = max(0.0, d_avg * lt + ss)
        return round(ss, 0), round(rop, 0)

    if not policy_f.empty and "demand_avg" in policy_f.columns:
        policy_f = policy_f.copy()
        recalc   = policy_f.apply(
            lambda r: pd.Series(
                _recalc_ss(r, g_lead_time, g_service_level),
                index=["ss_whatif", "rop_whatif"]
            ), axis=1
        )
        policy_f["ss_whatif"]  = recalc["ss_whatif"]
        policy_f["rop_whatif"] = recalc["rop_whatif"]
        policy_f["status_whatif"] = "OK"
        policy_f.loc[policy_f["on_hand_units"] <= policy_f["ss_whatif"],  "status_whatif"] = "CRITICAL"
        policy_f.loc[
            (policy_f["on_hand_units"] > policy_f["ss_whatif"]) &
            (policy_f["on_hand_units"] <= policy_f["rop_whatif"]),
            "status_whatif"
        ] = "REORDER"

    section("Inventory Policy KPIs")

    base_lt  = policy_f["lead_time_avg"].mean() if not policy_f.empty and "lead_time_avg" in policy_f.columns else 14
    base_sl  = policy_f["target_service_level"].mean() * 100 if not policy_f.empty and "target_service_level" in policy_f.columns else 95
    is_whatif = abs(g_lead_time - base_lt) > 0.5 or abs(g_service_level - base_sl) > 0.5

    if is_whatif:
        st.info(
            f"⚡ **What-if mode active** — Lead Time: **{g_lead_time}d** "
            f"(data avg: {base_lt:.1f}d) · Service Level: **{g_service_level}%** "
            f"(data avg: {base_sl:.1f}%) — Safety Stock & ROP recalculated below."
        )

    if not policy_f.empty:
        total_val = policy_f["inventory_value_vnd"].sum() / 1e9
        avg_dos   = policy_f["dos"].replace(999, np.nan).mean()
        n_repl    = policy_f["replenishment_needed"].sum() if "replenishment_needed" in policy_f.columns else 0
        avg_risk  = policy_f["stockout_risk"].mean() * 100
        avg_ss    = policy_f["ss_whatif"].mean()  if is_whatif and "ss_whatif"  in policy_f.columns else policy_f["safety_stock"].mean()
        avg_rop   = policy_f["rop_whatif"].mean() if is_whatif and "rop_whatif" in policy_f.columns else policy_f["reorder_point"].mean()
        avg_eoq   = policy_f["eoq"].mean()

        n_critical_whatif = (
            int((policy_f["status_whatif"] == "CRITICAL").sum())
            if is_whatif and "status_whatif" in policy_f.columns
            else int((policy_f["policy_status"] == "CRITICAL").sum())
        )

        ss_label  = f"Avg Safety Stock {'⚡' if is_whatif else ''}"
        rop_label = f"Avg ROP {'⚡' if is_whatif else ''}"

        cards  = '<div class="kpi-grid">'
        cards += kpi_card("Total Value",         f"{total_val:.2f}B",   "VND", "accent")
        cards += kpi_card("Avg DOS",             f"{avg_dos:.1f}",      "days of supply", "accent3")
        cards += kpi_card(ss_label,              f"{avg_ss:.0f}",       f"LT={g_lead_time}d SL={g_service_level}%", "accent")
        cards += kpi_card(rop_label,             f"{avg_rop:.0f}",      "units reorder point", "accent2")
        cards += kpi_card("Avg EOQ",             f"{avg_eoq:.0f}",      "units per order", "accent2")
        cards += kpi_card("Need Replenishment",  str(int(n_repl)),      "SKU×WH", "warn")
        cards += kpi_card("Avg Stockout Risk",   f"{avg_risk:.1f}%",    "", "danger")
        cards += kpi_card(f"Critical {'⚡' if is_whatif else ''}", str(n_critical_whatif), "SKU×WH", "danger")
        cards += '</div>'
        st.markdown(cards, unsafe_allow_html=True)
    else:
        st.info("No data matches current filters.")

    col7, col8 = st.columns(2)

    with col7:
        section("DOS Distribution by Status")
        if not policy_f.empty:
            fig = go.Figure()
            for status, color in STATUS_COLORS.items():
                sub = policy_f[policy_f["policy_status"] == status]
                if not sub.empty:
                    fig.add_trace(go.Histogram(
                        x=sub["dos"].replace(999, np.nan),
                        name=status, marker_color=color, opacity=0.75,
                        nbinsx=20,
                    ))
            fig.update_layout(**PLOTLY_LAYOUT, barmode="overlay", height=280,
                              xaxis_title="Days of Supply", yaxis_title="Count")
            st.plotly_chart(fig, use_container_width=True)

    with col8:
        section("Top 20 High-Risk SKUs — On Hand vs SS vs ROP")
        top20 = apply_global(DATA.get("policy", pd.DataFrame()).copy())
        top20 = top20.nlargest(20, "stockout_risk") if not top20.empty else top20
        if not top20.empty:
            top20 = top20.copy()
            top20["label"] = top20["sku"] + "\n" + top20["warehouse"]
            fig = go.Figure()
            fig.add_trace(go.Bar(name="On Hand",      x=top20["label"], y=top20["on_hand_units"],  marker_color="#00d4ff"))
            fig.add_trace(go.Bar(name="Safety Stock", x=top20["label"], y=top20["safety_stock"],   marker_color="#ff6b35"))
            fig.add_trace(go.Bar(name="ROP",          x=top20["label"], y=top20["reorder_point"],  marker_color="#ffb830", opacity=0.7))
            fig.update_layout(**PLOTLY_LAYOUT, barmode="overlay", height=320,
                              xaxis_tickangle=-45, xaxis_title="", yaxis_title="Units")
            st.plotly_chart(fig, use_container_width=True)

    section("Full Policy Table")
    if not policy_f.empty:
        show = policy_f[[
            "sku","warehouse","abc_class","policy_status",
            "on_hand_units","safety_stock","reorder_point","eoq","max_stock",
            "dos","stockout_risk","demand_avg","lead_time_avg",
            "target_service_level","inventory_value_vnd",
        ]].copy().sort_values(sort_inv, ascending=(sort_inv != "stockout_risk"))
        show["inventory_value_vnd"]  = (show["inventory_value_vnd"]/1e6).map(lambda x: f"{x:.1f}M")
        show["stockout_risk"]        = (show["stockout_risk"]*100).map(lambda x: f"{x:.1f}%")
        show["dos"]                  = show["dos"].map(lambda x: f"{x:.1f}d")
        show["target_service_level"] = (show["target_service_level"]*100).map(lambda x: f"{x:.1f}%")
        st.dataframe(show.reset_index(drop=True), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════
# TAB 4 — SUPPLY
# ══════════════════════════════════════════════════════════════════════════

with tab4:
    plan_base  = apply_global(DATA.get("plan", pd.DataFrame()).copy())
    sup_kpi_df = DATA.get("supply_kpis", pd.DataFrame())

    section("Supply Filters")
    sf1, sf2, sf3, sf4 = st.columns(4)

    with sf1:
        sup_opts = ["All"] + (sorted(plan_base["supplier_id"].unique().tolist())
                              if not plan_base.empty else [])
        sel_supplier = st.selectbox("Supplier", sup_opts, key="sup_supplier")

    with sf2:
        if not plan_base.empty and "expected_arrival" in plan_base.columns:
            arr_min = plan_base["expected_arrival"].min().date()
            arr_max = plan_base["expected_arrival"].max().date()
            sel_arrival = st.slider("Arrival Date Range", arr_min, arr_max,
                                    (arr_min, arr_max), key="sup_arrival")
        else:
            sel_arrival = (None, None)

    with sf3:
        min_order_val = st.number_input(
            "Min Order Value (M VND)", min_value=0, max_value=500, value=0, step=10,
            key="sup_minval"
        )

    with sf4:
        cap_only = st.toggle("Capacity Issues Only", value=False, key="sup_cap")

    plan_f = plan_base.copy()
    if not plan_f.empty:
        if sel_supplier != "All":
            plan_f = plan_f[plan_f["supplier_id"] == sel_supplier]
        if sel_arrival[0] is not None and "expected_arrival" in plan_f.columns:
            plan_f = plan_f[
                (plan_f["expected_arrival"].dt.date >= sel_arrival[0]) &
                (plan_f["expected_arrival"].dt.date <= sel_arrival[1])
            ]
        if min_order_val > 0:
            plan_f = plan_f[plan_f["order_value_vnd"] >= min_order_val * 1e6]
        if cap_only and "capacity_ok" in plan_f.columns:
            plan_f = plan_f[plan_f["capacity_ok"] == False]

    section("Replenishment Plan KPIs")
    sk = sup_kpi_df.iloc[0].to_dict() if not sup_kpi_df.empty else {}

    f_orders = len(plan_f) if not plan_f.empty else 0
    f_value  = plan_f["order_value_vnd"].sum() / 1e9 if not plan_f.empty else 0
    f_units  = plan_f["adjusted_qty"].sum() if not plan_f.empty else 0

    cards  = '<div class="kpi-grid">'
    cards += kpi_card("Total Orders (filtered)",  str(f_orders),        "to place now", "danger")
    cards += kpi_card("Order Value (filtered)",   f"{f_value:.2f}B",    "VND", "accent2")
    cards += kpi_card("Units to Order (filtered)",f"{int(f_units):,}",  "units", "accent")
    cards += kpi_card("Avg Lead Time",            f"{sk.get('avg_lead_time_days',0):.1f}", "days (all data)", "warn")
    cards += kpi_card("Suppliers Involved",       str(sk.get('n_suppliers_involved',0)),   "active", "accent3")
    cards += kpi_card("Safety Buffer",            f"{g_safety_buffer}%","from sidebar param", "warn")
    cards += '</div>'
    st.markdown(cards, unsafe_allow_html=True)

    col9, col10 = st.columns(2)

    with col9:
        section("Order Value by Warehouse")
        if not plan_f.empty:
            wh_plan = (plan_f.groupby("warehouse")
                       .agg(total_value=("order_value_vnd","sum"), n_orders=("order_id","count"))
                       .reset_index().sort_values("total_value", ascending=True))
            wh_plan["total_value_b"] = wh_plan["total_value"] / 1e9
            fig = go.Figure(go.Bar(
                x=wh_plan["total_value_b"], y=wh_plan["warehouse"], orientation="h",
                marker=dict(color=wh_plan["total_value_b"],
                            colorscale=[[0,"#1e2736"],[1,"#ff6b35"]], showscale=False),
                text=wh_plan.apply(lambda r: f"{r['total_value_b']:.2f}B ({r['n_orders']} orders)", axis=1),
                textposition="outside", textfont=dict(size=9),
            ))
            fig.update_layout(**PLOTLY_LAYOUT, height=300,
                              xaxis_title="Order Value (B VND)", yaxis_title="")
            st.plotly_chart(fig, use_container_width=True)

    with col10:
        section("DOS Before Order vs Stockout Risk")
        if not plan_f.empty:
            fig = px.scatter(
                plan_f, x="dos_before_order", y="stockout_risk",
                color="abc_class", size="adjusted_qty",
                hover_data=["sku","warehouse","order_value_vnd","expected_arrival"],
                color_discrete_map={"A":"#00d4ff","B":"#ffb830","C":"#7fff6b"},
            )
            fig.update_layout(**PLOTLY_LAYOUT, height=280,
                              xaxis_title="DOS Before Order (days)",
                              yaxis_title="Stockout Risk")
            st.plotly_chart(fig, use_container_width=True)

    col11, col12 = st.columns(2)

    with col11:
        section("Orders by Arrival Date")
        if not plan_f.empty and "expected_arrival" in plan_f.columns:
            arr_daily = (plan_f.groupby(plan_f["expected_arrival"].dt.date)
                         .agg(n_orders=("order_id","count"),
                              total_value=("order_value_vnd","sum"))
                         .reset_index())
            arr_daily["total_value_b"] = arr_daily["total_value"] / 1e9
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=arr_daily["expected_arrival"], y=arr_daily["n_orders"],
                name="Orders", marker_color="#00d4ff",
            ))
            fig.update_layout(**PLOTLY_LAYOUT, height=280,
                              xaxis_title="Expected Arrival", yaxis_title="Order Count")
            st.plotly_chart(fig, use_container_width=True)

    with col12:
        section("Order Value by Supplier")
        if not plan_f.empty:
            sup_plan = (plan_f.groupby("supplier_id")["order_value_vnd"]
                        .sum().reset_index().sort_values("order_value_vnd", ascending=False))
            sup_plan["value_b"] = sup_plan["order_value_vnd"] / 1e9
            fig = go.Figure(go.Pie(
                labels=sup_plan["supplier_id"], values=sup_plan["value_b"], hole=0.5,
                textinfo="label+percent",
                textfont=dict(family="DM Mono", size=10, color="#e2e8f0"),
            ))
            fig.update_layout(**PLOTLY_LAYOUT, height=280, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    section("Full Replenishment Plan")
    if not plan_f.empty:
        show_plan = plan_f[[
            "order_id","sku","warehouse","supplier_id","abc_class","policy_status",
            "on_hand_units","adjusted_qty","order_value_vnd",
            "order_date","expected_arrival","lead_time_days",
            "dos_before_order","stockout_risk","capacity_ok",
        ]].copy()
        show_plan["order_value_vnd"]  = (show_plan["order_value_vnd"]/1e6).map(lambda x: f"{x:.1f}M VND")
        show_plan["stockout_risk"]    = (show_plan["stockout_risk"]*100).map(lambda x: f"{x:.1f}%")
        show_plan["dos_before_order"] = show_plan["dos_before_order"].map(lambda x: f"{x:.1f}d")
        show_plan["order_date"]       = show_plan["order_date"].astype(str)
        show_plan["expected_arrival"] = show_plan["expected_arrival"].astype(str)
        show_plan["capacity_ok"]      = show_plan["capacity_ok"].map(lambda x: "✅" if x else "❌")
        st.dataframe(show_plan.reset_index(drop=True), use_container_width=True, hide_index=True)
    elif not plan_base.empty:
        st.info("No orders match current supply filters.")
    else:
        st.info("Run the pipeline to generate replenishment plan data.")
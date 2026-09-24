"""
Customer Churn Decision Dashboard
==================================
A three-page Streamlit + Plotly business-intelligence dashboard built on the
IBM Telco Customer Churn dataset.

Pages
-----
1. Executive Overview      – KPIs and key churn trends
2. Services & Revenue      – Churn drivers across services and payment methods
3. Risk, Opportunities & Actions – Model-based scoring, risk cards, action cards

Run
---
    streamlit run churn_dashboard_app.py
"""

import glob
import os
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Resolve data folder relative to this script's own directory so the app works
# regardless of the working directory from which streamlit is launched.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.join(_SCRIPT_DIR, "data")


def _find_csv() -> Optional[str]:
    """Return the path of the first *.csv found in the data/ folder, or None."""
    matches = glob.glob(os.path.join(_DATA_DIR, "*.csv"))
    return matches[0] if matches else None

# Brand-consistent color palette (used across all charts)
COLOR_CHURN = "#E05C5C"      # red  – churned / high-risk
COLOR_RETAIN = "#3B82D4"     # blue – retained / low-risk
COLOR_ACCENT = "#7C5CD8"     # purple – model / estimates
COLOR_NEUTRAL = "#57606A"    # muted grey – supporting text

TENURE_ORDER = ["0-12 months", "13-24 months", "25-48 months", "49+ months"]

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Customer Churn Decision Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Loading dataset …")
def load_data(csv_path: Optional[str] = None, uploaded_file=None) -> pd.DataFrame:
    """Load the Telco Churn CSV from disk (csv_path) or an uploaded file object."""
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    return pd.read_csv(csv_path)


@st.cache_data(show_spinner="Cleaning data …")
def clean_data(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all cleaning steps:
    - TotalCharges: strip whitespace, replace blanks with 0.0, cast to float.
      (Blank rows have tenure = 0, meaning no charges have accrued yet.)
    - Add SeniorCitizenLabel (Yes / No) for readable display.
    - Add TenureGroup categorical column.
    """
    df = raw.copy()

    # --- TotalCharges ---
    df["TotalCharges"] = df["TotalCharges"].astype(str).str.strip()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)

    # --- Senior citizen readable label ---
    df["SeniorCitizenLabel"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})

    # --- Tenure groups ---
    def _tenure_group(t):
        if t <= 12:
            return "0-12 months"
        elif t <= 24:
            return "13-24 months"
        elif t <= 48:
            return "25-48 months"
        else:
            return "49+ months"

    df["TenureGroup"] = df["tenure"].apply(_tenure_group)
    df["TenureGroup"] = pd.Categorical(df["TenureGroup"], categories=TENURE_ORDER, ordered=True)

    return df


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

# Numeric and categorical feature columns fed into the model.
# customerID and Churn are intentionally excluded (no data leakage).
# "gender" is intentionally excluded: it is a protected demographic attribute
# and shows negligible predictive value in this dataset.
MODEL_NUM_COLS = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
MODEL_CAT_COLS = [
    "Contract", "InternetService", "PaymentMethod", "TechSupport",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "PaperlessBilling", "Partner", "Dependents",
    "PhoneService", "MultipleLines", "StreamingTV", "StreamingMovies",
]
MODEL_ALL_COLS = MODEL_NUM_COLS + MODEL_CAT_COLS


@st.cache_resource(show_spinner="Training churn model …")
def train_model(df: pd.DataFrame):
    """
    Train a logistic regression model to predict churn probability.

    Preprocessing
    -------------
    - Numeric columns: StandardScaler
    - Categorical columns: OneHotEncoder (handle_unknown='ignore')
    Both steps are combined in a ColumnTransformer inside one Pipeline.
    customerID and Churn are never passed as inputs (no leakage).

    Returns
    -------
    pipeline : Fitted sklearn Pipeline
    metrics  : dict with accuracy, precision, recall, roc_auc on the held-out test set
    """
    X = df[MODEL_ALL_COLS].copy()
    y = (df["Churn"] == "Yes").astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), MODEL_NUM_COLS),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), MODEL_CAT_COLS),
    ])

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("clf", LogisticRegression(max_iter=1000, random_state=42)),
    ])
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy":  round(accuracy_score(y_test, y_pred),  4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall":    round(recall_score(y_test, y_pred),    4),
        "roc_auc":   round(roc_auc_score(y_test, y_proba),  4),
    }

    return pipeline, metrics


def score_active_customers(df: pd.DataFrame, pipeline) -> pd.DataFrame:
    """
    Score only currently active customers (Churn == 'No') using the trained
    pipeline. Adds a ChurnProbability column to the returned DataFrame.
    """
    active = df[df["Churn"] == "No"].copy()
    active["ChurnProbability"] = pipeline.predict_proba(active[MODEL_ALL_COLS])[:, 1]
    return active


# ---------------------------------------------------------------------------
# Helper: small KPI card using st.metric
# ---------------------------------------------------------------------------

def kpi_card(label: str, value: str, delta: str = None, delta_color: str = "normal"):
    """Render a single st.metric KPI card."""
    st.metric(label=label, value=value, delta=delta, delta_color=delta_color)


# ---------------------------------------------------------------------------
# Helper: churn rate bar chart (reused across pages)
# ---------------------------------------------------------------------------

def churn_rate_bar(df: pd.DataFrame, group_col: str, title: str, sort: bool = True) -> go.Figure:
    """
    Compute churn rate by *group_col* and return a Plotly bar chart.
    Bars are colour-coded by churn rate (high = red, low = blue).
    """
    grp = (
        df.groupby(group_col)["Churn"]
        .apply(lambda s: (s == "Yes").sum() / len(s) * 100)
        .reset_index()
        .rename(columns={"Churn": "ChurnRate"})
    )
    if sort:
        grp = grp.sort_values("ChurnRate", ascending=False)

    fig = px.bar(
        grp,
        x=group_col,
        y="ChurnRate",
        title=title,
        labels={"ChurnRate": "Churn Rate (%)"},
        color="ChurnRate",
        color_continuous_scale=[[0, COLOR_RETAIN], [1, COLOR_CHURN]],
        text=grp["ChurnRate"].map(lambda v: f"{v:.1f}%"),
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        coloraxis_showscale=False,
        plot_bgcolor="white",
        yaxis=dict(range=[0, grp["ChurnRate"].max() * 1.2]),
        margin=dict(t=50, b=40),
    )
    return fig


# ---------------------------------------------------------------------------
# PAGE 1 — Executive Overview
# ---------------------------------------------------------------------------

def page_executive_overview(df: pd.DataFrame):
    """Render Page 1: KPI cards, tenure group bar chart, contract mix donut."""

    st.title("📊 Executive Overview")
    st.caption(
        "⚠️ **Note:** All associations shown here are correlations observed in "
        "historical data, not proof of causation."
    )
    st.divider()

    # ── KPI Calculations ──────────────────────────────────────────────────
    total = len(df)
    churned_df = df[df["Churn"] == "Yes"]
    retained_df = df[df["Churn"] == "No"]

    churn_rate = len(churned_df) / total * 100
    retention_rate = 100 - churn_rate
    revenue_lost = churned_df["MonthlyCharges"].sum()
    avg_tenure_all = df["tenure"].mean()
    avg_tenure_churned = churned_df["tenure"].mean()
    avg_mc_all = df["MonthlyCharges"].mean()
    avg_mc_churned = churned_df["MonthlyCharges"].mean()

    # ── KPI Cards Row ─────────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    with c1:
        kpi_card("Overall Churn Rate", f"{churn_rate:.1f}%")
        kpi_card("Retention Rate", f"{retention_rate:.1f}%")
    with c2:
        kpi_card("Monthly Revenue Lost to Churn", f"${revenue_lost:,.0f}")
        kpi_card("Active Customers", f"{len(retained_df):,}")
    with c3:
        kpi_card(
            "Avg Tenure — Churned vs All",
            f"{avg_tenure_churned:.1f} mo",
            delta=f"{avg_tenure_churned - avg_tenure_all:.1f} mo vs avg",
            delta_color="inverse",
        )
        kpi_card(
            "Avg Monthly Charges — Churned vs All",
            f"${avg_mc_churned:.2f}",
            delta=f"${avg_mc_churned - avg_mc_all:.2f} vs avg",
            delta_color="inverse",
        )

    st.divider()

    # ── Headline Insight (computed from data) ─────────────────────────────
    contract_churn = (
        df.groupby("Contract")["Churn"]
        .apply(lambda s: (s == "Yes").sum() / len(s) * 100)
    )
    top_contract = contract_churn.idxmax()
    top_contract_rate = contract_churn[top_contract]
    st.info(
        f"💡 **Headline Insight:** {top_contract} customers churn at "
        f"**{top_contract_rate:.1f}%** — the highest of any contract type. "
        f"Reducing this single segment's churn is the highest-leverage "
        f"opportunity in the data."
    )

    # ── Charts ────────────────────────────────────────────────────────────
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.subheader("Churn Rate by Tenure Group")
        tg = (
            df.groupby("TenureGroup", observed=True)["Churn"]
            .apply(lambda s: (s == "Yes").sum() / len(s) * 100)
            .reset_index()
            .rename(columns={"Churn": "ChurnRate"})
        )
        tg_counts = df.groupby("TenureGroup", observed=True).size().reset_index(name="Count")
        tg = tg.merge(tg_counts, on="TenureGroup")

        fig_tg = px.bar(
            tg,
            x="TenureGroup",
            y="ChurnRate",
            text=tg["ChurnRate"].map(lambda v: f"{v:.1f}%"),
            color="ChurnRate",
            color_continuous_scale=[[0, COLOR_RETAIN], [1, COLOR_CHURN]],
            labels={"ChurnRate": "Churn Rate (%)", "TenureGroup": "Tenure Group"},
            hover_data={"Count": True},
            category_orders={"TenureGroup": TENURE_ORDER},
        )
        fig_tg.update_traces(textposition="outside")
        fig_tg.update_layout(
            coloraxis_showscale=False,
            plot_bgcolor="white",
            yaxis=dict(range=[0, tg["ChurnRate"].max() * 1.25]),
            margin=dict(t=30, b=40),
        )
        st.plotly_chart(fig_tg, use_container_width=True)

    with col_right:
        st.subheader("Contract Type Mix")
        contract_counts = df["Contract"].value_counts().reset_index()
        contract_counts.columns = ["Contract", "Count"]
        fig_donut = px.pie(
            contract_counts,
            names="Contract",
            values="Count",
            hole=0.5,
            color_discrete_sequence=[COLOR_RETAIN, COLOR_ACCENT, COLOR_CHURN],
        )
        fig_donut.update_traces(textinfo="percent+label")
        fig_donut.update_layout(margin=dict(t=30, b=20), showlegend=True)
        st.plotly_chart(fig_donut, use_container_width=True)


# ---------------------------------------------------------------------------
# PAGE 2 — Services & Revenue Analysis
# ---------------------------------------------------------------------------

def page_services_revenue(df: pd.DataFrame):
    """Render Page 2: Churn drivers by service, payment method and charges."""

    st.title("🔍 Services & Revenue Analysis")
    st.caption(
        "⚠️ All figures show associations in historical data — not causal relationships."
    )
    st.divider()

    # ── Three driver bar charts ───────────────────────────────────────────
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Internet Service")
        st.plotly_chart(
            churn_rate_bar(df, "InternetService", "Churn Rate by Internet Service"),
            use_container_width=True,
        )

    with col2:
        st.subheader("Payment Method")
        st.plotly_chart(
            churn_rate_bar(df, "PaymentMethod", "Churn Rate by Payment Method"),
            use_container_width=True,
        )

    with col3:
        st.subheader("Tech Support")
        st.plotly_chart(
            churn_rate_bar(df, "TechSupport", "Churn Rate by Tech Support"),
            use_container_width=True,
        )

    st.divider()

    # ── Monthly charges distribution ─────────────────────────────────────
    st.subheader("Monthly Charges Distribution — Churned vs Retained")

    bins = [0, 35, 65, 85, df["MonthlyCharges"].max() + 1]
    labels = ["Under $35", "$35–$64", "$65–$84", "$85+"]
    df2 = df.copy()
    df2["ChargesBand"] = pd.cut(df2["MonthlyCharges"], bins=bins, labels=labels, right=False)

    band_churn = (
        df2.groupby(["ChargesBand", "Churn"], observed=True)
        .size()
        .reset_index(name="Count")
    )
    fig_mc = px.bar(
        band_churn,
        x="ChargesBand",
        y="Count",
        color="Churn",
        barmode="group",
        color_discrete_map={"Yes": COLOR_CHURN, "No": COLOR_RETAIN},
        labels={"ChargesBand": "Monthly Charges Band", "Count": "Number of Customers"},
        text_auto=True,
    )
    fig_mc.update_layout(plot_bgcolor="white", margin=dict(t=30, b=40))
    st.plotly_chart(fig_mc, use_container_width=True)

    st.divider()

    # ── Top 5 Combinations Table ──────────────────────────────────────────
    st.subheader("Top 5 High-Churn Combinations (Contract × Internet × Payment Method)")
    st.caption("Filtered to combinations with ≥ 30 customers for statistical reliability.")

    combos = (
        df.groupby(["Contract", "InternetService", "PaymentMethod"])
        .agg(
            TotalCustomers=("Churn", "count"),
            Churned=("Churn", lambda s: (s == "Yes").sum()),
        )
        .reset_index()
    )
    combos["ChurnRate"] = combos["Churned"] / combos["TotalCustomers"] * 100
    combos = combos[combos["TotalCustomers"] >= 30].sort_values("ChurnRate", ascending=False)

    top5 = combos.head(5).reset_index(drop=True)
    top5.index = top5.index + 1
    top5["ChurnRate"] = top5["ChurnRate"].map(lambda v: f"{v:.1f}%")
    st.dataframe(
        top5[["Contract", "InternetService", "PaymentMethod", "TotalCustomers", "Churned", "ChurnRate"]],
        use_container_width=True,
    )


# ---------------------------------------------------------------------------
# PAGE 3 — Risk, Opportunities & Actions
# ---------------------------------------------------------------------------

def page_risk_actions(df: pd.DataFrame, pipeline, metrics: dict):
    """Render Page 3: heatmap, revenue at risk, model scores, action cards."""

    st.title("🎯 Risk, Opportunities & Recommended Actions")
    st.caption(
        "⚠️ All churn associations are correlational. Revenue-at-Risk is a "
        "**model estimate** based on logistic regression probabilities."
    )
    st.divider()

    # ── Sidebar sliders for impact estimates ─────────────────────────────
    st.sidebar.markdown("---")
    st.sidebar.subheader("📐 Impact Assumption Sliders")
    pct_contract = st.sidebar.slider(
        "% of month-to-month customers converted to annual contract",
        min_value=5, max_value=50, value=20, step=5,
    ) / 100
    pct_autopay = st.sidebar.slider(
        "% of electronic-check customers switched to auto-pay",
        min_value=5, max_value=50, value=20, step=5,
    ) / 100
    pct_techsupport = st.sidebar.slider(
        "% of unsupported internet customers upsold to tech support",
        min_value=5, max_value=50, value=20, step=5,
    ) / 100
    # High-risk threshold controls ONLY the "High-risk count" KPI.
    high_risk_threshold = st.sidebar.slider(
        "High-risk probability threshold",
        min_value=0.30, max_value=0.80, value=0.50, step=0.05,
    )

    # ── Score active customers ────────────────────────────────────────────
    active_scored = score_active_customers(df, pipeline)

    # Expected revenue at risk = Σ(P_churn × MonthlyCharges) over all active customers
    revenue_at_risk = (
        active_scored["ChurnProbability"] * active_scored["MonthlyCharges"]
    ).sum()

    # High-risk count is driven by the threshold slider above
    high_risk_count = (active_scored["ChurnProbability"] >= high_risk_threshold).sum()

    # ── KPI row ───────────────────────────────────────────────────────────
    churned_df = df[df["Churn"] == "Yes"]
    revenue_lost_monthly = churned_df["MonthlyCharges"].sum()

    k1, k2, k3 = st.columns(3)
    with k1:
        kpi_card("Monthly Revenue Lost to Churn (Actual)", f"${revenue_lost_monthly:,.0f}")
    with k2:
        kpi_card(
            "Estimated Revenue at Risk (Active Customers)",
            f"${revenue_at_risk:,.0f}",
        )
        st.caption(
            "**Estimated** · Formula: Σ(ChurnProbability × MonthlyCharges) "
            "over all active customers."
        )
    with k3:
        kpi_card(
            f"High-Risk Active Customers (≥ {high_risk_threshold:.0%})",
            f"{high_risk_count:,}",
        )
        st.caption(
            f"Active customers with predicted churn probability ≥ {high_risk_threshold:.0%}. "
            "Adjust the threshold slider in the sidebar."
        )

    # ── Model metrics expander ────────────────────────────────────────────
    with st.expander("📈 Model Performance (Logistic Regression — Test Set)"):
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy",  f"{metrics['accuracy']:.1%}")
        m2.metric("Precision", f"{metrics['precision']:.1%}")
        m3.metric("Recall",    f"{metrics['recall']:.1%}")
        m4.metric("ROC-AUC",   f"{metrics['roc_auc']:.3f}")
        miss_rate = round(1 - metrics["recall"], 4)
        st.caption(
            "Trained on 80% of all customers (stratified split, random_state=42). "
            "Preprocessing: StandardScaler on numeric columns; OneHotEncoder on "
            "categorical columns. gender excluded (protected attribute). "
            f"Scores above are on the held-out 20% test set. "
            f"At a 0.5 threshold the model misses **{miss_rate:.1%}** of actual churners — "
            f"use scores to prioritise outreach, not as a final decision."
        )

    st.divider()

    # ── Heatmap: Churn Rate by Contract × Tenure Group ───────────────────
    st.subheader("Churn Rate Heatmap — Contract Type × Tenure Group")

    pivot = (
        df.groupby(["Contract", "TenureGroup"], observed=True)["Churn"]
        .apply(lambda s: round((s == "Yes").sum() / len(s) * 100, 1))
        .unstack("TenureGroup")
        .reindex(columns=TENURE_ORDER)
    )

    fig_heat = px.imshow(
        pivot,
        text_auto=True,
        color_continuous_scale=[[0, "#EFF6FF"], [0.5, "#93C5FD"], [1, COLOR_CHURN]],
        labels={"color": "Churn Rate (%)"},
        aspect="auto",
    )
    fig_heat.update_layout(margin=dict(t=30, b=40))
    st.plotly_chart(fig_heat, use_container_width=True)

    st.divider()

    # ── Revenue Lost by Segment (Internet Service) ────────────────────────
    st.subheader("Monthly Revenue Lost by Internet Service Segment")
    seg_rev = (
        churned_df.groupby("InternetService")["MonthlyCharges"]
        .sum()
        .reset_index()
        .rename(columns={"MonthlyCharges": "RevenueLost"})
        .sort_values("RevenueLost", ascending=False)
    )
    fig_seg = px.bar(
        seg_rev,
        x="InternetService",
        y="RevenueLost",
        color="RevenueLost",
        color_continuous_scale=[[0, COLOR_RETAIN], [1, COLOR_CHURN]],
        text=seg_rev["RevenueLost"].map(lambda v: f"${v:,.0f}"),
        labels={"RevenueLost": "Monthly Revenue Lost ($)", "InternetService": "Internet Service"},
    )
    fig_seg.update_traces(textposition="outside")
    fig_seg.update_layout(
        coloraxis_showscale=False,
        plot_bgcolor="white",
        yaxis=dict(range=[0, seg_rev["RevenueLost"].max() * 1.2]),
        margin=dict(t=30, b=40),
    )
    st.plotly_chart(fig_seg, use_container_width=True)

    st.divider()

    # ── Top 20 Highest-Risk Active Customers ─────────────────────────────
    st.subheader("Top 20 Highest-Risk Active Customers (Model Estimate)")
    st.caption(
        "Risk is scored using logistic regression. These are current active customers "
        "ranked by predicted churn probability. Use for proactive outreach."
    )

    top20 = (
        active_scored[["customerID", "Contract", "tenure", "MonthlyCharges", "ChurnProbability"]]
        .sort_values("ChurnProbability", ascending=False)
        .head(20)
        .reset_index(drop=True)
    )
    top20.index = top20.index + 1
    top20["ChurnProbability"] = top20["ChurnProbability"].map(lambda v: f"{v:.1%}")
    top20["MonthlyCharges"] = top20["MonthlyCharges"].map(lambda v: f"${v:.2f}")

    st.dataframe(top20, use_container_width=True)

    # CSV download
    top20_raw = (
        active_scored[["customerID", "Contract", "tenure", "MonthlyCharges", "ChurnProbability"]]
        .sort_values("ChurnProbability", ascending=False)
        .head(20)
    )
    csv_bytes = top20_raw.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download Top-20 At-Risk List as CSV",
        data=csv_bytes,
        file_name="top20_at_risk_customers.csv",
        mime="text/csv",
    )

    st.divider()

    # ── Risk, Opportunity and Action Cards ───────────────────────────────
    # --- Compute card numbers from data ---

    # Risk 1: Fiber optic churn
    fiber_df = df[df["InternetService"] == "Fiber optic"]
    fiber_churn_rate = (fiber_df["Churn"] == "Yes").mean() * 100
    fiber_rev_lost = fiber_df[fiber_df["Churn"] == "Yes"]["MonthlyCharges"].sum()

    # Risk 2: New-customer churn
    new_cust = df[df["TenureGroup"] == "0-12 months"]
    new_churn_rate = (new_cust["Churn"] == "Yes").mean() * 100

    # Risk 3: Electronic check churn
    echeck = df[df["PaymentMethod"] == "Electronic check"]
    echeck_churn_rate = (echeck["Churn"] == "Yes").mean() * 100
    echeck_churned = (echeck["Churn"] == "Yes").sum()

    # Opportunity 1: Contract conversion
    # Formula: converted_count × avg_MC_of_segment × Δchurn_rate
    mtm = df[df["Contract"] == "Month-to-month"]
    mtm_churn_rate = (mtm["Churn"] == "Yes").mean() * 100
    oneyear_churn_rate = (df[df["Contract"] == "One year"]["Churn"] == "Yes").mean() * 100
    mtm_active = mtm[mtm["Churn"] == "No"]
    mtm_count = len(mtm_active)
    converted_customers = int(mtm_count * pct_contract)
    avg_mc_mtm = mtm_active["MonthlyCharges"].mean()  # mean over the whole active MTM segment
    estimated_saved_monthly = avg_mc_mtm * converted_customers * (mtm_churn_rate - oneyear_churn_rate) / 100

    # Opportunity 2: Auto-pay migration
    # Formula: converted_count × avg_MC_of_segment × Δchurn_rate
    echeck_active = df[(df["PaymentMethod"] == "Electronic check") & (df["Churn"] == "No")]
    autopay_churn_rate = (
        df[df["PaymentMethod"].isin(["Bank transfer (automatic)", "Credit card (automatic)"])]["Churn"]
        .apply(lambda s: s == "Yes").mean() * 100
    )
    echeck_active_count = len(echeck_active)
    autopay_converted = int(echeck_active_count * pct_autopay)
    avg_mc_echeck = echeck_active["MonthlyCharges"].mean()  # mean over the whole active echeck segment
    estimated_autopay_saved = avg_mc_echeck * autopay_converted * (echeck_churn_rate - autopay_churn_rate) / 100

    # Opportunity 3: Tech support upsell
    # Formula: upsold_count × avg_MC_of_segment × Δchurn_rate
    no_ts = df[
        (df["TechSupport"] == "No") &
        (df["InternetService"] != "No") &
        (df["Churn"] == "No")
    ]
    ts_yes_churn_rate = (df[df["TechSupport"] == "Yes"]["Churn"] == "Yes").mean() * 100
    no_ts_churn_rate = (df[df["TechSupport"] == "No"]["Churn"] == "Yes").mean() * 100
    no_ts_count = len(no_ts)
    ts_upsold = int(no_ts_count * pct_techsupport)
    avg_mc_no_ts = no_ts["MonthlyCharges"].mean()  # mean over the whole active no-tech-support segment
    estimated_ts_saved = avg_mc_no_ts * ts_upsold * (no_ts_churn_rate - ts_yes_churn_rate) / 100

    # Compute whether fiber optic has the highest churn among internet service types (don't assume)
    isp_churn = df.groupby("InternetService")["Churn"].apply(lambda s: (s == "Yes").mean() * 100)
    fiber_is_highest = isp_churn.idxmax() == "Fiber optic"

    # --- Render cards ---
    st.subheader("⚠️ Risks")
    r1, r2, r3 = st.columns(3)

    with r1:
        st.error(
            f"**R1 — Fiber Optic Churn**\n\n"
            f"Fiber optic customers show a churn rate of **{fiber_churn_rate:.1f}%**"
            + (" — the highest of any internet service type in this dataset." if fiber_is_highest else ".")
            + f" This segment is associated with **${fiber_rev_lost:,.0f}/month** in lost monthly revenue."
        )
    with r2:
        st.error(
            f"**R2 — New-Customer Drop-Off**\n\n"
            f"**{new_churn_rate:.1f}%** of customers in their first 12 months churn "
            f"({len(new_cust):,} customers in this band). "
            f"Early churn limits the customer's lifetime value before meaningful revenue "
            f"has been generated."
        )
    with r3:
        st.error(
            f"**R3 — Electronic Check Fragility**\n\n"
            f"Electronic check users churn at **{echeck_churn_rate:.1f}%**. "
            f"{echeck_churned:,} customers in this payment group have already churned. "
            f"Manual payment is associated with lower engagement and commitment."
        )

    st.subheader("✅ Opportunities")
    o1, o2, o3 = st.columns(3)

    with o1:
        st.success(
            f"**O1 — Annual Contract Conversion**\n\n"
            f"Month-to-month churn: **{mtm_churn_rate:.1f}%**. "
            f"One-year contract churn: **{oneyear_churn_rate:.1f}%**. "
            f"There are **{mtm_count:,}** active month-to-month customers. "
            f"At {int(pct_contract*100)}% conversion: "
            f"**~{converted_customers:,} customers** converted.\n\n"
            f"**Estimated** additional revenue retained: "
            f"**${estimated_saved_monthly:,.0f}/month**\n\n"
            f"*Formula: {int(pct_contract*100)}% × {mtm_count:,} customers "
            f"× avg monthly charge (${avg_mc_mtm:.2f}) "
            f"× ({mtm_churn_rate:.1f}% − {oneyear_churn_rate:.1f}%) churn rate reduction. "
            f"Adjust the slider to explore scenarios.*"
        )
    with o2:
        st.success(
            f"**O2 — Auto-Pay Migration**\n\n"
            f"Electronic check churn: **{echeck_churn_rate:.1f}%**. "
            f"Auto-pay churn: **{autopay_churn_rate:.1f}%**. "
            f"There are **{echeck_active_count:,}** active electronic-check customers.\n\n"
            f"**Estimated** additional revenue retained at {int(pct_autopay*100)}% conversion: "
            f"**${estimated_autopay_saved:,.0f}/month**\n\n"
            f"*Formula: {int(pct_autopay*100)}% × {echeck_active_count:,} customers "
            f"× avg monthly charge (${avg_mc_echeck:.2f}) "
            f"× ({echeck_churn_rate:.1f}% − {autopay_churn_rate:.1f}%) churn rate reduction. "
            f"Adjust the slider to explore scenarios.*"
        )
    with o3:
        st.success(
            f"**O3 — Tech Support Upsell**\n\n"
            f"No tech-support churn: **{no_ts_churn_rate:.1f}%**. "
            f"With tech-support churn: **{ts_yes_churn_rate:.1f}%**. "
            f"There are **{no_ts_count:,}** active internet customers without tech support.\n\n"
            f"**Estimated** additional revenue retained at {int(pct_techsupport*100)}% upsell: "
            f"**${estimated_ts_saved:,.0f}/month**\n\n"
            f"*Formula: {int(pct_techsupport*100)}% × {no_ts_count:,} customers "
            f"× avg monthly charge (${avg_mc_no_ts:.2f}) "
            f"× ({no_ts_churn_rate:.1f}% − {ts_yes_churn_rate:.1f}%) churn rate reduction. "
            f"Adjust the slider to explore scenarios.*"
        )

    st.subheader("🎯 Recommended Actions")
    a1, a2, a3 = st.columns(3)

    with a1:
        st.info(
            f"**A1 — Year-1 Loyalty Offer**\n\n"
            f"Target the **{len(new_cust):,}** customers in their first 12 months "
            f"(churn rate: {new_churn_rate:.1f}%). "
            f"Offer a discounted annual contract or a free add-on service during onboarding. "
            f"The goal is to convert the highest-risk window into a commitment period before "
            f"the customer evaluates alternatives."
        )
    with a2:
        dsl_churn_rate = (df[df["InternetService"] == "DSL"]["Churn"] == "Yes").mean() * 100
        st.info(
            f"**A2 — Investigate Fiber Optic Value Perception**\n\n"
            f"Survey or interview churned fiber optic customers. "
            f"The fiber optic segment shows a **{fiber_churn_rate:.1f}%** churn rate "
            f"compared to **{dsl_churn_rate:.1f}%** for DSL — a difference that warrants "
            f"qualitative investigation into pricing or service quality. "
            f"This segment is associated with **${fiber_rev_lost:,.0f}/month** in monthly revenue loss."
        )
    with a3:
        st.info(
            f"**A3 — Auto-Pay Incentive Campaign**\n\n"
            f"Offer a small monthly discount (e.g., $5) to the **{echeck_active_count:,}** "
            f"active electronic-check customers who switch to automatic payment. "
            f"Electronic-check churn ({echeck_churn_rate:.1f}%) is notably higher than "
            f"auto-pay churn ({autopay_churn_rate:.1f}%), suggesting that manual payment "
            f"is associated with lower engagement. "
            f"Low cost to implement, high signal value."
        )


# ---------------------------------------------------------------------------
# Main app — sidebar navigation
# ---------------------------------------------------------------------------

def main():
    """Entry point: loads data, trains model, routes to the selected page."""

    # ── Sidebar ────────────────────────────────────────────────────────────
    st.sidebar.title("📊 Churn Dashboard")
    st.sidebar.markdown("**Customer Churn Decision Dashboard**")
    st.sidebar.markdown("---")

    page = st.sidebar.radio(
        "Navigate to",
        ["Page 1 — Executive Overview", "Page 2 — Services & Revenue", "Page 3 — Risk & Actions"],
    )

    # ── Data loading ──────────────────────────────────────────────────────
    csv_path = _find_csv()
    uploaded = None
    if csv_path is None:
        st.warning(
            f"No CSV file found in `{_DATA_DIR}`. "
            "Place any CSV file there, or upload it below."
        )
        uploaded = st.file_uploader("Upload Telco-Customer-Churn.csv", type="csv")
        if uploaded is None:
            st.stop()

    raw = load_data(csv_path=csv_path, uploaded_file=uploaded)
    df = clean_data(raw)
    pipeline, metrics = train_model(df)

    # ── Page routing ──────────────────────────────────────────────────────
    if page == "Page 1 — Executive Overview":
        page_executive_overview(df)
    elif page == "Page 2 — Services & Revenue":
        page_services_revenue(df)
    elif page == "Page 3 — Risk & Actions":
        page_risk_actions(df, pipeline, metrics)

    # ── Footer ─────────────────────────────────────────────────────────────
    st.sidebar.markdown("---")
    st.sidebar.caption(
        "Data: IBM Telco Customer Churn (Kaggle)\n"
        "Model: Logistic Regression (scikit-learn)\n"
        "Built with Streamlit + Plotly"
    )


if __name__ == "__main__":
    main()

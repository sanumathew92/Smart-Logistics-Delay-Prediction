# ============================================================
# IMPORT LIBRARIES
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from pathlib import Path


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Inventory Analytics",
    page_icon="📦",
    layout="wide"
)


# ============================================================
# PROJECT PATH
# ============================================================
# Current file:
#
# Smart Logistic Dataset/
# └── streamlit_app/
#     └── pages/
#         └── 5_Inventory_Analytics.py
#
# parents[0] = pages
# parents[1] = streamlit_app
# parents[2] = Smart Logistic Dataset
#
# Therefore parents[2] is the PROJECT ROOT.
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]


# ============================================================
# CORRECT DATASET LOCATION
# ============================================================

DATA_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "logistics_feature_engineered.csv"
)


# ============================================================
# LOAD DATASET
# ============================================================

@st.cache_data
def load_data():

    if not DATA_PATH.exists():

        st.error(
            f"""
            Dataset not found.

            Expected location:

            {DATA_PATH}

            Please verify that
            logistics_feature_engineered.csv
            exists inside:

            data/processed/
            """
        )

        return None

    try:

        df = pd.read_csv(DATA_PATH)

        return df

    except Exception as e:

        st.error(
            f"Unable to load dataset.\n\nError: {e}"
        )

        return None


# ============================================================
# LOAD DATA
# ============================================================

inventory_df = load_data()


# ============================================================
# STOP IF DATASET IS NOT AVAILABLE
# ============================================================

if inventory_df is None:

    st.stop()


# ============================================================
# PAGE TITLE
# ============================================================

st.title("📦 Inventory Analytics")

st.markdown(
    """
    Analyze inventory availability, demand pressure, stock risk,
    inventory coverage, and their relationship with logistics delays.
    """
)


# ============================================================
# DATASET INFORMATION
# ============================================================

st.caption(
    f"Dataset: {DATA_PATH}"
)

st.caption(
    f"Records: {len(inventory_df):,} | "
    f"Columns: {len(inventory_df.columns):,}"
)


# ============================================================
# REQUIRED COLUMN CHECK
# ============================================================

required_columns = [
    "Inventory_Level",
    "Demand_Forecast",
    "Inventory_Coverage",
    "Stock_Risk",
    "Logistics_Delay"
]


missing_columns = [
    col
    for col in required_columns
    if col not in inventory_df.columns
]


if missing_columns:

    st.warning(
        "The following expected columns are missing: "
        + ", ".join(missing_columns)
    )


# ============================================================
# CREATE SAFE COPY
# ============================================================

df = inventory_df.copy()


# ============================================================
# DATA TYPE CLEANING
# ============================================================

numeric_columns = [
    "Inventory_Level",
    "Demand_Forecast",
    "Inventory_Coverage",
    "Logistics_Delay",
    "Waiting_Time",
    "Asset_Utilization",
    "Fleet_Load_Index"
]


for col in numeric_columns:

    if col in df.columns:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )


# ============================================================
# KPI SECTION
# ============================================================

st.subheader("📊 Inventory KPIs")


col1, col2, col3, col4 = st.columns(4)


# ------------------------------------------------------------
# Average Inventory
# ------------------------------------------------------------

with col1:

    if "Inventory_Level" in df.columns:

        avg_inventory = df["Inventory_Level"].mean()

        st.metric(
            "Average Inventory",
            f"{avg_inventory:,.1f}"
        )

    else:

        st.metric(
            "Average Inventory",
            "N/A"
        )


# ------------------------------------------------------------
# Average Demand
# ------------------------------------------------------------

with col2:

    if "Demand_Forecast" in df.columns:

        avg_demand = df["Demand_Forecast"].mean()

        st.metric(
            "Average Demand",
            f"{avg_demand:,.1f}"
        )

    else:

        st.metric(
            "Average Demand",
            "N/A"
        )


# ------------------------------------------------------------
# Average Inventory Coverage
# ------------------------------------------------------------

with col3:

    if "Inventory_Coverage" in df.columns:

        avg_coverage = df["Inventory_Coverage"].mean()

        st.metric(
            "Avg Inventory Coverage",
            f"{avg_coverage:,.2f}"
        )

    else:

        st.metric(
            "Avg Inventory Coverage",
            "N/A"
        )


# ------------------------------------------------------------
# Delay Rate
# ------------------------------------------------------------

with col4:

    if "Logistics_Delay" in df.columns:

        delay_rate = (
            df["Logistics_Delay"]
            .mean()
            * 100
        )

        st.metric(
            "Delay Rate",
            f"{delay_rate:.2f}%"
        )

    else:

        st.metric(
            "Delay Rate",
            "N/A"
        )


# ============================================================
# INVENTORY RISK ANALYSIS
# ============================================================

st.divider()

st.subheader("🚨 Inventory Risk Analysis")


if "Stock_Risk" in df.columns:

    risk_df = (
        df["Stock_Risk"]
        .astype(str)
        .str.strip()
        .str.title()
        .value_counts()
        .reset_index()
    )

    risk_df.columns = [
        "Stock_Risk",
        "Count"
    ]

    col1, col2 = st.columns(2)


    # --------------------------------------------------------
    # Risk Distribution
    # --------------------------------------------------------

    with col1:

        fig = px.bar(
            risk_df,
            x="Stock_Risk",
            y="Count",
            title="Stock Risk Distribution",
            text_auto=True
        )

        fig.update_layout(
            xaxis_title="Stock Risk",
            yaxis_title="Number of Records"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    # --------------------------------------------------------
    # High Risk Count
    # --------------------------------------------------------

    with col2:

        high_risk_labels = [
            "High",
            "Critical",
            "Critical Risk"
        ]

        high_risk_count = (
            df["Stock_Risk"]
            .astype(str)
            .str.strip()
            .str.lower()
            .isin(
                [
                    x.lower()
                    for x in high_risk_labels
                ]
            )
            .sum()
        )

        total_records = len(df)

        high_risk_percentage = (
            high_risk_count /
            total_records *
            100
            if total_records > 0
            else 0
        )

        st.metric(
            "High / Critical Risk Records",
            f"{high_risk_count:,}"
        )

        st.metric(
            "High / Critical Risk %",
            f"{high_risk_percentage:.2f}%"
        )

else:

    st.info(
        "Stock_Risk column is not available."
    )


# ============================================================
# INVENTORY VS DEMAND
# ============================================================

st.divider()

st.subheader("📈 Inventory vs Demand")


if (
    "Inventory_Level" in df.columns
    and
    "Demand_Forecast" in df.columns
):

    sample_df = df[
        [
            "Inventory_Level",
            "Demand_Forecast"
        ]
    ].dropna()

    if len(sample_df) > 5000:

        sample_df = sample_df.sample(
            5000,
            random_state=42
        )


    fig = px.scatter(
        sample_df,
        x="Demand_Forecast",
        y="Inventory_Level",
        title="Inventory Level vs Forecasted Demand",
        opacity=0.5
    )

    fig.update_layout(
        xaxis_title="Demand Forecast",
        yaxis_title="Inventory Level"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

else:

    st.info(
        "Inventory_Level or Demand_Forecast is unavailable."
    )


# ============================================================
# INVENTORY COVERAGE
# ============================================================

st.divider()

st.subheader("📦 Inventory Coverage")


if "Inventory_Coverage" in df.columns:

    coverage_df = df[
        ["Inventory_Coverage"]
    ].dropna()

    fig = px.histogram(
        coverage_df,
        x="Inventory_Coverage",
        nbins=40,
        title="Inventory Coverage Distribution"
    )

    fig.update_layout(
        xaxis_title="Inventory Coverage",
        yaxis_title="Number of Records"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

else:

    st.info(
        "Inventory_Coverage column is unavailable."
    )


# ============================================================
# INVENTORY COVERAGE VS DELAYS
# ============================================================

st.divider()

st.subheader("🚚 Inventory Coverage vs Logistics Delays")


if (
    "Inventory_Coverage" in df.columns
    and
    "Logistics_Delay" in df.columns
):

    coverage_delay = (
        df.groupby(
            "Logistics_Delay",
            observed=True
        )["Inventory_Coverage"]
        .mean()
        .reset_index()
    )

    coverage_delay[
        "Delay_Status"
    ] = coverage_delay[
        "Logistics_Delay"
    ].map(
        {
            0: "No Delay",
            1: "Delay"
        }
    )

    coverage_delay[
        "Delay_Status"
    ] = coverage_delay[
        "Delay_Status"
    ].fillna(
        coverage_delay[
            "Logistics_Delay"
        ].astype(str)
    )


    fig = px.bar(
        coverage_delay,
        x="Delay_Status",
        y="Inventory_Coverage",
        text_auto=".2f",
        title="Average Inventory Coverage by Delay Status"
    )

    fig.update_layout(
        xaxis_title="Delivery Status",
        yaxis_title="Average Inventory Coverage"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

else:

    st.info(
        "Required inventory/delay columns are unavailable."
    )


# ============================================================
# STOCK RISK VS DELAY RATE
# ============================================================

st.divider()

st.subheader("⚠️ Stock Risk vs Delay Rate")


if (
    "Stock_Risk" in df.columns
    and
    "Logistics_Delay" in df.columns
):

    risk_delay = (
        df.groupby(
            "Stock_Risk",
            observed=True
        )["Logistics_Delay"]
        .mean()
        .reset_index()
    )

    risk_delay[
        "Delay_Rate"
    ] = (
        risk_delay[
            "Logistics_Delay"
        ] * 100
    )


    risk_delay[
        "Stock_Risk"
    ] = (
        risk_delay[
            "Stock_Risk"
        ]
        .astype(str)
    )


    fig = px.bar(
        risk_delay,
        x="Stock_Risk",
        y="Delay_Rate",
        text_auto=".2f",
        title="Logistics Delay Rate by Stock Risk"
    )

    fig.update_layout(
        xaxis_title="Stock Risk",
        yaxis_title="Delay Rate (%)"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

else:

    st.info(
        "Stock_Risk or Logistics_Delay is unavailable."
    )


# ============================================================
# INVENTORY DEMAND GAP
# ============================================================

st.divider()

st.subheader("📉 Inventory-Demand Gap")


if (
    "Inventory_Level" in df.columns
    and
    "Demand_Forecast" in df.columns
):

    if "Inventory_Demand_Gap" not in df.columns:

        df["Inventory_Demand_Gap"] = (
            df["Inventory_Level"]
            -
            df["Demand_Forecast"]
        )


    gap_df = df[
        ["Inventory_Demand_Gap"]
    ].dropna()


    fig = px.histogram(
        gap_df,
        x="Inventory_Demand_Gap",
        nbins=50,
        title="Inventory-Demand Gap Distribution"
    )

    fig.update_layout(
        xaxis_title="Inventory − Demand",
        yaxis_title="Number of Records"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# INVENTORY INSIGHTS
# ============================================================

st.divider()

st.subheader("💡 Business Insights")


insights = []


# ------------------------------------------------------------
# Inventory / Demand pressure
# ------------------------------------------------------------

if (
    "Inventory_Level" in df.columns
    and
    "Demand_Forecast" in df.columns
):

    avg_inventory = df[
        "Inventory_Level"
    ].mean()

    avg_demand = df[
        "Demand_Forecast"
    ].mean()

    if avg_demand > avg_inventory:

        insights.append(
            "🔴 Forecasted demand exceeds average inventory, "
            "indicating potential inventory pressure."
        )

    else:

        insights.append(
            "🟢 Average inventory is currently above "
            "forecasted demand."
        )


# ------------------------------------------------------------
# Stock risk
# ------------------------------------------------------------

if "Stock_Risk" in df.columns:

    high_risk_count = (
        df["Stock_Risk"]
        .astype(str)
        .str.lower()
        .isin(
            [
                "high",
                "critical",
                "critical risk"
            ]
        )
        .sum()
    )

    high_risk_pct = (
        high_risk_count /
        len(df) *
        100
        if len(df) > 0
        else 0
    )

    insights.append(
        f"⚠️ {high_risk_pct:.2f}% of records "
        f"are classified as High/Critical stock risk."
    )


# ------------------------------------------------------------
# Delay relationship
# ------------------------------------------------------------

if (
    "Stock_Risk" in df.columns
    and
    "Logistics_Delay" in df.columns
):

    risk_delay = (
        df.groupby(
            "Stock_Risk",
            observed=True
        )["Logistics_Delay"]
        .mean()
    )

    if len(risk_delay) > 1:

        highest_risk = (
            risk_delay
            .idxmax()
        )

        highest_delay = (
            risk_delay
            .max()
            * 100
        )

        insights.append(
            f"🚚 The {highest_risk} stock-risk category "
            f"has the highest observed delay rate "
            f"({highest_delay:.2f}%)."
        )


# ------------------------------------------------------------
# Display insights
# ------------------------------------------------------------

for insight in insights:

    st.markdown(
        f"- {insight}"
    )


# ============================================================
# DATA PREVIEW
# ============================================================

with st.expander("🔍 View Inventory Dataset"):

    st.dataframe(
        df.head(100),
        use_container_width=True
    )
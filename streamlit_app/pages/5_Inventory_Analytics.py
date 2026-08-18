# Inventory_Analytics.py 
# Smart Logistics Delay Prediction & Decision Support System

import streamlit as st 
import pandas as pd 
import numpy as np 
import plotly.express as px 

# Page Configuration

st.set_page_config(
    page_title="Inventory Analytics",
    page_icon = "📦",
    layout="wide"
)

# Title

st.title("📦 Inventory Analytics")

st.markdown(
    """
    Analyze inventory availability, demand pressure, stock rise, inventory coverage, and their relationship 
    with logistics delays.
    """
)

# Load Data

@st.cache_data
def load_data():

    data_path = (
        "C:/Users/Sanu/OneDrive/Desktop/"
        "Smart Logistic Dataset/data/processed/"
        "logistics_feature_engineered.csv"
    )

    return pd.read_csv(data_path)

try:

    df = load_data()

except FileNotFoundError:

    st.error(
        "Dataset not found. Please verify that "
        "`logistics_feature_engineered.csv` exists in the "
        "project data folder."
    )

    st.stop()

# Data Validation

required_columns = [
    "Inventory_Level",
    "Demand_Forecast",
    "Inventory_Coverage",
    "Inventory_Demand_Gap",
    "Stock_Risk",
    "Logistics_Delay"
]

missing_columns = [
    col 
    for col in required_columns
    if col not in df.columns
]

if missing_columns:

    st.error(
        f"The following required columns are missing: "
        f"{missing_columns}"
    )

    st.stop()

# Data Preparation

inventory_df = df.copy()

numeric_columns = [
    "Inventory_Level",
    "Demand_Forecast",
    "Inventory_Coverage",
    "Inventory_Demand_Gap",
    "Logistics_Delay"
]

for col in numeric_columns:

    if col in inventory_df.columns:

        inventory_df[col]= pd.to_numeric(
            inventory_df[col],
            errors="coerce"
        )

inventory_df = inventory_df.dropna(
    subset=[
        "Inventory_Level",
        "Demand_Forecast",
        "Logistics_Delay"
    ]
)

# KPI Calculations

total_shipments = len(inventory_df)

delayed_shipments = int(
    inventory_df["Logistics_Delay"].sum()
)

delay_rate = (
    delayed_shipments / total_shipments * 100
    if total_shipments > 0
    else 0
)

avg_inventory = inventory_df[
    "Inventory_Level"
].mean()

avg_demand = inventory_df[
    "Demand_Forecast"
].mean()

avg_coverage = inventory_df[
    "Inventory_Coverage"
].mean()

high_risk_count = 0

# ============================================================
# STOCK RISK ANALYSIS
# ============================================================

if "Stock_Risk" in inventory_df.columns:

    stock_risk = (
        inventory_df["Stock_Risk"]
        .fillna("Unknown")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    high_risk_count = stock_risk.isin(
        ["high", "critical", "critical risk"]
    ).sum()

    risk_rate = (
        high_risk_count / len(inventory_df) * 100
        if len(inventory_df) > 0
        else 0
    )

else:

    high_risk_count = 0
    risk_rate = 0


col1, col2 = st.columns(2)

with col1:
    st.metric(
        "High/Critical Stock Risk",
        f"{high_risk_count:,}"
    )

with col2:
    st.metric(
        "High-Risk Rate",
        f"{risk_rate:.1f}%"
    )

# KPI Cards

st.subheader("Inventory Performance Overview")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:

    st.metric(
        "Average Inventory",
        f"{avg_inventory:,.1f}"
    )

with col2:

    st.metric(
        "Average Demand",
        f"{avg_demand:,.1f}"
    )

with col3:

    st.metric(
        "Avg Inventory Coverage",
        f"{avg_coverage:,.2f}"
    )

with col4:

    st.metric(
        "High-Risk Inventory",
        f"{high_risk_count:,}"
    )

with col5:

    st.metric(
        "Overall Delay Rate",
        f"{delay_rate:.1f}%"
    )

st.divider()

# Inventory vs Demand

st.subheader("Inventory vs Demand")

col1, col2 = st.columns(2)

with col1:

    sample_df = inventory_df[
        [
            "Inventory_Level",
            "Demand_Forecast",
            "Logistics_Delay"
        ]
    ].dropna()

    if len(sample_df) > 5000:

        sample_df = sample_df.sample(
            5000,
            random_state=42
        )

    sample_df["Delay_Status"] = (
        sample_df["Logistics_Delay"]
        .map({
            0: "No Delay",
            1: "Delay"
        })
        .fillna("Unknown")
    )

    fig = px.scatter(
        sample_df,
        x="Inventory_Level",
        y="Demand_Forecast",
        color="Delay_Status",
        title="Inventory Level vs Demand Forecast",
        opacity=0.65,
        labels={
            "Inventory_Level": "Inventory Level",
            "Demand_Forecast": "Demand Forecast"
        }
         
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

with col2:

    gap_df = inventory_df[
        [
            "Inventory_Demand_Gap",
            "Logistics_Delay"
        ]
    ].dropna()

    gap_df["Delay_Status"] = (
        gap_df["Logistics_Delay"]
        .map({
            0: "No Delay",
            1: "Delay"
        })
        .fillna("Unknown")
    )

    if len(gap_df) > 5000:

        gap_df = gap_df.sample(
            5000,
            random_state=42
        )

    fig = px.scatter(
        gap_df,
        x="Inventory_Demand_Gap",
        y="Delay_Status",
        color="Delay_Status",
        title="Inventory Demand Gap vs Logistics Delay",
        opacity = 0.65,
        labels={
            "Inventory_Demand_Gap": "Inventory Demand Gap",
            "Logistics_Delay": "Logistics Delay"
        }
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# Inventory Demand Gap Analysis

st.subheader("Inventory Demand Gap Analysis")

gap_analysis = (
    inventory_df
    .groupby("Logistics_Delay")
    .agg(
        Average_Inventory_Demand_Gap=(
            "Inventory_Demand_Gap",
            "mean"
        ),
        Average_Inventory=(
            "Inventory_Level",
            "mean"
        ),
        Average_Demand=(
            "Demand_Forecast",
            "mean"
        ),
        Shipment_Count=(
            "Logistics_Delay",
            "count"
        )
    )
    .reset_index()
)

gap_analysis["Delay_Status"] = (
    gap_analysis["Logistics_Delay"]
    .map({
        0: "No Delay",
        1: "Delay"
    })
)

gap_analysis = gap_analysis[
    [
        "Delay_Status",
        "Average_Inventory_Demand_Gap",
        "Average_Inventory",
        "Average_Demand",
        "Shipment_Count"
    ]
]

st.dataframe(
    gap_analysis.round(2),
    use_container_width=True,
    hide_index=True
)

# Inventory Coverage Analysis

st.subheader("Inventory Coverage and Delay Risk")

coverage_df = inventory_df[
    [
        "Inventory_Coverage",
        "Logistics_Delay"

    ]
].dropna()

coverage_df["Delay_Status"] = (
    coverage_df["Logistics_Delay"]
    .map({
        0:"No Delay",
        1: "Delay"
    })
)

fig = px.box(
    coverage_df,
    x="Delay_Status",
    y="Inventory_Coverage",
    color="Delay_Status",
    points="outliers",
    title="Inventory Coverage by Delay Status",
    labels={
        "Delay_Status": "Delivery Status",
        "Inventory_Coverage": "Inventory Coverage"
    }
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# Stock Risk Analysis

if "Stock_Risk" in inventory_df.columns:

    st.subheader("Stock Risk Distribution")

    stock_risk_df= (
        inventory_df["Stock_Risk"]
        .astype(str)
        .value_counts()
        .reset_index()
    )

    stock_risk_df.columns = [
        "Stock_Risk",
        "Shipment_Count"
    ]

    stock_risk_df["Percentage"] = (
        stock_risk_df["Shipment_Count"]
        / stock_risk_df["Shipment_Count"].sum()
        * 100
    ).round(2)

    col1, col2 = st.columns(2)

    with col1:

        st.dataframe(
            stock_risk_df,
            use_container_width=True,
            hide_index=True
        )

    with col2:

        fig = px.pie(
            stock_risk_df,
            names="Stock_Risk",
            values="Shipment_Count",
            title="Stock Risk Distribution"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

# Stock Risk vs Delay 

if "Stock_Risk" in inventory_df.columns:

    st.subheader("Stock Risk vs Logistics Delay")

    risk_delay= (
        inventory_df
        .groupby("Stock_Risk")
        .agg(
            Total_Shipments=(
                "Logistics_Delay",
                "count"
            ),
            Delayed_Shipments=(
                "Logistics_Delay",
                "sum"
            )
        )
        .reset_index()
    )

    risk_delay["Delay_Rate"]= (
        risk_delay["Delayed_Shipments"]
        / risk_delay["Total_Shipments"]
        * 100
    ).round(2)


    st.dataframe(
        risk_delay,
        use_container_width=True,
        hide_index=True
    )

    fig = px.bar(
        risk_delay,
        x="Stock_Risk",
        y="Delay_Rate",
        text="Delay_Rate",
        title="Logistics Delay Rate by Stock Risk"
    )

    fig.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside"
    )

    fig.update_layout(
        yaxis_title="Delay Rate (%)",
        xaxis_title="Stock Risk"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# Inventory Band Analysis

if "Inventory_Band" in inventory_df.columns:

    st.subheader("Inventory Band Performance")

    band_df = inventory_df[
        [
            "Inventory_Band",
            "Logistics_Delay"
        ]
    ].dropna().copy()

    # Prevent Plotly Interval serialization errors
    band_df["Inventory_Band"] = (
        band_df["Inventory_Band"]
        .astype(str)
    )

    band_summary = (
        band_df
        .groupby("Inventory_Band")
        .agg(
            Total_Shipments=(
                "Logistics_Delay",
                "count"
            ),
            Delayed_Shipments=(
                "Logistics_Delay",
                "count"
            )
        )
        .reset_index()
    )

    band_summary["Delay_Rate"] = (
        band_summary["Delayed_Shipments"]
        / band_summary["Total_Shipments"]
        * 100
    ).round(2)

    st.dataframe(
        band_summary,
        use_container_width=True,
        hide_index=True
    )

    fig = px.bar(
        band_summary,
        x="Inventory_Band",
        y="Delay_Rate",
        text="Delay_Rate",
        title="Delay Rate by Inventory Band"
    )

    fig.update_traces(
        texttemplate="%{text:.1f}%",
        textposition= "outside"
    )
    fig.update_layout(
        yaxis_title="Delay Rate (%)",
        xaxis_title="Inventory Band"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# Inventory Insights

st.subheader("💡 Inventory Management Insights")

st.markdown(
    """
    **Management should focus on four inventory signals.**

    - **Inventory-demand gap:** A negative or insufficient inventory-demand gap can indicate increasing
    stock pressure and potential operation distruption.

    - **Inventory covergae:** Low coverage indicates that available inventory may not be sufficient to
    support expected demand.

    - **Stock risk:** High-risk inventory segments should receive proactive replenishment and
    operational attention. 

    -- **Delay relationship:** Inventory conditions should be monitored alongside logistics delay 
    rather than independently.
    """
)

# Business Recommendations

st.subheader("🎯  Business Recommendations")

recommendations = [
    (
        "1. Monitor inventory-demand gaps",
        "Prioritize shipments where forecast demand is approaching "
        "or exceeding available inventory."
    ),
    (
        "2. Protect low-coverage inventory",
        "Use inventory ccoverage as an early-warning indicator for "
        "potential fulfillment and logistics disruption."
    ),
    (
        "3. Prioritize high stock-risk segments",
        "Review high-risk inventory segments before dispatch planning "
        "and replenish critical stock where appropriate."
    ),
    (
        "4. Combine inventory and logistics intelligence",
        "Inventory risk should be evaluated together with traffic, "
        "fleet utilization, operational stress and delay probability."
    ),
    (
        "5. Use predictive delay alerts",
        "The Random Forest delay model can be used alongside inventory "
        "analytics to prioritize shipments requiring operational attention."
    )
]


for title, recommendation in recommendations:

    st.markdown(
        f"**{title}** - {recommendation}"
    )


# Footer 

st.divider()

st.caption(
    "Smart Logistics Delay Prediction & Decision Support System | "
    "Inventory Analytics"
)
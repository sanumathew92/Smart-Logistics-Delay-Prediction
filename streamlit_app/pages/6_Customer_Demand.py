# ============================================================
# 6_Customer_Demand.py
# Customer Demand Analytics
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Customer Demand Analytics",
    page_icon="📈",
    layout="wide"
)


# ============================================================
# PATH CONFIGURATION
# ============================================================

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_PATH = BASE_DIR / "data" / "processed" / "logistics_feature_engineered.csv"


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    if not DATA_PATH.exists():

        st.error(
            f"""
            Dataset not found.

            Expected location:
            {DATA_PATH}
            """
        )

        return None

    data = pd.read_csv(DATA_PATH)

    return data


df = load_data()


if df is None:

    st.stop()


# ============================================================
# PAGE HEADER
# ============================================================

st.title("📈 Customer Demand Analytics")

st.markdown(
    """
    Analyze customer demand patterns, purchasing behavior,
    demand pressure, customer value and their relationship
    with logistics delays.
    """
)

st.divider()


# ============================================================
# DATA PREPARATION
# ============================================================

demand_df = df.copy()


# Convert timestamp if available

if "Timestamp" in demand_df.columns:

    demand_df["Timestamp"] = pd.to_datetime(
        demand_df["Timestamp"],
        errors="coerce"
    )


# ============================================================
# SIDEBAR FILTERS
# ============================================================

st.sidebar.header("🔎 Filters")


# Traffic filter

if "Traffic_Status" in demand_df.columns:

    traffic_options = sorted(
        demand_df["Traffic_Status"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    selected_traffic = st.sidebar.multiselect(
        "Traffic Status",
        traffic_options,
        default=traffic_options
    )

    if selected_traffic:

        demand_df = demand_df[
            demand_df["Traffic_Status"]
            .astype(str)
            .isin(selected_traffic)
        ]


# Customer segment filter

if "Customer_Value_Segment" in demand_df.columns:

    segment_options = sorted(
        demand_df["Customer_Value_Segment"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    selected_segments = st.sidebar.multiselect(
        "Customer Value Segment",
        segment_options,
        default=segment_options
    )

    if selected_segments:

        demand_df = demand_df[
            demand_df["Customer_Value_Segment"]
            .astype(str)
            .isin(selected_segments)
        ]


# ============================================================
# KPI CALCULATIONS
# ============================================================

total_transactions = len(demand_df)


if "User_Transaction_Amount" in demand_df.columns:

    total_transaction_value = (
        pd.to_numeric(
            demand_df["User_Transaction_Amount"],
            errors="coerce"
        )
        .sum()
    )

else:

    total_transaction_value = 0


if "User_Purchase_Frequency" in demand_df.columns:

    avg_purchase_frequency = (
        pd.to_numeric(
            demand_df["User_Purchase_Frequency"],
            errors="coerce"
        )
        .mean()
    )

else:

    avg_purchase_frequency = 0


if "Demand_Forecast" in demand_df.columns:

    avg_demand_forecast = (
        pd.to_numeric(
            demand_df["Demand_Forecast"],
            errors="coerce"
        )
        .mean()
    )

else:

    avg_demand_forecast = 0


if "Logistics_Delay" in demand_df.columns:

    delay_rate = (
        pd.to_numeric(
            demand_df["Logistics_Delay"],
            errors="coerce"
        )
        .mean()
        * 100
    )

else:

    delay_rate = 0


# ============================================================
# KPI CARDS
# ============================================================

col1, col2, col3, col4, col5 = st.columns(5)


with col1:

    st.metric(
        "Transactions",
        f"{total_transactions:,}"
    )


with col2:

    st.metric(
        "Transaction Value",
        f"{total_transaction_value:,.0f}"
    )


with col3:

    st.metric(
        "Avg Purchase Frequency",
        f"{avg_purchase_frequency:.2f}"
    )


with col4:

    st.metric(
        "Avg Demand Forecast",
        f"{avg_demand_forecast:.2f}"
    )


with col5:

    st.metric(
        "Delay Rate",
        f"{delay_rate:.1f}%"
    )


st.divider()


# ============================================================
# DEMAND OVERVIEW
# ============================================================

st.subheader("📊 Demand Overview")


col1, col2 = st.columns(2)


# ------------------------------------------------------------
# Demand Forecast Distribution
# ------------------------------------------------------------

with col1:

    if "Demand_Forecast" in demand_df.columns:

        plot_df = demand_df[
            ["Demand_Forecast"]
        ].copy()

        plot_df["Demand_Forecast"] = pd.to_numeric(
            plot_df["Demand_Forecast"],
            errors="coerce"
        )

        plot_df = plot_df.dropna()

        fig = px.histogram(
            plot_df,
            x="Demand_Forecast",
            nbins=30,
            title="Demand Forecast Distribution"
        )

        fig.update_layout(
            xaxis_title="Demand Forecast",
            yaxis_title="Number of Transactions"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    else:

        st.info(
            "Demand_Forecast column is not available."
        )


# ------------------------------------------------------------
# Demand Score Distribution
# ------------------------------------------------------------

with col2:

    if "Demand_Score" in demand_df.columns:

        plot_df = demand_df[
            ["Demand_Score"]
        ].copy()

        plot_df["Demand_Score"] = pd.to_numeric(
            plot_df["Demand_Score"],
            errors="coerce"
        )

        plot_df = plot_df.dropna()

        fig = px.histogram(
            plot_df,
            x="Demand_Score",
            nbins=30,
            title="Demand Score Distribution"
        )

        fig.update_layout(
            xaxis_title="Demand Score",
            yaxis_title="Number of Transactions"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    else:

        st.info(
            "Demand_Score column is not available."
        )


# ============================================================
# DEMAND BY TIME
# ============================================================

st.subheader("🕒 Demand by Time")


col1, col2 = st.columns(2)


# ------------------------------------------------------------
# Hourly Demand
# ------------------------------------------------------------

with col1:

    if (
        "Hour" in demand_df.columns
        and "Demand_Forecast" in demand_df.columns
    ):

        hourly_demand = (
            demand_df
            .groupby("Hour", as_index=False)
            .agg(
                Average_Demand=(
                    "Demand_Forecast",
                    "mean"
                ),
                Transactions=(
                    "Demand_Forecast",
                    "count"
                )
            )
        )

        hourly_demand["Hour"] = pd.to_numeric(
            hourly_demand["Hour"],
            errors="coerce"
        )

        hourly_demand = hourly_demand.dropna()

        hourly_demand = hourly_demand.sort_values(
            "Hour"
        )

        fig = px.line(
            hourly_demand,
            x="Hour",
            y="Average_Demand",
            markers=True,
            title="Average Demand by Hour"
        )

        fig.update_layout(
            xaxis_title="Hour of Day",
            yaxis_title="Average Demand"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    else:

        st.info(
            "Hour or Demand_Forecast column is not available."
        )


# ------------------------------------------------------------
# Day-of-Week Demand
# ------------------------------------------------------------

with col2:

    if (
        "Day_of_Week" in demand_df.columns
        and "Demand_Forecast" in demand_df.columns
    ):

        weekday_demand = (
            demand_df
            .groupby("Day_of_Week", as_index=False)
            .agg(
                Average_Demand=(
                    "Demand_Forecast",
                    "mean"
                ),
                Transactions=(
                    "Demand_Forecast",
                    "count"
                )
            )
        )

        weekday_demand["Day_of_Week"] = (
            weekday_demand["Day_of_Week"]
            .astype(str)
        )

        fig = px.bar(
            weekday_demand,
            x="Day_of_Week",
            y="Average_Demand",
            title="Average Demand by Day of Week",
            text_auto=".2f"
        )

        fig.update_layout(
            xaxis_title="Day of Week",
            yaxis_title="Average Demand"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    else:

        st.info(
            "Day_of_Week or Demand_Forecast column is not available."
        )


# ============================================================
# CUSTOMER VALUE ANALYSIS
# ============================================================

st.subheader("👥 Customer Value Analysis")


col1, col2 = st.columns(2)


# ------------------------------------------------------------
# Customer Value Segment
# ------------------------------------------------------------

with col1:

    if "Customer_Value_Segment" in demand_df.columns:

        segment_summary = (
            demand_df
            .groupby(
                "Customer_Value_Segment",
                dropna=False
            )
            .size()
            .reset_index(
                name="Customers"
            )
        )

        segment_summary[
            "Customer_Value_Segment"
        ] = (
            segment_summary[
                "Customer_Value_Segment"
            ]
            .fillna("Unknown")
            .astype(str)
        )

        fig = px.bar(
            segment_summary,
            x="Customer_Value_Segment",
            y="Customers",
            title="Customer Value Segments",
            text_auto=True
        )

        fig.update_layout(
            xaxis_title="Customer Value Segment",
            yaxis_title="Transactions"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    else:

        st.info(
            "Customer_Value_Segment is not available."
        )


# ------------------------------------------------------------
# Customer Value Index
# ------------------------------------------------------------

with col2:

    if "Customer_Value_Index" in demand_df.columns:

        customer_value = pd.to_numeric(
            demand_df["Customer_Value_Index"],
            errors="coerce"
        ).dropna()

        fig = px.histogram(
            x=customer_value,
            nbins=30,
            title="Customer Value Index Distribution"
        )

        fig.update_layout(
            xaxis_title="Customer Value Index",
            yaxis_title="Transactions"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    else:

        st.info(
            "Customer_Value_Index is not available."
        )


# ============================================================
# PURCHASE BEHAVIOR
# ============================================================

st.subheader("🛒 Customer Purchase Behaviour")


col1, col2 = st.columns(2)


# ------------------------------------------------------------
# Purchase Frequency
# ------------------------------------------------------------

with col1:

    if "User_Purchase_Frequency" in demand_df.columns:

        purchase_frequency = pd.to_numeric(
            demand_df["User_Purchase_Frequency"],
            errors="coerce"
        )

        purchase_frequency = purchase_frequency.dropna()

        fig = px.histogram(
            x=purchase_frequency,
            nbins=30,
            title="Purchase Frequency Distribution"
        )

        fig.update_layout(
            xaxis_title="Purchase Frequency",
            yaxis_title="Transactions"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    else:

        st.info(
            "User_Purchase_Frequency is not available."
        )


# ------------------------------------------------------------
# Transaction Amount
# ------------------------------------------------------------

with col2:

    if "User_Transaction_Amount" in demand_df.columns:

        transaction_amount = pd.to_numeric(
            demand_df["User_Transaction_Amount"],
            errors="coerce"
        )

        transaction_amount = transaction_amount.dropna()

        fig = px.histogram(
            x=transaction_amount,
            nbins=30,
            title="Transaction Amount Distribution"
        )

        fig.update_layout(
            xaxis_title="Transaction Amount",
            yaxis_title="Transactions"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


# ============================================================
# DEMAND VS INVENTORY
# ============================================================

st.subheader("📦 Demand vs Inventory Pressure")


if (
    "Inventory_Level" in demand_df.columns
    and "Demand_Forecast" in demand_df.columns
):

    sample_df = demand_df[
        [
            "Inventory_Level",
            "Demand_Forecast"
        ]
    ].copy()

    sample_df["Inventory_Level"] = pd.to_numeric(
        sample_df["Inventory_Level"],
        errors="coerce"
    )

    sample_df["Demand_Forecast"] = pd.to_numeric(
        sample_df["Demand_Forecast"],
        errors="coerce"
    )

    sample_df = sample_df.dropna()

    # Avoid excessively large browser payloads

    if len(sample_df) > 5000:

        sample_df = sample_df.sample(
            5000,
            random_state=42
        )

    fig = px.scatter(
        sample_df,
        x="Inventory_Level",
        y="Demand_Forecast",
        title="Inventory Level vs Demand Forecast",
        opacity=0.5
    )

    fig.update_layout(
        xaxis_title="Inventory Level",
        yaxis_title="Demand Forecast"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

else:

    st.info(
        "Inventory_Level or Demand_Forecast is not available."
    )


# ============================================================
# INVENTORY DEMAND GAP
# ============================================================

if "Inventory_Demand_Gap" in demand_df.columns:

    st.subheader("⚠️ Inventory Demand Gap")

    gap_df = demand_df[
        ["Inventory_Demand_Gap"]
    ].copy()

    gap_df["Inventory_Demand_Gap"] = pd.to_numeric(
        gap_df["Inventory_Demand_Gap"],
        errors="coerce"
    )

    gap_df = gap_df.dropna()

    if not gap_df.empty:

        fig = px.histogram(
            gap_df,
            x="Inventory_Demand_Gap",
            nbins=30,
            title="Inventory Demand Gap Distribution"
        )

        fig.update_layout(
            xaxis_title="Inventory Demand Gap",
            yaxis_title="Transactions"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


# ============================================================
# DEMAND AND LOGISTICS DELAY
# ============================================================

st.subheader("🚚 Demand vs Logistics Delays")


if (
    "Demand_Forecast" in demand_df.columns
    and "Logistics_Delay" in demand_df.columns
):

    delay_demand = (
        demand_df
        .groupby("Logistics_Delay", as_index=False)
        .agg(
            Average_Demand=(
                "Demand_Forecast",
                "mean"
            ),
            Transactions=(
                "Demand_Forecast",
                "count"
            )
        )
    )

    delay_demand["Logistics_Delay"] = (
        pd.to_numeric(
            delay_demand["Logistics_Delay"],
            errors="coerce"
        )
    )

    delay_demand = delay_demand.dropna()

    delay_demand["Delay_Status"] = (
        delay_demand["Logistics_Delay"]
        .map({
            0: "No Delay",
            1: "Delay"
        })
        .fillna(
            delay_demand["Logistics_Delay"]
            .astype(str)
        )
    )

    fig = px.bar(
        delay_demand,
        x="Delay_Status",
        y="Average_Demand",
        title="Average Demand: Delayed vs Non-Delayed",
        text_auto=".2f"
    )

    fig.update_layout(
        xaxis_title="Delivery Status",
        yaxis_title="Average Demand"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# TRAFFIC AND DEMAND
# ============================================================

if (
    "Traffic_Status" in demand_df.columns
    and "Demand_Forecast" in demand_df.columns
):

    st.subheader("🚦 Traffic Conditions and Demand")

    traffic_demand = (
        demand_df
        .groupby(
            "Traffic_Status",
            dropna=False
        )
        .agg(
            Average_Demand=(
                "Demand_Forecast",
                "mean"
            ),
            Transactions=(
                "Demand_Forecast",
                "count"
            )
        )
        .reset_index()
    )

    traffic_demand["Traffic_Status"] = (
        traffic_demand["Traffic_Status"]
        .fillna("Unknown")
        .astype(str)
    )

    fig = px.bar(
        traffic_demand,
        x="Traffic_Status",
        y="Average_Demand",
        title="Average Demand by Traffic Status",
        text_auto=".2f"
    )

    fig.update_layout(
        xaxis_title="Traffic Status",
        yaxis_title="Average Demand"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# KEY BUSINESS INSIGHTS
# ============================================================

st.divider()

st.subheader("💡 Key Business Insights")


insights = []


# Demand insight

if "Demand_Forecast" in demand_df.columns:

    demand_values = pd.to_numeric(
        demand_df["Demand_Forecast"],
        errors="coerce"
    ).dropna()

    if not demand_values.empty:

        high_demand_threshold = demand_values.quantile(
            0.75
        )

        high_demand_pct = (
            demand_values.ge(
                high_demand_threshold
            ).mean()
            * 100
        )

        insights.append(
            f"• Approximately **{high_demand_pct:.1f}%** "
            "of observations fall within the high-demand range."
        )


# Inventory pressure insight

if "Inventory_Demand_Gap" in demand_df.columns:

    gap_values = pd.to_numeric(
        demand_df["Inventory_Demand_Gap"],
        errors="coerce"
    ).dropna()

    if not gap_values.empty:

        negative_gap_pct = (
            gap_values.lt(0).mean()
            * 100
        )

        insights.append(
            f"• **{negative_gap_pct:.1f}%** of observations "
            "show an inventory-demand gap below zero, "
            "indicating potential inventory pressure."
        )


# Customer value insight

if "Customer_Value_Segment" in demand_df.columns:

    segment_counts = (
        demand_df[
            "Customer_Value_Segment"
        ]
        .fillna("Unknown")
        .astype(str)
        .value_counts()
    )

    if not segment_counts.empty:

        top_segment = segment_counts.index[0]

        insights.append(
            f"• The largest customer value segment is "
            f"**{top_segment}**."
        )


# Delay insight

if "Logistics_Delay" in demand_df.columns:

    delay_values = pd.to_numeric(
        demand_df["Logistics_Delay"],
        errors="coerce"
    )

    valid_delay = delay_values.dropna()

    if not valid_delay.empty:

        delay_percentage = (
            valid_delay.mean()
            * 100
        )

        insights.append(
            f"• Overall logistics delay exposure "
            f"is approximately **{delay_percentage:.1f}%**."
        )


for insight in insights:

    st.markdown(insight)


# ============================================================
# MANAGEMENT RECOMMENDATIONS
# ============================================================

st.subheader("🎯 Management Recommendations")

st.markdown(
    """
    **1. Prepare capacity for high-demand periods**

    Use hourly and daily demand patterns to align fleet,
    inventory and operational staffing with expected demand.

    **2. Monitor inventory-demand pressure**

    Negative inventory-demand gaps should receive early
    operational attention because insufficient inventory
    relative to demand can increase service risk.

    **3. Prioritize high-value customers**

    Customer Value Index and customer segments can help
    management prioritize service quality and capacity
    allocation for high-value customer groups.

    **4. Incorporate traffic conditions into planning**

    Demand should be interpreted together with traffic
    conditions because elevated traffic can increase
    operational pressure and potential delays.

    **5. Connect demand intelligence with delay prediction**

    The demand indicators shown here complement the
    Random Forest delay prediction model used elsewhere
    in the dashboard.
    """
)


# ============================================================
# DATA TABLE
# ============================================================

with st.expander("📋 View Customer Demand Data"):

    st.dataframe(
        demand_df.head(500),
        use_container_width=True
    )


# ============================================================
# FOOTER
# ============================================================

st.caption(
    "Customer Demand Analytics | Smart Logistics "
    "Predictive Decision Support System"
)
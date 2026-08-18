# ============================================================
# 8_Delay_Diagnostics.py
# Logistics Delay Diagnostics
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
    page_title="Delay Diagnostics",
    page_icon="🚨",
    layout="wide"
)


# ============================================================
# CUSTOM HEADER
# ============================================================

st.title("🚨 Delay Diagnostics")
st.markdown(
    """
    ### Understanding the drivers and operational patterns behind logistics delays

    This page provides diagnostic analysis of historical logistics delays,
    identifying major operational, traffic, inventory and demand-related
    factors associated with delayed shipments.
    """
)

st.divider()


# ============================================================
# PATH CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_PATH = (
    BASE_DIR
    / "data" / "processed"
    / "logistics_feature_engineered.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    if not DATA_PATH.exists():
        return None

    data = pd.read_csv(DATA_PATH)

    return data


df = load_data()


# ============================================================
# DATA VALIDATION
# ============================================================

if df is None:

    st.error(
        "Dataset not found.\n\n"
        f"Expected location:\n`{DATA_PATH}`"
    )

    st.stop()


if df.empty:

    st.error("The dataset is empty.")

    st.stop()


# ============================================================
# DATA PREPARATION
# ============================================================

# Convert Timestamp if available
if "Timestamp" in df.columns:

    df["Timestamp"] = pd.to_datetime(
        df["Timestamp"],
        errors="coerce"
    )


# Validate target column
if "Logistics_Delay" not in df.columns:

    st.error(
        "Required column `Logistics_Delay` "
        "is missing from the dataset."
    )

    st.stop()


# Convert delay target to numeric
df["Logistics_Delay"] = pd.to_numeric(
    df["Logistics_Delay"],
    errors="coerce"
).fillna(0)


# Ensure binary target
df["Logistics_Delay"] = (
    df["Logistics_Delay"] > 0
).astype(int)


# ============================================================
# PAGE INFORMATION
# ============================================================

with st.expander("ℹ️ About this page"):

    st.markdown(
        """
        **Business purpose**

        Delay diagnostics helps management understand:

        - How frequently logistics delays occur
        - Which traffic conditions contribute to delays
        - Which shipment statuses experience more delays
        - The major reasons for delayed shipments
        - Whether operational stress is associated with delays
        - Whether inventory and demand conditions contribute to delay risk

        These insights support proactive logistics planning and operational
        intervention.
        """
    )


# ============================================================
# SECTION 1 — EXECUTIVE DELAY KPIs
# ============================================================

st.header("📊 Delay Performance Overview")


Total_Shipments = len(df)


Total_Delays = int(
    df["Logistics_Delay"].sum()
)


On_Time_Shipments = (
    Total_Shipments - Total_Delays
)


Delay_Rate = (
    Total_Delays / Total_Shipments * 100
    if Total_Shipments > 0
    else 0
)


On_Time_Rate = (
    On_Time_Shipments / Total_Shipments * 100
    if Total_Shipments > 0
    else 0
)


col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "Total Shipments",
        f"{Total_Shipments:,}"
    )


with col2:

    st.metric(
        "Delayed Shipments",
        f"{Total_Delays:,}"
    )


with col3:

    st.metric(
        "Delay Rate",
        f"{Delay_Rate:.1f}%"
    )


with col4:

    st.metric(
        "On-Time Rate",
        f"{On_Time_Rate:.1f}%"
    )


st.divider()


# ============================================================
# SECTION 2 — DELAY VS ON-TIME
# ============================================================

st.header("📦 Shipment Outcome Distribution")


delay_distribution = pd.DataFrame(
    {
        "Status": [
            "On-Time",
            "Delayed"
        ],
        "Shipments": [
            On_Time_Shipments,
            Total_Delays
        ]
    }
)


fig_delay_distribution = px.pie(
    delay_distribution,
    names="Status",
    values="Shipments",
    hole=0.45,
    title="On-Time vs Delayed Shipments"
)


fig_delay_distribution.update_layout(
    height=450
)


st.plotly_chart(
    fig_delay_distribution,
    use_container_width=True
)


# ============================================================
# SECTION 3 — DELAY BY TRAFFIC STATUS
# ============================================================

st.header("🚦 Traffic Impact on Delays")


if "Traffic_Status" in df.columns:

    traffic_summary = (
        df.groupby(
            "Traffic_Status",
            dropna=False
        )
        .agg(
            Total_Shipments=(
                "Logistics_Delay",
                "size"
            ),
            Delays=(
                "Logistics_Delay",
                "sum"
            )
        )
        .reset_index()
    )


    traffic_summary["Traffic_Status"] = (
        traffic_summary["Traffic_Status"]
        .fillna("Unknown")
        .astype(str)
    )


    traffic_summary["Delay_Rate"] = (
        traffic_summary["Delays"]
        / traffic_summary["Total_Shipments"]
        * 100
    )


    traffic_summary = traffic_summary.sort_values(
        "Delay_Rate",
        ascending=False
    )


    fig_traffic = px.bar(
        traffic_summary,
        x="Traffic_Status",
        y="Delay_Rate",
        text="Delay_Rate",
        title="Delay Rate by Traffic Status"
    )


    fig_traffic.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside"
    )


    fig_traffic.update_layout(
        yaxis_title="Delay Rate (%)",
        xaxis_title="Traffic Status",
        yaxis_range=[
            0,
            max(
                traffic_summary["Delay_Rate"].max() * 1.2,
                10
            )
        ]
    )


    st.plotly_chart(
        fig_traffic,
        use_container_width=True
    )


    st.dataframe(
        traffic_summary,
        use_container_width=True,
        hide_index=True
    )


else:

    st.warning(
        "`Traffic_Status` column is not available."
    )


st.divider()


# ============================================================
# SECTION 4 — DELAY BY SHIPMENT STATUS
# ============================================================

st.header("🚚 Shipment Status and Delay")


if "Shipment_Status" in df.columns:

    shipment_summary = (
        df.groupby(
            "Shipment_Status",
            dropna=False
        )
        .agg(
            Total_Shipments=(
                "Logistics_Delay",
                "size"
            ),
            Delays=(
                "Logistics_Delay",
                "sum"
            )
        )
        .reset_index()
    )


    shipment_summary["Shipment_Status"] = (
        shipment_summary["Shipment_Status"]
        .fillna("Unknown")
        .astype(str)
    )


    shipment_summary["Delay_Rate"] = (
        shipment_summary["Delays"]
        / shipment_summary["Total_Shipments"]
        * 100
    )


    shipment_summary = shipment_summary.sort_values(
        "Delay_Rate",
        ascending=False
    )


    fig_status = px.bar(
        shipment_summary,
        x="Shipment_Status",
        y="Delay_Rate",
        text="Delay_Rate",
        title="Delay Rate by Shipment Status"
    )


    fig_status.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside"
    )


    fig_status.update_layout(
        yaxis_title="Delay Rate (%)",
        xaxis_title="Shipment Status"
    )


    st.plotly_chart(
        fig_status,
        use_container_width=True
    )


else:

    st.warning(
        "`Shipment_Status` column is not available."
    )


st.divider()


# ============================================================
# SECTION 5 — DELAY REASONS
# ============================================================

st.header("🔎 Primary Logistics Delay Reasons")


if "Logistics_Delay_Reason" in df.columns:

    reason_df = df[
        df["Logistics_Delay"] == 1
    ].copy()


    if len(reason_df) > 0:

        reason_summary = (
            reason_df[
                "Logistics_Delay_Reason"
            ]
            .fillna("Unknown")
            .astype(str)
            .value_counts()
            .reset_index()
        )


        reason_summary.columns = [
            "Delay_Reason",
            "Delayed_Shipments"
        ]


        reason_summary["Percentage"] = (
            reason_summary["Delayed_Shipments"]
            / reason_summary["Delayed_Shipments"].sum()
            * 100
        )


        top_reasons = reason_summary.head(10)


        fig_reasons = px.bar(
            top_reasons,
            x="Delayed_Shipments",
            y="Delay_Reason",
            orientation="h",
            text="Delayed_Shipments",
            title="Top 10 Delay Reasons"
        )


        fig_reasons.update_traces(
            textposition="outside"
        )


        fig_reasons.update_layout(
            yaxis={
                "categoryorder": "total ascending"
            },
            xaxis_title="Delayed Shipments",
            yaxis_title="Delay Reason"
        )


        st.plotly_chart(
            fig_reasons,
            use_container_width=True
        )


        st.dataframe(
            reason_summary,
            use_container_width=True,
            hide_index=True
        )


    else:

        st.info(
            "No delayed shipments are available for "
            "delay-reason analysis."
        )


else:

    st.warning(
        "`Logistics_Delay_Reason` column is not available."
    )


st.divider()


# ============================================================
# SECTION 6 — OPERATIONAL STRESS ANALYSIS
# ============================================================

st.header("⚠️ Operational Stress and Delay")


if "Operational_Stress_Level" in df.columns:

    stress_summary = (
        df.groupby(
            "Operational_Stress_Level",
            dropna=False
        )
        .agg(
            Total_Shipments=(
                "Logistics_Delay",
                "size"
            ),
            Delays=(
                "Logistics_Delay",
                "sum"
            )
        )
        .reset_index()
    )


    stress_summary[
        "Operational_Stress_Level"
    ] = (
        stress_summary[
            "Operational_Stress_Level"
        ]
        .fillna("Unknown")
        .astype(str)
    )


    stress_summary["Delay_Rate"] = (
        stress_summary["Delays"]
        / stress_summary["Total_Shipments"]
        * 100
    )


    fig_stress = px.bar(
        stress_summary,
        x="Operational_Stress_Level",
        y="Delay_Rate",
        text="Delay_Rate",
        title="Delay Rate by Operational Stress Level"
    )


    fig_stress.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside"
    )


    fig_stress.update_layout(
        yaxis_title="Delay Rate (%)",
        xaxis_title="Operational Stress Level"
    )


    st.plotly_chart(
        fig_stress,
        use_container_width=True
    )


    st.dataframe(
        stress_summary,
        use_container_width=True,
        hide_index=True
    )


else:

    st.warning(
        "`Operational_Stress_Level` column is not available."
    )


st.divider()


# ============================================================
# SECTION 7 — INVENTORY RISK AND DELAYS
# ============================================================

st.header("📦 Inventory Risk and Delay")


if "Stock_Risk" in df.columns:

    inventory_summary = (
        df.groupby(
            "Stock_Risk",
            dropna=False
        )
        .agg(
            Total_Shipments=(
                "Logistics_Delay",
                "size"
            ),
            Delays=(
                "Logistics_Delay",
                "sum"
            )
        )
        .reset_index()
    )


    inventory_summary["Stock_Risk"] = (
        inventory_summary["Stock_Risk"]
        .fillna("Unknown")
        .astype(str)
    )


    inventory_summary["Delay_Rate"] = (
        inventory_summary["Delays"]
        / inventory_summary["Total_Shipments"]
        * 100
    )


    inventory_summary = inventory_summary.sort_values(
        "Delay_Rate",
        ascending=False
    )


    fig_inventory = px.bar(
        inventory_summary,
        x="Stock_Risk",
        y="Delay_Rate",
        text="Delay_Rate",
        title="Delay Rate by Inventory Risk"
    )


    fig_inventory.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside"
    )


    fig_inventory.update_layout(
        yaxis_title="Delay Rate (%)",
        xaxis_title="Stock Risk"
    )


    st.plotly_chart(
        fig_inventory,
        use_container_width=True
    )


else:

    st.warning(
        "`Stock_Risk` column is not available."
    )


st.divider()


# ============================================================
# SECTION 8 — DELAY BY TIME PERIOD
# ============================================================

st.header("🕐 Temporal Delay Patterns")


if "Hour" in df.columns:

    hourly_summary = (
        df.groupby("Hour")
        .agg(
            Total_Shipments=(
                "Logistics_Delay",
                "size"
            ),
            Delays=(
                "Logistics_Delay",
                "sum"
            )
        )
        .reset_index()
    )


    hourly_summary["Delay_Rate"] = (
        hourly_summary["Delays"]
        / hourly_summary["Total_Shipments"]
        * 100
    )


    fig_hour = px.line(
        hourly_summary,
        x="Hour",
        y="Delay_Rate",
        markers=True,
        title="Hourly Delay Rate"
    )


    fig_hour.update_layout(
        xaxis_title="Hour of Day",
        yaxis_title="Delay Rate (%)"
    )


    st.plotly_chart(
        fig_hour,
        use_container_width=True
    )


# ============================================================
# DAY OF WEEK
# ============================================================

if "Day_of_Week" in df.columns:

    weekday_summary = (
        df.groupby(
            "Day_of_Week",
            dropna=False
        )
        .agg(
            Total_Shipments=(
                "Logistics_Delay",
                "size"
            ),
            Delays=(
                "Logistics_Delay",
                "sum"
            )
        )
        .reset_index()
    )


    weekday_summary["Day_of_Week"] = (
        weekday_summary["Day_of_Week"]
        .fillna("Unknown")
        .astype(str)
    )


    weekday_summary["Delay_Rate"] = (
        weekday_summary["Delays"]
        / weekday_summary["Total_Shipments"]
        * 100
    )


    fig_weekday = px.bar(
        weekday_summary,
        x="Day_of_Week",
        y="Delay_Rate",
        text="Delay_Rate",
        title="Delay Rate by Day of Week"
    )


    fig_weekday.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside"
    )


    fig_weekday.update_layout(
        yaxis_title="Delay Rate (%)",
        xaxis_title="Day of Week"
    )


    st.plotly_chart(
        fig_weekday,
        use_container_width=True
    )


st.divider()


# ============================================================
# SECTION 9 — WAITING TIME AND DELAYS
# ============================================================

st.header("⏳ Waiting Time and Delay Relationship")


if "Waiting_Time" in df.columns:

    waiting_df = df[
        [
            "Waiting_Time",
            "Logistics_Delay"
        ]
    ].copy()


    waiting_df["Waiting_Time"] = pd.to_numeric(
        waiting_df["Waiting_Time"],
        errors="coerce"
    )


    waiting_df = waiting_df.dropna()


    if len(waiting_df) > 0:

        waiting_df["Waiting_Band"] = pd.cut(
            waiting_df["Waiting_Time"],
            bins=5,
            duplicates="drop"
        ).astype(str)


        waiting_summary = (
            waiting_df.groupby(
                "Waiting_Band",
                observed=True
            )
            .agg(
                Total_Shipments=(
                    "Logistics_Delay",
                    "size"
                ),
                Delays=(
                    "Logistics_Delay",
                    "sum"
                )
            )
            .reset_index()
        )


        waiting_summary["Delay_Rate"] = (
            waiting_summary["Delays"]
            / waiting_summary["Total_Shipments"]
            * 100
        )


        # Convert categorical interval to string
        waiting_summary["Waiting_Band"] = (
            waiting_summary["Waiting_Band"]
            .astype(str)
        )


        fig_waiting = px.bar(
            waiting_summary,
            x="Waiting_Band",
            y="Delay_Rate",
            text="Delay_Rate",
            title="Delay Rate by Waiting-Time Band"
        )


        fig_waiting.update_traces(
            texttemplate="%{text:.1f}%",
            textposition="outside"
        )


        fig_waiting.update_layout(
            xaxis_title="Waiting Time Band",
            yaxis_title="Delay Rate (%)"
        )


        st.plotly_chart(
            fig_waiting,
            use_container_width=True
        )


st.divider()


# ============================================================
# SECTION 10 — DISTANCE FROM CENTER
# ============================================================

st.header("📍 Distance and Delay Analysis")


if "Distance_From_Center" in df.columns:

    distance_df = df[
        [
            "Distance_From_Center",
            "Logistics_Delay"
        ]
    ].copy()


    distance_df["Distance_From_Center"] = pd.to_numeric(
        distance_df["Distance_From_Center"],
        errors="coerce"
    )


    distance_df = distance_df.dropna()


    if len(distance_df) > 0:

        # Sample to keep dashboard responsive
        sample_size = min(
            5000,
            len(distance_df)
        )


        sample_df = distance_df.sample(
            sample_size,
            random_state=42
        )


        fig_distance = px.scatter(
            sample_df,
            x="Distance_From_Center",
            y="Logistics_Delay",
            title="Distance from Center vs Delay",
            opacity=0.5
        )


        fig_distance.update_layout(
            xaxis_title="Distance From Center",
            yaxis_title="Logistics Delay"
        )


        st.plotly_chart(
            fig_distance,
            use_container_width=True
        )


else:

    st.info(
        "`Distance_From_Center` is not available."
    )


st.divider()


# ============================================================
# SECTION 11 — DEMAND VS DELAY
# ============================================================

st.header("📈 Demand Conditions and Delay")


if "Demand_Forecast" in df.columns:

    demand_df = df[
        [
            "Demand_Forecast",
            "Logistics_Delay"
        ]
    ].copy()


    demand_df["Demand_Forecast"] = pd.to_numeric(
        demand_df["Demand_Forecast"],
        errors="coerce"
    )


    demand_df = demand_df.dropna()


    if len(demand_df) > 0:

        sample_size = min(
            5000,
            len(demand_df)
        )


        sample_demand = demand_df.sample(
            sample_size,
            random_state=42
        )


        fig_demand = px.scatter(
            sample_demand,
            x="Demand_Forecast",
            y="Logistics_Delay",
            opacity=0.5,
            title="Demand Forecast vs Logistics Delay"
        )


        fig_demand.update_layout(
            xaxis_title="Demand Forecast",
            yaxis_title="Logistics Delay"
        )


        st.plotly_chart(
            fig_demand,
            use_container_width=True
        )


st.divider()


# ============================================================
# SECTION 12 — DELAY DIAGNOSTIC SUMMARY
# ============================================================

st.header("💡 Business Diagnostic Summary")


# Determine highest traffic delay category
traffic_insight = None

if (
    "Traffic_Status" in df.columns
    and len(traffic_summary) > 0
):

    highest_traffic = traffic_summary.iloc[0]

    traffic_insight = (
        f"**Traffic:** "
        f"{highest_traffic['Traffic_Status']} traffic "
        f"has the highest observed delay rate "
        f"({highest_traffic['Delay_Rate']:.1f}%)."
    )


# Determine highest stress level
stress_insight = None

if (
    "Operational_Stress_Level" in df.columns
    and len(stress_summary) > 0
):

    highest_stress = stress_summary.iloc[
        stress_summary["Delay_Rate"].argmax()
    ]

    stress_insight = (
        f"**Operational Stress:** "
        f"{highest_stress['Operational_Stress_Level']} "
        f"stress conditions show a delay rate of "
        f"{highest_stress['Delay_Rate']:.1f}%."
    )


# Determine top delay reason
reason_insight = None

if (
    "Logistics_Delay_Reason" in df.columns
    and "reason_summary" in locals()
    and len(reason_summary) > 0
):

    top_reason = reason_summary.iloc[0]

    reason_insight = (
        f"**Delay Reason:** "
        f"{top_reason['Delay_Reason']} is the most frequent "
        f"recorded delay reason, representing "
        f"{top_reason['Percentage']:.1f}% of delayed shipments."
    )


# Display insights

if traffic_insight:

    st.markdown(
        f"🔴 {traffic_insight}"
    )


if stress_insight:

    st.markdown(
        f"🟠 {stress_insight}"
    )


if reason_insight:

    st.markdown(
        f"🟡 {reason_insight}"
    )


st.markdown(
    f"""
    ### Management Takeaway

    The current dataset contains **{Total_Shipments:,} shipments**,
    of which **{Total_Delays:,} are classified as delayed**.

    The overall historical delay rate is
    **{Delay_Rate:.1f}%**.

    Delay diagnostics should be used together with the predictive
    model to identify high-risk shipments before dispatch and support
    proactive operational intervention.
    """
)


# ============================================================
# SECTION 13 — DATA QUALITY INFORMATION
# ============================================================

with st.expander("🔍 Dataset Information"):

    info_col1, info_col2, info_col3 = st.columns(3)


    with info_col1:

        st.metric(
            "Rows",
            f"{df.shape[0]:,}"
        )


    with info_col2:

        st.metric(
            "Columns",
            f"{df.shape[1]:,}"
        )


    with info_col3:

        missing_values = int(
            df.isna().sum().sum()
        )

        st.metric(
            "Missing Values",
            f"{missing_values:,}"
        )


    st.write(
        "Available columns:"
    )

    st.write(
        list(df.columns)
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Logistics Delay Prediction & Business Diagnostics Dashboard"
)
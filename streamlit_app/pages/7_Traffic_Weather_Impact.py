# ============================================================
# 7_TRAFFIC_WEATHER_IMPACT.PY
# Traffic & Weather Impact Dashboard
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
    page_title="Traffic & Weather Impact",
    page_icon="🌦️",
    layout="wide"
)


# ============================================================
# PATH CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_PATH = (
    BASE_DIR
    / "data"/ "processed"
    / "logistics_feature_engineered.csv"
)


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
            `{DATA_PATH}`

            Please verify that the feature-engineered dataset
            exists inside the project's data folder.
            """
        )

        st.stop()

    df = pd.read_csv(DATA_PATH)

    return df


df = load_data()


# ============================================================
# PAGE TITLE
# ============================================================

st.title("🌦️ Traffic & Weather Impact")

st.markdown(
    """
    ### Operational Impact Analysis

    This page analyzes how **traffic conditions, temperature,
    humidity and operational factors** influence logistics
    delays.

    The objective is to identify environmental and traffic
    conditions associated with elevated delay risk.
    """
)


# ============================================================
# DATA VALIDATION
# ============================================================

required_columns = [
    "Traffic_Status",
    "Traffic_Level",
    "Temperature",
    "Humidity",
    "Logistics_Delay"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:

    st.warning(
        "The following expected columns are missing: "
        + ", ".join(missing_columns)
    )

    st.info(
        "The dashboard will use only the columns available "
        "in the dataset."
    )


# ============================================================
# PREPARE ANALYSIS DATA
# ============================================================

analysis_df = df.copy()


# ------------------------------------------------------------
# Ensure target is numeric
# ------------------------------------------------------------

if "Logistics_Delay" in analysis_df.columns:

    analysis_df["Logistics_Delay"] = pd.to_numeric(
        analysis_df["Logistics_Delay"],
        errors="coerce"
    )

    analysis_df = analysis_df.dropna(
        subset=["Logistics_Delay"]
    )


# ------------------------------------------------------------
# Convert numeric environmental variables
# ------------------------------------------------------------

for col in [
    "Temperature",
    "Humidity",
    "Waiting_Time",
    "Asset_Utilization",
    "Demand_Forecast",
    "Operational_Stress_Score"
]:

    if col in analysis_df.columns:

        analysis_df[col] = pd.to_numeric(
            analysis_df[col],
            errors="coerce"
        )


# ============================================================
# SIDEBAR FILTERS
# ============================================================

st.sidebar.header("🔎 Filters")


# ------------------------------------------------------------
# Traffic filter
# ------------------------------------------------------------

if "Traffic_Status" in analysis_df.columns:

    traffic_values = sorted(
        analysis_df["Traffic_Status"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    selected_traffic = st.sidebar.multiselect(
        "Traffic Status",
        traffic_values,
        default=traffic_values
    )

    analysis_df = analysis_df[
        analysis_df["Traffic_Status"]
        .astype(str)
        .isin(selected_traffic)
    ]


# ------------------------------------------------------------
# Temperature filter
# ------------------------------------------------------------

if "Temperature" in analysis_df.columns:

    temp_series = analysis_df["Temperature"].dropna()

    if not temp_series.empty:

        temp_min = float(temp_series.min())
        temp_max = float(temp_series.max())

        selected_temp = st.sidebar.slider(
            "Temperature Range",
            min_value=temp_min,
            max_value=temp_max,
            value=(temp_min, temp_max)
        )

        analysis_df = analysis_df[
            analysis_df["Temperature"].between(
                selected_temp[0],
                selected_temp[1]
            )
        ]


# ------------------------------------------------------------
# Humidity filter
# ------------------------------------------------------------

if "Humidity" in analysis_df.columns:

    humidity_series = analysis_df["Humidity"].dropna()

    if not humidity_series.empty:

        humidity_min = float(humidity_series.min())
        humidity_max = float(humidity_series.max())

        selected_humidity = st.sidebar.slider(
            "Humidity Range",
            min_value=humidity_min,
            max_value=humidity_max,
            value=(humidity_min, humidity_max)
        )

        analysis_df = analysis_df[
            analysis_df["Humidity"].between(
                selected_humidity[0],
                selected_humidity[1]
            )
        ]


# ============================================================
# KPI SECTION
# ============================================================

st.subheader("📊 Key Environmental & Traffic Indicators")

kpi1, kpi2, kpi3, kpi4 = st.columns(4)


# ------------------------------------------------------------
# Overall delay rate
# ------------------------------------------------------------

if "Logistics_Delay" in analysis_df.columns:

    overall_delay_rate = (
        analysis_df["Logistics_Delay"]
        .mean() * 100
    )

else:

    overall_delay_rate = np.nan


# ------------------------------------------------------------
# Average temperature
# ------------------------------------------------------------

if "Temperature" in analysis_df.columns:

    avg_temperature = (
        analysis_df["Temperature"]
        .mean()
    )

else:

    avg_temperature = np.nan


# ------------------------------------------------------------
# Average humidity
# ------------------------------------------------------------

if "Humidity" in analysis_df.columns:

    avg_humidity = (
        analysis_df["Humidity"]
        .mean()
    )

else:

    avg_humidity = np.nan


# ------------------------------------------------------------
# High traffic delay rate
# ------------------------------------------------------------

high_traffic_delay = np.nan

if (
    "Traffic_Status" in analysis_df.columns
    and "Logistics_Delay" in analysis_df.columns
):

    traffic_text = (
        analysis_df["Traffic_Status"]
        .astype(str)
        .str.lower()
    )

    high_mask = traffic_text.str.contains(
        "heavy|high|congested",
        regex=True,
        na=False
    )

    if high_mask.any():

        high_traffic_delay = (
            analysis_df.loc[
                high_mask,
                "Logistics_Delay"
            ].mean() * 100
        )


# ============================================================
# DISPLAY KPIs
# ============================================================

with kpi1:

    st.metric(
        "Overall Delay Rate",
        (
            f"{overall_delay_rate:.1f}%"
            if not np.isnan(overall_delay_rate)
            else "N/A"
        )
    )


with kpi2:

    st.metric(
        "Average Temperature",
        (
            f"{avg_temperature:.1f}"
            if not np.isnan(avg_temperature)
            else "N/A"
        )
    )


with kpi3:

    st.metric(
        "Average Humidity",
        (
            f"{avg_humidity:.1f}%"
            if not np.isnan(avg_humidity)
            else "N/A"
        )
    )


with kpi4:

    st.metric(
        "High Traffic Delay Rate",
        (
            f"{high_traffic_delay:.1f}%"
            if not np.isnan(high_traffic_delay)
            else "N/A"
        )
    )


# ============================================================
# SECTION 1
# TRAFFIC IMPACT
# ============================================================

st.divider()

st.subheader("🚦 Traffic Impact on Logistics Delays")


if (
    "Traffic_Status" in analysis_df.columns
    and "Logistics_Delay" in analysis_df.columns
):

    traffic_summary = (
        analysis_df
        .groupby("Traffic_Status", dropna=False)
        .agg(
            Total_Shipments=("Logistics_Delay", "size"),
            Delayed_Shipments=("Logistics_Delay", "sum"),
            Delay_Rate=("Logistics_Delay", "mean")
        )
        .reset_index()
    )

    traffic_summary["Delay_Rate"] = (
        traffic_summary["Delay_Rate"] * 100
    )

    traffic_summary["Traffic_Status"] = (
        traffic_summary["Traffic_Status"]
        .astype(str)
    )

    col1, col2 = st.columns(2)


    # --------------------------------------------------------
    # Delay rate by traffic
    # --------------------------------------------------------

    with col1:

        fig = px.bar(
            traffic_summary,
            x="Traffic_Status",
            y="Delay_Rate",
            text="Delay_Rate",
            title="Delay Rate by Traffic Condition"
        )

        fig.update_traces(
            texttemplate="%{text:.1f}%",
            textposition="outside"
        )

        fig.update_layout(
            yaxis_title="Delay Rate (%)",
            xaxis_title="Traffic Condition"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    # --------------------------------------------------------
    # Shipment volume
    # --------------------------------------------------------

    with col2:

        fig = px.bar(
            traffic_summary,
            x="Traffic_Status",
            y="Total_Shipments",
            text="Total_Shipments",
            title="Shipment Volume by Traffic Condition"
        )

        fig.update_layout(
            yaxis_title="Shipments",
            xaxis_title="Traffic Condition"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    # --------------------------------------------------------
    # Summary table
    # --------------------------------------------------------

    display_traffic = traffic_summary.copy()

    display_traffic["Delay_Rate"] = (
        display_traffic["Delay_Rate"]
        .round(2)
    )

    st.dataframe(
        display_traffic,
        use_container_width=True,
        hide_index=True
    )

else:

    st.info(
        "Traffic_Status or Logistics_Delay is not available."
    )


# ============================================================
# SECTION 2
# TRAFFIC LEVEL IMPACT
# ============================================================

if (
    "Traffic_Level" in analysis_df.columns
    and "Logistics_Delay" in analysis_df.columns
):

    st.subheader("📈 Delay Risk by Traffic Level")

    traffic_level_summary = (
        analysis_df
        .groupby("Traffic_Level", dropna=False)
        .agg(
            Shipments=("Logistics_Delay", "size"),
            Delay_Rate=("Logistics_Delay", "mean")
        )
        .reset_index()
    )

    traffic_level_summary["Delay_Rate"] *= 100

    traffic_level_summary["Traffic_Level"] = (
        traffic_level_summary["Traffic_Level"]
        .astype(str)
    )

    fig = px.bar(
        traffic_level_summary,
        x="Traffic_Level",
        y="Delay_Rate",
        text="Delay_Rate",
        title="Logistics Delay Rate by Traffic Level"
    )

    fig.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside"
    )

    fig.update_layout(
        yaxis_title="Delay Rate (%)",
        xaxis_title="Traffic Level"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# SECTION 3
# TEMPERATURE IMPACT
# ============================================================

st.divider()

st.subheader("🌡️ Temperature Impact")


if (
    "Temperature" in analysis_df.columns
    and "Logistics_Delay" in analysis_df.columns
):

    temp_df = analysis_df[
        [
            "Temperature",
            "Logistics_Delay"
        ]
    ].dropna()

    if len(temp_df) > 0:

        # Create temperature bands
        temp_min = temp_df["Temperature"].min()
        temp_max = temp_df["Temperature"].max()

        if temp_min < temp_max:

            temp_df["Temperature_Band"] = pd.cut(
                temp_df["Temperature"],
                bins=5,
                include_lowest=True
            )

            temp_summary = (
                temp_df
                .groupby(
                    "Temperature_Band",
                    observed=True
                )
                .agg(
                    Shipments=("Logistics_Delay", "size"),
                    Delay_Rate=("Logistics_Delay", "mean")
                )
                .reset_index()
            )

            temp_summary["Delay_Rate"] *= 100

            # Convert Interval to string
            # This prevents Plotly JSON serialization errors
            temp_summary["Temperature_Band"] = (
                temp_summary["Temperature_Band"]
                .astype(str)
            )

            fig = px.bar(
                temp_summary,
                x="Temperature_Band",
                y="Delay_Rate",
                text="Delay_Rate",
                title="Delay Rate Across Temperature Ranges"
            )

            fig.update_traces(
                texttemplate="%{text:.1f}%",
                textposition="outside"
            )

            fig.update_layout(
                xaxis_title="Temperature Range",
                yaxis_title="Delay Rate (%)"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )


# ============================================================
# SECTION 4
# HUMIDITY IMPACT
# ============================================================

st.subheader("💧 Humidity Impact")


if (
    "Humidity" in analysis_df.columns
    and "Logistics_Delay" in analysis_df.columns
):

    humidity_df = analysis_df[
        [
            "Humidity",
            "Logistics_Delay"
        ]
    ].dropna()

    if len(humidity_df) > 0:

        humidity_min = humidity_df["Humidity"].min()
        humidity_max = humidity_df["Humidity"].max()

        if humidity_min < humidity_max:

            humidity_df["Humidity_Band"] = pd.cut(
                humidity_df["Humidity"],
                bins=5,
                include_lowest=True
            )

            humidity_summary = (
                humidity_df
                .groupby(
                    "Humidity_Band",
                    observed=True
                )
                .agg(
                    Shipments=("Logistics_Delay", "size"),
                    Delay_Rate=("Logistics_Delay", "mean")
                )
                .reset_index()
            )

            humidity_summary["Delay_Rate"] *= 100

            # IMPORTANT:
            # Convert pandas Interval to string
            humidity_summary["Humidity_Band"] = (
                humidity_summary["Humidity_Band"]
                .astype(str)
            )

            fig = px.bar(
                humidity_summary,
                x="Humidity_Band",
                y="Delay_Rate",
                text="Delay_Rate",
                title="Delay Rate Across Humidity Ranges"
            )

            fig.update_traces(
                texttemplate="%{text:.1f}%",
                textposition="outside"
            )

            fig.update_layout(
                xaxis_title="Humidity Range",
                yaxis_title="Delay Rate (%)"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )


# ============================================================
# SECTION 5
# TEMPERATURE VS HUMIDITY
# ============================================================

st.divider()

st.subheader("🌡️💧 Temperature vs Humidity")


if (
    "Temperature" in analysis_df.columns
    and "Humidity" in analysis_df.columns
    and "Logistics_Delay" in analysis_df.columns
):

    scatter_df = analysis_df[
        [
            "Temperature",
            "Humidity",
            "Logistics_Delay"
        ]
    ].dropna()

    # Limit chart size for Streamlit performance
    if len(scatter_df) > 5000:

        scatter_df = scatter_df.sample(
            5000,
            random_state=42
        )

    scatter_df["Delay_Status"] = (
        scatter_df["Logistics_Delay"]
        .map({
            0: "No Delay",
            1: "Delay"
        })
        .fillna(
            scatter_df["Logistics_Delay"]
            .astype(str)
        )
    )

    fig = px.scatter(
        scatter_df,
        x="Temperature",
        y="Humidity",
        color="Delay_Status",
        title="Temperature and Humidity vs Delay Status",
        opacity=0.65
    )

    fig.update_layout(
        xaxis_title="Temperature",
        yaxis_title="Humidity"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# SECTION 6
# WEATHER CONDITIONS + OPERATIONAL STRESS
# ============================================================

if (
    "Operational_Stress_Score" in analysis_df.columns
    and "Logistics_Delay" in analysis_df.columns
):

    st.divider()

    st.subheader(
        "⚙️ Environmental Conditions & Operational Stress"
    )

    stress_df = analysis_df[
        [
            "Operational_Stress_Score",
            "Logistics_Delay"
        ]
    ].dropna()

    if len(stress_df) > 0:

        stress_min = stress_df[
            "Operational_Stress_Score"
        ].min()

        stress_max = stress_df[
            "Operational_Stress_Score"
        ].max()

        if stress_min < stress_max:

            stress_df["Stress_Band"] = pd.cut(
                stress_df["Operational_Stress_Score"],
                bins=5,
                include_lowest=True
            )

            stress_summary = (
                stress_df
                .groupby(
                    "Stress_Band",
                    observed=True
                )
                .agg(
                    Shipments=("Logistics_Delay", "size"),
                    Delay_Rate=("Logistics_Delay", "mean")
                )
                .reset_index()
            )

            stress_summary["Delay_Rate"] *= 100

            # Prevent Plotly Interval serialization error
            stress_summary["Stress_Band"] = (
                stress_summary["Stress_Band"]
                .astype(str)
            )

            fig = px.line(
                stress_summary,
                x="Stress_Band",
                y="Delay_Rate",
                markers=True,
                title="Delay Rate by Operational Stress"
            )

            fig.update_layout(
                xaxis_title="Operational Stress Range",
                yaxis_title="Delay Rate (%)"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )


# ============================================================
# SECTION 7
# WAITING TIME IMPACT
# ============================================================

if (
    "Waiting_Time" in analysis_df.columns
    and "Logistics_Delay" in analysis_df.columns
):

    st.divider()

    st.subheader("⏱️ Waiting Time vs Delay Risk")

    wait_df = analysis_df[
        [
            "Waiting_Time",
            "Logistics_Delay"
        ]
    ].dropna()

    if len(wait_df) > 0:

        wait_min = wait_df["Waiting_Time"].min()
        wait_max = wait_df["Waiting_Time"].max()

        if wait_min < wait_max:

            wait_df["Waiting_Time_Band"] = pd.cut(
                wait_df["Waiting_Time"],
                bins=5,
                include_lowest=True
            )

            wait_summary = (
                wait_df
                .groupby(
                    "Waiting_Time_Band",
                    observed=True
                )
                .agg(
                    Shipments=("Logistics_Delay", "size"),
                    Delay_Rate=("Logistics_Delay", "mean")
                )
                .reset_index()
            )

            wait_summary["Delay_Rate"] *= 100

            wait_summary["Waiting_Time_Band"] = (
                wait_summary["Waiting_Time_Band"]
                .astype(str)
            )

            fig = px.bar(
                wait_summary,
                x="Waiting_Time_Band",
                y="Delay_Rate",
                text="Delay_Rate",
                title="Delay Rate by Waiting Time"
            )

            fig.update_traces(
                texttemplate="%{text:.1f}%",
                textposition="outside"
            )

            fig.update_layout(
                xaxis_title="Waiting Time Range",
                yaxis_title="Delay Rate (%)"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )


# ============================================================
# BUSINESS INSIGHTS
# ============================================================

st.divider()

st.subheader("💡 Business Diagnostics")


business_messages = []


# ------------------------------------------------------------
# Traffic insight
# ------------------------------------------------------------

if not np.isnan(high_traffic_delay):

    business_messages.append(
        f"""
        **🚦 Traffic:** High/heavy traffic conditions show a
        delay rate of approximately **{high_traffic_delay:.1f}%**.
        Traffic congestion should therefore be considered during
        dispatch planning.
        """
    )


# ------------------------------------------------------------
# Temperature insight
# ------------------------------------------------------------

if (
    "Temperature" in analysis_df.columns
    and "Logistics_Delay" in analysis_df.columns
):

    temp_corr = analysis_df[
        ["Temperature", "Logistics_Delay"]
    ].corr().iloc[0, 1]

    if pd.notna(temp_corr):

        direction = (
            "positive"
            if temp_corr > 0
            else "negative"
        )

        business_messages.append(
            f"""
            **🌡️ Temperature:** The correlation between
            temperature and delay occurrence is
            **{temp_corr:.2f} ({direction})**.
            Temperature should be monitored as an environmental
            risk indicator rather than used in isolation.
            """
        )


# ------------------------------------------------------------
# Humidity insight
# ------------------------------------------------------------

if (
    "Humidity" in analysis_df.columns
    and "Logistics_Delay" in analysis_df.columns
):

    humidity_corr = analysis_df[
        ["Humidity", "Logistics_Delay"]
    ].corr().iloc[0, 1]

    if pd.notna(humidity_corr):

        business_messages.append(
            f"""
            **💧 Humidity:** Humidity has a correlation of
            **{humidity_corr:.2f}** with delay occurrence.
            Combining humidity with traffic and operational
            conditions can provide stronger risk signals.
            """
        )


# ------------------------------------------------------------
# Operational stress insight
# ------------------------------------------------------------

if "Operational_Stress_Score" in analysis_df.columns:

    high_stress = analysis_df[
        "Operational_Stress_Score"
    ].quantile(0.75)

    high_stress_df = analysis_df[
        analysis_df["Operational_Stress_Score"]
        >= high_stress
    ]

    if (
        len(high_stress_df) > 0
        and "Logistics_Delay" in high_stress_df.columns
    ):

        high_stress_delay = (
            high_stress_df["Logistics_Delay"]
            .mean() * 100
        )

        business_messages.append(
            f"""
            **⚙️ Operational Stress:** The highest 25% of
            operational-stress observations show a delay rate
            of approximately **{high_stress_delay:.1f}%**.
            High-stress periods should receive additional
            operational attention.
            """
        )


if business_messages:

    for message in business_messages:

        st.markdown(message)

else:

    st.info(
        "Insufficient data is available to generate "
        "business diagnostics."
    )


# ============================================================
# MANAGEMENT RECOMMENDATIONS
# ============================================================

st.subheader("🎯 Management Recommendations")

recommendations = [
    "Monitor heavy traffic conditions closely during dispatch planning.",
    "Use traffic and weather indicators together rather than relying on a single environmental variable.",
    "Increase operational readiness during periods of elevated operational stress.",
    "Combine environmental signals with inventory, utilization and demand indicators for proactive delay prevention.",
    "Use the predictive delay model as an early-warning system rather than waiting for confirmed delivery failures."
]

for i, recommendation in enumerate(
    recommendations,
    start=1
):

    st.markdown(
        f"**{i}.** {recommendation}"
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Traffic & Weather Impact | Smart Logistics Delay "
    "Prediction & Decision Support System"
)
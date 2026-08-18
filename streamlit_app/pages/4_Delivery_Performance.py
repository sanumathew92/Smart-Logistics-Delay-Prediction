# Delivery Performance
# Smart Logistics Delay Prediction Syatem

import streamlit as st
import pandas as pd 
import numpy as np 
import plotly.express as px 
from pathlib import Path 

# Page Configuration

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_PATH = BASE_DIR / "data" / "processed" / "logistics_feature_engineered.csv"

# Load Data

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
            exists inside the data folder.
            """
        )

        return None

    data = pd.read_csv(DATA_PATH)

    return data 

df = load_data()

if df is None:
    st.stop()

# Page Header

st.title("📦 Delivery Performance")

st.markdown(
    """

    ### Delivery execution and logistics service-performance analysis

    This page helps management understand:

    - Overall delivery performance
    - Delay frequency
    - Shipment-status performance
    - Delivery performance by traffic conditions
    - Delivery performance bu delay reason
    - Waiting-time impact
    - Delivery performance across operational periods
    """
)

st.divider()

# Data Validation

expected_columns= [
    "Logistics_Delay",
    "Shipment_Status",
    "Traffic_Status",
    "Logistics_Delay_Reason",
    "Waiting_Time",
    "Asset_Utilization",
    "Fleet_Load_Index",
    "Operational_Stress_Score"
]

missing_columns = [
    col 
    for col in expected_columns
    if col not in df.columns
]

if missing_columns:

    st.warning(
        "Some expected columns are not available: "
        + ", ".join(missing_columns)
    )


# Section 1 - Delivery KPI cards

st.subheader("📊 Delivery Performance Overview")

total_records = len(df)

if "Logistics_Delay" in df.columns:

    total_delays = int(
        df["Logistics_Delay"].sum()
    )

    delay_rate = (
        df["Logistics_Delay"].mean() * 100
    )

    on_time_rate = 100 - delay_rate

else:

    total_delays = 0
    delay_rate = np.nan
    on_time_rate = np.nan 

if "Waiting_Time" in df.columns:

    avg_waiting = df["Waiting_Time"].mean()

else:

    avg_waiting = np.nan 

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total_Shipments",
    f"{total_records:,}"
)

col2.metric(
    "On-Time Rate",
    (
        f"{on_time_rate:.1f}%"
        if not pd.isna(on_time_rate)
        else "N/A"
    )
)

col3.metric(
    "Delay Rate",
(
    f"{delay_rate:.1f}%"
    if not pd.isna(delay_rate)
    else "N/A"
)
)

col4.metric(
    "Avg Waiting Time",
    (
        f"{avg_waiting:.2f}"
        if not pd.isna(avg_waiting)
        else "N/A"
    )
)

st.divider()

# Section 2 - Delivery Status 

# ============================================================
# SHIPMENT STATUS SUMMARY
# ============================================================

st.subheader("Shipment Status Performance")

if "Shipment_Status" in df.columns:

    status_summary = (
        df["Shipment_Status"]
        .value_counts(dropna=False)
        .reset_index()
    )

    status_summary.columns = [
        "Shipment_Status",
        "Shipment_Count"
    ]

    status_summary["Percentage"] = (
        status_summary["Shipment_Count"]
        / status_summary["Shipment_Count"].sum()
        * 100
    )

    status_summary["Percentage"] = (
        status_summary["Percentage"].round(2)
    )

    st.dataframe(
        status_summary,
        use_container_width=True,
        hide_index=True
    )

else:
    st.warning(
        "Shipment_Status column is not available in the dataset."
    )

# Section 3 - Traffic Impact 

# ============================================================
# TRAFFIC STATUS ANALYSIS
# ============================================================

st.subheader("Traffic Conditions and Delivery Delays")

if "Traffic_Status" in df.columns and "Logistics_Delay" in df.columns:

    # --------------------------------------------------------
    # Clean required columns
    # --------------------------------------------------------

    traffic_data = df[
        ["Traffic_Status", "Logistics_Delay"]
    ].copy()

    traffic_data = traffic_data.dropna(
        subset=["Traffic_Status", "Logistics_Delay"]
    )

    # --------------------------------------------------------
    # Explicit aggregation
    # --------------------------------------------------------

    traffic_summary = (
        traffic_data
        .groupby("Traffic_Status")
        .agg(
            Total_Shipments=("Logistics_Delay", "count"),
            Delayed_Shipments=("Logistics_Delay", "sum")
        )
        .reset_index()
    )

    # --------------------------------------------------------
    # Calculate delay rate explicitly
    # --------------------------------------------------------

    traffic_summary["Delay_Rate"] = (
        traffic_summary["Delayed_Shipments"]
        / traffic_summary["Total_Shipments"]
        * 100
    )

    traffic_summary["Delay_Rate"] = (
        traffic_summary["Delay_Rate"].round(2)
    )

    # --------------------------------------------------------
    # Display table
    # --------------------------------------------------------

    st.dataframe(
        traffic_summary,
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------------
    # Visualization
    # --------------------------------------------------------

    fig = px.bar(
        traffic_summary,
        x="Traffic_Status",
        y="Delay_Rate",
        text="Delay_Rate",
        title="Logistics Delay Rate by Traffic Condition"
    )

    fig.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside"
    )

    fig.update_layout(
        xaxis_title="Traffic Condition",
        yaxis_title="Delay Rate (%)",
        yaxis=dict(range=[0, 100])
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

else:

    missing = [
        col
        for col in ["Traffic_Status", "Logistics_Delay"]
        if col not in df.columns
    ]

    st.warning(
        f"Required columns are missing: {missing}"
    )

# Section 4- Delay Reasons

st.subheader("⚠️ Logistics Delay Reasons")

if (
    "Logistics_Delay_Reason" in df.columns
    and
    "Logistics_Delay" in df.columns
):

    reason_df = df[
        df["Logistics_Delay"] == 1
    ].copy()

    if len(reason_df) > 0:

        reason_summary = (
            reason_df[
                "Logistics_Delay_Reason"
            ]
            .value_counts()
            .reset_index()
        )

        reason_summary.columns = [
            "Delay_Reason",
            "Delayed_Shipments"
        ]

        col1, col2  = st.columns(2)

        with col1:

            fig = px.bar(
                reason_summary,
                x="Delay_Reason",
                y="Delayed_Shipments",
                title="Delayed Shipments by Reason.",
                text_auto=True
            )

            fig.update_xaxes(
                tickangle=-30
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        with col2:

            fig = px.pie(
                reason_summary,
                names="Delay_Reason",
                values="Delayed_Shipments",
                title = "Delay Reason Contribution",
                hole = 0.45
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    else:

        st.info(
            "No delayed shipments are available "
            "for delay-reason analysis."
        )

else:

    st.info(
        "Logistics_Delay_Reason is not available."
    )

# Section 5- Waiting Time Impact

st.subheader("⏱️ Waiting Time and Delivery Delays")

if (
    "Waiting_Time" in df.columns
    and
    "Logistics_Delay" in df.columns
):

    waiting_summary = (
        df.groupby("Logistics_Delay")
        ["Waiting_Time"]
        .agg(
            [
                "mean",
                "median",
                "min",
                "max"
            ]
        )
        .reset_index()
    )

    waiting_summary["Logistics_Delay"] = (
        waiting_summary["Logistics_Delay"]
        .map({
            0: "No Delay",
            1: "Delay"
        })
    )

    col1, col2 = st.columns(2)

    with col1:

        fig = px.box(
            df,
            x="Logistics_Delay",
            y="Waiting_Time",
            title= "Waiting Time Distribution by Delay Status"
        )

        fig.update_xaxes(
            tickvals = [0, 1],
            ticktext = [
                "No Delay",
                "Delay"
            ]
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        fig = px.bar(
            waiting_summary,
            x="Logistics_Delay",
            y="mean",
            title="Average Waiting Time",
            text_auto=".2f"
        )

        fig.update_yaxes(
            title="Average Waitiing Time"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

else:

    st.info(
        "Waiting_Time or Logistics_Delay is not available."
    )


# Section 6 - Distance from Center

# ============================================================
# DISTANCE FROM CENTER VS LOGISTICS DELAY
# ============================================================

st.subheader("Distance from Center vs Logistics Delay")

DISTANCE_COL = "Distance_From_Center"

if (
    DISTANCE_COL in df.columns
    and "Logistics_Delay" in df.columns
):

    # --------------------------------------------------------
    # Create a clean sample for visualization
    # --------------------------------------------------------

    sample_df = df[
        [
            DISTANCE_COL,
            "Logistics_Delay"
        ]
    ].dropna().copy()

    # Limit points for better Streamlit performance
    if len(sample_df) > 5000:
        sample_df = sample_df.sample(
            5000,
            random_state=42
        )

    # --------------------------------------------------------
    # Convert delay to business-friendly labels
    # --------------------------------------------------------

    sample_df["Delay_Status"] = (
        sample_df["Logistics_Delay"]
        .map({
            0: "No Delay",
            1: "Delay"
        })
        .fillna(
            sample_df["Logistics_Delay"].astype(str)
        )
    )

    # --------------------------------------------------------
    # Scatter plot
    # --------------------------------------------------------

    fig = px.scatter(
        sample_df,
        x=DISTANCE_COL,
        y="Logistics_Delay",
        color="Delay_Status",
        title="Distance from Center vs Logistics Delay",
        labels={
            DISTANCE_COL: "Distance From Center",
            "Logistics_Delay": "Logistics Delay"
        },
        hover_data=[
            DISTANCE_COL,
            "Logistics_Delay",
            "Delay_Status"
        ],
        opacity=0.6
    )

    fig.update_layout(
        xaxis_title="Distance From Center",
        yaxis_title="Logistics Delay"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

else:

    missing_columns = [
        col
        for col in [
            DISTANCE_COL,
            "Logistics_Delay"
        ]
        if col not in df.columns
    ]

    st.warning(
        f"Required columns are missing: {missing_columns}"
    )

# Section 7 - Operational Conditions

# ============================================================
# INVENTORY / OPERATIONAL BAND ANALYSIS
# ============================================================

st.subheader("Operational Performance by Inventory Band")

BAND_COL = "Inventory_Band"

if BAND_COL in df.columns and "Logistics_Delay" in df.columns:

    band_df = df[
        [BAND_COL, "Logistics_Delay"]
    ].dropna().copy()

    # IMPORTANT:
    # Convert pandas Interval/category objects to strings
    band_df[BAND_COL] = (
        band_df[BAND_COL]
        .astype(str)
    )

    # Convert delay to numeric
    band_df["Logistics_Delay"] = pd.to_numeric(
        band_df["Logistics_Delay"],
        errors="coerce"
    )

    band_df = band_df.dropna(
        subset=["Logistics_Delay"]
    )

    # --------------------------------------------------------
    # Aggregate
    # --------------------------------------------------------

    band_summary = (
        band_df
        .groupby(BAND_COL, as_index=False)
        .agg(
            Total_Shipments=("Logistics_Delay", "count"),
            Delayed_Shipments=("Logistics_Delay", "sum")
        )
    )

    band_summary["Delay_Rate"] = (
        band_summary["Delayed_Shipments"]
        / band_summary["Total_Shipments"]
        * 100
    ).round(2)

    # --------------------------------------------------------
    # Display summary
    # --------------------------------------------------------

    st.dataframe(
        band_summary,
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    fig = px.bar(
        band_summary,
        x=BAND_COL,
        y="Delay_Rate",
        text="Delay_Rate",
        title="Logistics Delay Rate by Inventory Band",
        labels={
            BAND_COL: "Inventory Band",
            "Delay_Rate": "Delay Rate (%)"
        }
    )

    fig.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside"
    )

    fig.update_layout(
        yaxis=dict(
            title="Delay Rate (%)",
            range=[0, 100]
        ),
        xaxis_title="Inventory Band"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

else:

    st.info(
        "Inventory_Band or Logistics_Delay "
        "is not available."
    )

# Section 8 - Daily / Time Performance

st.subheader("📅 Delivery Performance Over Time")

if "Timestamp" in df.columns:

    time_df = df.copy()

    time_df["Timestamp"] = pd.to_datetime(
        time_df["Timestamp"],
        errors="coerce"
    )

    time_df = time_df.dropna(
        subset=["Timestamp"]
    )

    if "Logistics_Delay" in time_df.columns:

        daily_performance = (
            time_df
            .set_index("Timestamp")
            .resample("D")
            ["Logistics_Delay"]
            .agg(
                [
                    "count",
                    "mean"
                ]
            )
            .reset_index()
        )

        daily_performance["Delay_Rate"] = (
            daily_performance["mean"]
            * 100
        )

        fig = px.line(
            daily_performance,
            x="Timestamp",
            y= "Delay_Rate",
            title = "Daily Logistics DelayRate"
        )

        fig.update_yaxes(
            title="Delay Rate (%)"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

else:

    st.info(
        "Timestamp is not available for time-based analysis."
    )


# Section 9 - Performance Summary Table

st.subheader("📋 Delivery Performance Summary")

summary_data = []

if "Logistics_Delay" in df.columns:

    summary_data.append(
        {
            "Metric": "Total_Shipments",
            "Value": len(df)
        }
    )

    summary_data.append(
        {

            "Metric": "Delayed Shipments",
            "Value": int(
                df["Logistics_Delay"].sum()
            )
        }
    )

    summary_data.append(
        {
            "Metric": "Delay_Rate",
            "Value": f"{delay_rate:.2f}%"
        }

    )

    summary_data.append(
        {
            "Metric": "On-Time Rate",
            "Value": f"{on_time_rate:.2f}%"
        }
    )

if "Waiting_Time" in df.columns:

    summary_data.append(
        {
            "Metric": "Average Waiting Time",
            "Value": f"{df['Waiting_Time'].mean():.2f}"
        }
    )


if "Distance_From_Center" in df.columns:

    summary_data.append(
        {
            "Metric": "Average Distance From Center",
            "Value": f"{df['Distance_From_Center'].mean():.2f}"
        }
    )

summary_table = pd.DataFrame(
    summary_data
)


st.dataframe(
    summary_table,
    use_container_width=True,
    hide_index=True
)

# Section 10 - Management Insights

st.subheader("🎯 Management Insights")

insights = []

# Delay rate
if "Logistics_Delay" in df.columns:

    if delay_rate >= 50:

        insights.append(
            f"🔴 Overall delay rate is {delay_rate:.1f}%, "
            "indicating a significant delivery-performance risk."
        )

    else:

        insights.append(
            f"🟢 Overall delay rate is {delay_rate:.1f}%. "
            "Delivery operations should continue to be monitored "
            "for emerging risk patterns."
        )

# Traffic
if (
    "Traffic_Status" in df.columns
    and
    "Logistics_Delay" in df.columns
):

    traffic_delay =(
        df.groupby("Traffic_Status")
        ["Logistics_Delay"]
        .mean()
        * 100
    )

    highest_traffic = traffic_delay.idxmax()

    highest_traffic_rate = traffic_delay.max()

    insights.append(
        f"🚦 {highest_traffic} traffic conditions show "
        f"the highest observed delay rate "
        f"({highest_traffic_rate:.1f}%)."
    )

# Waiting Time
if (
    "Waiting_Time" in df.columns
    and
    "Logistics_Delay" in df.columns
):

    wait_by_status = (
        df.groupby("Logistics_Delay")
        ["Waiting_Time"]
        .mean()
    )

    if (
        0 in wait_by_status.index
        and 
        1 in wait_by_status.index
    ):

        wait_difference = (
            wait_by_status[1]
            - wait_by_status[0]
        )

        insights.append(
            f"⏱️ Delayed shipments have an average "
            f"waiting-time difference of "
            f"{wait_difference:.2f} compared with "
            "non-delayed shipments."
        )

# Delay reason
if (
    "Logistics_Delay_Reason" in df.columns
    and
    "Logistics_Delay" in df.columns
):

    delayed = df[
        df["Logistics_Delay"] == 1
    ]

    if len(delayed) > 0:

        top_reason = (
            delayed[
                "Logistics_Delay_Reason"
            ]
            .value_counts()
            .idxmax()
        )

        insights.append(
            f"⚠️  {top_reason} is the most frequently "
            "observed delay reason among delayed shipments."
        )

for insights in insights:

    st.markdown(
        insights
    )

# Model business context

st.divider()

st.subheader("🤖 Predictive Decision Support")

st.info(
    """
    The deployed logistics-delay prediction system uses a Random Forest model with a classification 
    threhold of 0.48.

    The threshold was selected with a recall-first business objective and a minimum precision 
    requirement of 0.70

    Final validation performance:

    Precision : 72.07%
    Recall : 74.07%
    F1 Score: 73.06%
    ROC-AUC: 77.48%
    PR-AUC: 86.30%

    The lower threshold increases the system's ability to identify potential delays earlier, while
    accepting  higher number of false alerts.
    """
)

# Footer

st.divider()

st.caption(
    "Smart Logistics Delay Prediction System | Delivery Performance"
)




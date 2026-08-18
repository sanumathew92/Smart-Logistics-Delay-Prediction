# ============================================================
# FLEET ANALYTICS
# Smart Logistics Delay Prediction System
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
    page_title="Fleet Analytics",
    page_icon="🚚",
    layout="wide"
)


# ============================================================
# PATH CONFIGURATION
# ============================================================

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

            Please verify that
            logistics_feature_engineered.csv
            exists inside the data folder.
            """
        )

        return None

    return pd.read_csv(DATA_PATH)


df = load_data()


if df is None:

    st.stop()


# ============================================================
# PAGE TITLE
# ============================================================

st.title("🚚 Fleet Analytics")

st.markdown(
    """
    ### Fleet performance, utilization and logistics-delay diagnostics

    This page provides management with visibility into:

    - Fleet utilization
    - Fleet load
    - Operational stress
    - Inventory pressure
    - Waiting time
    - Demand pressure
    - Delay patterns across fleet operations
    """
)


st.divider()


# ============================================================
# DATA VALIDATION
# ============================================================

required_columns = [
    "Asset_ID",
    "Asset_Utilization",
    "Fleet_Load_Index",
    "Fleet_Load_Index_Normalized",
    "Operational_Stress_Score",
    "Waiting_Time",
    "Inventory_Level",
    "Inventory_Demand_Gap",
    "Logistics_Delay"
]

available_columns = [
    col for col in required_columns
    if col in df.columns
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:

    st.warning(
        "The following expected columns are not available: "
        + ", ".join(missing_columns)
    )


# ============================================================
# SECTION 1 — FLEET KPIs
# ============================================================

st.subheader("📊 Fleet Performance Overview")


col1, col2, col3, col4 = st.columns(4)


# Fleet count
if "Asset_ID" in df.columns:

    fleet_count = df["Asset_ID"].nunique()

else:

    fleet_count = 0


# Average utilization
if "Asset_Utilization" in df.columns:

    avg_utilization = df["Asset_Utilization"].mean()

else:

    avg_utilization = np.nan


# Average fleet load
if "Fleet_Load_Index" in df.columns:

    avg_fleet_load = df["Fleet_Load_Index"].mean()

else:

    avg_fleet_load = np.nan


# Average operational stress
if "Operational_Stress_Score" in df.columns:

    avg_stress = df["Operational_Stress_Score"].mean()

else:

    avg_stress = np.nan


col1.metric(
    "Fleet Assets",
    f"{fleet_count:,}"
)

col2.metric(
    "Avg Utilization",
    f"{avg_utilization:.2f}" if not pd.isna(avg_utilization) else "N/A"
)

col3.metric(
    "Avg Fleet Load",
    f"{avg_fleet_load:.2f}" if not pd.isna(avg_fleet_load) else "N/A"
)

col4.metric(
    "Avg Operational Stress",
    f"{avg_stress:.2f}" if not pd.isna(avg_stress) else "N/A"
)


st.divider()


# ============================================================
# SECTION 2 — FLEET UTILIZATION
# ============================================================

st.subheader("⚙️ Fleet Utilization Analysis")


if "Asset_Utilization" in df.columns:

    utilization = df["Asset_Utilization"].dropna()

    col1, col2 = st.columns(2)

    with col1:

        fig = px.histogram(
            df,
            x="Asset_Utilization",
            nbins=30,
            title="Fleet Utilization Distribution",
            marginal="box"
        )

        fig.update_layout(
            xaxis_title="Asset Utilization",
            yaxis_title="Number of Records"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        utilization_bins = pd.cut(
            utilization,
            bins=[-np.inf, 0.25, 0.50, 0.75, np.inf],
            labels=[
                "Low",
                "Moderate",
                "High",
                "Very High"
            ]
        )

        utilization_summary = (
            utilization_bins
            .value_counts()
            .sort_index()
            .reset_index()
        )

        utilization_summary.columns = [
            "Utilization Level",
            "Records"
        ]

        fig = px.bar(
            utilization_summary,
            x="Utilization Level",
            y="Records",
            title="Fleet Utilization Categories",
            text="Records"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

else:

    st.info(
        "Asset_Utilization is not available in the dataset."
    )


# ============================================================
# SECTION 3 — FLEET LOAD ANALYSIS
# ============================================================

st.subheader("📦 Fleet Load Analysis")


if "Fleet_Load_Index" in df.columns:

    col1, col2 = st.columns(2)

    with col1:

        fig = px.box(
            df,
            y="Fleet_Load_Index",
            title="Fleet Load Distribution"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        if "Logistics_Delay" in df.columns:

            load_delay = (
                df.groupby("Logistics_Delay")[
                    "Fleet_Load_Index"
                ]
                .mean()
                .reset_index()
            )

            load_delay["Logistics_Delay"] = (
                load_delay["Logistics_Delay"]
                .map({
                    0: "No Delay",
                    1: "Delay"
                })
                .fillna(
                    load_delay["Logistics_Delay"].astype(str)
                )
            )

            fig = px.bar(
                load_delay,
                x="Logistics_Delay",
                y="Fleet_Load_Index",
                title="Average Fleet Load by Delay Status",
                text_auto=".2f"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

else:

    st.info(
        "Fleet_Load_Index is not available."
    )


# ============================================================
# SECTION 4 — OPERATIONAL STRESS
# ============================================================

st.subheader("⚠️ Operational Stress Analysis")


if "Operational_Stress_Score" in df.columns:

    col1, col2 = st.columns(2)

    with col1:

        fig = px.histogram(
            df,
            x="Operational_Stress_Score",
            nbins=30,
            title="Operational Stress Distribution"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        if "Logistics_Delay" in df.columns:

            stress_delay = (
                df.groupby("Logistics_Delay")[
                    "Operational_Stress_Score"
                ]
                .mean()
                .reset_index()
            )

            stress_delay["Logistics_Delay"] = (
                stress_delay["Logistics_Delay"]
                .map({
                    0: "No Delay",
                    1: "Delay"
                })
                .fillna(
                    stress_delay["Logistics_Delay"].astype(str)
                )
            )

            fig = px.bar(
                stress_delay,
                x="Logistics_Delay",
                y="Operational_Stress_Score",
                title="Operational Stress by Delay Status",
                text_auto=".2f"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

else:

    st.info(
        "Operational_Stress_Score is not available."
    )


# ============================================================
# SECTION 5 — WAITING TIME
# ============================================================

st.subheader("⏱️ Waiting Time Analysis")


if "Waiting_Time" in df.columns:

    col1, col2 = st.columns(2)

    with col1:

        fig = px.histogram(
            df,
            x="Waiting_Time",
            nbins=30,
            title="Waiting Time Distribution"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        if "Logistics_Delay" in df.columns:

            waiting_delay = (
                df.groupby("Logistics_Delay")[
                    "Waiting_Time"
                ]
                .mean()
                .reset_index()
            )

            waiting_delay["Logistics_Delay"] = (
                waiting_delay["Logistics_Delay"]
                .map({
                    0: "No Delay",
                    1: "Delay"
                })
                .fillna(
                    waiting_delay["Logistics_Delay"].astype(str)
                )
            )

            fig = px.bar(
                waiting_delay,
                x="Logistics_Delay",
                y="Waiting_Time",
                title="Average Waiting Time by Delay Status",
                text_auto=".2f"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

else:

    st.info(
        "Waiting_Time is not available."
    )


# ============================================================
# SECTION 6 — INVENTORY PRESSURE
# ============================================================

st.subheader("📦 Inventory Pressure")


if (
    "Inventory_Level" in df.columns
    and
    "Inventory_Demand_Gap" in df.columns
):

    col1, col2 = st.columns(2)

    with col1:

        fig = px.scatter(
            df.sample(
                min(5000, len(df)),
                random_state=42
            ),
            x="Inventory_Level",
            y="Inventory_Demand_Gap",
            color=(
                "Logistics_Delay"
                if "Logistics_Delay" in df.columns
                else None
            ),
            title="Inventory Level vs Demand Gap",
            opacity=0.6
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        inventory_summary = pd.DataFrame({
            "Metric": [
                "Average Inventory",
                "Average Demand Gap",
                "Minimum Inventory",
                "Maximum Inventory"
            ],
            "Value": [
                df["Inventory_Level"].mean(),
                df["Inventory_Demand_Gap"].mean(),
                df["Inventory_Level"].min(),
                df["Inventory_Level"].max()
            ]
        })

        st.dataframe(
            inventory_summary,
            use_container_width=True,
            hide_index=True
        )

else:

    st.info(
        "Inventory_Level or Inventory_Demand_Gap "
        "is not available."
    )


# ============================================================
# SECTION 7 — ASSET LEVEL ANALYSIS
# ============================================================

st.subheader("🚛 Asset-Level Fleet Performance")


if "Asset_ID" in df.columns:

    asset_metrics = (
        df.groupby("Asset_ID")
        .agg(
            Records=("Asset_ID", "size"),
            Avg_Utilization=(
                "Asset_Utilization",
                "mean"
            ) if "Asset_Utilization" in df.columns
            else ("Asset_ID", "size"),
            Avg_Waiting_Time=(
                "Waiting_Time",
                "mean"
            ) if "Waiting_Time" in df.columns
            else ("Asset_ID", "size"),
            Avg_Stress=(
                "Operational_Stress_Score",
                "mean"
            ) if "Operational_Stress_Score" in df.columns
            else ("Asset_ID", "size")
        )
        .reset_index()
    )

    if "Logistics_Delay" in df.columns:

        delay_rate = (
            df.groupby("Asset_ID")["Logistics_Delay"]
            .mean()
            .reset_index()
        )

        delay_rate.rename(
            columns={
                "Logistics_Delay": "Delay_Rate"
            },
            inplace=True
        )

        asset_metrics = asset_metrics.merge(
            delay_rate,
            on="Asset_ID",
            how="left"
        )

    else:

        asset_metrics["Delay_Rate"] = np.nan


    asset_metrics["Delay_Rate"] = (
        asset_metrics["Delay_Rate"] * 100
    )


    st.dataframe(
        asset_metrics.sort_values(
            "Delay_Rate",
            ascending=False
        ).head(20),
        use_container_width=True,
        hide_index=True
    )


else:

    st.info(
        "Asset_ID is not available."
    )


# ============================================================
# SECTION 8 — DELAY RATE BY FLEET UTILIZATION
# ============================================================

st.subheader("📈 Delay Risk by Fleet Utilization")


if (
    "Asset_Utilization" in df.columns
    and
    "Logistics_Delay" in df.columns
):

    temp = df.copy()

    temp["Utilization_Band"] = pd.cut(
        temp["Asset_Utilization"],
        bins=[
            -np.inf,
            0.25,
            0.50,
            0.75,
            np.inf
        ],
        labels=[
            "Low",
            "Moderate",
            "High",
            "Very High"
        ]
    )

    utilization_delay = (
        temp.groupby(
            "Utilization_Band",
            observed=False
        )["Logistics_Delay"]
        .mean()
        .reset_index()
    )

    utilization_delay["Delay_Rate"] = (
        utilization_delay["Logistics_Delay"] * 100
    )

    fig = px.bar(
        utilization_delay,
        x="Utilization_Band",
        y="Delay_Rate",
        title="Delay Rate by Fleet Utilization",
        text_auto=".1f"
    )

    fig.update_yaxes(
        title="Delay Rate (%)"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# SECTION 9 — BUSINESS DIAGNOSTICS
# ============================================================

st.subheader("💼 Fleet Business Diagnostics")


diagnostics = []


if (
    "Asset_Utilization" in df.columns
    and
    "Logistics_Delay" in df.columns
):

    high_util = df[
        df["Asset_Utilization"]
        >= df["Asset_Utilization"].quantile(0.75)
    ]

    if len(high_util) > 0:

        high_util_delay = high_util[
            "Logistics_Delay"
        ].mean() * 100

        diagnostics.append(
            f"High-utilization fleet records show a "
            f"{high_util_delay:.1f}% delay rate."
        )


if (
    "Waiting_Time" in df.columns
    and
    "Logistics_Delay" in df.columns
):

    high_wait = df[
        df["Waiting_Time"]
        >= df["Waiting_Time"].quantile(0.75)
    ]

    if len(high_wait) > 0:

        high_wait_delay = high_wait[
            "Logistics_Delay"
        ].mean() * 100

        diagnostics.append(
            f"High waiting-time records show a "
            f"{high_wait_delay:.1f}% delay rate."
        )


if (
    "Operational_Stress_Score" in df.columns
    and
    "Logistics_Delay" in df.columns
):

    high_stress = df[
        df["Operational_Stress_Score"]
        >= df["Operational_Stress_Score"].quantile(0.75)
    ]

    if len(high_stress) > 0:

        high_stress_delay = high_stress[
            "Logistics_Delay"
        ].mean() * 100

        diagnostics.append(
            f"High operational-stress records show a "
            f"{high_stress_delay:.1f}% delay rate."
        )


if diagnostics:

    for item in diagnostics:

        st.info(
            "🔎 " + item
        )

else:

    st.info(
        "Insufficient fleet variables are available "
        "to generate business diagnostics."
    )


# ============================================================
# SECTION 10 — MANAGEMENT RECOMMENDATIONS
# ============================================================

st.subheader("🎯 Management Recommendations")


recommendations = [
    (
        "Monitor fleet utilization",
        "Identify assets operating consistently at "
        "high utilization and review whether capacity "
        "balancing or fleet reallocation is required."
    ),

    (
        "Control operational stress",
        "High operational stress should trigger "
        "additional operational review before it "
        "translates into service delays."
    ),

    (
        "Reduce excessive waiting time",
        "Investigate assets and routes with elevated "
        "waiting times because prolonged waiting can "
        "indicate operational bottlenecks."
    ),

    (
        "Watch inventory-demand imbalance",
        "Inventory-demand gaps should be monitored "
        "to identify potential fulfillment and "
        "logistics pressure."
    ),

    (
        "Prioritize early delay detection",
        "The deployed Random Forest model uses a "
        "0.48 probability threshold to prioritize "
        "early detection of potential logistics delays."
    )
]


for title, description in recommendations:

    st.markdown(
        f"""
        **{title}**

        {description}
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Smart Logistics Delay Prediction System | Fleet Analytics"
)
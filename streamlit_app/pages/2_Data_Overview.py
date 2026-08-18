# ============================================================
# DATA OVERVIEW
# Smart Logistics Delay Prediction System
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Data Overview",
    page_icon="📋",
    layout="wide"
)


# ============================================================
# PATH CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = BASE_DIR / "data" / "processed"

# Change this filename ONLY if your actual feature-engineered
# dataset has a different name.
DATA_PATH = DATA_DIR / "logistics_feature_engineered.csv"


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    if not DATA_PATH.exists():
        return None

    return pd.read_csv(DATA_PATH)


df = load_data()


# ============================================================
# PAGE HEADER
# ============================================================

st.title("📋 Data Overview")

st.markdown(
    """
    ### Understanding the Logistics Dataset

    This page provides an overview of the data used to develop the
    **Smart Logistics Delay Prediction System**.

    The objective is to understand shipment characteristics,
    operational conditions, logistics activity and the distribution
    of the target variable before applying predictive modelling.
    """
)


# ============================================================
# DATA AVAILABILITY CHECK
# ============================================================

if df is None:

    st.error(
        f"""
        **Dataset not found.**

        Expected location:

        `{DATA_PATH}`

        Please verify that the feature-engineered dataset exists
        inside the project's `data` folder.
        """
    )

    st.stop()


# ============================================================
# BASIC DATASET INFORMATION
# ============================================================

st.subheader("📊 Dataset Snapshot")


total_rows = len(df)
total_columns = len(df.columns)

duplicate_rows = df.duplicated().sum()

missing_cells = df.isna().sum().sum()

missing_percentage = (
    missing_cells / (total_rows * total_columns) * 100
    if total_rows > 0
    else 0
)


col1, col2, col3, col4 = st.columns(4)


with col1:
    st.metric(
        "Total Records",
        f"{total_rows:,}"
    )


with col2:
    st.metric(
        "Total Features",
        f"{total_columns:,}"
    )


with col3:
    st.metric(
        "Duplicate Rows",
        f"{duplicate_rows:,}"
    )


with col4:
    st.metric(
        "Missing Data",
        f"{missing_percentage:.2f}%"
    )


# ============================================================
# TARGET VARIABLE
# ============================================================

st.subheader("🎯 Target Variable")


TARGET = "Logistics_Delay"


if TARGET in df.columns:

    target_counts = df[TARGET].value_counts(dropna=False)

    target_percentage = (
        df[TARGET]
        .value_counts(normalize=True, dropna=False)
        * 100
    )

    target_summary = pd.DataFrame(
        {
            "Class": target_counts.index.astype(str),
            "Records": target_counts.values,
            "Percentage": target_percentage.values
        }
    )

    target_summary["Percentage"] = target_summary[
        "Percentage"
    ].round(2)

    target_col1, target_col2 = st.columns(2)

    with target_col1:

        st.dataframe(
            target_summary,
            use_container_width=True,
            hide_index=True
        )

    with target_col2:

        st.bar_chart(
            target_counts
        )

else:

    st.warning(
        f"Target column `{TARGET}` was not found in the dataset."
    )


# ============================================================
# BUSINESS INTERPRETATION OF TARGET
# ============================================================

if TARGET in df.columns:

    delay_rate = df[TARGET].mean()

    st.info(
        f"""
        **Delay Rate:** `{delay_rate:.1%}`

        The target variable identifies whether a shipment experienced
        a logistics delay.

        A value of **1** represents a delay and a value of **0**
        represents no delay.

        The target distribution is important because the model is
        evaluated primarily using **Precision, Recall, F1, ROC-AUC
        and PR-AUC**, rather than relying only on accuracy.
        """
    )


# ============================================================
# DATA TYPES
# ============================================================

st.subheader("🧬 Feature Composition")


numeric_columns = df.select_dtypes(
    include=np.number
).columns.tolist()

categorical_columns = df.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

datetime_columns = df.select_dtypes(
    include=["datetime"]
).columns.tolist()


type_col1, type_col2, type_col3 = st.columns(3)


with type_col1:

    st.metric(
        "Numerical Features",
        len(numeric_columns)
    )


with type_col2:

    st.metric(
        "Categorical Features",
        len(categorical_columns)
    )


with type_col3:

    st.metric(
        "Datetime Features",
        len(datetime_columns)
    )


feature_type_df = pd.DataFrame(
    {
        "Feature Type": [
            "Numerical",
            "Categorical",
            "Datetime"
        ],
        "Count": [
            len(numeric_columns),
            len(categorical_columns),
            len(datetime_columns)
        ]
    }
)


st.bar_chart(
    feature_type_df.set_index("Feature Type")
)


# ============================================================
# FEATURE LIST
# ============================================================

st.subheader("📝 Dataset Features")


feature_table = pd.DataFrame(
    {
        "Feature": df.columns,
        "Data Type": [
            str(df[col].dtype)
            for col in df.columns
        ],
        "Missing Values": [
            df[col].isna().sum()
            for col in df.columns
        ],
        "Unique Values": [
            df[col].nunique(dropna=True)
            for col in df.columns
        ]
    }
)


st.dataframe(
    feature_table,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# MISSING VALUE ANALYSIS
# ============================================================

st.subheader("🔎 Missing Value Analysis")


missing_df = pd.DataFrame(
    {
        "Feature": df.columns,
        "Missing Values": [
            df[col].isna().sum()
            for col in df.columns
        ]
    }
)

missing_df["Missing %"] = (
    missing_df["Missing Values"]
    / len(df)
    * 100
)

missing_df = missing_df.sort_values(
    "Missing Values",
    ascending=False
)


missing_features = missing_df[
    missing_df["Missing Values"] > 0
]


if len(missing_features) == 0:

    st.success(
        "✓ No missing values were detected in the dataset."
    )

else:

    st.warning(
        f"{len(missing_features)} features contain missing values."
    )

    st.dataframe(
        missing_features.style.format(
            {
                "Missing %": "{:.2f}%"
            }
        ),
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# DUPLICATE ANALYSIS
# ============================================================

st.subheader("♻️ Duplicate Records")


if duplicate_rows == 0:

    st.success(
        "✓ No duplicate records were detected."
    )

else:

    st.warning(
        f"{duplicate_rows:,} duplicate records were detected."
    )


# ============================================================
# SHIPMENT STATUS
# ============================================================

if "Shipment_Status" in df.columns:

    st.subheader("🚚 Shipment Status Distribution")

    shipment_status = (
        df["Shipment_Status"]
        .value_counts()
        .reset_index()
    )

    shipment_status.columns = [
        "Shipment Status",
        "Records"
    ]

    status_col1, status_col2 = st.columns(2)

    with status_col1:

        st.dataframe(
            shipment_status,
            use_container_width=True,
            hide_index=True
        )

    with status_col2:

        st.bar_chart(
            shipment_status.set_index(
                "Shipment Status"
            )
        )


# ============================================================
# TRAFFIC STATUS
# ============================================================

if "Traffic_Status" in df.columns:

    st.subheader("🚦 Traffic Conditions")

    traffic_counts = (
        df["Traffic_Status"]
        .value_counts()
        .reset_index()
    )

    traffic_counts.columns = [
        "Traffic Status",
        "Records"
    ]

    traffic_col1, traffic_col2 = st.columns(2)

    with traffic_col1:

        st.dataframe(
            traffic_counts,
            use_container_width=True,
            hide_index=True
        )

    with traffic_col2:

        st.bar_chart(
            traffic_counts.set_index(
                "Traffic Status"
            )
        )


# ============================================================
# DELAY RATE BY TRAFFIC STATUS
# ============================================================

if (
    "Traffic_Status" in df.columns
    and TARGET in df.columns
):

    st.subheader("🚦 Delay Rate by Traffic Condition")

    traffic_delay = (
        df.groupby("Traffic_Status")[TARGET]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )

    traffic_delay["Delay Rate"] = (
        traffic_delay[TARGET] * 100
    )

    traffic_delay = traffic_delay[
        ["Traffic_Status", "Delay Rate"]
    ]

    traffic_delay.columns = [
        "Traffic Status",
        "Delay Rate (%)"
    ]

    st.dataframe(
        traffic_delay.style.format(
            {
                "Delay Rate (%)": "{:.2f}%"
            }
        ),
        use_container_width=True,
        hide_index=True
    )

    st.bar_chart(
        traffic_delay.set_index(
            "Traffic Status"
        )
    )


# ============================================================
# DELAY RATE BY SHIPMENT STATUS
# ============================================================

if (
    "Shipment_Status" in df.columns
    and TARGET in df.columns
):

    st.subheader("📦 Delay Rate by Shipment Status")

    shipment_delay = (
        df.groupby("Shipment_Status")[TARGET]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )

    shipment_delay["Delay Rate"] = (
        shipment_delay[TARGET] * 100
    )

    shipment_delay = shipment_delay[
        ["Shipment_Status", "Delay Rate"]
    ]

    shipment_delay.columns = [
        "Shipment Status",
        "Delay Rate (%)"
    ]

    st.dataframe(
        shipment_delay.style.format(
            {
                "Delay Rate (%)": "{:.2f}%"
            }
        ),
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# DELAY REASON
# ============================================================

if "Logistics_Delay_Reason" in df.columns:

    st.subheader("⚠️ Logistics Delay Reasons")

    reason_counts = (
        df["Logistics_Delay_Reason"]
        .value_counts(dropna=False)
        .reset_index()
    )

    reason_counts.columns = [
        "Delay Reason",
        "Records"
    ]

    st.dataframe(
        reason_counts,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# NUMERICAL FEATURE SUMMARY
# ============================================================

st.subheader("📐 Numerical Feature Summary")


if len(numeric_columns) > 0:

    numerical_summary = (
        df[numeric_columns]
        .describe()
        .T
        .reset_index()
    )

    numerical_summary.rename(
        columns={
            "index": "Feature"
        },
        inplace=True
    )

    st.dataframe(
        numerical_summary,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# IMPORTANT OPERATIONAL VARIABLES
# ============================================================

st.subheader("🏭 Key Operational Variables")


operational_features = [
    "Inventory_Level",
    "Temperature",
    "Humidity",
    "Traffic_Status",
    "Waiting_Time",
    "User_Transaction_Amount",
    "User_Purchase_Frequency",
    "Asset_Utilization",
    "Demand_Forecast",
    "Operational_Stress_Score",
    "Customer_Value_Index",
    "Inventory_Demand_Gap",
    "Inventory_Coverage",
    "Fleet_Load_Index_Normalized",
    "Utilization_Normalized",
    "Utilization_Score"
]


available_operational = [
    col
    for col in operational_features
    if col in df.columns
]


if len(available_operational) > 0:

    operational_summary = []

    for col in available_operational:

        operational_summary.append(
            {
                "Feature": col,
                "Data Type": str(df[col].dtype),
                "Unique Values": df[col].nunique(
                    dropna=True
                ),
                "Missing Values": df[col].isna().sum()
            }
        )

    operational_summary = pd.DataFrame(
        operational_summary
    )

    st.dataframe(
        operational_summary,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# DATA QUALITY SUMMARY
# ============================================================

st.subheader("✅ Data Quality Summary")


quality_checks = pd.DataFrame(
    {
        "Quality Check": [
            "Records Available",
            "Features Available",
            "Duplicate Records",
            "Missing Cells",
            "Target Available",
            "Categorical Features",
            "Numerical Features"
        ],

        "Result": [
            f"{total_rows:,}",
            f"{total_columns:,}",
            f"{duplicate_rows:,}",
            f"{missing_cells:,}",
            "Yes" if TARGET in df.columns else "No",
            len(categorical_columns),
            len(numeric_columns)
        ]
    }
)


st.dataframe(
    quality_checks,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# BUSINESS TAKEAWAYS
# ============================================================

st.subheader("💡 Business Takeaways")


takeaways = []


if TARGET in df.columns:

    takeaways.append(
        f"The dataset contains {total_rows:,} records available "
        "for logistics delay analysis."
    )

    takeaways.append(
        f"The observed logistics delay rate is approximately "
        f"{delay_rate:.1%}."
    )


if "Traffic_Status" in df.columns:

    heavy_count = (
        df["Traffic_Status"]
        .astype(str)
        .str.lower()
        .eq("heavy")
        .sum()
    )

    if heavy_count > 0:

        takeaways.append(
            f"{heavy_count:,} records are associated with "
            "heavy traffic conditions."
        )


if duplicate_rows == 0:

    takeaways.append(
        "No duplicate records were detected, supporting the "
        "integrity of the modelling dataset."
    )


if len(missing_features) > 0:

    takeaways.append(
        "Missing-value handling is required for selected features "
        "before model training and prediction."
    )


for i, takeaway in enumerate(takeaways, start=1):

    st.markdown(
        f"**{i}.** {takeaway}"
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Smart Logistics Delay Prediction System | "
    "Data Overview"
)
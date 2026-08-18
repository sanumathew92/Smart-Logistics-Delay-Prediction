import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Executive Summary",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]
ASSETS_DIR = BASE_DIR / "streamlit_assets"

MODEL_PATH = ASSETS_DIR / "final_model.pkl"
FEATURE_NAMES_PATH = ASSETS_DIR / "feature_names.pkl"


# ============================================================
# BUSINESS CONFIGURATION
# ============================================================

CHAMPION_MODEL = "Random Forest"

FINAL_THRESHOLD = 0.48

MIN_PRECISION = 0.70

BUSINESS_OBJECTIVE = (
    "Prioritize early detection of potential logistics delays"
)


# Final model performance
ACCURACY = 0.7050
PRECISION = 0.7207
RECALL = 0.7407
F1 = 0.7306
ROC_AUC = 0.7748
PR_AUC = 0.8630


# Baseline performance
BASELINE_ACCURACY = 0.6000
BASELINE_PRECISION = 0.5972
BASELINE_RECALL = 0.7963
BASELINE_F1 = 0.6825
BASELINE_ROC_AUC = 0.7600
BASELINE_PR_AUC = 0.8533


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():
        return None

    return joblib.load(MODEL_PATH)


@st.cache_resource
def load_feature_names():

    if not FEATURE_NAMES_PATH.exists():
        return None

    return joblib.load(FEATURE_NAMES_PATH)


final_model = load_model()
feature_names = load_feature_names()


# ============================================================
# TITLE
# ============================================================

st.title("📊 Executive Summary")

st.markdown(
    """
    ### What does management need to know?

    This dashboard summarizes the logistics delay prediction system
    from a **business and operational perspective**.

    The deployed model is designed to identify shipments that have a
    higher probability of experiencing a logistics delay, allowing
    operations teams to intervene earlier.
    """
)


# ============================================================
# EXECUTIVE ALERT
# ============================================================

st.info(
    """
    **Business Objective**

    The model is configured for **early detection of potential logistics
    delays**. A recall-first classification threshold of **0.48** is used
    so that more potential delay cases are detected while maintaining a
    minimum precision requirement of **70%**.
    """
)


# ============================================================
# KPI SECTION
# ============================================================

st.subheader("🎯 Current Model Performance")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Accuracy",
        f"{ACCURACY:.1%}"
    )

with col2:
    st.metric(
        "Precision",
        f"{PRECISION:.1%}"
    )

with col3:
    st.metric(
        "Recall",
        f"{RECALL:.1%}"
    )

with col4:
    st.metric(
        "F1 Score",
        f"{F1:.1%}"
    )


col5, col6, col7 = st.columns(3)

with col5:
    st.metric(
        "ROC-AUC",
        f"{ROC_AUC:.4f}"
    )

with col6:
    st.metric(
        "PR-AUC",
        f"{PR_AUC:.4f}"
    )

with col7:
    st.metric(
        "Decision Threshold",
        f"{FINAL_THRESHOLD:.2f}"
    )


# ============================================================
# MODEL INFORMATION
# ============================================================

st.subheader("🏆 Champion Model")

model_col1, model_col2, model_col3 = st.columns(3)

with model_col1:

    st.markdown("### Random Forest")

    st.caption(
        "Selected optimized model"
    )


with model_col2:

    st.markdown("### Recall-First")

    st.caption(
        "Business-oriented prediction strategy"
    )


with model_col3:

    st.markdown("### Threshold: 0.48")

    st.caption(
        "Lower threshold for earlier detection"
    )


# ============================================================
# BUSINESS INTERPRETATION
# ============================================================

st.subheader("💼 What the Model Means for Management")

business_points = [
    (
        "Early Warning",
        "The model identifies shipments with elevated probability "
        "of experiencing a logistics delay before the delay becomes "
        "a larger operational problem."
    ),
    (
        "Higher Delay Detection",
        f"The recall of {RECALL:.1%} means the model detects approximately "
        f"{RECALL:.0%} of actual delay cases in the evaluated test set."
    ),
    (
        "Controlled False Alerts",
        f"Precision of {PRECISION:.1%} means that approximately "
        f"{PRECISION:.0%} of shipments flagged as potential delays "
        "are actually delayed."
    ),
    (
        "Business-Oriented Threshold",
        f"The threshold was reduced to {FINAL_THRESHOLD:.2f} to prioritize "
        "early detection rather than maximizing overall accuracy."
    )
]


for title, description in business_points:

    st.markdown(f"**{title}**")

    st.write(description)


# ============================================================
# OPERATIONAL CONFUSION MATRIX
# ============================================================

st.subheader("🚚 Operational Prediction Outcomes")

cm_data = pd.DataFrame(
    {
        "Actual No Delay": [61, 31],
        "Actual Delay": [28, 80]
    },
    index=[
        "Predicted No Delay",
        "Predicted Delay"
    ]
)

st.dataframe(
    cm_data,
    use_container_width=True
)


st.markdown(
    """
    **Operational interpretation:**

    - **80** actual delays were correctly identified.
    - **28** actual delays were missed.
    - **31** shipments received a delay warning but were not actually delayed.
    - **61** shipments were correctly identified as having no delay.
    """
)


# ============================================================
# BASELINE VS FINAL MODEL
# ============================================================

st.subheader("📈 Improvement from Baseline")

comparison = pd.DataFrame(
    {
        "Metric": [
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score",
            "ROC-AUC",
            "PR-AUC"
        ],

        "Baseline Gradient Boosting": [
            BASELINE_ACCURACY,
            BASELINE_PRECISION,
            BASELINE_RECALL,
            BASELINE_F1,
            BASELINE_ROC_AUC,
            BASELINE_PR_AUC
        ],

        "Final Random Forest": [
            ACCURACY,
            PRECISION,
            RECALL,
            F1,
            ROC_AUC,
            PR_AUC
        ]
    }
)

comparison["Change"] = (
    comparison["Final Random Forest"]
    - comparison["Baseline Gradient Boosting"]
)

comparison


# ============================================================
# KEY BUSINESS CHANGES
# ============================================================

st.subheader("📌 Key Business Changes")

change_col1, change_col2, change_col3 = st.columns(3)

with change_col1:

    precision_change = PRECISION - BASELINE_PRECISION

    st.metric(
        "Precision Improvement",
        f"{precision_change:+.1%}"
    )

with change_col2:

    f1_change = F1 - BASELINE_F1

    st.metric(
        "F1 Improvement",
        f"{f1_change:+.1%}"
    )

with change_col3:

    roc_change = ROC_AUC - BASELINE_ROC_AUC

    st.metric(
        "ROC-AUC Improvement",
        f"{roc_change:+.4f}"
    )


# ============================================================
# MAJOR DELAY DRIVERS
# ============================================================

st.subheader("⚠️ Major Delay Drivers")

st.markdown(
    """
    Model explainability identified several variables that are strongly
    associated with the model's delay predictions.
    """
)


feature_data = pd.DataFrame(
    {
        "Feature": [
            "Traffic_Status_Heavy",
            "Traffic_Status_Clear",
            "Traffic_Status_Detour",
            "Latitude",
            "Inventory_Level",
            "Distance_From_Center",
            "User_Transaction_Amount",
            "Longitude",
            "Operational_Stress_Score",
            "Customer_Value_Index",
            "Inventory_Demand_Gap",
            "Utilization_Waiting_Interaction",
            "Temperature",
            "Humidity",
            "Inventory_Coverage"
        ],

        "Importance": [
            0.152801,
            0.050079,
            0.042494,
            0.029920,
            0.025460,
            0.024654,
            0.024473,
            0.024092,
            0.023842,
            0.023799,
            0.023796,
            0.023699,
            0.023442,
            0.022610,
            0.021966
        ]
    }
)

feature_data["Importance %"] = (
    feature_data["Importance"] * 100
)

st.dataframe(
    feature_data.style.format(
        {
            "Importance": "{:.4f}",
            "Importance %": "{:.2f}%"
        }
    ),
    use_container_width=True
)


# ============================================================
# TRAFFIC INSIGHT
# ============================================================

st.warning(
    """
    **Key diagnostic finding:**

    `Traffic_Status = Heavy` is the strongest identified feature in the
    model explainability analysis.

    This indicates that traffic conditions should be treated as an
    important operational warning signal when managing potential
    logistics delays.
    """
)


# ============================================================
# MANAGEMENT RECOMMENDATIONS
# ============================================================

st.subheader("🎯 Recommended Management Actions")

recommendations = [
    (
        "1. Prioritize Heavy-Traffic Shipments",
        "Flag shipments operating under heavy traffic conditions for "
        "early operational review."
    ),

    (
        "2. Introduce Early-Warning Alerts",
        "Use the model probability to identify shipments requiring "
        "attention before the expected delivery window is affected."
    ),

    (
        "3. Protect High-Value Shipments",
        "Combine model predictions with transaction value and customer "
        "value information to prioritize interventions."
    ),

    (
        "4. Monitor Inventory Risk",
        "Inventory level, inventory demand gap and inventory coverage "
        "should be monitored alongside predicted logistics risk."
    ),

    (
        "5. Use the 0.48 Threshold for Early Detection",
        "The current threshold supports the stated business objective "
        "of detecting potential delays earlier while maintaining the "
        "minimum precision constraint."
    ),

    (
        "6. Continuously Monitor Model Performance",
        "Track precision, recall, false alerts and missed delays after "
        "deployment to ensure that the model continues to support "
        "operational objectives."
    )
]


for title, description in recommendations:

    st.markdown(f"**{title}**")

    st.write(description)


# ============================================================
# BUSINESS TRADE-OFF
# ============================================================

st.subheader("⚖️ Business Trade-Off")

trade_col1, trade_col2 = st.columns(2)

with trade_col1:

    st.success(
        """
        ### Benefit

        **Higher delay detection**

        The recall-first threshold helps identify more actual delay
        cases and provides operations teams with earlier warning.
        """
    )


with trade_col2:

    st.warning(
        """
        ### Cost

        **More false alerts**

        Lowering the threshold means some shipments will be flagged
        even though they ultimately experience no delay.
        """
    )


# ============================================================
# EXECUTIVE TAKEAWAY
# ============================================================

st.subheader("📌 Executive Takeaway")

st.success(
    f"""
    ### Management Decision

    The final **Random Forest** model should be used as an **early-warning
    logistics risk system**, rather than as a definitive statement that
    a shipment will be delayed.

    With a decision threshold of **{FINAL_THRESHOLD:.2f}**, the system
    achieves:

    **{RECALL:.1%} recall | {PRECISION:.1%} precision | {F1:.1%} F1 |
    {PR_AUC:.4f} PR-AUC**

    The recommended operational strategy is to investigate flagged
    shipments proactively, particularly where **heavy traffic,
    inventory pressure, operational stress and customer/value factors**
    indicate elevated risk.
    """
)


# ============================================================
# MODEL STATUS
# ============================================================

st.divider()

status_col1, status_col2, status_col3 = st.columns(3)

with status_col1:

    if final_model is not None:
        st.success("✓ Final model loaded")
    else:
        st.error("✗ Final model not found")


with status_col2:

    if feature_names is not None:
        st.success(
            f"✓ {len(feature_names)} transformed features loaded"
        )
    else:
        st.error("✗ Feature names not found")


with status_col3:

    st.success(
        f"✓ Threshold configured: {FINAL_THRESHOLD:.2f}"
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Smart Logistics Delay Prediction System | "
    "Executive Decision Support"
)
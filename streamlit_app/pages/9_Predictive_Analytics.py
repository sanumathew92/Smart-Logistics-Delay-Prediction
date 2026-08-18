# ============================================================
# 9_Predictive_Analytics.py
# Smart Logistics Delay Prediction Dashboard
# ============================================================

import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Predictive Analytics",
    page_icon="🔮",
    layout="wide"
)


# ============================================================
# PATH CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_PATH = BASE_DIR / "data" /"processed" / "logistics_feature_engineered.csv"
MODEL_PATH = BASE_DIR / "streamlit_app" /"streamlit_assets" / "final_model.pkl"
FEATURE_PATH = BASE_DIR / "streamlit_app" / "streamlit_assets" / "feature_names.pkl"


# ============================================================
# CONSTANTS
# ============================================================

FINAL_THRESHOLD = 0.48
MIN_PRECISION = 0.70

TARGET_COLUMN = "Logistics_Delay"


# ============================================================
# PAGE HEADER
# ============================================================

st.title("🔮 Predictive Analytics")

st.markdown(
    """
    ### Predictive Logistics Delay Intelligence

    This page uses the **optimized Random Forest model** to estimate
    the probability that a logistics shipment or operation will
    experience a delay.

    The model is configured for a **recall-first business objective**:

    > **Prioritize early detection of potential logistics delays.**

    A classification threshold of **0.48** is used to increase delay
    detection while maintaining the required minimum precision of **70%**.
    """
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    if not DATA_PATH.exists():
        return None

    df = pd.read_csv(DATA_PATH)

    return df


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():
        return None

    return joblib.load(MODEL_PATH)


# ============================================================
# LOAD FEATURE NAMES
# ============================================================

@st.cache_resource
def load_feature_names():

    if not FEATURE_PATH.exists():
        return None

    return joblib.load(FEATURE_PATH)


df = load_data()
final_model = load_model()
feature_names = load_feature_names()


# ============================================================
# VALIDATION
# ============================================================

if df is None:

    st.error(
        f"""
        Dataset not found.

        Expected location:

        `{DATA_PATH}`
        """
    )

    st.stop()


if final_model is None:

    st.error(
        f"""
        Final model not found.

        Expected location:

        `{MODEL_PATH}`
        """
    )

    st.stop()


if feature_names is None:

    st.error(
        f"""
        Feature names file not found.

        Expected location:

        `{FEATURE_PATH}`
        """
    )

    st.stop()


# ============================================================
# BASIC DATA INFORMATION
# ============================================================

total_records = len(df)

if TARGET_COLUMN in df.columns:

    delay_count = int(df[TARGET_COLUMN].sum())

    no_delay_count = int(
        (df[TARGET_COLUMN] == 0).sum()
    )

    overall_delay_rate = (
        delay_count / total_records * 100
        if total_records > 0
        else 0
    )

else:

    delay_count = None
    no_delay_count = None
    overall_delay_rate = None


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Prediction Settings")

threshold = st.sidebar.slider(
    "Prediction Threshold",
    min_value=0.20,
    max_value=0.80,
    value=FINAL_THRESHOLD,
    step=0.01
)

st.sidebar.markdown(
    f"""
    **Recommended threshold:** `{FINAL_THRESHOLD:.2f}`

    **Minimum precision constraint:** `{MIN_PRECISION:.0%}`
    """
)


# ============================================================
# KPI SECTION
# ============================================================

st.subheader("📊 Predictive Overview")

col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "Total Records",
        f"{total_records:,}"
    )


with col2:

    if overall_delay_rate is not None:

        st.metric(
            "Historical Delay Rate",
            f"{overall_delay_rate:.2f}%"
        )

    else:

        st.metric(
            "Historical Delay Rate",
            "N/A"
        )


with col3:

    st.metric(
        "Model",
        "Random Forest"
    )


with col4:

    st.metric(
        "Prediction Threshold",
        f"{threshold:.2f}"
    )


# ============================================================
# MODEL BUSINESS OBJECTIVE
# ============================================================

st.markdown("---")

st.subheader("🎯 Model Objective")

obj_col1, obj_col2 = st.columns(2)


with obj_col1:

    st.info(
        """
        **Business Objective**

        Prioritize early detection of potential logistics delays.

        The model intentionally accepts more false alerts in exchange
        for identifying a greater proportion of actual delay cases.
        """
    )


with obj_col2:

    st.success(
        f"""
        **Optimization Configuration**

        - Champion Model: **Random Forest**
        - Threshold: **{FINAL_THRESHOLD:.2f}**
        - Minimum Precision: **{MIN_PRECISION:.0%}**
        - Strategy: **Recall First**
        """
    )


# ============================================================
# SINGLE RECORD PREDICTION
# ============================================================

st.markdown("---")

st.subheader("🚚 Shipment Delay Prediction")

st.write(
    """
    Enter operational information below to estimate the probability
    of a logistics delay.
    """
)


# ============================================================
# IDENTIFY INPUT FEATURES
# ============================================================

# Remove target from possible prediction inputs
input_features = [
    col
    for col in df.columns
    if col != TARGET_COLUMN
]


# ============================================================
# INPUT FORM
# ============================================================

with st.form("prediction_form"):

    st.markdown("### Operational Inputs")

    input_values = {}

    # Divide features into manageable columns
    feature_cols = st.columns(3)

    for i, column in enumerate(input_features):

        current_col = feature_cols[i % 3]

        with current_col:

            series = df[column]

            # ------------------------------------------------
            # Numeric feature
            # ------------------------------------------------

            if pd.api.types.is_numeric_dtype(series):

                median_value = series.median()

                if pd.isna(median_value):
                    median_value = 0

                min_value = series.min()
                max_value = series.max()

                if pd.isna(min_value):
                    min_value = 0

                if pd.isna(max_value):
                    max_value = 1

                if min_value == max_value:

                    input_values[column] = st.number_input(
                        column,
                        value=float(median_value)
                    )

                else:

                    input_values[column] = st.number_input(
                        column,
                        value=float(median_value)
                    )

            # ------------------------------------------------
            # Categorical feature
            # ------------------------------------------------

            else:

                values = (
                    series
                    .dropna()
                    .astype(str)
                    .unique()
                    .tolist()
                )

                values = sorted(values)

                if len(values) == 0:

                    input_values[column] = st.text_input(
                        column,
                        value=""
                    )

                else:

                    input_values[column] = st.selectbox(
                        column,
                        values
                    )


    predict_button = st.form_submit_button(
        "🔮 Predict Delay Risk",
        use_container_width=True
    )


# ============================================================
# PREDICTION
# ============================================================

if predict_button:

    input_df = pd.DataFrame(
        [input_values]
    )

    # --------------------------------------------------------
    # Ensure original feature order
    # --------------------------------------------------------

    missing_columns = [
        col
        for col in input_features
        if col not in input_df.columns
    ]

    if missing_columns:

        st.error(
            f"Missing input columns: {missing_columns}"
        )

        st.stop()


    input_df = input_df[input_features]


    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    try:

        probability = final_model.predict_proba(
            input_df
        )[:, 1][0]

        prediction = int(
            probability >= threshold
        )

    except Exception as e:

        st.error(
            f"""
            Prediction failed.

            Error:

            `{e}`
            """
        )

        st.stop()


    # ========================================================
    # DISPLAY PREDICTION
    # ========================================================

    st.markdown("---")

    st.subheader("Prediction Result")

    result_col1, result_col2, result_col3 = st.columns(3)


    with result_col1:

        st.metric(
            "Delay Probability",
            f"{probability:.2%}"
        )


    with result_col2:

        st.metric(
            "Threshold",
            f"{threshold:.2f}"
        )


    with result_col3:

        if prediction == 1:

            st.metric(
                "Prediction",
                "⚠️ DELAY"
            )

        else:

            st.metric(
                "Prediction",
                "✅ NO DELAY"
            )


    # --------------------------------------------------------
    # Business interpretation
    # --------------------------------------------------------

    if prediction == 1:

        st.error(
            f"""
            ### ⚠️ Potential Delay Detected

            Estimated delay probability:

            **{probability:.2%}**

            Since the probability is greater than or equal to
            the selected threshold of **{threshold:.2f}**, the
            shipment is classified as a potential delay.

            **Recommended action:**

            - Review shipment status
            - Check traffic conditions
            - Review inventory availability
            - Check fleet utilization
            - Prepare operational resources
            - Consider proactive customer communication
            """
        )

    else:

        st.success(
            f"""
            ### ✅ No Delay Predicted

            Estimated delay probability:

            **{probability:.2%}**

            The probability is below the selected threshold
            of **{threshold:.2f}**.

            The shipment is currently classified as
            **No Delay**.
            """
        )


# ============================================================
# BATCH PREDICTION
# ============================================================

st.markdown("---")

st.subheader("📦 Batch Prediction")

st.write(
    """
    Apply the model to the existing logistics dataset to identify
    records with elevated delay risk.
    """
)


if st.button(
    "Run Delay Risk Analysis",
    use_container_width=True
):

    try:

        X_batch = df.drop(
            columns=[TARGET_COLUMN],
            errors="ignore"
        )

        probabilities = final_model.predict_proba(
            X_batch
        )[:, 1]

        predictions = (
            probabilities >= threshold
        ).astype(int)


        prediction_df = df.copy()

        prediction_df[
            "Predicted_Delay_Probability"
        ] = probabilities

        prediction_df[
            "Predicted_Delay"
        ] = predictions


        # ----------------------------------------------------
        # Risk categories
        # ----------------------------------------------------

        prediction_df["Risk_Level"] = pd.cut(
            probabilities,
            bins=[
                -np.inf,
                0.30,
                0.48,
                0.70,
                np.inf
            ],
            labels=[
                "Low",
                "Moderate",
                "High",
                "Critical"
            ]
        )


        # ----------------------------------------------------
        # Batch KPIs
        # ----------------------------------------------------

        predicted_delays = int(
            predictions.sum()
        )

        predicted_delay_rate = (
            predicted_delays /
            len(predictions) *
            100
            if len(predictions) > 0
            else 0
        )

        avg_probability = (
            probabilities.mean() * 100
        )


        batch_col1, batch_col2, batch_col3 = st.columns(3)


        with batch_col1:

            st.metric(
                "Predicted Delays",
                f"{predicted_delays:,}"
            )


        with batch_col2:

            st.metric(
                "Predicted Delay Rate",
                f"{predicted_delay_rate:.2f}%"
            )


        with batch_col3:

            st.metric(
                "Average Delay Probability",
                f"{avg_probability:.2f}%"
            )


        # ----------------------------------------------------
        # Risk distribution
        # ----------------------------------------------------

        st.subheader("Risk Distribution")

        risk_summary = (
            prediction_df["Risk_Level"]
            .value_counts()
            .reset_index()
        )

        risk_summary.columns = [
            "Risk_Level",
            "Count"
        ]

        fig_risk = px.bar(
            risk_summary,
            x="Risk_Level",
            y="Count",
            title="Predicted Logistics Risk Distribution",
            text_auto=True
        )

        st.plotly_chart(
            fig_risk,
            use_container_width=True
        )


        # ----------------------------------------------------
        # Probability distribution
        # ----------------------------------------------------

        st.subheader("Delay Probability Distribution")

        fig_prob = px.histogram(
            prediction_df,
            x="Predicted_Delay_Probability",
            nbins=30,
            title="Distribution of Predicted Delay Probability"
        )

        fig_prob.add_vline(
            x=threshold,
            line_dash="dash",
            annotation_text=f"Threshold = {threshold:.2f}"
        )

        st.plotly_chart(
            fig_prob,
            use_container_width=True
        )


        # ----------------------------------------------------
        # Highest-risk records
        # ----------------------------------------------------

        st.subheader("🚨 Highest-Risk Shipments")

        display_columns = []

        preferred_columns = [
            "Timestamp",
            "Asset_ID",
            "Shipment_Status",
            "Traffic_Status",
            "Inventory_Level",
            "Waiting_Time",
            "Asset_Utilization",
            "Operational_Stress_Score",
            "Fleet_Load_Index_Normalized",
            "Distance_From_Center"
        ]


        for column in preferred_columns:

            if column in prediction_df.columns:

                display_columns.append(column)


        display_columns += [
            "Predicted_Delay_Probability",
            "Predicted_Delay",
            "Risk_Level"
        ]


        high_risk_df = (
            prediction_df[
                display_columns
            ]
            .sort_values(
                "Predicted_Delay_Probability",
                ascending=False
            )
            .head(20)
        )


        st.dataframe(
            high_risk_df,
            use_container_width=True
        )


        # ----------------------------------------------------
        # Download predictions
        # ----------------------------------------------------

        csv_data = prediction_df.to_csv(
            index=False
        ).encode("utf-8")


        st.download_button(
            label="⬇️ Download Prediction Results",
            data=csv_data,
            file_name="logistics_delay_predictions.csv",
            mime="text/csv"
        )


    except Exception as e:

        st.error(
            f"""
            Batch prediction failed.

            Error:

            `{e}`
            """
        )


# ============================================================
# THRESHOLD ANALYSIS
# ============================================================

st.markdown("---")

st.subheader("⚖️ Threshold Business Analysis")

st.write(
    """
    A lower threshold increases sensitivity to potential delays.
    This is appropriate when the cost of missing a delay is greater
    than the cost of investigating a false alert.
    """
)


threshold_values = np.arange(
    0.30,
    0.71,
    0.02
)


if TARGET_COLUMN in df.columns:

    try:

        X_eval = df.drop(
            columns=[TARGET_COLUMN],
            errors="ignore"
        )

        y_eval = df[TARGET_COLUMN].astype(int)

        probabilities_eval = final_model.predict_proba(
            X_eval
        )[:, 1]


        threshold_results = []


        for t in threshold_values:

            preds = (
                probabilities_eval >= t
            ).astype(int)


            tp = (
                (preds == 1) &
                (y_eval.values == 1)
            ).sum()


            fp = (
                (preds == 1) &
                (y_eval.values == 0)
            ).sum()


            fn = (
                (preds == 0) &
                (y_eval.values == 1)
            ).sum()


            precision = (
                tp / (tp + fp)
                if (tp + fp) > 0
                else 0
            )


            recall = (
                tp / (tp + fn)
                if (tp + fn) > 0
                else 0
            )


            threshold_results.append(
                {
                    "Threshold": t,
                    "Precision": precision,
                    "Recall": recall
                }
            )


        threshold_df = pd.DataFrame(
            threshold_results
        )


        fig_threshold = px.line(
            threshold_df,
            x="Threshold",
            y=["Precision", "Recall"],
            markers=True,
            title="Precision vs Recall Across Prediction Thresholds"
        )


        fig_threshold.add_vline(
            x=FINAL_THRESHOLD,
            line_dash="dash",
            annotation_text=f"Recommended = {FINAL_THRESHOLD:.2f}"
        )


        st.plotly_chart(
            fig_threshold,
            use_container_width=True
        )


    except Exception as e:

        st.warning(
            f"Threshold analysis unavailable: {e}"
        )


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

st.markdown("---")

st.subheader("🔍 Model Feature Importance")

st.write(
    """
    The following chart shows which transformed features contribute
    most strongly to the Random Forest model's predictive decisions.
    """
)


try:

    model = final_model


    # --------------------------------------------------------
    # Extract final estimator
    # --------------------------------------------------------

    if hasattr(model, "named_steps"):

        estimator = model.steps[-1][1]

    else:

        estimator = model


    # --------------------------------------------------------
    # Feature importance
    # --------------------------------------------------------

    if hasattr(estimator, "feature_importances_"):

        importance_values = (
            estimator.feature_importances_
        )


        saved_features = list(
            feature_names
        )


        if len(importance_values) == len(saved_features):

            importance_df = pd.DataFrame(
                {
                    "Feature": saved_features,
                    "Importance": importance_values
                }
            )


            importance_df = (
                importance_df
                .sort_values(
                    "Importance",
                    ascending=False
                )
                .head(20)
            )


            fig_importance = px.bar(
                importance_df.sort_values(
                    "Importance"
                ),
                x="Importance",
                y="Feature",
                orientation="h",
                title="Top 20 Predictive Features"
            )


            st.plotly_chart(
                fig_importance,
                use_container_width=True
            )


            st.dataframe(
                importance_df,
                use_container_width=True
            )


        else:

            st.warning(
                f"""
                Feature count mismatch.

                Model features: {len(importance_values)}

                Saved feature names: {len(saved_features)}
                """
            )


    else:

        st.info(
            "Feature importance is not available for this model."
        )


except Exception as e:

    st.warning(
        f"Feature importance could not be displayed: {e}"
    )


# ============================================================
# BUSINESS RECOMMENDATIONS
# ============================================================

st.markdown("---")

st.subheader("💼 Business Recommendations")

rec_col1, rec_col2 = st.columns(2)


with rec_col1:

    st.markdown(
        """
        ### 🚨 High-Risk Operations

        When predicted delay probability is above the threshold:

        - Prioritize shipment monitoring
        - Review fleet availability
        - Check traffic conditions
        - Review inventory position
        - Investigate operational stress
        - Prepare alternative routing
        """
    )


with rec_col2:

    st.markdown(
        """
        ### 📊 Management Actions

        Management can use the predictions to:

        - Allocate fleet resources proactively
        - Prepare warehouse and dispatch teams
        - Identify emerging operational bottlenecks
        - Reduce unexpected customer delays
        - Improve delivery reliability
        - Support proactive customer communication
        """
    )


# ============================================================
# FINAL MODEL SUMMARY
# ============================================================

st.markdown("---")

st.subheader("🏆 Final Model Configuration")

summary_df = pd.DataFrame(
    {
        "Configuration": [
            "Champion Model",
            "Business Objective",
            "Classification Strategy",
            "Selected Threshold",
            "Minimum Precision",
            "Baseline Model",
            "Baseline F1",
            "Optimized F1",
            "Optimized Recall",
            "Optimized Precision",
            "ROC-AUC",
            "PR-AUC"
        ],
        "Value": [
            "Random Forest",
            "Early detection of potential delays",
            "Recall First",
            "0.48",
            "0.70",
            "Gradient Boosting",
            "0.6825",
            "0.7306",
            "0.7407",
            "0.7207",
            "0.7748",
            "0.8630"
        ]
    }
)


st.dataframe(
    summary_df,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    """
    Smart Logistics Predictive Analytics | Random Forest |
    Recall-First Delay Detection | Threshold = 0.48
    """
)
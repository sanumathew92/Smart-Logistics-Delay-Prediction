# ============================================================
# SMART LOGISTICS DELAY PREDICTION SYSTEM
# Streamlit Application
# ============================================================

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Smart Logistics Delay Prediction",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PATH CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

ASSET_DIR = BASE_DIR / "streamlit_assets"

MODEL_PATH = ASSET_DIR / "final_model.pkl"
FEATURE_LIST_PATH = ASSET_DIR / "feature_names.pkl"

CONFIG_PATH = ASSET_DIR / "model_config.json"
METADATA_PATH = ASSET_DIR / "input_metadata.json"

PERFORMANCE_PATH = ASSET_DIR / "model_performance.csv"
IMPORTANCE_PATH = ASSET_DIR / "feature_importance.csv"
KPI_PATH = ASSET_DIR / "dashboard_kpis.json"


# ============================================================
# REQUIRED FILE VALIDATION
# ============================================================

required_files = {
    "Final Model": MODEL_PATH,
    "Feature Names": FEATURE_LIST_PATH,
    "Model Configuration": CONFIG_PATH,
    "Input Metadata": METADATA_PATH,
    "Model Performance": PERFORMANCE_PATH,
    "Feature Importance": IMPORTANCE_PATH,
    "Dashboard KPIs": KPI_PATH
}

missing_files = [
    f"{name}: {path}"
    for name, path in required_files.items()
    if not path.exists()
]

if missing_files:

    st.error("❌ Required application files are missing.")

    for item in missing_files:
        st.write(f"- {item}")

    st.info(
        "Please verify that streamlit_assets is inside "
        "the streamlit_app folder."
    )

    st.stop()


# ============================================================
# LOAD FUNCTIONS
# ============================================================

@st.cache_resource
def load_model():

    return joblib.load(MODEL_PATH)


@st.cache_data
def load_feature_names():

    return joblib.load(FEATURE_LIST_PATH)


@st.cache_data
def load_json(path):

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_csv(path):

    return pd.read_csv(path)


# ============================================================
# LOAD ASSETS
# ============================================================

final_model = load_model()

feature_names = load_feature_names()

model_config = load_json(CONFIG_PATH)

input_metadata = load_json(METADATA_PATH)

performance_df = load_csv(PERFORMANCE_PATH)

importance_df = load_csv(IMPORTANCE_PATH)

kpi_data = load_json(KPI_PATH)


# ============================================================
# FINAL BUSINESS CONFIGURATION
# ============================================================

CHAMPION_MODEL = model_config.get(
    "champion_model",
    "Random Forest"
)

FINAL_THRESHOLD = float(
    model_config.get(
        "threshold",
        0.48
    )
)

MIN_PRECISION = float(
    model_config.get(
        "minimum_precision",
        0.70
    )
)

BUSINESS_OBJECTIVE = model_config.get(
    "business_objective",
    "Prioritize early detection of potential logistics delays"
)


# ============================================================
# FINAL MODEL PERFORMANCE
# ============================================================

FINAL_ACCURACY = float(
    model_config.get("accuracy", 0.7050)
)

FINAL_PRECISION = float(
    model_config.get("precision", 0.7207)
)

FINAL_RECALL = float(
    model_config.get("recall", 0.7407)
)

FINAL_F1 = float(
    model_config.get("f1", 0.7306)
)

FINAL_ROC_AUC = float(
    model_config.get("roc_auc", 0.7748)
)

FINAL_PR_AUC = float(
    model_config.get("pr_auc", 0.8630)
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🚚 Smart Logistics")

st.sidebar.markdown(
    """
    ### Logistics Delay Prediction

    Predict potential logistics delays using the
    optimized Random Forest model.
    """
)

st.sidebar.divider()

page = st.sidebar.radio(
    "Navigate",
    [
        "Executive Summary",
        "Delay Prediction",
        "Model Performance",
        "Model Explainability",
        "Business Diagnostics",
        "Data Explorer",
        "About"
    ]
)

st.sidebar.divider()

st.sidebar.metric(
    "Champion Model",
    CHAMPION_MODEL
)

st.sidebar.metric(
    "Decision Threshold",
    f"{FINAL_THRESHOLD:.2f}"
)

st.sidebar.metric(
    "Recall",
    f"{FINAL_RECALL:.2%}"
)


# ============================================================
# HEADER
# ============================================================

st.title("🚚 Smart Logistics Delay Prediction")

st.caption(
    "Predictive analytics and business decision-support system "
    "for proactive logistics delay detection."
)


# ============================================================
# PAGE 1 — EXECUTIVE SUMMARY
# ============================================================

if page == "Executive Summary":

    st.header("📊 Executive Summary")

    st.markdown(
        """
        ### Business Objective

        The system is designed to identify potential logistics
        delays early enough to support proactive operational action.

        The final model uses a **recall-first classification strategy**
        with a threshold of **0.48**.
        """
    )

    st.divider()

    # --------------------------------------------------------
    # KPI ROW
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Accuracy",
            f"{FINAL_ACCURACY:.2%}"
        )

    with col2:

        st.metric(
            "Precision",
            f"{FINAL_PRECISION:.2%}"
        )

    with col3:

        st.metric(
            "Recall",
            f"{FINAL_RECALL:.2%}"
        )

    with col4:

        st.metric(
            "F1 Score",
            f"{FINAL_F1:.2%}"
        )

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "ROC-AUC",
            f"{FINAL_ROC_AUC:.4f}"
        )

    with col2:

        st.metric(
            "PR-AUC",
            f"{FINAL_PR_AUC:.4f}"
        )

    with col3:

        st.metric(
            "Minimum Precision",
            f"{MIN_PRECISION:.0%}"
        )

    st.divider()

    # --------------------------------------------------------
    # MODEL SUMMARY
    # --------------------------------------------------------

    st.subheader("🏆 Final Model")

    summary_df = pd.DataFrame({
        "Component": [
            "Champion Model",
            "Business Objective",
            "Decision Threshold",
            "Minimum Precision",
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score",
            "ROC-AUC",
            "PR-AUC"
        ],
        "Value": [
            CHAMPION_MODEL,
            BUSINESS_OBJECTIVE,
            FINAL_THRESHOLD,
            MIN_PRECISION,
            FINAL_ACCURACY,
            FINAL_PRECISION,
            FINAL_RECALL,
            FINAL_F1,
            FINAL_ROC_AUC,
            FINAL_PR_AUC
        ]
    })

    st.dataframe(
        summary_df,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.subheader("💼 Business Interpretation")

    st.success(
        """
        The model prioritizes early detection of potential logistics
        delays. The lower threshold increases the number of shipments
        identified as potential delays while maintaining the required
        minimum precision constraint.
        """
    )

    st.warning(
        """
        Operational trade-off: more false alerts are accepted in
        exchange for reducing missed delay cases.
        """
    )


# ============================================================
# PAGE 2 — DELAY PREDICTION
# ============================================================

elif page == "Delay Prediction":

    st.header("🔮 Logistics Delay Prediction")

    st.markdown(
        f"""
        Enter shipment and operational information below.

        **Decision threshold:** `{FINAL_THRESHOLD:.2f}`

        A probability greater than or equal to this threshold
        is classified as a potential logistics delay.
        """
    )

    st.divider()

    user_input = {}

    with st.form("delay_prediction_form"):

        st.subheader("📋 Shipment & Operational Information")

        metadata_items = list(input_metadata.items())

        for start in range(0, len(metadata_items), 2):

            row = metadata_items[start:start + 2]

            columns = st.columns(len(row))

            for column, (feature, metadata) in zip(
                columns,
                row
            ):

                with column:

                    feature_type = metadata.get(
                        "type",
                        "numeric"
                    )

                    if feature_type == "categorical":

                        values = metadata.get(
                            "values",
                            []
                        )

                        if len(values) > 0:

                            user_input[feature] = st.selectbox(
                                feature,
                                values
                            )

                    else:

                        min_value = float(
                            metadata.get(
                                "min",
                                0
                            )
                        )

                        max_value = float(
                            metadata.get(
                                "max",
                                100
                            )
                        )

                        median_value = float(
                            metadata.get(
                                "median",
                                (min_value + max_value) / 2
                            )
                        )

                        user_input[feature] = st.number_input(
                            feature,
                            min_value=min_value,
                            max_value=max_value,
                            value=median_value
                        )

        st.divider()

        submitted = st.form_submit_button(
            "🔍 Predict Logistics Delay",
            use_container_width=True
        )

    if submitted:

        try:

            input_df = pd.DataFrame(
                [user_input]
            )

            # ------------------------------------------------
            # Ensure expected raw columns
            # ------------------------------------------------

            if hasattr(
                final_model,
                "feature_names_in_"
            ):

                expected_columns = list(
                    final_model.feature_names_in_
                )

                for column in expected_columns:

                    if column not in input_df.columns:

                        input_df[column] = np.nan

                input_df = input_df[
                    expected_columns
                ]

            # ------------------------------------------------
            # Prediction
            # ------------------------------------------------

            probability = float(
                final_model.predict_proba(
                    input_df
                )[0, 1]
            )

            prediction = int(
                probability >= FINAL_THRESHOLD
            )

            st.divider()

            st.subheader("🎯 Prediction Result")

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "Delay Probability",
                    f"{probability:.2%}"
                )

            with col2:

                st.metric(
                    "Decision Threshold",
                    f"{FINAL_THRESHOLD:.2f}"
                )

            with col3:

                if prediction == 1:

                    st.metric(
                        "Prediction",
                        "⚠️ Potential Delay"
                    )

                else:

                    st.metric(
                        "Prediction",
                        "✅ No Delay"
                    )

            if prediction == 1:

                st.error(
                    """
                    ⚠️ Potential logistics delay detected.

                    Consider proactive operational intervention,
                    shipment monitoring and resource allocation.
                    """
                )

            else:

                st.success(
                    """
                    ✅ No potential delay detected at the selected
                    decision threshold.
                    """
                )

            # ------------------------------------------------
            # Probability gauge
            # ------------------------------------------------

            st.progress(
                min(probability, 1.0)
            )

        except Exception as e:

            st.error(
                "Prediction failed."
            )

            st.exception(e)


# ============================================================
# PAGE 3 — MODEL PERFORMANCE
# ============================================================

elif page == "Model Performance":

    st.header("📈 Model Performance")

    st.markdown(
        """
        Comparison of the models evaluated during the
        optimization process.
        """
    )

    st.divider()

    if not performance_df.empty:

        st.dataframe(
            performance_df,
            use_container_width=True,
            hide_index=True
        )

        numeric_columns = performance_df.select_dtypes(
            include=np.number
        ).columns

        if len(numeric_columns) > 0:

            st.subheader("Performance Comparison")

            chart_columns = [
                c for c in [
                    "Accuracy",
                    "Precision",
                    "Recall",
                    "F1",
                    "ROC_AUC",
                    "PR_AUC"
                ]
                if c in performance_df.columns
            ]

            if chart_columns:

                chart_df = performance_df[
                    chart_columns
                ].copy()

                if "Model" in performance_df.columns:

                    chart_df.index = performance_df[
                        "Model"
                    ]

                st.bar_chart(
                    chart_df
                )

    st.divider()

    st.subheader("🏆 Final Champion")

    st.success(
        f"""
        **{CHAMPION_MODEL}** was selected as the final champion
        model for the recall-first business objective.

        The final classification threshold is **{FINAL_THRESHOLD:.2f}**.
        """
    )


# ============================================================
# PAGE 4 — MODEL EXPLAINABILITY
# ============================================================

elif page == "Model Explainability":

    st.header("🔎 Model Explainability")

    st.markdown(
        """
        This section explains which features contribute most to
        the model's logistics delay predictions.
        """
    )

    st.divider()

    # --------------------------------------------------------
    # Feature importance
    # --------------------------------------------------------

    if not importance_df.empty:

        st.subheader("📊 Feature Importance")

        st.dataframe(
            importance_df.head(20),
            use_container_width=True,
            hide_index=True
        )

        # Determine feature/value columns
        feature_column = None
        importance_column = None

        for column in importance_df.columns:

            lower = column.lower()

            if (
                "feature" in lower
                or "variable" in lower
            ):

                feature_column = column

            if (
                "importance" in lower
                or "shap" in lower
            ):

                importance_column = column

        if (
            feature_column is not None
            and importance_column is not None
        ):

            plot_df = importance_df[
                [
                    feature_column,
                    importance_column
                ]
            ].copy()

            plot_df = plot_df.head(15)

            plot_df = plot_df.set_index(
                feature_column
            )

            st.subheader(
                "Top 15 Important Features"
            )

            st.bar_chart(
                plot_df
            )

    st.divider()

    st.subheader("🔍 Key Explainability Findings")

    st.markdown(
        """
        Based on the feature importance analysis, traffic-related
        variables are among the strongest contributors to delay
        prediction.

        In particular, the transformed feature:

        **`cat__Traffic_Status_Heavy`**

        was identified as the most influential feature in the
        available feature-importance results.

        Other important operational and environmental variables
        include inventory level, distance from center, transaction
        amount, operational stress and customer-value related
        variables.
        """
    )

    st.info(
        """
        Explainability should be interpreted as model behavior,
        not as proof of causation.
        """
    )


# ============================================================
# PAGE 5 — BUSINESS DIAGNOSTICS
# ============================================================

elif page == "Business Diagnostics":

    st.header("💼 Business Diagnostics")

    st.markdown(
        """
        Business diagnostics translate model predictions into
        operational insights for logistics management.
        """
    )

    st.divider()

    # --------------------------------------------------------
    # Business objective
    # --------------------------------------------------------

    st.subheader("🎯 Business Objective")

    st.info(
        BUSINESS_OBJECTIVE
    )

    # --------------------------------------------------------
    # Threshold
    # --------------------------------------------------------

    st.subheader("⚖️ Decision Threshold")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Threshold",
            f"{FINAL_THRESHOLD:.2f}"
        )

    with col2:

        st.metric(
            "Minimum Precision",
            f"{MIN_PRECISION:.0%}"
        )

    with col3:

        st.metric(
            "Recall",
            f"{FINAL_RECALL:.2%}"
        )

    st.markdown(
        """
        The threshold was deliberately lowered to support
        early detection of potential logistics delays.

        This increases sensitivity to potential delay cases,
        while accepting additional false alerts.
        """
    )

    st.divider()

    # --------------------------------------------------------
    # Operational interpretation
    # --------------------------------------------------------

    st.subheader("🚨 Operational Interpretation")

    diagnostics = pd.DataFrame({
        "Area": [
            "Delay Detection",
            "False Alerts",
            "Resource Planning",
            "Shipment Monitoring",
            "Traffic Management",
            "Inventory Management"
        ],
        "Recommended Action": [
            "Prioritize shipments flagged as potential delays.",
            "Validate high-risk alerts before costly intervention.",
            "Prepare additional resources for high-risk shipments.",
            "Increase monitoring frequency for flagged shipments.",
            "Investigate heavy traffic conditions proactively.",
            "Monitor inventory pressure associated with delay risk."
        ]
    })

    st.dataframe(
        diagnostics,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    # --------------------------------------------------------
    # Feature diagnostics
    # --------------------------------------------------------

    st.subheader("🔬 Key Business Drivers")

    business_drivers = pd.DataFrame({
        "Feature": [
            "Traffic_Status_Heavy",
            "Traffic_Status_Clear",
            "Traffic_Status_Detour",
            "Inventory_Level",
            "Distance_From_Center",
            "User_Transaction_Amount",
            "Operational_Stress_Score",
            "Customer_Value_Index",
            "Inventory_Demand_Gap",
            "Temperature"
        ],
        "Business Meaning": [
            "Heavy traffic conditions may substantially increase delay risk.",
            "Clear traffic provides comparatively lower operational pressure.",
            "Detours may increase route complexity and travel time.",
            "Low or stressed inventory can affect fulfilment operations.",
            "Longer distances can increase exposure to disruptions.",
            "Transaction activity may indicate shipment intensity.",
            "Higher operational stress can indicate capacity pressure.",
            "Customer-value characteristics may affect operational priority.",
            "Demand pressure can create fulfilment stress.",
            "Environmental conditions can influence logistics operations."
        ]
    })

    st.dataframe(
        business_drivers,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    # --------------------------------------------------------
    # Confusion matrix interpretation
    # --------------------------------------------------------

    st.subheader("📌 Model Trade-off")

    col1, col2 = st.columns(2)

    with col1:

        st.success(
            f"""
            **Recall: {FINAL_RECALL:.2%}**

            The model identifies a relatively high proportion of
            actual delay cases, supporting the objective of early
            intervention.
            """
        )

    with col2:

        st.warning(
            f"""
            **Precision: {FINAL_PRECISION:.2%}**

            Some predicted delay cases will not become actual
            delays. These false alerts are the operational cost
            of using a recall-first strategy.
            """
        )


# ============================================================
# PAGE 6 — DATA EXPLORER
# ============================================================

elif page == "Data Explorer":

    st.header("🔍 Data Explorer")

    st.markdown(
        """
        Explore the model's stored performance and feature
        importance information.
        """
    )

    tab1, tab2 = st.tabs(
        [
            "Model Performance",
            "Feature Importance"
        ]
    )

    with tab1:

        st.subheader(
            "Model Performance Dataset"
        )

        st.dataframe(
            performance_df,
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "⬇️ Download Model Performance",
            performance_df.to_csv(
                index=False
            ),
            file_name="model_performance.csv",
            mime="text/csv"
        )

    with tab2:

        st.subheader(
            "Feature Importance Dataset"
        )

        st.dataframe(
            importance_df,
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "⬇️ Download Feature Importance",
            importance_df.to_csv(
                index=False
            ),
            file_name="feature_importance.csv",
            mime="text/csv"
        )

    st.divider()

    st.subheader("Model Feature Count")

    st.metric(
        "Saved Feature Names",
        len(feature_names)
    )


# ============================================================
# PAGE 7 — ABOUT
# ============================================================

elif page == "About":

    st.header("ℹ️ About the Project")

    st.markdown(
        """
        ## Smart Logistics Delay Prediction

        This project develops a machine-learning based decision
        support system for predicting potential logistics delays.

        ### Machine Learning Workflow

        The project includes:

        - Data understanding
        - Exploratory data analysis
        - Data preprocessing
        - Feature engineering
        - Leakage validation
        - Chronological train/test split
        - Baseline machine learning
        - Time-series cross-validation
        - Model optimization
        - Threshold optimization
        - Model explainability
        - Business diagnostics
        - Streamlit deployment

        ### Final Model

        **Random Forest**

        ### Business Strategy

        The final model uses a recall-first threshold of:

        **0.48**

        with a minimum precision constraint of:

        **70%**

        ### Objective

        The primary business objective is:

        **Never miss a potential logistics delay wherever
        operationally possible, while maintaining an acceptable
        level of precision.**
        """
    )

    st.divider()

    st.subheader("Final Model Metrics")

    final_metrics = pd.DataFrame({
        "Metric": [
            "Accuracy",
            "Precision",
            "Recall",
            "F1",
            "ROC-AUC",
            "PR-AUC"
        ],
        "Value": [
            FINAL_ACCURACY,
            FINAL_PRECISION,
            FINAL_RECALL,
            FINAL_F1,
            FINAL_ROC_AUC,
            FINAL_PR_AUC
        ]
    })

    final_metrics["Value"] = final_metrics[
        "Value"
    ].map(
        lambda x: f"{x:.4f}"
    )

    st.dataframe(
        final_metrics,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.caption(
        "Smart Logistics Delay Prediction System"
    )

    st.caption(
        "Machine Learning • Explainability • "
        "Business Diagnostics • Decision Support"
    )
# ============================================================
# 9_SHAP_EXPLAINABILITY.PY
# ============================================================
# Purpose:
#     Explain why the Random Forest model predicts a logistics
#     shipment as "Delay" or "No Delay".
#
# Model:
#     Random Forest
#
# Business threshold:
#     0.48
#
# Saved files:
#     streamlit_assets/final_model.pkl
#     streamlit_assets/feature_names.pkl
#
# ============================================================


# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import os
import joblib
import warnings

import numpy as np
import pandas as pd

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

import shap

warnings.filterwarnings("ignore")


# ============================================================
# 2. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="SHAP Explainability",
    page_icon="🔍",
    layout="wide"
)


# ============================================================
# 3. PAGE TITLE
# ============================================================

st.title("🔍 SHAP Explainability")

st.markdown(
    """
    ### Understanding Why the Model Predicts a Delay

    This page explains the **Random Forest logistics delay model**
    using **SHAP (SHapley Additive exPlanations)**.

    The objective is to answer:

    > **Why did the model predict a potential logistics delay?**

    The analysis provides both:

    - **Global explainability** → What features influence the model most?
    - **Local explainability** → Why did the model make this particular prediction?
    """
)


# ============================================================
# 4. DEFINE PROJECT DIRECTORIES
# ============================================================

# Get the directory containing this Python file.
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Move one level upward from:
#
# Smart Logistic Dataset/
#       streamlit_app/
#           pages/
#
# to:
#
# Smart Logistic Dataset/
#
PROJECT_ROOT = os.path.abspath(
    os.path.join(CURRENT_DIR, "..", "..")
)

# Location of Streamlit assets.
ASSETS_DIR = os.path.join(
    PROJECT_ROOT,
    "streamlit_app",
    "streamlit_assets"
)

# Location of the feature-engineered dataset.
DATA_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed"
)


# ============================================================
# 5. DEFINE FILE PATHS
# ============================================================

MODEL_PATH = os.path.join(
    ASSETS_DIR,
    "final_model.pkl"
)

FEATURE_NAMES_PATH = os.path.join(
    ASSETS_DIR,
    "feature_names.pkl"
)

DATA_PATH = os.path.join(
    DATA_DIR,
    "logistics_feature_engineered.csv"
)


# ============================================================
# 6. DISPLAY FILE PATHS FOR DEBUGGING
# ============================================================

with st.expander("🔧 File Configuration", expanded=False):

    st.write("Model path:")
    st.code(MODEL_PATH)

    st.write("Feature names path:")
    st.code(FEATURE_NAMES_PATH)

    st.write("Dataset path:")
    st.code(DATA_PATH)


# ============================================================
# 7. CHECK WHETHER REQUIRED FILES EXIST
# ============================================================

if not os.path.exists(MODEL_PATH):

    st.error(
        f"""
        ❌ Final model not found.

        Expected location:

        `{MODEL_PATH}`

        Please make sure `final_model.pkl` exists inside:

        `streamlit_app/streamlit_assets/`
        """
    )

    st.stop()


if not os.path.exists(FEATURE_NAMES_PATH):

    st.error(
        f"""
        ❌ Feature names file not found.

        Expected location:

        `{FEATURE_NAMES_PATH}`

        Please make sure `feature_names.pkl` exists.
        """
    )

    st.stop()


if not os.path.exists(DATA_PATH):

    st.error(
        f"""
        ❌ Dataset not found.

        Expected location:

        `{DATA_PATH}`

        Please make sure the feature-engineered dataset exists.
        """
    )

    st.stop()


# ============================================================
# 8. LOAD FINAL MODEL
# ============================================================

@st.cache_resource
def load_model():

    """
    Load the complete trained pipeline.

    The pipeline should contain:

        preprocessing
              ↓
        Random Forest

    Therefore, we should NOT manually preprocess the raw data
    before passing it to the model.
    """

    return joblib.load(MODEL_PATH)


# Load model.
final_model = load_model()


# ============================================================
# 9. LOAD FEATURE NAMES
# ============================================================

@st.cache_resource
def load_feature_names():

    """
    Load the transformed feature names.

    These should correspond exactly to the columns generated
    by the fitted preprocessing pipeline.
    """

    names = joblib.load(FEATURE_NAMES_PATH)

    # Convert to list to make downstream processing predictable.
    return list(names)


feature_names = load_feature_names()


# ============================================================
# 10. LOAD DATASET
# ============================================================

@st.cache_data
def load_dataset():

    """
    Load the feature-engineered dataset.

    This dataset contains the original business variables plus
    engineered variables such as:

        Operational_Stress_Score
        Inventory_Demand_Gap
        Fleet_Load_Index
        Customer_Value_Index
        Traffic_Utilization_Interaction
        etc.
    """

    return pd.read_csv(DATA_PATH)


df = load_dataset()


# ============================================================
# 11. DISPLAY BASIC MODEL INFORMATION
# ============================================================

st.subheader("📌 Model Information")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Model",
        "Random Forest"
    )

with col2:
    st.metric(
        "Decision Threshold",
        "0.48"
    )

with col3:
    st.metric(
        "Transformed Features",
        len(feature_names)
    )

with col4:
    st.metric(
        "Dataset Rows",
        f"{len(df):,}"
    )


# ============================================================
# 12. BUSINESS INTERPRETATION
# ============================================================

st.info(
    """
    **Business objective:** Prioritize early detection of potential
    logistics delays.

    The selected probability threshold is **0.48**.

    A shipment is classified as a potential delay when:

    **Predicted probability ≥ 0.48**

    This lower threshold was selected to improve delay detection
    while maintaining the required minimum precision of 0.70.
    """
)


# ============================================================
# 13. IDENTIFY PREPROCESSOR
# ============================================================

# The saved model is expected to be a sklearn Pipeline.

if not hasattr(final_model, "named_steps"):

    st.error(
        """
        ❌ The saved model does not appear to be a sklearn Pipeline.

        SHAP analysis in this page expects the saved model to contain
        a preprocessing step and a Random Forest model.
        """
    )

    st.stop()


# ============================================================
# 14. DISPLAY PIPELINE STRUCTURE
# ============================================================

with st.expander("🔎 Model Pipeline Structure", expanded=False):

    st.write(
        list(final_model.named_steps.keys())
    )


# ============================================================
# 15. FIND PREPROCESSOR
# ============================================================

if "preprocessor" not in final_model.named_steps:

    st.error(
        """
        ❌ `preprocessor` was not found in the saved model pipeline.

        Expected pipeline structure:

        Pipeline(
            steps=[
                ("preprocessor", ...),
                ("model", RandomForestClassifier(...))
            ]
        )
        """
    )

    st.stop()


preprocessor = final_model.named_steps["preprocessor"]


# ============================================================
# 16. FIND FINAL MODEL
# ============================================================

# Your optimization workflow used a Random Forest champion model.

if "model" in final_model.named_steps:

    rf_model = final_model.named_steps["model"]

else:

    # Fallback:
    # Assume the last step is the model.
    rf_model = final_model.steps[-1][1]


# ============================================================
# 17. VERIFY MODEL TYPE
# ============================================================

st.subheader("🌲 Final Model")

st.write(
    f"Final estimator: **{type(rf_model).__name__}**"
)


# ============================================================
# 18. VERIFY FEATURE COUNT
# ============================================================

# Transform the complete dataset through the fitted preprocessor.
#
# IMPORTANT:
#     We do NOT fit the preprocessor again.
#
# We only call transform().
# This ensures that the exact same encoding/scaling used during
# model training is used during SHAP analysis.

try:

    X_transformed = preprocessor.transform(df.drop(
        columns=["Logistics_Delay"],
        errors="ignore"
    ))

except Exception as e:

    st.error(
        f"""
        ❌ Unable to transform the dataset.

        Error:

        `{e}`
        """
    )

    st.stop()


# Convert sparse matrix if necessary.
if hasattr(X_transformed, "toarray"):

    X_transformed = X_transformed.toarray()


# Convert to NumPy array.
X_transformed = np.asarray(
    X_transformed,
    dtype=float
)


actual_feature_count = X_transformed.shape[1]

saved_feature_count = len(feature_names)


# ============================================================
# 19. FEATURE COUNT VALIDATION
# ============================================================

if actual_feature_count != saved_feature_count:

    st.error(
        f"""
        ❌ Feature mismatch detected.

        Actual transformed features:
        `{actual_feature_count}`

        Saved feature names:
        `{saved_feature_count}`

        SHAP analysis has been stopped to prevent incorrect
        feature attribution.
        """
    )

    st.stop()


st.success(
    f"""
    ✓ Feature validation successful.

    Transformed features: **{actual_feature_count}**

    Feature names: **{saved_feature_count}**
    """
)


# ============================================================
# 20. CREATE TRANSFORMED DATAFRAME
# ============================================================

X_transformed_df = pd.DataFrame(
    X_transformed,
    columns=feature_names
)


# ============================================================
# 21. LIMIT SHAP DATASET SIZE
# ============================================================

# SHAP calculations can become computationally expensive.
#
# We therefore use a sample of the dataset for interactive
# explainability.

MAX_SHAP_ROWS = 1000


if len(X_transformed_df) > MAX_SHAP_ROWS:

    shap_sample = X_transformed_df.sample(
        MAX_SHAP_ROWS,
        random_state=42
    )

else:

    shap_sample = X_transformed_df.copy()


# ============================================================
# 22. CREATE SHAP EXPLAINER
# ============================================================

st.subheader("🧠 SHAP Model Explainer")

st.write(
    """
    SHAP measures how each feature contributes to the model's
    prediction.

    Positive SHAP values generally push the prediction toward
    **Delay**.

    Negative SHAP values generally push the prediction toward
    **No Delay**.
    """
)


@st.cache_resource
def create_shap_explainer(_model):

    """
    Create a TreeExplainer for the Random Forest model.

    Random Forest is a tree-based model, therefore
    TreeExplainer is appropriate.
    """

    return shap.TreeExplainer(_model)


try:

    explainer = create_shap_explainer(
        rf_model
    )

except Exception as e:

    st.error(
        f"""
        ❌ Unable to create SHAP TreeExplainer.

        Error:

        `{e}`
        """
    )

    st.stop()


# ============================================================
# 23. CALCULATE SHAP VALUES
# ============================================================

@st.cache_data
def calculate_shap_values(
    _explainer,
    X_data
):

    """
    Calculate SHAP values.

    For binary classification, SHAP can return either:

        list of arrays
    or
        a 3-dimensional array

    depending on the installed SHAP version.

    This function normalizes those formats.
    """

    values = _explainer.shap_values(
        X_data
    )

    return values


try:

    raw_shap_values = calculate_shap_values(
        explainer,
        shap_sample
    )

except Exception as e:

    st.error(
        f"""
        ❌ SHAP calculation failed.

        Error:

        `{e}`
        """
    )

    st.stop()


# ============================================================
# 24. EXTRACT SHAP VALUES FOR DELAY CLASS
# ============================================================

def extract_delay_shap_values(
    shap_values,
    n_features
):

    """
    Extract SHAP values corresponding to class 1 = Delay.

    Handles common SHAP output formats.

    Returns:

        2-dimensional NumPy array
    """

    # --------------------------------------------------------
    # Case 1:
    # list of arrays
    # --------------------------------------------------------

    if isinstance(shap_values, list):

        if len(shap_values) >= 2:

            return np.asarray(
                shap_values[1]
            )

        return np.asarray(
            shap_values[0]
        )

    # Convert to NumPy.
    values = np.asarray(
        shap_values
    )

    # --------------------------------------------------------
    # Case 2:
    # 3-dimensional array
    # --------------------------------------------------------

    if values.ndim == 3:

        # Possible shape:
        #
        # samples × features × classes
        #

        if values.shape[2] == 2:

            return values[:, :, 1]

        # Possible shape:
        #
        # samples × classes × features
        #

        if values.shape[1] == 2:

            return values[:, 1, :]

    # --------------------------------------------------------
    # Case 3:
    # 2-dimensional array
    # --------------------------------------------------------

    if values.ndim == 2:

        return values

    raise ValueError(
        f"Unsupported SHAP output shape: {values.shape}"
    )


try:

    shap_delay = extract_delay_shap_values(
        raw_shap_values,
        len(feature_names)
    )

except Exception as e:

    st.error(
        f"""
        ❌ Unable to interpret SHAP output.

        Error:

        `{e}`
        """
    )

    st.stop()


# ============================================================
# 25. FINAL SHAP SHAPE VALIDATION
# ============================================================

if shap_delay.shape[1] != len(feature_names):

    st.error(
        f"""
        ❌ SHAP feature mismatch.

        SHAP features:
        `{shap_delay.shape[1]}`

        Feature names:
        `{len(feature_names)}`
        """
    )

    st.stop()


st.success(
    f"""
    ✓ SHAP calculation successful.

    Samples explained: **{shap_delay.shape[0]:,}**

    Features explained: **{shap_delay.shape[1]:,}**
    """
)


# ============================================================
# 26. GLOBAL SHAP FEATURE IMPORTANCE
# ============================================================

st.header("📊 Global Feature Importance")

st.write(
    """
    Global SHAP importance shows which variables have the largest
    overall influence on the Random Forest's delay predictions.

    Importance is calculated as:

    **Mean(|SHAP value|)**
    """
)


mean_abs_shap = np.abs(
    shap_delay
).mean(axis=0)


# ------------------------------------------------------------
# Create DataFrame
# ------------------------------------------------------------

shap_importance = pd.DataFrame(
    {
        "Feature": feature_names,
        "Mean_Absolute_SHAP": mean_abs_shap
    }
)


# ------------------------------------------------------------
# Sort by importance
# ------------------------------------------------------------

shap_importance = shap_importance.sort_values(
    "Mean_Absolute_SHAP",
    ascending=False
).reset_index(drop=True)


# ------------------------------------------------------------
# Display top features
# ------------------------------------------------------------

TOP_N = 20

top_shap = shap_importance.head(
    TOP_N
)


# ============================================================
# 27. SHAP BAR CHART
# ============================================================

fig_importance = px.bar(
    top_shap.sort_values(
        "Mean_Absolute_SHAP"
    ),
    x="Mean_Absolute_SHAP",
    y="Feature",
    orientation="h",
    title="Top 20 Features by Mean Absolute SHAP Value",
    labels={
        "Mean_Absolute_SHAP": "Mean |SHAP Value|",
        "Feature": "Feature"
    }
)

fig_importance.update_layout(
    height=700
)

st.plotly_chart(
    fig_importance,
    use_container_width=True
)


# ============================================================
# 28. DISPLAY IMPORTANCE TABLE
# ============================================================

with st.expander(
    "📋 View SHAP Feature Importance Table"
):

    display_importance = top_shap.copy()

    display_importance[
        "Mean_Absolute_SHAP"
    ] = display_importance[
        "Mean_Absolute_SHAP"
    ].round(6)

    st.dataframe(
        display_importance,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# 29. SHAP BEESWARM
# ============================================================

st.header("🐝 SHAP Beeswarm Plot")

st.write(
    """
    The beeswarm plot provides more detail than the simple
    importance ranking.

    Interpretation:

    - **Right side** → feature pushes prediction toward Delay
    - **Left side** → feature pushes prediction toward No Delay
    - **Red / high values** → higher feature values
    - **Blue / low values** → lower feature values
    """
)


# ------------------------------------------------------------
# Create matplotlib SHAP plot
# ------------------------------------------------------------

try:

    shap_fig = plt.figure(
        figsize=(10, 8)
    )

except NameError:

    import matplotlib.pyplot as plt

    shap_fig = plt.figure(
        figsize=(10, 8)
    )


shap.summary_plot(
    shap_delay,
    shap_sample,
    feature_names=feature_names,
    max_display=20,
    show=False
)

st.pyplot(
    shap_fig,
    clear_figure=True
)


# ============================================================
# 30. BUSINESS INTERPRETATION OF TOP FEATURES
# ============================================================

st.header("💼 Business Interpretation")


# ------------------------------------------------------------
# Extract top 10 features
# ------------------------------------------------------------

top_features = shap_importance.head(
    10
)["Feature"].tolist()


st.write(
    "The model's strongest global drivers are:"
)


for i, feature in enumerate(
    top_features,
    start=1
):

    st.write(
        f"**{i}. {feature}**"
    )


# ============================================================
# 31. FEATURE CATEGORY INTERPRETATION
# ============================================================

st.subheader(
    "Operational Interpretation"
)


def interpret_feature(
    feature
):

    """
    Convert technical feature names into business language.
    """

    feature_lower = feature.lower()

    if "traffic_status" in feature_lower:

        return (
            "Traffic conditions are influencing delay risk. "
            "Heavy or disrupted traffic may require proactive "
            "routing or dispatch adjustments."
        )

    if "inventory" in feature_lower:

        return (
            "Inventory conditions are influencing delay risk. "
            "Low inventory coverage or inventory-demand imbalance "
            "may create operational pressure."
        )

    if "utilization" in feature_lower:

        return (
            "Fleet or operational utilization is influencing "
            "delay risk. High utilization may indicate limited "
            "operational capacity."
        )

    if "waiting" in feature_lower:

        return (
            "Waiting time is contributing to delay risk. "
            "Longer waiting periods may indicate bottlenecks."
        )

    if "temperature" in feature_lower:

        return (
            "Temperature is contributing to the model's decision "
            "and may represent weather-related operational impact."
        )

    if "humidity" in feature_lower:

        return (
            "Humidity is contributing to delay risk and may "
            "represent environmental operating conditions."
        )

    if "demand" in feature_lower:

        return (
            "Demand conditions are influencing delay probability. "
            "Higher demand may increase operational pressure."
        )

    if "stress" in feature_lower:

        return (
            "Operational stress is influencing the prediction. "
            "Higher stress indicates greater potential for "
            "delivery disruption."
        )

    if "fleet_load" in feature_lower:

        return (
            "Fleet load is influencing delay probability. "
            "High fleet load may indicate limited spare capacity."
        )

    if "distance" in feature_lower:

        return (
            "Distance from the operational center is influencing "
            "delay risk and may affect travel and dispatch time."
        )

    if "customer" in feature_lower:

        return (
            "Customer value or customer behavior is influencing "
            "the model's prediction."
        )

    return (
        "This variable contributes to the model's delay-risk "
        "prediction."
    )


for feature in top_features:

    st.markdown(
        f"**{feature}** — {interpret_feature(feature)}"
    )


# ============================================================
# 32. INDIVIDUAL PREDICTION EXPLANATION
# ============================================================

st.header("🎯 Individual Shipment Explanation")

st.write(
    """
    Select an individual shipment to understand why the model
    classified it as a potential delay or no-delay case.
    """
)


# ------------------------------------------------------------
# Select row
# ------------------------------------------------------------

row_number = st.number_input(
    "Select shipment row",
    min_value=0,
    max_value=len(df) - 1,
    value=0,
    step=1
)


selected_row = df.iloc[
    int(row_number)
]


# ============================================================
# 33. PREPARE SELECTED ROW
# ============================================================

# Remove target column if present.
selected_input = pd.DataFrame(
    [selected_row]
).drop(
    columns=["Logistics_Delay"],
    errors="ignore"
)


# ============================================================
# 34. MODEL PREDICTION
# ============================================================

try:

    probability = final_model.predict_proba(
        selected_input
    )[0, 1]

except Exception as e:

    st.error(
        f"""
        ❌ Unable to generate prediction.

        Error:

        `{e}`
        """
    )

    st.stop()


# Business threshold.
FINAL_THRESHOLD = 0.48


prediction = int(
    probability >= FINAL_THRESHOLD
)


# ============================================================
# 35. DISPLAY PREDICTION
# ============================================================

col1, col2, col3 = st.columns(3)


with col1:

    st.metric(
        "Delay Probability",
        f"{probability:.1%}"
    )


with col2:

    st.metric(
        "Decision Threshold",
        f"{FINAL_THRESHOLD:.2f}"
    )


with col3:

    if prediction == 1:

        st.error(
            "⚠️ Potential Delay"
        )

    else:

        st.success(
            "✓ No Delay"
        )


# ============================================================
# 36. TRANSFORM SELECTED ROW
# ============================================================

selected_transformed = preprocessor.transform(
    selected_input
)


if hasattr(
    selected_transformed,
    "toarray"
):

    selected_transformed = (
        selected_transformed.toarray()
    )


selected_transformed = np.asarray(
    selected_transformed,
    dtype=float
)


selected_transformed_df = pd.DataFrame(
    selected_transformed,
    columns=feature_names
)


# ============================================================
# 37. CALCULATE LOCAL SHAP VALUES
# ============================================================

try:

    local_raw = explainer.shap_values(
        selected_transformed_df
    )

    local_shap = extract_delay_shap_values(
        local_raw,
        len(feature_names)
    )[0]

except Exception as e:

    st.error(
        f"""
        ❌ Unable to calculate local SHAP explanation.

        Error:

        `{e}`
        """
    )

    st.stop()


# ============================================================
# 38. CREATE LOCAL SHAP DATAFRAME
# ============================================================

local_explanation = pd.DataFrame(
    {
        "Feature": feature_names,
        "SHAP_Value": local_shap,
        "Feature_Value": selected_transformed_df.iloc[0].values
    }
)


# ============================================================
# 39. ADD DIRECTION
# ============================================================

local_explanation["Direction"] = np.where(
    local_explanation["SHAP_Value"] >= 0,
    "Increases Delay Risk",
    "Reduces Delay Risk"
)


# ============================================================
# 40. SORT BY ABSOLUTE IMPACT
# ============================================================

local_explanation[
    "Absolute_SHAP"
] = np.abs(
    local_explanation["SHAP_Value"]
)


local_explanation = local_explanation.sort_values(
    "Absolute_SHAP",
    ascending=False
).reset_index(drop=True)


# ============================================================
# 41. DISPLAY TOP LOCAL DRIVERS
# ============================================================

st.subheader(
    "Top Factors Affecting This Prediction"
)


TOP_LOCAL = 15


local_top = local_explanation.head(
    TOP_LOCAL
)


# ============================================================
# 42. LOCAL SHAP BAR CHART
# ============================================================

plot_local = local_top.sort_values(
    "SHAP_Value"
)


fig_local = px.bar(
    plot_local,
    x="SHAP_Value",
    y="Feature",
    orientation="h",
    title="Individual Prediction Drivers",
    labels={
        "SHAP_Value": "SHAP Value",
        "Feature": "Feature"
    }
)


fig_local.add_vline(
    x=0,
    line_width=1
)


fig_local.update_layout(
    height=650
)


st.plotly_chart(
    fig_local,
    use_container_width=True
)


# ============================================================
# 43. LOCAL EXPLANATION TABLE
# ============================================================

st.dataframe(
    local_top[
        [
            "Feature",
            "SHAP_Value",
            "Direction"
        ]
    ],
    use_container_width=True,
    hide_index=True
)


# ============================================================
# 44. BUSINESS MESSAGE FOR SELECTED SHIPMENT
# ============================================================

st.subheader(
    "🚚 Operational Recommendation"
)


positive_features = local_explanation[
    local_explanation["SHAP_Value"] > 0
].head(5)


negative_features = local_explanation[
    local_explanation["SHAP_Value"] < 0
].head(5)


if prediction == 1:

    st.warning(
        """
        **Potential delay detected.**

        The model identifies this shipment as a potential delay
        because the predicted probability is above the 0.48
        operational threshold.

        Management should review the strongest positive SHAP
        contributors and determine whether operational
        intervention is required.
        """
    )

else:

    st.success(
        """
        **No potential delay detected.**

        The model's predicted probability is below the 0.48
        operational threshold.

        This does not guarantee on-time delivery; it indicates
        that the model currently estimates the delay probability
        below the selected intervention threshold.
        """
    )


# ============================================================
# 45. TOP POSITIVE CONTRIBUTORS
# ============================================================

if len(positive_features) > 0:

    st.markdown(
        "### Factors Increasing Delay Risk"
    )

    for _, row in positive_features.iterrows():

        st.write(
            f"🔴 **{row['Feature']}** "
            f"(SHAP: {row['SHAP_Value']:.4f})"
        )


# ============================================================
# 46. TOP NEGATIVE CONTRIBUTORS
# ============================================================

if len(negative_features) > 0:

    st.markdown(
        "### Factors Reducing Delay Risk"
    )

    for _, row in negative_features.iterrows():

        st.write(
            f"🟢 **{row['Feature']}** "
            f"(SHAP: {row['SHAP_Value']:.4f})"
        )


# ============================================================
# 47. RAW SHIPMENT INFORMATION
# ============================================================

st.header(
    "📋 Shipment Information"
)


# Display the original business variables.
#
# We do NOT show the 79 transformed model variables here because
# management users should primarily see the original operational
# information.

display_columns = [
    "Timestamp",
    "Asset_ID",
    "Latitude",
    "Longitude",
    "Inventory_Level",
    "Shipment_Status",
    "Temperature",
    "Humidity",
    "Traffic_Status",
    "Waiting_Time",
    "User_Transaction_Amount",
    "User_Purchase_Frequency",
    "Logistics_Delay_Reason",
    "Asset_Utilization",
    "Demand_Forecast",
    "Operational_Stress_Score",
    "Inventory_Demand_Gap",
    "Fleet_Load_Index",
    "Customer_Value_Index",
    "Distance_From_Center"
]


available_display_columns = [
    col
    for col in display_columns
    if col in selected_row.index
]


shipment_information = pd.DataFrame(
    {
        "Variable": available_display_columns,
        "Value": [
            selected_row[col]
            for col in available_display_columns
        ]
    }
)


st.dataframe(
    shipment_information,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# 48. MANAGEMENT SUMMARY
# ============================================================

st.header(
    "📌 Management Summary"
)


st.markdown(
    f"""
    ### What management should know

    **1. Model**
    
    The deployed predictive model is a **Random Forest classifier**.

    **2. Business objective**
    
    The model is configured to prioritize **early detection of
    potential logistics delays**.

    **3. Decision threshold**
    
    The operational threshold is **0.48** rather than the default
    0.50. This makes the model more sensitive to potential delays.

    **4. Delay probability**
    
    For the selected shipment, the estimated delay probability is
    **{probability:.1%}**.

    **5. Model decision**
    
    The current shipment is classified as:

    **{"Potential Delay" if prediction == 1 else "No Delay"}**

    **6. Explainability**
    
    SHAP identifies the operational factors that most strongly
    influenced the model decision.

    **7. Operational use**
    
    The strongest positive SHAP contributors can be investigated
    by operations teams for possible intervention.
    """
)


# ============================================================
# 49. DISCLAIMER / MODEL GOVERNANCE
# ============================================================

with st.expander(
    "ℹ️ Model Governance & Interpretation Notes"
):

    st.markdown(
        """
        ### Important interpretation notes

        SHAP explains **model behavior**, not necessarily
        causality.

        A feature with a large SHAP value means that the feature
        strongly influenced the model prediction. It does not
        automatically mean that changing that variable will cause
        the delivery outcome to change.

        The model should therefore be used as a **decision-support
        system**, with operational teams validating important
        alerts before taking action.

        The 0.48 threshold reflects the project's recall-first
        business objective and minimum precision constraint.
        """
    )


# ============================================================
# 50. FOOTER
# ============================================================

st.markdown("---")

st.caption(
    """
    Logistics Delay Prediction & Explainability Dashboard |
    Random Forest | Threshold = 0.48 | SHAP Explainability
    """
)
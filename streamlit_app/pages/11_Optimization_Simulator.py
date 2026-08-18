# ============================================================
# 9_Optimization_Simulator.py
# ============================================================
#
# LOGISTICS DELAY PREDICTION
# OPTIMIZATION / WHAT-IF SIMULATOR
#
# Champion Model:
#     Random Forest
#
# Business Objective:
#     Prioritize early detection of potential delays
#
# Final Threshold:
#     0.48
#
# Minimum Precision Constraint:
#     0.70
#
# ============================================================


# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import os
import warnings
import joblib

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

warnings.filterwarnings("ignore")


# ============================================================
# 2. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Optimization Simulator",
    page_icon="⚙️",
    layout="wide"
)


# ============================================================
# 3. PROJECT PATHS
# ============================================================

# Current file:
#
# Smart Logistic Dataset/
#     streamlit_app/
#         pages/
#             9_Optimization_Simulator.py


CURRENT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


# streamlit_app
BASE_DIR = os.path.dirname(
    CURRENT_DIR
)


# Project root
PROJECT_ROOT = os.path.dirname(
    BASE_DIR
)


# ============================================================
# 4. ASSET PATHS
# ============================================================

ASSETS_DIR = os.path.join(
    BASE_DIR,
    "streamlit_assets"
)


MODEL_PATH = os.path.join(
    ASSETS_DIR,
    "final_model.pkl"
)


FEATURE_NAMES_PATH = os.path.join(
    ASSETS_DIR,
    "feature_names.pkl"
)


# ============================================================
# 5. DATASET PATH
# ============================================================

DATA_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "logistics_feature_engineered.csv"
)


# ============================================================
# 6. BUSINESS CONFIGURATION
# ============================================================

CHAMPION_MODEL = "Random Forest"

FINAL_THRESHOLD = 0.48

MIN_PRECISION = 0.70


# ============================================================
# 7. CHECK REQUIRED FILES
# ============================================================

required_files = {
    "Final Model": MODEL_PATH,
    "Feature Names": FEATURE_NAMES_PATH,
    "Dataset": DATA_PATH
}


missing_files = []

for name, path in required_files.items():

    if not os.path.exists(path):

        missing_files.append(
            f"{name}: {path}"
        )


if missing_files:

    st.error(
        "One or more required files could not be found."
    )

    for item in missing_files:

        st.code(item)

    st.stop()


# ============================================================
# 8. LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    return joblib.load(
        MODEL_PATH
    )


# ============================================================
# 9. LOAD FEATURE NAMES
# ============================================================

@st.cache_resource
def load_feature_names():

    return joblib.load(
        FEATURE_NAMES_PATH
    )


# ============================================================
# 10. LOAD DATASET
# ============================================================

@st.cache_data
def load_data():

    return pd.read_csv(
        DATA_PATH
    )


# ============================================================
# 11. LOAD OBJECTS
# ============================================================

try:

    model = load_model()

    feature_names = load_feature_names()

    df = load_data()

except Exception as e:

    st.error(
        f"Unable to load model/data: {e}"
    )

    st.stop()


# ============================================================
# 12. GET FITTED PREPROCESSOR
# ============================================================
#
# IMPORTANT
# ---------
# The model is a Pipeline.
#
# Example:
#
# Pipeline(
#     steps=[
#         ("preprocessor", ColumnTransformer(...)),
#         ("classifier", RandomForestClassifier(...))
#     ]
# )
#
# We retrieve the fitted preprocessor so that we know exactly
# which ORIGINAL columns the model expects.
#
# ============================================================

preprocessor = None

expected_columns = []


if hasattr(model, "named_steps"):

    if "preprocessor" in model.named_steps:

        preprocessor = (
            model.named_steps[
                "preprocessor"
            ]
        )


if preprocessor is not None:

    try:

        expected_columns = list(
            preprocessor.feature_names_in_
        )

    except Exception:

        expected_columns = []


# ============================================================
# 13. FALLBACK TO DATASET COLUMNS
# ============================================================

if not expected_columns:

    expected_columns = list(
        df.columns
    )


# ============================================================
# 14. IDENTIFY NUMERIC / CATEGORICAL FEATURES
# ============================================================

numeric_columns = []

categorical_columns = []


for column in expected_columns:

    if column in df.columns:

        if pd.api.types.is_numeric_dtype(
            df[column]
        ):

            numeric_columns.append(
                column
            )

        else:

            categorical_columns.append(
                column
            )

    else:

        # If the column doesn't exist in the dataset,
        # we cannot infer its type from the data.
        #
        # Use feature-name conventions as a fallback.

        if column in [
            "Hour",
            "Day",
            "Day_of_Week",
            "Month",
            "Week_of_Year",
            "Is_Weekend"
        ]:

            numeric_columns.append(
                column
            )

        else:

            categorical_columns.append(
                column
            )


# ============================================================
# 15. HELPER: GET NUMERIC DEFAULT
# ============================================================

def get_numeric_default(
    column,
    fallback=0.0
):

    if column not in df.columns:

        return fallback

    series = df[column]

    numeric_series = pd.to_numeric(
        series,
        errors="coerce"
    )

    if numeric_series.notna().sum() == 0:

        return fallback

    return float(
        numeric_series.median()
    )


# ============================================================
# 16. HELPER: GET CATEGORICAL DEFAULT
# ============================================================

def get_categorical_default(
    column,
    fallback="Unknown"
):

    if column not in df.columns:

        return fallback

    series = (
        df[column]
        .dropna()
        .astype(str)
    )

    if len(series) == 0:

        return fallback

    return str(
        series.mode().iloc[0]
    )


# ============================================================
# 17. HELPER: GET DEFAULT VALUE
# ============================================================

def get_default_value(
    column
):

    if column in numeric_columns:

        return get_numeric_default(
            column
        )

    return get_categorical_default(
        column
    )


# ============================================================
# 18. PAGE HEADER
# ============================================================

st.title(
    "⚙️ Optimization Simulator"
)

st.markdown(
    """
    ### What happens if operational conditions change?

    Use this simulator to test different logistics scenarios
    and estimate the probability of a potential delivery delay.
    """
)


# ============================================================
# 19. BUSINESS OBJECTIVE
# ============================================================

st.info(
    f"""
    **Champion Model:** {CHAMPION_MODEL}

    **Business Objective:** Prioritize early detection of
    potential logistics delays.

    **Selected Threshold:** {FINAL_THRESHOLD:.2f}

    **Minimum Precision Constraint:** {MIN_PRECISION:.2f}

    A lower threshold is intentionally used to increase delay
    detection and reduce missed delay cases.
    """
)


# ============================================================
# 20. MODEL / DATA INFORMATION
# ============================================================

with st.expander(
    "📂 Model & Dataset Information"
):

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Dataset Rows",
            f"{len(df):,}"
        )

    with col2:

        st.metric(
            "Dataset Columns",
            f"{df.shape[1]:,}"
        )

    with col3:

        st.metric(
            "Model Features",
            len(expected_columns)
        )

    with col4:

        st.metric(
            "Threshold",
            f"{FINAL_THRESHOLD:.2f}"
        )


# ============================================================
# 21. SIDEBAR
# ============================================================

st.sidebar.header(
    "🎛️ Scenario Controls"
)

st.sidebar.caption(
    "Adjust operational variables and evaluate potential delay risk."
)


# ============================================================
# 22. INVENTORY LEVEL
# ============================================================

inventory_level = st.sidebar.number_input(
    "Inventory Level",
    min_value=0.0,
    value=get_numeric_default(
        "Inventory_Level",
        100
    ),
    step=1.0
)


# ============================================================
# 23. DEMAND FORECAST
# ============================================================

demand_forecast = st.sidebar.number_input(
    "Demand Forecast",
    min_value=0.0,
    value=get_numeric_default(
        "Demand_Forecast",
        100
    ),
    step=1.0
)


# ============================================================
# 24. ASSET UTILIZATION
# ============================================================

utilization_default = get_numeric_default(
    "Asset_Utilization",
    0.50
)


# If dataset stores utilization as percentages
# instead of 0-1 values, convert for slider.

if utilization_default > 1:

    utilization_default = (
        utilization_default / 100
    )


asset_utilization = st.sidebar.slider(
    "Asset Utilization",
    min_value=0.0,
    max_value=1.0,
    value=float(
        np.clip(
            utilization_default,
            0,
            1
        )
    ),
    step=0.01
)


# ============================================================
# 25. WAITING TIME
# ============================================================

waiting_time = st.sidebar.number_input(
    "Waiting Time",
    min_value=0.0,
    value=get_numeric_default(
        "Waiting_Time",
        10
    ),
    step=1.0
)


# ============================================================
# 26. TEMPERATURE
# ============================================================

temperature = st.sidebar.number_input(
    "Temperature",
    value=get_numeric_default(
        "Temperature",
        20
    ),
    step=0.5
)


# ============================================================
# 27. HUMIDITY
# ============================================================

humidity = st.sidebar.slider(
    "Humidity (%)",
    min_value=0.0,
    max_value=100.0,
    value=float(
        np.clip(
            get_numeric_default(
                "Humidity",
                50
            ),
            0,
            100
        )
    ),
    step=1.0
)


# ============================================================
# 28. TRANSACTION AMOUNT
# ============================================================

transaction_amount = st.sidebar.number_input(
    "User Transaction Amount",
    min_value=0.0,
    value=get_numeric_default(
        "User_Transaction_Amount",
        100
    ),
    step=1.0
)


# ============================================================
# 29. PURCHASE FREQUENCY
# ============================================================

purchase_frequency = st.sidebar.number_input(
    "User Purchase Frequency",
    min_value=0.0,
    value=get_numeric_default(
        "User_Purchase_Frequency",
        5
    ),
    step=1.0
)


# ============================================================
# 30. DISTANCE FROM CENTER
# ============================================================

distance_from_center = st.sidebar.number_input(
    "Distance From Center",
    min_value=0.0,
    value=get_numeric_default(
        "Distance_From_Center",
        10
    ),
    step=0.5
)


# ============================================================
# 31. TRAFFIC STATUS
# ============================================================

if "Traffic_Status" in df.columns:

    traffic_options = sorted(
        df["Traffic_Status"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

else:

    traffic_options = [
        "Clear",
        "Heavy",
        "Detour"
    ]


traffic_status = st.sidebar.selectbox(
    "Traffic Status",
    traffic_options
)


# ============================================================
# 32. SHIPMENT STATUS
# ============================================================

if "Shipment_Status" in df.columns:

    shipment_options = sorted(
        df["Shipment_Status"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

else:

    shipment_options = [
        "In Transit",
        "Pending",
        "Delivered"
    ]


shipment_status = st.sidebar.selectbox(
    "Shipment Status",
    shipment_options
)


# ============================================================
# 33. DELAY REASON
# ============================================================

if "Logistics_Delay_Reason" in df.columns:

    delay_reason_options = sorted(
        df["Logistics_Delay_Reason"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

else:

    delay_reason_options = []


if "None" not in delay_reason_options:

    delay_reason_options.insert(
        0,
        "None"
    )


delay_reason = st.sidebar.selectbox(
    "Logistics Delay Reason",
    delay_reason_options
)


# ============================================================
# 34. TIME VARIABLES
# ============================================================

st.sidebar.markdown("---")

st.sidebar.subheader(
    "🕒 Time Conditions"
)


hour = st.sidebar.slider(
    "Hour",
    0,
    23,
    12
)


day = st.sidebar.slider(
    "Day",
    1,
    31,
    15
)


day_of_week = st.sidebar.slider(
    "Day of Week",
    0,
    6,
    2
)


month = st.sidebar.slider(
    "Month",
    1,
    12,
    6
)


# ============================================================
# 35. BUILD SCENARIO
# ============================================================

scenario = {}


scenario["Inventory_Level"] = (
    inventory_level
)

scenario["Demand_Forecast"] = (
    demand_forecast
)

scenario["Asset_Utilization"] = (
    asset_utilization
)

scenario["Waiting_Time"] = (
    waiting_time
)

scenario["Temperature"] = (
    temperature
)

scenario["Humidity"] = (
    humidity
)

scenario["User_Transaction_Amount"] = (
    transaction_amount
)

scenario["User_Purchase_Frequency"] = (
    purchase_frequency
)

scenario["Distance_From_Center"] = (
    distance_from_center
)

scenario["Traffic_Status"] = (
    traffic_status
)

scenario["Shipment_Status"] = (
    shipment_status
)

scenario["Logistics_Delay_Reason"] = (
    delay_reason
)

scenario["Hour"] = hour

scenario["Day"] = day

scenario["Day_of_Week"] = day_of_week

scenario["Month"] = month

scenario["Is_Weekend"] = int(
    day_of_week >= 5
)


# ============================================================
# 36. DERIVED FEATURES
# ============================================================

# Month name

scenario["Month_Name"] = pd.Timestamp(
    2025,
    month,
    1
).month_name()


# ============================================================
# 37. UTILIZATION BAND
# ============================================================

if asset_utilization < 0.40:

    scenario["Utilization_Band"] = "Low"

elif asset_utilization < 0.70:

    scenario["Utilization_Band"] = "Medium"

else:

    scenario["Utilization_Band"] = "High"


# ============================================================
# 38. INVENTORY BAND
# ============================================================

inventory_q33 = (
    df["Inventory_Level"].quantile(0.33)
    if "Inventory_Level" in df.columns
    else 50
)

inventory_q66 = (
    df["Inventory_Level"].quantile(0.66)
    if "Inventory_Level" in df.columns
    else 150
)


if inventory_level <= inventory_q33:

    scenario["Inventory_Band"] = "Low"

elif inventory_level <= inventory_q66:

    scenario["Inventory_Band"] = "Medium"

else:

    scenario["Inventory_Band"] = "High"


# ============================================================
# 39. PURCHASE FREQUENCY BAND
# ============================================================

frequency_q33 = (
    df["User_Purchase_Frequency"].quantile(0.33)
    if "User_Purchase_Frequency" in df.columns
    else 3
)

frequency_q66 = (
    df["User_Purchase_Frequency"].quantile(0.66)
    if "User_Purchase_Frequency" in df.columns
    else 7
)


if purchase_frequency <= frequency_q33:

    scenario[
        "Purchase_Frequency_Band"
    ] = "Low"

elif purchase_frequency <= frequency_q66:

    scenario[
        "Purchase_Frequency_Band"
    ] = "Medium"

else:

    scenario[
        "Purchase_Frequency_Band"
    ] = "High"


# ============================================================
# 40. NORMALIZED FEATURES
# ============================================================

def safe_zscore(
    value,
    column
):

    if column not in df.columns:

        return 0.0

    numeric_series = pd.to_numeric(
        df[column],
        errors="coerce"
    )

    mean = numeric_series.mean()

    std = numeric_series.std()

    if (
        pd.isna(std)
        or std == 0
    ):

        return 0.0

    return float(
        (value - mean) / std
    )


scenario["Demand_Normalized"] = safe_zscore(
    demand_forecast,
    "Demand_Forecast"
)


scenario[
    "Utilization_Normalized"
] = safe_zscore(
    asset_utilization,
    "Asset_Utilization"
)


scenario[
    "Waiting_Normalized"
] = safe_zscore(
    waiting_time,
    "Waiting_Time"
)


# ============================================================
# 41. INVENTORY COVERAGE
# ============================================================

if demand_forecast > 0:

    scenario[
        "Inventory_Coverage"
    ] = (
        inventory_level /
        demand_forecast
    )

else:

    scenario[
        "Inventory_Coverage"
    ] = 0.0


# ============================================================
# 42. INVENTORY DEMAND GAP
# ============================================================

scenario[
    "Inventory_Demand_Gap"
] = (
    inventory_level -
    demand_forecast
)


# ============================================================
# 43. STOCK RISK
# ============================================================

if inventory_level < demand_forecast:

    scenario["Stock_Risk"] = "High"

else:

    scenario["Stock_Risk"] = "Low"


# ============================================================
# 44. TRAFFIC LEVEL
# ============================================================

traffic_text = str(
    traffic_status
).lower()


if (
    "heavy" in traffic_text
    or
    "detour" in traffic_text
):

    scenario["Traffic_Level"] = "High"

else:

    scenario["Traffic_Level"] = "Low"


# ============================================================
# 45. TRAFFIC INTERACTION
# ============================================================

traffic_flag = int(
    scenario["Traffic_Level"] == "High"
)


scenario[
    "Traffic_Utilization_Interaction"
] = (
    asset_utilization *
    traffic_flag
)


scenario[
    "Demand_Traffic_Interaction"
] = (
    demand_forecast *
    traffic_flag
)


scenario[
    "Utilization_Waiting_Interaction"
] = (
    asset_utilization *
    waiting_time
)


# ============================================================
# 46. FLEET LOAD INDEX
# ============================================================

scenario[
    "Fleet_Load_Index"
] = (
    asset_utilization *
    demand_forecast
)


scenario[
    "Fleet_Load_Index_Normalized"
] = safe_zscore(
    scenario["Fleet_Load_Index"],
    "Fleet_Load_Index"
)


# ============================================================
# 47. SCORES
# ============================================================

scenario["Demand_Score"] = (
    scenario["Demand_Normalized"]
)


scenario["Utilization_Score"] = (
    scenario["Utilization_Normalized"]
)


scenario["Waiting_Score"] = (
    scenario["Waiting_Normalized"]
)


# ============================================================
# 48. CUSTOMER VALUE
# ============================================================

scenario[
    "Customer_Value_Index"
] = (
    transaction_amount *
    purchase_frequency
)


if (
    scenario["Customer_Value_Index"]
    <= df[
        "Customer_Value_Index"
    ].quantile(0.33)
    if "Customer_Value_Index" in df.columns
    else 100
):

    scenario[
        "Customer_Value_Segment"
    ] = "Low"

elif (
    scenario["Customer_Value_Index"]
    <= df[
        "Customer_Value_Index"
    ].quantile(0.66)
    if "Customer_Value_Index" in df.columns
    else 500
):

    scenario[
        "Customer_Value_Segment"
    ] = "Medium"

else:

    scenario[
        "Customer_Value_Segment"
    ] = "High"


# ============================================================
# 49. OPERATIONAL STRESS
# ============================================================

scenario[
    "Operational_Stress_Score"
] = (
    abs(
        scenario["Demand_Normalized"]
    )
    +
    abs(
        scenario["Waiting_Normalized"]
    )
    +
    abs(
        scenario["Utilization_Normalized"]
    )
)


if (
    scenario[
        "Operational_Stress_Score"
    ] < 1
):

    scenario[
        "Operational_Stress_Level"
    ] = "Low"

elif (
    scenario[
        "Operational_Stress_Score"
    ] < 2
):

    scenario[
        "Operational_Stress_Level"
    ] = "Medium"

else:

    scenario[
        "Operational_Stress_Level"
    ] = "High"


# ============================================================
# 50. PRE-DISPATCH STRESS
# ============================================================

scenario[
    "Pre_Dispatch_Stress_Score"
] = (
    scenario[
        "Operational_Stress_Score"
    ]
    +
    traffic_flag
)


# ============================================================
# 51. WEEK OF YEAR
# ============================================================

try:

    scenario["Week_of_Year"] = int(
        pd.Timestamp(
            2025,
            month,
            min(day, 28)
        ).isocalendar().week
    )

except Exception:

    scenario["Week_of_Year"] = 1


# ============================================================
# 52. TIME PERIOD
# ============================================================

if hour < 6:

    scenario["Time_Period"] = "Night"

elif hour < 12:

    scenario["Time_Period"] = "Morning"

elif hour < 18:

    scenario["Time_Period"] = "Afternoon"

else:

    scenario["Time_Period"] = "Evening"


# ============================================================
# 53. CREATE RAW SCENARIO DATAFRAME
# ============================================================

scenario_df = pd.DataFrame(
    [scenario]
)


# ============================================================
# 54. ALIGN EXACTLY TO MODEL INPUT
# ============================================================
#
# THIS IS THE IMPORTANT FIX.
#
# We do NOT use feature_names.pkl here because that file
# contains the transformed features.
#
# The Pipeline expects ORIGINAL features.
#
# ============================================================

def prepare_model_input(
    raw_scenario
):

    model_input = raw_scenario.copy()


    # Add every missing model feature.

    for column in expected_columns:

        if column not in model_input.columns:

            model_input[column] = (
                get_default_value(
                    column
                )
            )


    # Remove columns not expected by model.

    model_input = model_input[
        expected_columns
    ].copy()


    # ========================================================
    # FORCE CORRECT DATA TYPES
    # ========================================================

    # Numeric columns MUST be numeric.

    for column in numeric_columns:

        if column in model_input.columns:

            model_input[column] = pd.to_numeric(
                model_input[column],
                errors="coerce"
            )


            # If conversion generated NaN,
            # replace with numeric median.

            if model_input[
                column
            ].isna().any():

                model_input[
                    column
                ] = model_input[
                    column
                ].fillna(
                    get_numeric_default(
                        column
                    )
                )


    # Categorical columns MUST be strings.

    for column in categorical_columns:

        if column in model_input.columns:

            model_input[column] = (
                model_input[column]
                .astype(str)
            )


    return model_input


# ============================================================
# 55. PREPARE FINAL INPUT
# ============================================================

model_input = prepare_model_input(
    scenario_df
)


# ============================================================
# 56. DEBUG INFORMATION
# ============================================================

with st.expander(
    "🔍 Technical Input Validation"
):

    st.write(
        "Expected model input columns:"
    )

    st.write(
        len(expected_columns)
    )

    st.write(
        "Numeric columns:"
    )

    st.write(
        len(numeric_columns)
    )

    st.write(
        "Categorical columns:"
    )

    st.write(
        len(categorical_columns)
    )

    st.write(
        "Final input shape:"
    )

    st.write(
        model_input.shape
    )


# ============================================================
# 57. PREDICTION
# ============================================================

try:

    y_probability = model.predict_proba(
        model_input
    )[0, 1]

except Exception as e:

    st.error(
        "Prediction could not be generated."
    )

    st.code(
        str(e)
    )

    st.write(
        "Model input data:"
    )

    st.dataframe(
        model_input,
        use_container_width=True
    )

    st.stop()


# ============================================================
# 58. CLASSIFICATION
# ============================================================

predicted_class = int(
    y_probability >= FINAL_THRESHOLD
)


# ============================================================
# 59. RISK LEVEL
# ============================================================

if y_probability < 0.30:

    risk_level = "Low Risk"

elif y_probability < FINAL_THRESHOLD:

    risk_level = "Moderate Risk"

elif y_probability < 0.70:

    risk_level = "High Risk"

else:

    risk_level = "Critical Risk"


# ============================================================
# 60. KPI CARDS
# ============================================================

st.markdown("---")

st.subheader(
    "📊 Scenario Prediction"
)


col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "Delay Probability",
        f"{y_probability:.1%}"
    )


with col2:

    st.metric(
        "Threshold",
        f"{FINAL_THRESHOLD:.2f}"
    )


with col3:

    st.metric(
        "Risk Level",
        risk_level
    )


with col4:

    if predicted_class == 1:

        st.metric(
            "Prediction",
            "⚠️ Potential Delay"
        )

    else:

        st.metric(
            "Prediction",
            "✅ No Delay"
        )


# ============================================================
# 61. BUSINESS INTERPRETATION
# ============================================================

if predicted_class == 1:

    st.error(
        f"""
        ### ⚠️ Potential Delay Detected

        Estimated delay probability:

        **{y_probability:.1%}**

        This is above the selected business threshold of
        **{FINAL_THRESHOLD:.2f}**.

        **Recommended action:** initiate proactive operational
        review.
        """
    )

else:

    st.success(
        f"""
        ### ✅ No Delay Signal Detected

        Estimated delay probability:

        **{y_probability:.1%}**

        This is below the selected business threshold of
        **{FINAL_THRESHOLD:.2f}**.

        Continue normal operational monitoring.
        """
    )


# ============================================================
# 62. PROBABILITY GAUGE
# ============================================================

st.subheader(
    "🎯 Delay Probability"
)


gauge = go.Figure(
    go.Indicator(
        mode="gauge+number",
        value=y_probability * 100,
        number={
            "suffix": "%"
        },
        title={
            "text": "Potential Delay Probability"
        },
        gauge={
            "axis": {
                "range": [0, 100]
            },
            "threshold": {
                "line": {
                    "width": 4
                },
                "value":
                    FINAL_THRESHOLD * 100
            }
        }
    )
)


gauge.update_layout(
    height=350
)


st.plotly_chart(
    gauge,
    use_container_width=True
)


# ============================================================
# 63. CURRENT SCENARIO SUMMARY
# ============================================================

st.markdown("---")

st.subheader(
    "📋 Current Scenario"
)


display_variables = [
    "Inventory_Level",
    "Demand_Forecast",
    "Asset_Utilization",
    "Waiting_Time",
    "Temperature",
    "Humidity",
    "Traffic_Status",
    "Shipment_Status",
    "User_Transaction_Amount",
    "User_Purchase_Frequency",
    "Distance_From_Center",
    "Operational_Stress_Level",
    "Stock_Risk"
]


scenario_display = []


for column in display_variables:

    if column in scenario:

        value = scenario[column]

        if (
            column ==
            "Asset_Utilization"
        ):

            value = (
                f"{value:.1%}"
            )

        scenario_display.append(
            {
                "Variable":
                    column,
                "Value":
                    value
            }
        )


st.dataframe(
    pd.DataFrame(
        scenario_display
    ),
    use_container_width=True,
    hide_index=True
)


# ============================================================
# 64. WHAT-IF FUNCTION
# ============================================================

def predict_probability(
    scenario_dictionary
):

    temp_df = pd.DataFrame(
        [scenario_dictionary]
    )

    temp_df = prepare_model_input(
        temp_df
    )

    return float(
        model.predict_proba(
            temp_df
        )[0, 1]
    )


# ============================================================
# 65. ALTERNATIVE SCENARIOS
# ============================================================

alternative_scenarios = []


# Current scenario

alternative_scenarios.append(
    {
        "Scenario": "Current",
        "Data": scenario.copy()
    }
)


# ============================================================
# CLEAR TRAFFIC SCENARIO
# ============================================================

clear_traffic = scenario.copy()

if "Traffic_Status" in df.columns:

    clear_traffic[
        "Traffic_Status"
    ] = "Clear"


clear_traffic[
    "Traffic_Level"
] = "Low"


clear_traffic[
    "Traffic_Utilization_Interaction"
] = 0


clear_traffic[
    "Demand_Traffic_Interaction"
] = 0


alternative_scenarios.append(
    {
        "Scenario":
            "Clear Traffic",
        "Data":
            clear_traffic
    }
)


# ============================================================
# REDUCED WAITING SCENARIO
# ============================================================

reduced_waiting = scenario.copy()

reduced_waiting[
    "Waiting_Time"
] = waiting_time * 0.50


reduced_waiting[
    "Waiting_Normalized"
] = safe_zscore(
    reduced_waiting[
        "Waiting_Time"
    ],
    "Waiting_Time"
)


reduced_waiting[
    "Waiting_Score"
] = reduced_waiting[
    "Waiting_Normalized"
]


reduced_waiting[
    "Utilization_Waiting_Interaction"
] = (
    asset_utilization *
    reduced_waiting[
        "Waiting_Time"
    ]
)


alternative_scenarios.append(
    {
        "Scenario":
            "50% Lower Waiting Time",
        "Data":
            reduced_waiting
    }
)


# ============================================================
# HIGHER INVENTORY SCENARIO
# ============================================================

higher_inventory = scenario.copy()

higher_inventory[
    "Inventory_Level"
] = inventory_level * 1.25


higher_inventory[
    "Inventory_Demand_Gap"
] = (
    higher_inventory[
        "Inventory_Level"
    ]
    -
    demand_forecast
)


higher_inventory[
    "Inventory_Coverage"
] = (
    higher_inventory[
        "Inventory_Level"
    ]
    /
    demand_forecast
    if demand_forecast > 0
    else 0
)


higher_inventory[
    "Stock_Risk"
] = (
    "High"
    if higher_inventory[
        "Inventory_Level"
    ] < demand_forecast
    else "Low"
)


alternative_scenarios.append(
    {
        "Scenario":
            "25% Higher Inventory",
        "Data":
            higher_inventory
    }
)


# ============================================================
# 66. RUN WHAT-IF PREDICTIONS
# ============================================================

comparison_results = []


for item in alternative_scenarios:

    try:

        probability = predict_probability(
            item["Data"]
        )

        comparison_results.append(
            {
                "Scenario":
                    item["Scenario"],
                "Delay Probability":
                    probability
            }
        )

    except Exception:

        continue


# ============================================================
# 67. DISPLAY COMPARISON
# ============================================================

if comparison_results:

    comparison_df = pd.DataFrame(
        comparison_results
    )


    comparison_df[
        "Delay Probability %"
    ] = (
        comparison_df[
            "Delay Probability"
        ] * 100
    ).round(2)


    st.markdown("---")

    st.subheader(
        "🔄 What-If Scenario Comparison"
    )


    st.dataframe(
        comparison_df[
            [
                "Scenario",
                "Delay Probability %"
            ]
        ],
        use_container_width=True,
        hide_index=True
    )


    # ========================================================
    # 68. WHAT-IF CHART
    # ========================================================

    fig = px.bar(
        comparison_df,
        x="Scenario",
        y="Delay Probability %",
        text="Delay Probability %",
        title="Potential Delay Risk Under Alternative Scenarios"
    )


    fig.add_hline(
        y=FINAL_THRESHOLD * 100,
        line_dash="dash",
        annotation_text=(
            f"Threshold = "
            f"{FINAL_THRESHOLD:.2f}"
        )
    )


    fig.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside"
    )


    fig.update_layout(
        yaxis_title="Delay Probability (%)",
        xaxis_title="Scenario"
    )


    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# 69. OPERATIONAL RECOMMENDATIONS
# ============================================================

st.markdown("---")

st.subheader(
    "💡 Recommended Operational Actions"
)


recommendations = []


# Traffic

if (
    "heavy" in traffic_text
    or
    "detour" in traffic_text
):

    recommendations.append(
        "🚦 Traffic conditions are elevated. "
        "Consider alternate routing or proactive dispatch planning."
    )


# Inventory

if inventory_level < demand_forecast:

    recommendations.append(
        "📦 Inventory is below forecast demand. "
        "Consider replenishment or redistribution."
    )


# Waiting

waiting_median = get_numeric_default(
    "Waiting_Time",
    waiting_time
)


if waiting_time > waiting_median:

    recommendations.append(
        "⏱️ Waiting time is above the historical median. "
        "Investigate dispatch or processing bottlenecks."
    )


# Utilization

if asset_utilization > 0.80:

    recommendations.append(
        "🚚 Fleet utilization is high. "
        "Consider asset reallocation or additional capacity."
    )


# Demand

if "Demand_Forecast" in df.columns:

    demand_75 = df[
        "Demand_Forecast"
    ].quantile(0.75)

    if demand_forecast > demand_75:

        recommendations.append(
            "📈 Demand forecast is elevated. "
            "Prepare additional fleet and operational capacity."
        )


# Model prediction

if predicted_class == 1:

    recommendations.append(
        "⚠️ The model identifies potential delay risk. "
        "Prioritize proactive intervention."
    )


if not recommendations:

    recommendations.append(
        "✅ No major operational risk indicators were detected. "
        "Continue standard monitoring."
    )


for recommendation in recommendations:

    st.write(
        recommendation
    )


# ============================================================
# 70. BUSINESS TRADE-OFF
# ============================================================

st.markdown("---")

st.subheader(
    "⚖️ Business Trade-off"
)


col1, col2 = st.columns(2)


with col1:

    st.markdown(
        """
        ### Recall-First Strategy

        The threshold of **0.48** was selected to improve
        detection of potential delay cases.

        This reduces the number of missed delays.
        """
    )


with col2:

    st.markdown(
        """
        ### Operational Cost

        The trade-off is that more false alerts may be generated.

        Operations teams should therefore prioritize alerts
        according to business impact and available capacity.
        """
    )


# ============================================================
# 71. FINAL MODEL PERFORMANCE
# ============================================================

st.markdown("---")

st.subheader(
    "🏆 Final Model Performance"
)


performance = pd.DataFrame(
    {
        "Metric": [
            "Accuracy",
            "Precision",
            "Recall",
            "F1",
            "ROC-AUC",
            "PR-AUC"
        ],
        "Value": [
            0.7050,
            0.7207,
            0.7407,
            0.7306,
            0.7748,
            0.8630
        ]
    }
)


performance["Value"] = (
    performance["Value"]
    * 100
).round(2).astype(str) + "%"


st.dataframe(
    performance,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# 72. MANAGEMENT TAKEAWAY
# ============================================================

st.markdown("---")

st.subheader(
    "🎯 Management Takeaway"
)


if predicted_class == 1:

    st.warning(
        """
        The current scenario indicates elevated potential
        delay risk.

        Management should consider proactive intervention
        involving traffic management, inventory availability,
        fleet capacity and dispatch planning.

        The model is intentionally configured for early
        detection rather than maximum precision.
        """
    )

else:

    st.success(
        """
        The current scenario does not cross the delay-risk
        threshold.

        Normal operational monitoring can continue while
        maintaining readiness for changing traffic, demand,
        inventory and utilization conditions.
        """
    )


# ============================================================
# 73. FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Logistics Delay Prediction & Decision Support System | "
    "Random Forest | Recall-First Threshold = 0.48"
)
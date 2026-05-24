import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import numpy as np

# =============================
# PAGE CONFIG
# =============================

st.set_page_config(
    page_title="AI Solar Monitoring Dashboard",
    layout="wide"
)

st.title("⚡ AI Solar Energy Monitoring Dashboard")
st.markdown("### Predictive Fault Detection System")

# =============================
# FILE UPLOAD
# =============================

file = st.file_uploader(
    "📂 Upload CSV File",
    type=["csv"]
)

# =============================
# LOAD DATA
# =============================

if file:

    data = pd.read_csv(file)

    st.success("✅ Dataset Loaded Successfully")

    # =============================
    # CREATE SEQUENCE FEATURES
    # =============================

    sequence_cols = [
        "V_PV",
        "V_Batt",
        "Amp",
        "Power_W",
        "Temp_C",
        "UV_Actual"
    ]

    for col in sequence_cols:

        data[f"{col}_prev1"] = data[col].shift(1)
        data[f"{col}_prev2"] = data[col].shift(2)

        data[f"{col}_diff1"] = (
            data[col] - data[f"{col}_prev1"]
        )

        data[f"{col}_avg3"] = (
            data[col].rolling(3).mean()
        )

    data = data.dropna()

    # =============================
    # SIDEBAR
    # =============================

    st.sidebar.title("⚙️ Dashboard Filters")

    if "Label" in data.columns:

        fault_filter = st.sidebar.multiselect(
            "Select Fault Type",
            options=data["Label"].unique(),
            default=data["Label"].unique()
        )

        data = data[
            data["Label"].isin(fault_filter)
        ]

    # =============================
    # KPIs
    # =============================

    st.subheader("📌 System KPIs")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "⚡ Avg Power",
        f"{data['Power_W'].mean():.2f} W"
    )

    col2.metric(
        "🔋 Avg Battery",
        f"{data['V_Batt'].mean():.2f} V"
    )

    col3.metric(
        "🌡️ Avg Temp",
        f"{data['Temp_C'].mean():.2f} °C"
    )

    col4.metric(
        "🧹 Avg Dust",
        f"{data['Dust_Ratio'].mean():.2f} %"
    )

    # =============================
    # POWER ANALYSIS
    # =============================

    st.subheader("📈 Power Analysis")

    fig1 = px.line(
        data,
        y=["Power_W", "Ideal_Power_W"],
        title="Actual vs Ideal Power"
    )

    st.plotly_chart(
        fig1,
        use_container_width=True
    )

    # =============================
    # FAULT DISTRIBUTION
    # =============================

    st.subheader("🚨 Fault Distribution")

    fig2 = px.pie(
        data,
        names="Label",
        title="Detected Fault Types"
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

    # =============================
    # TEMP VS POWER
    # =============================

    st.subheader("🌡️ Temperature vs Power")

    fig3 = px.scatter(
        data,
        x="Temp_C",
        y="Power_W",
        color="Label",
        title="Temperature Effect on Power"
    )

    st.plotly_chart(
        fig3,
        use_container_width=True
    )

    # =============================
    # UV ANALYSIS
    # =============================

    st.subheader("☀️ UV Analysis")

    fig4 = px.line(
        data,
        y=["UV_Ideal", "UV_Actual"],
        title="UV Ideal vs Actual"
    )

    st.plotly_chart(
        fig4,
        use_container_width=True
    )

    # =============================
    # BATTERY STATUS
    # =============================

    if "Battery_Alert" in data.columns:

        st.subheader("🔋 Battery Status")

        fig5 = px.histogram(
            data,
            x="Battery_Alert",
            color="Battery_Alert"
        )

        st.plotly_chart(
            fig5,
            use_container_width=True
        )

    # =============================
    # SEQUENCE ANALYSIS
    # =============================

    st.subheader("📉 Sequential Trend Analysis")

    fig6 = px.line(
        data,
        y=[
            "Power_W",
            "Power_W_avg3"
        ],
        title="Power Trend & Moving Average"
    )

    st.plotly_chart(
        fig6,
        use_container_width=True
    )

    # =============================
    # PREDICTION SYSTEM
    # =============================

    st.subheader("🤖 AI Fault Prediction")

    try:

        model = joblib.load("rf_predictor.pkl")

        st.success("✅ AI Model Loaded")

        input_data = {}

        features = [

            # Original Features
            "Latitude",
            "V_PV",
            "V_Batt",
            "Amp",
            "Power_W",
            "Ideal_Power_W",
            "Temp_C",
            "UV_Ideal",
            "UV_Actual",
            "Dust_Ratio",
            "V_Diff",
            "SOC_Percentage",

            # Sequence Features
            "V_PV_prev1",
            "V_PV_prev2",
            "V_PV_diff1",
            "V_PV_avg3",

            "V_Batt_prev1",
            "V_Batt_prev2",
            "V_Batt_diff1",
            "V_Batt_avg3",

            "Amp_prev1",
            "Amp_prev2",
            "Amp_diff1",
            "Amp_avg3",

            "Power_W_prev1",
            "Power_W_prev2",
            "Power_W_diff1",
            "Power_W_avg3",

            "Temp_C_prev1",
            "Temp_C_prev2",
            "Temp_C_diff1",
            "Temp_C_avg3",

            "UV_Actual_prev1",
            "UV_Actual_prev2",
            "UV_Actual_diff1",
            "UV_Actual_avg3"
        ]

        st.markdown("### Enter Live Sensor Readings")

        cols = st.columns(3)

        for i, feature in enumerate(features):

            input_data[feature] = cols[i % 3].number_input(
                feature,
                value=float(data[feature].mean())
            )

        # =============================
        # PREDICT BUTTON
        # =============================

        if st.button("🔍 Predict Fault"):

            input_df = pd.DataFrame([input_data])

            prediction = model.predict(input_df)[0]

            st.error(
                f"⚠️ Predicted Fault Type: {prediction}"
            )

            # =============================
            # RISK ANALYSIS
            # =============================

            if prediction == "Normal":

                st.success(
                    "✅ System Operating Normally"
                )

            else:

                st.warning(
                    "⚠️ Potential Fault Detected!"
                )

                st.info(
                    "📌 AI detected abnormal sequential behavior in the system."
                )

    except Exception as e:

        st.error("❌ Error in prediction system")

        st.exception(e)

# =============================
# NO FILE
# =============================

else:

    st.info(
        "📂 Please upload a CSV dataset to start."
    )

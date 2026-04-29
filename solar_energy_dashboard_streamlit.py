import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import os
import numpy as np

# -----------------------------
# إعداد الصفحة
# -----------------------------

st.set_page_config(page_title="Solar Dashboard", layout="wide")
st.title("⚡ Solar Energy Monitoring Dashboard")

# -----------------------------
# تحميل الملف
# -----------------------------

file = st.file_uploader("Upload CSV File", type=["csv"])

if file:
    data = pd.read_csv(file)

    st.success("Data Loaded Successfully ✅")

    # -----------------------------
    # تنظيف البيانات + Feature Engineering
    # -----------------------------

    if "Power_W" in data.columns and "Ideal_Power_W" in data.columns:
        data["Efficiency"] = data["Power_W"] / data["Ideal_Power_W"]

    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.dropna(inplace=True)

    # -----------------------------
    # الفلاتر
    # -----------------------------

    st.sidebar.header("Filters")

    if "Label" in data.columns:
        fault_filter = st.sidebar.multiselect(
            "Select Fault Type",
            options=data["Label"].unique(),
            default=data["Label"].unique()
        )
        data = data[data["Label"].isin(fault_filter)]

    # -----------------------------
    # KPIs
    # -----------------------------

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Avg Power", f"{data['Power_W'].mean():.2f} W")
    col2.metric("Max Power", f"{data['Power_W'].max():.2f} W")
    col3.metric("Avg Battery", f"{data['V_Batt'].mean():.2f} V")
    col4.metric("Avg Dust", f"{data['Dust_Ratio'].mean():.2f} %")

    # -----------------------------
    # الرسومات
    # -----------------------------

    st.subheader("📊 Power vs Ideal Power")
    fig1 = px.line(data, y=["Power_W", "Ideal_Power_W"])
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("🚨 Fault Distribution")
    if "Label" in data.columns:
        fig2 = px.pie(data, names="Label")
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("🌡️ Temperature vs Power")
    fig3 = px.scatter(data, x="Temp_C", y="Power_W", color="Label")
    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("☀️ UV Ideal vs Actual")
    fig4 = px.line(data, y=["UV_Ideal", "UV_Actual"])
    st.plotly_chart(fig4, use_container_width=True)

    st.subheader("🔋 Battery Status")
    if "Battery_Alert" in data.columns:
        fig5 = px.histogram(data, x="Battery_Alert", color="Battery_Alert")
        st.plotly_chart(fig5, use_container_width=True)

    # -----------------------------
    # Prediction Section
    # -----------------------------

    st.subheader("🤖 Predict Fault")

    try:
        # تحميل الموديل
        if os.path.exists("best_model.pkl"):
            model = joblib.load("best_model.pkl")
        else:
            st.error("❌ Model file not found! Upload best_model.pkl to GitHub")
            st.stop()

        input_data = {}

        features = [
            "Latitude","V_PV","V_Batt","Amp","Power_W",
            "Ideal_Power_W","Temp_C","UV_Ideal","UV_Actual",
            "Dust_Ratio","V_Diff","SOC_Percentage","Efficiency"
        ]

        cols = st.columns(3)

        for i, feature in enumerate(features):
            if feature in data.columns:
                default_val = float(data[feature].mean())
            else:
                default_val = 0.0

            input_data[feature] = cols[i % 3].number_input(
                feature, value=default_val
            )

        if st.button("Predict"):
            input_df = pd.DataFrame([input_data])

            prediction = model.predict(input_df)[0]

            # Confidence
            if hasattr(model, "predict_proba"):
                confidence = np.max(model.predict_proba(input_df)) * 100
                st.success(f"Prediction: {prediction}")
                st.info(f"Confidence: {confidence:.2f}%")
            else:
                st.success(f"Prediction: {prediction}")

            # Smart Alert
            if prediction != "Normal":
                st.error("⚠️ Warning: Potential Fault Detected!")

    except Exception as e:
        st.error("❌ Error in prediction system")
        st.write(e)

else:
    st.info("Please upload a CSV file to start.")

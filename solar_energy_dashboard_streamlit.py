import streamlit as st
import pandas as pd
import plotly.express as px
import joblib

# -----------------------------
# Load Data
# -----------------------------

st.set_page_config(page_title="Solar Dashboard", layout="wide")

st.title("⚡ Solar Energy Monitoring Dashboard")

file = st.file_uploader("Upload CSV File", type=["csv"])

if file:
    data = pd.read_csv(file)

    st.success("Data Loaded Successfully ✅")

    # -----------------------------
    # Sidebar Filters
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
    # Power Chart
    # -----------------------------

    st.subheader("📊 Power vs Ideal Power")

    fig1 = px.line(data, y=["Power_W", "Ideal_Power_W"], title="Power Comparison")
    st.plotly_chart(fig1, use_container_width=True)

    # -----------------------------
    # Fault Distribution
    # -----------------------------

    st.subheader("🚨 Fault Distribution")

    if "Label" in data.columns:
        fig2 = px.pie(data, names="Label", title="Fault Types")
        st.plotly_chart(fig2, use_container_width=True)

    # -----------------------------
    # Temperature vs Power
    # -----------------------------

    st.subheader("🌡️ Temperature vs Power")

    fig3 = px.scatter(data, x="Temp_C", y="Power_W", color="Label")
    st.plotly_chart(fig3, use_container_width=True)

    # -----------------------------
    # UV Analysis
    # -----------------------------

    st.subheader("☀️ UV Ideal vs Actual")

    fig4 = px.line(data, y=["UV_Ideal", "UV_Actual"])
    st.plotly_chart(fig4, use_container_width=True)

    # -----------------------------
    # Battery Status
    # -----------------------------

    st.subheader("🔋 Battery Status")

    if "Battery_Alert" in data.columns:
        fig5 = px.histogram(data, x="Battery_Alert", color="Battery_Alert")
        st.plotly_chart(fig5, use_container_width=True)

    # -----------------------------
    # Prediction Section
    # -----------------------------

    st.subheader("🤖 Predict Fault")

    try:
        model = joblib.load("rf_predictor.pkl")

        input_data = {}
        features = [
            "Latitude","V_PV","V_Batt","Amp","Power_W",
            "Ideal_Power_W","Temp_C","UV_Ideal","UV_Actual",
            "Dust_Ratio","V_Diff","SOC_Percentage"
        ]

        cols = st.columns(3)

        for i, feature in enumerate(features):
            input_data[feature] = cols[i % 3].number_input(feature, value=float(data[feature].mean()))

        if st.button("Predict"):
            input_df = pd.DataFrame([input_data])
            prediction = model.predict(input_df)[0]
            st.error(f"⚠️ Predicted Fault: {prediction}")

    except:
        st.warning("Model file not found. Place rf_predictor.pkl in the same folder.")

else:
    st.info("Please upload a CSV file to start.")

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Smart Irrigation Dashboard", page_icon="🌱", layout="wide")
st.title("🌱 Smart Irrigation Monitoring System")
st.caption("Simulated sensor dashboard • ML-based irrigation recommendation")

DATA_PATH = "data/simulated_irrigation_data.csv"
MODEL_PATH = "models/irrigation_model.joblib"
FEATURES = ["temperature_c", "humidity_pct", "light_lux", "soil_moisture_pct"]

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

df = load_data()
model_bundle = load_model()

st.sidebar.header("Sensor input")
temperature = st.sidebar.slider("Temperature (°C)", 0.0, 50.0, 31.0, 0.5)
humidity = st.sidebar.slider("Humidity (%)", 0.0, 100.0, 45.0, 1.0)
light = st.sidebar.slider("Light (lux)", 0, 100000, 65000, 1000)
soil = st.sidebar.slider("Soil moisture (%)", 0.0, 100.0, 25.0, 0.5)

st.subheader("Live sensor readings (simulated)")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Temperature", f"{temperature:.1f} °C")
c2.metric("Humidity", f"{humidity:.1f}%")
c3.metric("Light", f"{light:,.0f} lux")
c4.metric("Soil moisture", f"{soil:.1f}%")

input_df = pd.DataFrame([[temperature, humidity, light, soil]], columns=FEATURES)

st.subheader("Irrigation decision")
if model_bundle is None:
    st.warning("Model not found. Run `python train_model.py` first.")
else:
    model = model_bundle["model"]
    probability = float(model.predict_proba(input_df)[0][1])
    prediction = int(probability >= 0.5)
    left, right = st.columns(2)
    with left:
        if prediction:
            st.error("💧 Irrigation recommended")
        else:
            st.success("✅ Irrigation not required")
        st.metric("Predicted irrigation probability", f"{probability*100:.1f}%")
    with right:
        st.progress(probability)
        st.caption("Decision threshold: 50%")

    # Demonstration-only control signal. No physical pump is connected.
    pump_signal = 1 if prediction else 0
    st.info(f"Simulated pump control signal: **{pump_signal}** ({'ON' if pump_signal else 'OFF'})")
    st.caption("This is a software simulation only. Do not connect it directly to a real pump.")

st.divider()
st.subheader("Historical sensor data")
st.dataframe(df.tail(20), use_container_width=True)
chart_data = df.copy()
chart_data["timestamp"] = pd.to_datetime(chart_data["timestamp"])
chart_data = chart_data.set_index("timestamp")
st.line_chart(chart_data[["temperature_c", "humidity_pct", "soil_moisture_pct"]].tail(300))

st.subheader("Historical analysis")
a, b, c = st.columns(3)
a.metric("Rows in dataset", f"{len(df):,}")
b.metric("Irrigation-positive samples", f"{int(df['irrigation_required'].sum()):,}")
c.metric("Positive class rate", f"{df['irrigation_required'].mean()*100:.1f}%")

st.download_button(
    "Download simulated dataset",
    data=df.to_csv(index=False).encode("utf-8"),
    file_name="simulated_irrigation_data.csv",
    mime="text/csv"
)

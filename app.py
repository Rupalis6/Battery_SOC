import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import plotly.graph_objects as go

# ===============================
# LOAD MODEL & SCALER
# ===============================
@st.cache_resource
def load_artifacts():
    model = load_model("ann_model.h5", compile=False)
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_artifacts()

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(page_title="Battery SOC Dashboard", page_icon="🔋")

st.title("🔋 Battery SOC Monitoring Dashboard")
st.markdown("Predict SOC using ANN (Charging & Discharging supported)")

# ===============================
# INPUT SECTION
# ===============================
st.subheader("⚙️ Input Parameters")

col1, col2 = st.columns(2)

with col1:
    voltage = st.slider("Battery Voltage (V)", 10.0, 14.0, 12.5)

with col2:
    current = st.slider(
        "Battery Current (A)",
        -27.0, 27.0, 2.0   # negative → charging, positive → discharging
    )

# ===============================
# INPUT VALIDATION (important)
# ===============================
if abs(current) > 15:
    st.warning("⚠️ Current is outside typical training range. Prediction may be less accurate.")

# ===============================
# MODEL PREDICTION
# ===============================
input_data = np.array([[current, voltage]])   # correct order
input_scaled = scaler.transform(input_data)

prediction = model.predict(input_scaled)
soc = float(prediction.flatten()[0])

# Clip SOC to valid range
soc = max(0, min(100, soc))

# ===============================
# SOC GAUGE
# ===============================
st.subheader("🔋 State of Charge (SOC)")

fig = go.Figure(go.Indicator(
    mode="gauge+number",
    value=soc,
    title={'text': "SOC (%)"},
    gauge={
        'axis': {'range': [0, 100]},
        'bar': {'color': "green"},
        'steps': [
            {'range': [0, 40], 'color': "#ff4d4d"},
            {'range': [40, 80], 'color': "#ffcc00"},
            {'range': [80, 100], 'color': "#66ff66"}
        ],
    }
))

st.plotly_chart(fig, use_container_width=True)

# ===============================
# STATUS DISPLAY
# ===============================
st.subheader("📊 System Status")

st.write(f"**Voltage:** {voltage:.2f} V")
st.write(f"**Current:** {current:.2f} A")
st.write(f"**Predicted SOC:** {soc:.2f} %")

# Automatic mode detection
if current < 0:
    st.info("⚡ Battery Charging Mode")
elif current > 0:
    st.info("🔻 Battery Discharging Mode (Load Connected)")
else:
    st.info("⏸️ Battery Idle")

# SOC condition
if soc > 80:
    st.success("✅ Battery Highly Charged")
elif soc > 40:
    st.warning("⚡ Battery Moderately Charged")
else:
    st.error("🔴 Battery Low — Needs Charging")

# ===============================
# FOOTNOTE / INFO
# ===============================
st.markdown("---")
st.info(
    "ℹ️ The ANN model is trained on both charging (negative current) "
    "and discharging (positive current) datasets. Current sign determines battery mode."
)
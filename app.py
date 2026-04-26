import streamlit as st
import numpy as np
import joblib
import os
from tensorflow.keras.models import load_model
import plotly.graph_objects as go

# ===============================
# LOAD MODEL & SCALER (SAFE)
# ===============================
@st.cache_resource
def load_artifacts():
    model = load_model("ann_model.keras", compile=False, safe_mode=False)
    scaler_path = "scaler.pkl"

    # Check if files exist
    if not os.path.exists(model_path):
        st.error("❌ Model file 'ann_model.h5' not found. Upload it to the same folder as app.py")
        st.stop()

    if not os.path.exists(scaler_path):
        st.error("❌ Scaler file 'scaler.pkl' not found. Upload it to the same folder as app.py")
        st.stop()

    model = load_model(model_path, compile=False)
    scaler = joblib.load(scaler_path)

    return model, scaler

model, scaler = load_artifacts()

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(page_title="Battery SOC Dashboard", page_icon="🔋")

st.title("🔋 Battery SOC Monitoring Dashboard")
st.markdown("ANN-based SOC prediction using Voltage & Current")

# ===============================
# INPUT SECTION
# ===============================
st.subheader("⚙️ Input Parameters")

col1, col2 = st.columns(2)

with col1:
    voltage = st.slider("Battery Voltage (V)", 10.0, 15.5, 12.5)

with col2:
    current = st.slider(
        "Battery Current (A)",
        -20.0, 20.0, 2.0
    )

# ===============================
# WARNING FOR OUT OF RANGE
# ===============================
if abs(current) > 15:
    st.warning("⚠️ Current is outside typical training range. Prediction may be less accurate.")

# ===============================
# MODEL PREDICTION
# ===============================
try:
    input_data = np.array([[current, voltage]])   # correct order
    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)
    soc = float(prediction.flatten()[0])

    # Clip SOC
    soc = max(0, min(100, soc))

except Exception as e:
    st.error("❌ Error during prediction")
    st.write(str(e))
    st.stop()

# ===============================
# SOC GAUGE
# ===============================
st.subheader("🔋 State of Charge")

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

# Mode detection
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
# DEBUG (OPTIONAL)
# ===============================
with st.expander("🔍 Debug Info"):
    st.write("Files in directory:", os.listdir())

# ===============================
# FOOTNOTE
# ===============================
st.markdown("---")
st.info(
    "ℹ️ Model trained on charging (negative current) and discharging (positive current) datasets. "
    "Current sign determines battery mode automatically."
)

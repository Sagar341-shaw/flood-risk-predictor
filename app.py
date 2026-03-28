
import streamlit as st
import joblib
import numpy as np

model  = joblib.load('flood_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("Flood Risk Predictor")
st.write("Adjust the sliders and click Predict!")

rainfall = st.slider("Rainfall (mm/day)", 0, 300, 80)
river    = st.slider("River water level (m)", 0.0, 10.0, 3.0)
soil     = st.slider("Soil moisture (%)", 0, 100, 45)
drain    = st.slider("Drainage capacity (%)", 0, 100, 60)
slope    = st.slider("Slope (degrees)", 0, 45, 10)
days     = st.slider("Days of continuous rain", 0, 14, 3)

if st.button("Predict Flood Risk"):
    inp  = scaler.transform([[rainfall, river, soil, drain, slope, days]])
    pred = model.predict(inp)[0]
    prob = model.predict_proba(inp)[0][1]

    if pred == 1:
        st.error(f"HIGH FLOOD RISK - {round(prob*100)}% probability")
    else:
        st.success(f"LOW FLOOD RISK - {round(prob*100)}% probability")

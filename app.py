import streamlit as st
import joblib
import numpy as np

st.set_page_config(page_title="Flood Risk Predictor", page_icon="🌊", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Syne:wght@700;800&display=swap');
* { font-family: 'Space Grotesk', sans-serif; }
html, body, [class*="css"] { background-color: #0a0f1e; color: #e0e8ff; }
.stApp { background: linear-gradient(135deg, #0a0f1e 0%, #0d1b2a 50%, #0a1628 100%); min-height: 100vh; }
.hero { text-align: center; padding: 2.5rem 1rem 1.5rem; }
.hero-icon { font-size: 3.5rem; display: block; margin-bottom: 0.5rem; animation: float 3s ease-in-out infinite; }
@keyframes float { 0%, 100% { transform: translateY(0px); } 50% { transform: translateY(-10px); } }
.hero h1 { font-family: 'Syne', sans-serif; font-size: 2.8rem; font-weight: 800; background: linear-gradient(90deg, #4fc3f7, #81d4fa, #4dd0e1); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin: 0; }
.hero p { color: #7090b0; font-size: 1rem; margin-top: 0.5rem; }
.section-label { font-size: 0.7rem; font-weight: 600; letter-spacing: 0.15em; color: #4fc3f7; text-transform: uppercase; margin-bottom: 1rem; margin-top: 1.5rem; }
.stButton > button { width: 100%; background: linear-gradient(135deg, #4fc3f7, #0288d1) !important; color: white !important; border: none !important; border-radius: 14px !important; padding: 0.9rem 2rem !important; font-size: 1.1rem !important; font-weight: 700 !important; margin-top: 1.5rem !important; box-shadow: 0 4px 20px rgba(79,195,247,0.35) !important; }
.result-high { background: linear-gradient(135deg, #1a0a0a, #2d0f0f); border: 1px solid #e53935; border-radius: 16px; padding: 1.5rem; text-align: center; margin-top: 1.5rem; animation: pulse-red 2s infinite; }
@keyframes pulse-red { 0%,100%{box-shadow:0 0 30px rgba(229,57,53,0.2);} 50%{box-shadow:0 0 50px rgba(229,57,53,0.4);} }
.result-low { background: linear-gradient(135deg, #0a1a0a, #0f2d18); border: 1px solid #43a047; border-radius: 16px; padding: 1.5rem; text-align: center; margin-top: 1.5rem; animation: pulse-green 2s infinite; }
@keyframes pulse-green { 0%,100%{box-shadow:0 0 30px rgba(67,160,71,0.2);} 50%{box-shadow:0 0 50px rgba(67,160,71,0.4);} }
.result-emoji { font-size: 3rem; display: block; margin-bottom: 0.5rem; }
.result-title { font-family: 'Syne', sans-serif; font-size: 1.8rem; font-weight: 800; }
.result-high .result-title { color: #ef5350; }
.result-low .result-title { color: #66bb6a; }
.result-prob { font-size: 1rem; color: #7090b0; margin-top: 0.3rem; }
.divider { height: 1px; background: linear-gradient(90deg, transparent, rgba(79,195,247,0.3), transparent); margin: 1.5rem 0; }
.footer { text-align: center; color: #304050; font-size: 0.75rem; padding: 2rem 0 1rem; }
</style>
""", unsafe_allow_html=True)

model  = joblib.load('flood_model.pkl')
scaler = joblib.load('scaler.pkl')

st.markdown("""
<div class="hero">
    <span class="hero-icon">🌊</span>
    <h1>Flood Risk Predictor</h1>
    <p>AI-powered early warning system for flood detection</p>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown('<div class="section-label">Environmental Parameters</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    rainfall = st.slider("Rainfall (mm/day)", 0, 300, 80)
    soil     = st.slider("Soil moisture (%)", 0, 100, 45)
    slope    = st.slider("Slope (degrees)", 0, 45, 10)
with col2:
    river    = st.slider("River water level (m)", 0.0, 10.0, 3.0)
    drain    = st.slider("Drainage capacity (%)", 0, 100, 60)
    days     = st.slider("Days of continuous rain", 0, 14, 3)

if st.button("Predict Flood Risk"):
    inp  = scaler.transform([[rainfall, river, soil, drain, slope, days]])
    pred = model.predict(inp)[0]
    prob = model.predict_proba(inp)[0][1]
    pct  = round(prob * 100)

    if pred == 1:
        st.markdown(f"""
        <div class="result-high">
            <span class="result-emoji">🚨</span>
            <div class="result-title">HIGH FLOOD RISK</div>
            <div class="result-prob">Flood probability: <strong style="color:#ef5350">{pct}%</strong> — Immediate action recommended</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-low">
            <span class="result-emoji">✅</span>
            <div class="result-title">LOW FLOOD RISK</div>
            <div class="result-prob">Flood probability: <strong style="color:#66bb6a">{pct}%</strong> — Conditions are safe</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown('<div class="footer">Built with Machine Learning · Random Forest Classifier · Deployed on Streamlit Cloud</div>', unsafe_allow_html=True)

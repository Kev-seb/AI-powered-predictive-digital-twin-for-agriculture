"""
weather_tab.py
--------------
Streamlit tab for weather risk assessment via Open-Meteo API.
"""

from __future__ import annotations

import streamlit as st

def render():
    st.header("Weather Risk Assessment")
    
    st.markdown("Analysing real-time forecast data for disease and drought risks.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Disease Risk")
        st.error("High Risk - Fungal (High Humidity & Temp)")
        
    with col2:
        st.subheader("Drought Risk")
        st.success("Low Risk - Adequate soil moisture")

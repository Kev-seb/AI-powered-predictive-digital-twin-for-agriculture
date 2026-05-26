"""
temporal_tab.py
---------------
Streamlit tab for temporal change detection and growth stage tracking.
"""

from __future__ import annotations

import streamlit as st

def render():
    st.header("Temporal Change & Growth Tracking")
    
    st.markdown("Track crop development across multiple surveys.")
    
    st.line_chart([0.2, 0.35, 0.6, 0.75, 0.8, 0.5])
    st.info("Mock NDVI time-series tracking over the season.")

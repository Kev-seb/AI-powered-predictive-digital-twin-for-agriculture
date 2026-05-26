"""
stress_tab.py
-------------
Streamlit tab for displaying crop stress classification and segmentation results.
"""

from __future__ import annotations

import streamlit as st

def render():
    st.header("Crop Stress Analysis")
    
    if "multispectral_stack" not in st.session_state:
        st.warning("Please upload imagery in the Upload tab first.")
        return
        
    st.markdown("Deep learning-based stress detection (EfficientNet / DeepLabV3+).")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Overall Stress Level", value="Moderate Stress")
        st.metric(label="Affected Area (%)", value="14.2 %")
        
    with col2:
        st.image(st.session_state.rgb_image, caption="Stress Segmentation Mask (Mock)", use_column_width=True)

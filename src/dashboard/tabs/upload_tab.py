"""
upload_tab.py
-------------
Streamlit tab for uploading multispectral UAV imagery.
"""

from __future__ import annotations

import streamlit as st
import numpy as np

def render():
    st.header("Upload Multispectral Imagery")
    st.markdown("Upload a calibrated multispectral orthomosaic (TIF) containing Blue, Green, Red, RedEdge, and NIR bands.")
    
    uploaded_file = st.file_uploader("Upload Multispectral Image (.tif, .tiff)", type=["tif", "tiff"])
    
    if uploaded_file is not None:
        st.success(f"File uploaded: {uploaded_file.name}")
        st.info("In a full application, this would parse the TIF file using rasterio, perform radiometric calibration, and store the tensor in session_state.")
        
        # Mocking the session state update for demonstration
        if "multispectral_stack" not in st.session_state:
            st.session_state.multispectral_stack = np.random.rand(4, 512, 512).astype(np.float32)
            st.session_state.rgb_image = (np.random.rand(512, 512, 3) * 255).astype(np.uint8)
            st.success("Mock image stack loaded into session.")
    else:
        st.warning("Please upload a file to proceed.")

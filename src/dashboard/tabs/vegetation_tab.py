"""
vegetation_tab.py
-----------------
Streamlit tab for displaying calculated vegetation indices.
"""

from __future__ import annotations

import streamlit as st

def render():
    st.header("Vegetation Indices")
    
    if "multispectral_stack" not in st.session_state:
        st.warning("Please upload imagery in the Upload tab first.")
        return
        
    st.markdown("Visualising computed vegetation indices (NDVI, NDRE, MSAVI2, etc.).")
    
    index_choice = st.selectbox("Select Index", ["NDVI", "NDRE", "GNDVI", "MSAVI2"])
    
    # Mock visualisation
    st.image(st.session_state.rgb_image, caption=f"Mock {index_choice} Map", use_column_width=True, clamp=True)
    
    st.info(f"{index_choice} indicates overall crop vigour and canopy density.")

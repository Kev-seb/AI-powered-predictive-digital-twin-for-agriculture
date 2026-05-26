"""
overlays.py
-----------
Streamlit components for displaying image overlays (e.g. GradCAM, stress masks).
"""

from __future__ import annotations

import streamlit as st
import numpy as np

def render_image_comparison(img1: np.ndarray, img2: np.ndarray, title1: str, title2: str) -> None:
    """Render two images side-by-side."""
    col1, col2 = st.columns(2)
    with col1:
        st.image(img1, caption=title1, use_column_width=True)
    with col2:
        st.image(img2, caption=title2, use_column_width=True)

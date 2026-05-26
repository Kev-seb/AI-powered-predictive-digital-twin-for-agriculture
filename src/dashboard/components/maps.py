"""
maps.py
-------
Folium map rendering components for Streamlit.
"""

from __future__ import annotations

import streamlit as st

try:
    from streamlit_folium import st_folium
    HAS_ST_FOLIUM = True
except ImportError:
    HAS_ST_FOLIUM = False

def render_folium_map(m, height: int = 500, key: str = "map") -> None:
    """
    Render a Folium map in Streamlit.
    """
    if m is None:
        st.warning("Map object is None.")
        return
        
    if HAS_ST_FOLIUM:
        st_folium(m, height=height, width="100%", returned_objects=[], key=key)
    else:
        # Fallback to HTML embedding if streamlit-folium is not installed
        st.components.v1.html(m._repr_html_(), height=height)

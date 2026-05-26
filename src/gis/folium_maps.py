"""
Interactive Folium maps: stress overlay, zone choropleth, index heatmaps.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import folium
import folium.plugins
import geopandas as gpd
import numpy as np
from loguru import logger


def zone_choropleth(
    gdf:        gpd.GeoDataFrame,
    center:     Optional[tuple[float, float]] = None,
    zoom:       int = 15,
    output_path: Optional[str | Path] = None,
) -> folium.Map:
    """
    Render management zones as a colour-coded Folium choropleth.
    """
    if center is None:
        centroid = gdf.geometry.unary_union.centroid
        center   = (centroid.y, centroid.x)

    m = folium.Map(location=center, zoom_start=zoom, tiles="Esri.WorldImagery")

    # Draw each zone as a filled polygon
    for _, row in gdf.iterrows():
        folium.GeoJson(
            row.geometry.__geo_interface__,
            style_function=lambda feat, c=row.color: {
                "fillColor":   c,
                "color":       "black",
                "weight":      1,
                "fillOpacity": 0.55,
            },
            tooltip=folium.Tooltip(
                f"<b>{row.label}</b><br>Mean stress: {row.mean_stress:.2f}"
            ),
        ).add_to(m)

    # Layer control + minimap
    folium.LayerControl().add_to(m)
    folium.plugins.MiniMap().add_to(m)

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        m.save(str(output_path))
        logger.info(f"Zone map saved → {output_path}")

    return m


def stress_heatmap(
    lats:        np.ndarray,
    lons:        np.ndarray,
    stress_vals: np.ndarray,
    center:      Optional[tuple[float, float]] = None,
    output_path: Optional[str | Path] = None,
) -> folium.Map:
    """
    Render a continuous stress heatmap using Folium HeatMap plugin.

    Parameters
    ----------
    lats, lons, stress_vals : 1-D arrays of equal length (sampled pixel coordinates)
    """
    center = center or (float(lats.mean()), float(lons.mean()))
    m = folium.Map(location=center, zoom_start=17, tiles="Esri.WorldImagery")

    data = list(zip(lats.tolist(), lons.tolist(), stress_vals.tolist()))
    folium.plugins.HeatMap(data, min_opacity=0.3, radius=15, blur=20).add_to(m)

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        m.save(str(output_path))
        logger.info(f"Heatmap saved → {output_path}")

    return m
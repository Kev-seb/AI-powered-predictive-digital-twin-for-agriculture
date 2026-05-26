"""
src/gis/geojson_export.py
Export GeoDataFrame to GeoJSON.
"""

from __future__ import annotations
from pathlib import Path
import geopandas as gpd
from loguru import logger


def export_geojson(gdf: gpd.GeoDataFrame, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(output_path, driver="GeoJSON")
    logger.info(f"GeoJSON exported → {output_path}")
    return output_path
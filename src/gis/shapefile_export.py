"""
src/gis/shapefile_export.py
Export GeoDataFrame to ESRI Shapefile.
"""

from __future__ import annotations
from pathlib import Path
import geopandas as gpd
from loguru import logger


def export_shapefile(gdf: gpd.GeoDataFrame, output_dir: str | Path, name: str = "zones") -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{name}.shp"
    # Shapefiles truncate column names to 10 chars
    gdf_copy = gdf.copy()
    gdf_copy.columns = [c[:10] for c in gdf_copy.columns]
    gdf_copy.to_file(out_path, driver="ESRI Shapefile")
    logger.info(f"Shapefile exported → {out_path}")
    return out_path
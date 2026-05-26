"""
satellite_loader.py
-------------------
Google Earth Engine (GEE) integration pipeline for querying and downloading
large-scale Sentinel-2 multi-spectral data. Handles temporal compositing,
cloud-masking, and resolution harmonization to fuse macro-scale satellite 
scans with micro-scale UAV imagery.
"""

import os
import requests
import zipfile
import io
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

try:
    import ee
    HAS_EE = True
except ImportError:
    HAS_EE = False
    print("[WARNING] earthengine-api not installed.")

class Sentinel2Engine:
    def __init__(self, project_id: Optional[str] = None):
        self.initialized = False
        self.error_msg = ""
        if not HAS_EE:
            self.error_msg = "earthengine-api not installed."
            return
            
        try:
            # Simple local auth relies on the user having run `earthengine authenticate`
            if project_id:
                ee.Initialize(project=project_id)
            else:
                ee.Initialize()
            self.initialized = True
            print("[INFO] Successfully initialized Google Earth Engine (Sentinel-2).")
        except Exception as e:
            self.error_msg = str(e)
            print(f"[ERROR] Failed to initialize GEE. Details: {e}")

    def mask_s2_clouds(self, image: ee.Image) -> ee.Image:
        """
        Filters out clouds and cloud shadows using the Sentinel-2 SCL (Scene Classification Layer).
        SCL classes:
        3: Cloud Shadows, 8: Cloud Medium Probability, 9: Cloud High Probability, 10: Thin Cirrus, 11: Snow
        """
        scl = image.select('SCL')
        # Keep pixels that are NOT shadows, clouds, or snow.
        # We allow 4 (Vegetation), 5 (Bare soils), 6 (Water), 7 (Unclassified)
        mask = scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10)).And(scl.neq(11))
        return image.updateMask(mask)

    def fetch_satellite_composite(
        self, 
        roi_polygon: list, 
        start_date: str, 
        end_date: str
    ) -> Optional[ee.Image]:
        """
        Fetches a cloud-free median composite of Sentinel-2 L2A data.
        roi_polygon: List of [lon, lat] coordinates defining the bounding box.
        """
        if not self.initialized:
            return None
            
        try:
            # Define Region of Interest
            roi = ee.Geometry.Polygon([roi_polygon])
            
            # Fetch Sentinel-2 Surface Reflectance Harmonized dataset
            dataset = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                      .filterBounds(roi)
                      .filterDate(start_date, end_date)
                      .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
                      .map(self.mask_s2_clouds))
            
            # Create a median temporal composite to remove transient anomalies
            composite = dataset.median()
            
            # Calculate Indices
            # 1. NDVI = (B8 - B4) / (B8 + B4)
            ndvi = composite.normalizedDifference(['B8', 'B4']).rename('NDVI')
            
            # 2. NDRE = (B8 - B5) / (B8 + B5)
            ndre = composite.normalizedDifference(['B8', 'B5']).rename('NDRE')
            
            # 3. NDWI = (B3 - B8) / (B3 + B8)
            ndwi = composite.normalizedDifference(['B3', 'B8']).rename('NDWI')
            
            # 4. EVI = 2.5 * ((NIR - Red) / (NIR + 6 * Red - 7.5 * Blue + 1))
            # S2 L2A is scaled by 10000, so the '1' constant is 10000
            evi = composite.expression(
                '2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 10000))', {
                    'NIR': composite.select('B8'),
                    'RED': composite.select('B4'),
                    'BLUE': composite.select('B2')
                }
            ).rename('EVI')
            
            # 5. Chlorophyll Index (CIre) = (NIR / RedEdge) - 1
            cire = composite.expression(
                '(NIR / REDEDGE) - 1.0', {
                    'NIR': composite.select('B8'),
                    'REDEDGE': composite.select('B5')
                }
            ).rename('CIRE')
            
            # Add indices as bands and clip to ROI
            final_image = composite.addBands([ndvi, ndre, ndwi, evi, cire]).clip(roi)
            return final_image
            
        except Exception as e:
            print(f"[ERROR] Failed to fetch GEE composite: {e}")
            return None

    def get_rgb_thumbnail_url(self, image: ee.Image, roi_polygon: list) -> Optional[str]:
        """
        Returns a signed URL to download a lightweight visual RGB preview.
        """
        if not self.initialized or image is None:
            return None
            
        try:
            roi = ee.Geometry.Polygon([roi_polygon])
            url = image.getThumbURL({
                'bands': ['B4', 'B3', 'B2'],
                'min': 0,
                'max': 3000,
                'region': roi,
                'scale': 10,
                'format': 'png'
            })
            return url
        except Exception:
            return None

    def get_index_thumbnail_url(self, image: ee.Image, roi_polygon: list, index_name: str) -> Optional[str]:
        """
        Returns a signed URL to download a colored heatmap preview for ANY calculated index.
        """
        if not self.initialized or image is None:
            return None
            
        params = {
            'NDVI': {'min': -0.1, 'max': 0.9, 'palette': ['red', 'yellow', 'green']},
            'NDRE': {'min': 0.0, 'max': 0.7, 'palette': ['red', 'orange', 'lightgreen', 'darkgreen']},
            'NDWI': {'min': -0.5, 'max': 0.5, 'palette': ['brown', 'white', 'blue']},
            'EVI': {'min': 0.0, 'max': 1.0, 'palette': ['saddlebrown', 'yellow', 'green']},
            'CIRE': {'min': 0.0, 'max': 4.0, 'palette': ['red', 'yellow', 'green', 'cyan']}
        }
        
        if index_name not in params:
            return None
            
        try:
            roi = ee.Geometry.Polygon([roi_polygon])
            url = image.getThumbURL({
                'bands': [index_name],
                'min': params[index_name]['min'],
                'max': params[index_name]['max'],
                'palette': params[index_name]['palette'],
                'region': roi,
                'scale': 10,
                'format': 'png'
            })
            return url
        except Exception:
            return None

    def download_raw_data(self, image: ee.Image, roi_polygon: list, download_dir: str = "outputs/satellite") -> Optional[str]:
        """
        Downloads the actual GeoTIFF array data for mathematical fusion with UAV data.
        Returns the path to the downloaded GeoTIFF.
        """
        if not self.initialized or image is None:
            return None
            
        Path(download_dir).mkdir(parents=True, exist_ok=True)
        roi = ee.Geometry.Polygon([roi_polygon])
        
        try:
            # We want specific bands for fusion
            export_img = image.select(['B3', 'B4', 'B5', 'B8', 'NDVI', 'NDRE', 'NDWI', 'EVI', 'CIRE'])
            url = export_img.getDownloadURL({
                'scale': 10,
                'crs': 'EPSG:4326',
                'region': roi,
                'format': 'GEO_TIFF'
            })
            
            # Download and extract the zip file
            response = requests.get(url)
            if response.status_code == 200:
                with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                    extracted_files = z.namelist()
                    z.extractall(download_dir)
                    if extracted_files:
                        return os.path.join(download_dir, extracted_files[0])
            return None
        except Exception as e:
            print(f"[ERROR] GEE Download failed: {e}")
            return None

    def calculate_spatiotemporal_fusion(self, satellite_index: np.ndarray, uav_index: np.ndarray) -> np.ndarray:
        """
        Calculate the difference/fusion mask between the macroscopic satellite data
        and the high-resolution UAV data. Highlights micro-anomalies that the satellite missed.
        Assumes arrays are already re-projected to the same spatial grid.
        """
        # 1. Normalize both to 0-1
        sat_norm = (satellite_index - np.min(satellite_index)) / (np.max(satellite_index) - np.min(satellite_index) + 1e-8)
        uav_norm = (uav_index - np.min(uav_index)) / (np.max(uav_index) - np.min(uav_index) + 1e-8)
        
        # 2. Compute absolute residual delta
        fusion_delta = np.abs(uav_norm - sat_norm)
        
        # 3. Enhance high-discrepancy zones (micro-stress invisible to satellite)
        enhanced_fusion = np.clip(fusion_delta * 2.0, 0, 1)
        return enhanced_fusion

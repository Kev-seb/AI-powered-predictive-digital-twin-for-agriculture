"""
generate_sample_tiff.py
------------------------
Generates a simulated 4-band Multispectral GeoTIFF for platform testing and file uploads.
Bands: Green (1), Red (2), Red Edge (3), NIR (4).
CRS: EPSG:4326.
Geotransform centered around Latitude 11.0, Longitude 79.0.
"""

from pathlib import Path
import numpy as np
import rasterio
from rasterio.transform import from_origin

def generate_sample_tiff(output_path: str, size: int = 256):
    # Base bands initialization
    np.random.seed(1337)
    
    # Simulate normal distributions for bands
    green = np.random.normal(0.35, 0.05, (size, size)).clip(0.01, 1.0)
    red = np.random.normal(0.20, 0.04, (size, size)).clip(0.01, 1.0)
    red_edge = np.random.normal(0.50, 0.06, (size, size)).clip(0.01, 1.0)
    nir = np.random.normal(0.70, 0.08, (size, size)).clip(0.01, 1.0)
    
    # Inject circular crop stress / disease pockets (Low NIR, high Red, low RedEdge)
    for _ in range(5):
        cy, cx = np.random.randint(40, size - 40), np.random.randint(40, size - 40)
        radius = np.random.randint(15, 30)
        
        yy, xx = np.ogrid[:size, :size]
        mask = (yy - cy)**2 + (xx - cx)**2 < radius**2
        
        stress_level = np.random.uniform(0.4, 0.75)
        nir[mask] *= (1.0 - stress_level)
        red[mask] *= (1.0 + stress_level * 0.4)
        red_edge[mask] *= (1.0 - stress_level * 0.3)
        green[mask] *= (1.0 - stress_level * 0.2)
        
    # Stack bands (Green, Red, RedEdge, NIR)
    data = np.stack([green, red, red_edge, nir]).astype(np.float32)
    
    # Set up geotransform centered at lat 11.0, lon 79.0 (0.05 meters per pixel)
    gsd = 0.05
    lat_deg_per_meter = 1.0 / 111120.0
    lon_deg_per_meter = 1.0 / (111120.0 * np.cos(np.radians(11.0)))
    
    pixel_height_deg = gsd * lat_deg_per_meter
    pixel_width_deg = gsd * lon_deg_per_meter
    
    top_left_lon = 79.0 - (size / 2.0) * pixel_width_deg
    top_left_lat = 11.0 + (size / 2.0) * pixel_height_deg
    
    transform = from_origin(
        top_left_lon,
        top_left_lat,
        pixel_width_deg,
        pixel_height_deg
    )
    
    # Save using rasterio
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    
    metadata = {
        'driver': 'GTiff',
        'dtype': 'float32',
        'nodata': None,
        'width': size,
        'height': size,
        'count': 4,
        'crs': 'EPSG:4326',
        'transform': transform
    }
    
    with rasterio.open(out_file, 'w', **metadata) as dst:
        for b in range(4):
            dst.write(data[b], b + 1)
            dst.set_band_description(b + 1, ["Green", "Red", "Red Edge", "NIR"][b])
            
    print(f"[SUCCESS] Sample multispectral GeoTIFF saved to: {out_file.absolute()}")

if __name__ == "__main__":
    generate_sample_tiff("data/samples/sample_paddy.tif")

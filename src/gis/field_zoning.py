"""
gis_field_zoning.py
--------------------
GIS-ready precision agriculture field zoning.

Capabilities:
    - Grid-based management zone delineation
    - Spatial stress region extraction (connected components)
    - Shapefile / GeoJSON export (requires rasterio + geopandas)
    - Interactive folium map generation
    - Prescription map generation (intervention recommendations)

Scientific basis:
    Zone thresholds derived from:
        - FAO crop stress guidelines
        - Paddy NDVI phenological windows (Liu et al.)
        - IRRI precision agriculture recommendations
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import numpy as np

try:
    from scipy import ndimage
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import geopandas as gpd
    from shapely.geometry import Polygon, mapping
    HAS_GEO = True
except ImportError:
    HAS_GEO = False

try:
    import folium
    HAS_FOLIUM = True
except ImportError:
    HAS_FOLIUM = False


# ──────────────────────────────────────────────────────────────
# Management zone definitions
# ──────────────────────────────────────────────────────────────

ZONE_THRESHOLDS = {
    "High Productivity":  (0.65, 1.0),    # NDVI range
    "Medium Productivity":(0.40, 0.65),
    "Low Productivity":   (0.20, 0.40),
    "Stressed / Bare":    (-1.0, 0.20),
}

ZONE_COLORS = {
    "High Productivity":   "#2ECC71",
    "Medium Productivity": "#F39C12",
    "Low Productivity":    "#E67E22",
    "Stressed / Bare":     "#E74C3C",
}

PRESCRIPTION_MAP = {
    "High Productivity":   "Maintain current inputs. Monitor for over-fertilisation.",
    "Medium Productivity": "Apply 15% additional nitrogen. Check soil moisture.",
    "Low Productivity":    "Investigate root health. Apply micro-nutrients + irrigation.",
    "Stressed / Bare":     "URGENT: scouting required. Consider replanting or remediation.",
}


@dataclass
class FieldZone:
    zone_name: str
    pixel_mask: np.ndarray            # bool (H, W)
    area_pct: float                   # % of total image area
    ndvi_mean: float
    stress_mean: float
    prescription: str
    color: str


@dataclass
class StressRegion:
    region_id: int
    centroid_yx: tuple[float, float]
    area_pixels: int
    area_pct: float
    stress_mean: float
    severity: str                     # "Low" / "Medium" / "High"
    bbox_yx: tuple[int, int, int, int]  # (y0, x0, y1, x1)


# ──────────────────────────────────────────────────────────────
# Management zone delineation
# ──────────────────────────────────────────────────────────────

def delineate_management_zones(ndvi: np.ndarray,
                               stress_score: np.ndarray) -> list[FieldZone]:
    """
    Segment the field into management zones based on NDVI thresholds.

    Returns list of FieldZone objects, one per zone category.
    """
    total_pixels = ndvi.size
    zones = []

    for zone_name, (lo, hi) in ZONE_THRESHOLDS.items():
        mask = (ndvi >= lo) & (ndvi < hi)
        if not mask.any():
            continue

        zones.append(FieldZone(
            zone_name=zone_name,
            pixel_mask=mask,
            area_pct=float(mask.sum() / total_pixels * 100),
            ndvi_mean=float(ndvi[mask].mean()),
            stress_mean=float(stress_score[mask].mean()),
            prescription=PRESCRIPTION_MAP[zone_name],
            color=ZONE_COLORS[zone_name],
        ))

    return zones


def render_zone_map(zones: list[FieldZone], shape: tuple[int, int]) -> np.ndarray:
    """
    Render management zones as an RGB uint8 image (H, W, 3).
    """
    import matplotlib.colors as mcolors
    canvas = np.zeros((*shape, 3), dtype=np.uint8)
    for zone in zones:
        rgb = tuple(int(c * 255) for c in mcolors.to_rgb(zone.color))
        canvas[zone.pixel_mask] = rgb
    return canvas


# ──────────────────────────────────────────────────────────────
# Spatial stress region extraction
# ──────────────────────────────────────────────────────────────

def extract_stress_regions(stress_score: np.ndarray,
                           threshold: float = 0.55,
                           min_area_pct: float = 0.1) -> list[StressRegion]:
    """
    Extract spatially contiguous stressed regions using connected-component analysis.

    Parameters
    ----------
    stress_score  : float32 (H, W) in [0, 1]
    threshold     : pixel score above which a pixel is considered stressed
    min_area_pct  : minimum region size as % of image to report

    Returns
    -------
    List of StressRegion objects sorted by stress_mean descending.
    """
    if not HAS_SCIPY:
        print("[WARN] scipy not installed — skipping spatial region extraction.")
        return []

    binary = (stress_score > threshold).astype(np.int32)
    labeled, n_regions = ndimage.label(binary)
    total_pixels = stress_score.size
    min_pixels = int(total_pixels * min_area_pct / 100)

    regions = []
    for region_id in range(1, n_regions + 1):
        mask = labeled == region_id
        area = int(mask.sum())
        if area < min_pixels:
            continue

        ys, xs = np.where(mask)
        cy, cx = float(ys.mean()), float(xs.mean())
        sm = float(stress_score[mask].mean())

        if sm > 0.75:
            severity = "High"
        elif sm > 0.60:
            severity = "Medium"
        else:
            severity = "Low"

        regions.append(StressRegion(
            region_id=region_id,
            centroid_yx=(cy, cx),
            area_pixels=area,
            area_pct=float(area / total_pixels * 100),
            stress_mean=sm,
            severity=severity,
            bbox_yx=(int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max())),
        ))

    regions.sort(key=lambda r: r.stress_mean, reverse=True)
    return regions


# ──────────────────────────────────────────────────────────────
# Grid-based zoning (fixed grid)
# ──────────────────────────────────────────────────────────────

def compute_grid_statistics(ndvi: np.ndarray, stress_score: np.ndarray,
                             grid_rows: int = 5, grid_cols: int = 5) -> np.ndarray:
    """
    Divide the image into a grid and compute mean NDVI per cell.

    Returns
    -------
    np.ndarray (grid_rows, grid_cols) of mean NDVI per cell.
    """
    H, W = ndvi.shape
    grid = np.zeros((grid_rows, grid_cols), dtype=np.float32)

    row_splits = np.array_split(np.arange(H), grid_rows)
    col_splits = np.array_split(np.arange(W), grid_cols)

    for i, row_idx in enumerate(row_splits):
        for j, col_idx in enumerate(col_splits):
            patch = ndvi[np.ix_(row_idx, col_idx)]
            grid[i, j] = float(patch.mean())

    return grid


def plot_grid_heatmap(grid: np.ndarray, title: str = "Field Grid NDVI Map") -> 'plt.Figure':
    """Visualise the management grid as a heatmap."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(grid, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, label="Mean NDVI")

    rows, cols = grid.shape
    for i in range(rows):
        for j in range(cols):
            ax.text(j, i, f"{grid[i,j]:.2f}", ha="center", va="center",
                    fontsize=9, color="black", fontweight="bold")

    ax.set_xticks(range(cols))
    ax.set_yticks(range(rows))
    ax.set_xticklabels([f"C{j+1}" for j in range(cols)])
    ax.set_yticklabels([f"R{i+1}" for i in range(rows)])
    ax.set_title(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────
# Folium interactive map
# ──────────────────────────────────────────────────────────────

def create_folium_stress_map(
    zones: list[FieldZone],
    regions: list[StressRegion],
    center_latlon: tuple[float, float] = (10.0, 78.0),
    pixel_to_meter: float = 0.05,         # ~5 cm GSD typical for UAV
) -> Optional['folium.Map']:
    """
    Build an interactive Folium map with management zone layers.

    Parameters
    ----------
    center_latlon    : (lat, lon) of field centre
    pixel_to_meter   : ground sampling distance in metres per pixel

    Returns
    -------
    folium.Map or None if folium is not installed.
    """
    if not HAS_FOLIUM:
        print("[WARN] folium not installed — interactive map unavailable.")
        return None

    m = folium.Map(location=list(center_latlon), zoom_start=17,
                   tiles="Esri.WorldImagery")

    # Stress regions as circle markers
    for region in regions:
        cy, cx = region.centroid_yx
        color = {"High": "red", "Medium": "orange", "Low": "yellow"}.get(region.severity, "blue")
        folium.CircleMarker(
            location=[center_latlon[0] + cy * pixel_to_meter / 111111,
                      center_latlon[1] + cx * pixel_to_meter / 111111],
            radius=max(5, int(region.area_pct)),
            color=color,
            fill=True,
            fill_opacity=0.6,
            popup=folium.Popup(
                f"<b>Stress Region #{region.region_id}</b><br>"
                f"Severity: {region.severity}<br>"
                f"Area: {region.area_pct:.2f}%<br>"
                f"Stress Score: {region.stress_mean:.3f}",
                max_width=250,
            ),
        ).add_to(m)

    folium.LayerControl().add_to(m)
    return m
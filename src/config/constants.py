"""
constants.py
------------
Global constants for the UAV Crop Stress Intelligence system.

Covers:
    - Band ordering for multispectral cameras
    - NDVI / index clipping ranges
    - Default model hyperparameters
    - FAO / IRRI-based agronomic thresholds
"""

from __future__ import annotations

# ──────────────────────────────────────────────────────────────
# Spectral band ordering (MicaSense RedEdge / Parrot Sequoia)
# ──────────────────────────────────────────────────────────────

BAND_ORDER = ["Blue", "Green", "Red", "RedEdge", "NIR"]
BAND_IDX   = {name: i for i, name in enumerate(BAND_ORDER)}

# ──────────────────────────────────────────────────────────────
# Vegetation index clip ranges  [min, max]
# ──────────────────────────────────────────────────────────────

INDEX_RANGES = {
    "NDVI":   (-1.0,  1.0),
    "EVI":    (-1.0,  1.0),
    "SAVI":   (-1.0,  1.0),
    "MSAVI2": (-1.0,  1.0),
    "NDWI":   (-1.0,  1.0),
    "NDRE":   (-1.0,  1.0),
    "CIRE":   ( 0.0, 20.0),
}

# ──────────────────────────────────────────────────────────────
# Agronomic stress thresholds  (IRRI / FAO references)
# ──────────────────────────────────────────────────────────────

# NDVI thresholds for paddy rice growth stages
NDVI_STAGE_THRESHOLDS = {
    "Nursery":    (0.10, 0.30),
    "Vegetative": (0.30, 0.60),
    "Flowering":  (0.55, 0.80),
    "Mature":     (0.30, 0.60),
}

# Stress severity (composite stress score 0–1)
STRESS_SEVERITY_THRESHOLDS = {
    "No Stress":       (0.00, 0.25),
    "Low Stress":      (0.25, 0.50),
    "Moderate Stress": (0.50, 0.75),
    "High Stress":     (0.75, 1.00),
}

# ──────────────────────────────────────────────────────────────
# Model defaults
# ──────────────────────────────────────────────────────────────

DEFAULT_IMG_SIZE     = 224          # pixels, input to classifier
DEFAULT_BATCH_SIZE   = 16
DEFAULT_EPOCHS       = 50
DEFAULT_LR           = 1e-4
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_IN_CHANNELS  = 4           # Green, Red, RedEdge, NIR

NUM_STAGE_CLASSES  = 4
NUM_STRESS_CLASSES = 4

# ──────────────────────────────────────────────────────────────
# GIS / spatial defaults
# ──────────────────────────────────────────────────────────────

DEFAULT_CRS          = "EPSG:4326"  # WGS-84
DEFAULT_GSD_METERS   = 0.05         # Ground Sampling Distance ~5 cm (UAV typical)
DEFAULT_GRID_ROWS    = 5
DEFAULT_GRID_COLS    = 5
MIN_STRESS_REGION_PCT = 0.1         # % of image area

# ──────────────────────────────────────────────────────────────
# Weather / climate constants
# ──────────────────────────────────────────────────────────────

TEMP_STRESS_HIGH_C   = 35.0         # °C — heat stress threshold for paddy
TEMP_STRESS_LOW_C    = 15.0         # °C — cold stress threshold
RAINFALL_DROUGHT_MM  = 5.0          # mm/day below which drought risk activates
HUMIDITY_HIGH_PCT    = 85.0         # % — high humidity fungal risk threshold
WIND_SPEED_HIGH_MS   = 10.0         # m/s — lodging risk

# ──────────────────────────────────────────────────────────────
# Colour palettes
# ──────────────────────────────────────────────────────────────

STRESS_PALETTE = {
    "No Stress":       "#2ECC71",
    "Low Stress":      "#F1C40F",
    "Moderate Stress": "#E67E22",
    "High Stress":     "#E74C3C",
}

STAGE_PALETTE = {
    "Nursery":    "#AED6F1",
    "Vegetative": "#27AE60",
    "Flowering":  "#F39C12",
    "Mature":     "#8E44AD",
}

"""
indices.py
----------
Vegetation index computation for UAV multispectral imagery.
Supports NDVI, NDWI, NDRE, EVI, SAVI, MSAVI2, GNDVI, CIre.

Band convention (0-indexed):
    0 = Green
    1 = Red
    2 = Red Edge
    3 = NIR

All inputs are numpy arrays (H, W) of float32.
All outputs are numpy arrays (H, W) of float32, clipped to [-1, 1] unless noted.
"""

import numpy as np


EPS = 1e-8  # Avoid division by zero


# ──────────────────────────────────────────────────────────────
# Core indices
# ──────────────────────────────────────────────────────────────

def compute_ndvi(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
    """Normalized Difference Vegetation Index.
    NDVI = (NIR - Red) / (NIR + Red)
    Range: [-1, 1]. Healthy canopy: 0.6–0.9
    """
    return np.clip((nir - red) / (nir + red + EPS), -1, 1)


def compute_ndwi(green: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """Normalized Difference Water Index (Gao 1996).
    NDWI = (Green - NIR) / (Green + NIR)
    Positive values: open water / canopy water stress.
    """
    return np.clip((green - nir) / (green + nir + EPS), -1, 1)


def compute_ndre(nir: np.ndarray, red_edge: np.ndarray) -> np.ndarray:
    """Normalized Difference Red Edge Index.
    NDRE = (NIR - RedEdge) / (NIR + RedEdge)
    Sensitive to chlorophyll content and early stress.
    """
    return np.clip((nir - red_edge) / (nir + red_edge + EPS), -1, 1)


def compute_gndvi(nir: np.ndarray, green: np.ndarray) -> np.ndarray:
    """Green NDVI — sensitive to chlorophyll concentration.
    GNDVI = (NIR - Green) / (NIR + Green)
    """
    return np.clip((nir - green) / (nir + green + EPS), -1, 1)


def compute_evi(nir: np.ndarray, red: np.ndarray, blue: np.ndarray,
                G: float = 2.5, C1: float = 6.0, C2: float = 7.5, L: float = 1.0) -> np.ndarray:
    """Enhanced Vegetation Index (Huete 1997).
    EVI = G * (NIR - Red) / (NIR + C1*Red - C2*Blue + L)
    Reduces atmospheric and soil background noise vs NDVI.
    """
    denom = nir + C1 * red - C2 * blue + L + EPS
    return np.clip(G * (nir - red) / denom, -1, 1)


def compute_savi(nir: np.ndarray, red: np.ndarray, L: float = 0.5) -> np.ndarray:
    """Soil Adjusted Vegetation Index (Huete 1988).
    SAVI = (NIR - Red) / (NIR + Red + L) * (1 + L)
    L=0.5 is standard for intermediate canopy density.
    """
    return np.clip((nir - red) / (nir + red + L + EPS) * (1 + L), -1, 1)


def compute_msavi2(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
    """Modified SAVI-2 (Qi 1994) — self-adjusting soil correction.
    MSAVI2 = (2*NIR + 1 - sqrt((2*NIR+1)^2 - 8*(NIR-Red))) / 2
    """
    inner = np.maximum((2 * nir + 1) ** 2 - 8 * (nir - red), 0)
    return np.clip((2 * nir + 1 - np.sqrt(inner)) / 2, -1, 1)


def compute_cire(nir: np.ndarray, red_edge: np.ndarray) -> np.ndarray:
    """Chlorophyll Index Red Edge (Gitelson 2003).
    CIre = (NIR / RedEdge) - 1
    Strongly correlated with canopy chlorophyll content.
    Typical range: 0–5.
    """
    return np.clip((nir / (red_edge + EPS)) - 1, 0, 10)


# ──────────────────────────────────────────────────────────────
# Composite stress score
# ──────────────────────────────────────────────────────────────

def compute_stress_score(ndvi: np.ndarray, ndre: np.ndarray, ndwi: np.ndarray) -> np.ndarray:
    """
    Composite pixel-level stress index derived from three indices.
    Higher score → higher probability of crop stress.

    Formula (per-pixel):
        stress = 0.5*(1 - ndvi) + 0.3*(1 - ndre) + 0.2*ndwi_positive

    All components normalized to [0, 1].
    Returns float32 array in [0, 1].
    """
    ndvi_n  = np.clip((1 - ndvi) / 2, 0, 1)   # low NDVI = high stress
    ndre_n  = np.clip((1 - ndre) / 2, 0, 1)   # low NDRE = high stress
    ndwi_n  = np.clip(ndwi,           0, 1)    # positive NDWI = water stress / waterlogging

    score = 0.5 * ndvi_n + 0.3 * ndre_n + 0.2 * ndwi_n
    return score.astype(np.float32)


# ──────────────────────────────────────────────────────────────
# Batch pipeline helper
# ──────────────────────────────────────────────────────────────

def compute_all_indices(bands: dict) -> dict:
    """
    Compute all indices from a band dictionary.

    Parameters
    ----------
    bands : dict with keys 'green', 'red', 'red_edge', 'nir'
            (optionally 'blue').  Values are float32 (H, W) arrays.

    Returns
    -------
    dict with keys: ndvi, ndwi, ndre, gndvi, evi, savi, msavi2, cire, stress_score
    """
    g   = bands["green"].astype(np.float32)
    r   = bands["red"].astype(np.float32)
    re  = bands["red_edge"].astype(np.float32)
    nir = bands["nir"].astype(np.float32)
    b   = bands.get("blue", np.zeros_like(g))

    ndvi  = compute_ndvi(nir, r)
    ndwi  = compute_ndwi(g, nir)
    ndre  = compute_ndre(nir, re)

    return {
        "ndvi":         ndvi,
        "ndwi":         ndwi,
        "ndre":         ndre,
        "gndvi":        compute_gndvi(nir, g),
        "evi":          compute_evi(nir, r, b),
        "savi":         compute_savi(nir, r),
        "msavi2":       compute_msavi2(nir, r),
        "cire":         compute_cire(nir, re),
        "stress_score": compute_stress_score(ndvi, ndre, ndwi),
    }
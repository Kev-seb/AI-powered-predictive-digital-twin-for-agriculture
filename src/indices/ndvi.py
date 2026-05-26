"""
ndvi.py
--------
Normalized Difference Vegetation Index (NDVI)

Formula:
    NDVI = (NIR - Red) / (NIR + Red)

Used for:
- vegetation health
- canopy vigor
- crop stress detection
- photosynthetic activity
"""

import numpy as np

EPS = 1e-8


def compute_ndvi(
    nir: np.ndarray,
    red: np.ndarray
) -> np.ndarray:
    """
    Compute NDVI from NIR and Red bands.

    Parameters
    ----------
    nir : np.ndarray
        Near Infrared band

    red : np.ndarray
        Red band

    Returns
    -------
    np.ndarray
        NDVI image in range [-1, 1]
    """

    ndvi = (nir - red) / (nir + red + EPS)

    return np.clip(
        ndvi,
        -1.0,
        1.0
    ).astype(np.float32)
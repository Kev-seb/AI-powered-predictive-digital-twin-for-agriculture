"""
ndwi.py
--------
Normalized Difference Water Index (NDWI)

Formula:
    NDWI = (Green - NIR) / (Green + NIR)

Used for:
- water stress analysis
- irrigation monitoring
- canopy water content
"""

import numpy as np

EPS = 1e-8


def compute_ndwi(
    green: np.ndarray,
    nir: np.ndarray
) -> np.ndarray:

    ndwi = (
        (green - nir) /
        (green + nir + EPS)
    )

    return np.clip(
        ndwi,
        -1.0,
        1.0
    ).astype(np.float32)
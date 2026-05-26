"""
ndre.py
--------
Normalized Difference Red Edge Index (NDRE)

Formula:
    NDRE = (NIR - RedEdge) / (NIR + RedEdge)

Used for:
- chlorophyll estimation
- nitrogen stress
- early stress detection
"""

import numpy as np

EPS = 1e-8


def compute_ndre(
    nir: np.ndarray,
    red_edge: np.ndarray
) -> np.ndarray:

    ndre = (
        (nir - red_edge) /
        (nir + red_edge + EPS)
    )

    return np.clip(
        ndre,
        -1.0,
        1.0
    ).astype(np.float32)
"""
savi.py
--------
Soil Adjusted Vegetation Index (SAVI)

Useful for:
- sparse vegetation
- early crop stage
- soil correction
"""

import numpy as np

EPS = 1e-8


def compute_savi(
    nir: np.ndarray,
    red: np.ndarray,
    L: float = 0.5
) -> np.ndarray:

    savi = (
        ((nir - red) / (nir + red + L + EPS))
        * (1 + L)
    )

    return np.clip(
        savi,
        -1.0,
        1.0
    ).astype(np.float32)
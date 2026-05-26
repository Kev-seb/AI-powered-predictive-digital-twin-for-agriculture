"""
evi.py
-------
Enhanced Vegetation Index (EVI)

Used for:
- dense canopy analysis
- atmospheric correction
- soil noise reduction
"""

import numpy as np

EPS = 1e-8


def compute_evi(
    nir: np.ndarray,
    red: np.ndarray,
    blue: np.ndarray,
    G: float = 2.5,
    C1: float = 6.0,
    C2: float = 7.5,
    L: float = 1.0
) -> np.ndarray:

    denominator = (
        nir +
        C1 * red -
        C2 * blue +
        L +
        EPS
    )

    evi = (
        G * (nir - red)
    ) / denominator

    return np.clip(
        evi,
        -1.0,
        1.0
    ).astype(np.float32)
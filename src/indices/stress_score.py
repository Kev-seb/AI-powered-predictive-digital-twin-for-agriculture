"""
stress_score.py
----------------
Composite crop stress score.

Combines:
- NDVI
- NDRE
- NDWI

Higher score:
    Higher crop stress probability
"""

import numpy as np


def compute_stress_score(
    ndvi: np.ndarray,
    ndre: np.ndarray,
    ndwi: np.ndarray
) -> np.ndarray:

    # Normalize indices
    ndvi_n = np.clip((1 - ndvi) / 2, 0, 1)
    ndre_n = np.clip((1 - ndre) / 2, 0, 1)
    ndwi_n = np.clip(ndwi, 0, 1)

    stress = (
        0.5 * ndvi_n +
        0.3 * ndre_n +
        0.2 * ndwi_n
    )

    return stress.astype(np.float32)
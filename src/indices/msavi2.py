"""
msavi2.py
----------
Modified Soil Adjusted Vegetation Index 2

Used for:
- sparse vegetation
- strong soil background correction
"""

import numpy as np


def compute_msavi2(
    nir: np.ndarray,
    red: np.ndarray
) -> np.ndarray:

    inner = np.maximum(
        (2 * nir + 1) ** 2 -
        8 * (nir - red),
        0
    )

    msavi2 = (
        (2 * nir + 1 - np.sqrt(inner))
        / 2
    )

    return np.clip(
        msavi2,
        -1.0,
        1.0
    ).astype(np.float32)
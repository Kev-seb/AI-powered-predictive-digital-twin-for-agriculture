"""
cire.py
--------
Chlorophyll Index Red Edge (CIre)

Formula:
    CIre = (NIR / RedEdge) - 1

Used for:
- chlorophyll estimation
- nitrogen monitoring
"""

import numpy as np

EPS = 1e-8


def compute_cire(
    nir: np.ndarray,
    red_edge: np.ndarray
) -> np.ndarray:

    cire = (
        (nir / (red_edge + EPS))
        - 1
    )

    return np.clip(
        cire,
        0.0,
        10.0
    ).astype(np.float32)
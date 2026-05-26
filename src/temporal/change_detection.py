"""
change_detection.py
--------------------
Temporal change detection between multi-date UAV multispectral surveys.

Methods:
    - Pixel-wise NDVI difference maps
    - Image differencing with statistical thresholding (z-score)
    - Change Vector Analysis (CVA) for multi-band imagery
    - Binary change mask generation
    - Change magnitude and direction visualisation

Scientific basis:
    - CVA: Malila (1980), Lambin & Strahler (1994)
    - Threshold selection: Bruzzone & Prieto (2000) — EM-based adaptive threshold
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


# ──────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────

@dataclass
class ChangeResult:
    diff_map:       np.ndarray    # signed (H, W) float32 — positive = improvement
    magnitude_map:  np.ndarray    # absolute change magnitude (H, W) float32
    change_mask:    np.ndarray    # bool (H, W) — True = significant change
    pct_changed:    float         # % pixels flagged as changed
    mean_change:    float         # mean signed change over field
    max_magnitude:  float


# ──────────────────────────────────────────────────────────────
# Pixel-wise NDVI difference
# ──────────────────────────────────────────────────────────────

def ndvi_difference(ndvi_t1: np.ndarray, ndvi_t2: np.ndarray,
                    threshold: float = 0.10) -> ChangeResult:
    """
    Compute signed NDVI change (t2 - t1) and flag significant pixels.

    Parameters
    ----------
    ndvi_t1   : NDVI map at time 1  (H, W) float32 [-1, 1]
    ndvi_t2   : NDVI map at time 2  (H, W) float32 [-1, 1]
    threshold : absolute ΔNDVI above which change is flagged

    Returns
    -------
    ChangeResult
    """
    diff      = (ndvi_t2 - ndvi_t1).astype(np.float32)
    magnitude = np.abs(diff)
    mask      = magnitude > threshold

    return ChangeResult(
        diff_map=diff,
        magnitude_map=magnitude,
        change_mask=mask,
        pct_changed=float(mask.mean() * 100),
        mean_change=float(diff.mean()),
        max_magnitude=float(magnitude.max()),
    )


# ──────────────────────────────────────────────────────────────
# Z-score statistical thresholding
# ──────────────────────────────────────────────────────────────

def zscore_change_detection(diff_map: np.ndarray,
                             z_threshold: float = 2.0) -> np.ndarray:
    """
    Apply z-score thresholding to flag statistically significant changes.

    Parameters
    ----------
    diff_map    : signed (H, W) difference map
    z_threshold : number of standard deviations for change detection

    Returns
    -------
    np.ndarray (H, W) bool — True = significant change
    """
    mu    = float(diff_map.mean())
    sigma = float(diff_map.std())
    if sigma < 1e-8:
        return np.zeros_like(diff_map, dtype=bool)
    z_map = np.abs((diff_map - mu) / sigma)
    return z_map > z_threshold


# ──────────────────────────────────────────────────────────────
# Change Vector Analysis (multi-band)
# ──────────────────────────────────────────────────────────────

def change_vector_analysis(stack_t1: np.ndarray, stack_t2: np.ndarray,
                            threshold_pct: float = 90.0
                            ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Multi-band Change Vector Analysis.

    Computes per-pixel change magnitude and direction in band space.

    Parameters
    ----------
    stack_t1       : (C, H, W) float32 at time 1
    stack_t2       : (C, H, W) float32 at time 2
    threshold_pct  : percentile of magnitudes used as change threshold

    Returns
    -------
    magnitude_map  : (H, W) float32 — Euclidean distance in band space
    direction_map  : (H, W) float32 — angle (radians) of primary change axis
    change_mask    : (H, W) bool    — pixels above percentile threshold
    """
    delta     = (stack_t2 - stack_t1).astype(np.float32)              # (C, H, W)
    magnitude = np.linalg.norm(delta, axis=0)                         # (H, W)

    # Direction using first two bands as primary axes (proxy angle)
    direction = np.arctan2(delta[1], delta[0]).astype(np.float32)     # (H, W)

    threshold_val = np.percentile(magnitude, threshold_pct)
    change_mask   = magnitude > threshold_val

    return magnitude, direction, change_mask


# ──────────────────────────────────────────────────────────────
# Visualisation helpers
# ──────────────────────────────────────────────────────────────

def plot_change_maps(result: ChangeResult,
                     t1_label: str = "Date 1",
                     t2_label: str = "Date 2") -> 'plt.Figure':
    """Render signed change map and magnitude map side by side."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
    except ImportError:
        raise ImportError("matplotlib required for plot_change_maps()")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Signed difference (diverging colourmap)
    vabs = max(0.01, float(np.abs(result.diff_map).max()))
    im0  = axes[0].imshow(result.diff_map, cmap="RdYlGn", vmin=-vabs, vmax=vabs)
    plt.colorbar(im0, ax=axes[0])
    axes[0].set_title(f"ΔNDVI ({t2_label} − {t1_label})", fontweight="bold")
    axes[0].axis("off")

    # Magnitude map
    im1 = axes[1].imshow(result.magnitude_map, cmap="hot", vmin=0)
    plt.colorbar(im1, ax=axes[1])
    axes[1].set_title("Change Magnitude", fontweight="bold")
    axes[1].axis("off")

    # Binary change mask
    axes[2].imshow(result.change_mask, cmap="gray")
    axes[2].set_title(
        f"Change Mask  ({result.pct_changed:.1f}% changed)", fontweight="bold")
    axes[2].axis("off")

    fig.suptitle(
        f"Temporal Change Detection  |  Mean ΔNDVI: {result.mean_change:+.3f}",
        fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig

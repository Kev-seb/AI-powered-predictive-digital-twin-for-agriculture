"""
preprocessing.py
----------------
Multispectral image preprocessing pipeline.

Steps:
    1. Radiometric calibration (DN → reflectance via panel factor)
    2. Atmospheric correction proxy (Dark Object Subtraction)
    3. Per-band normalisation / standardisation
    4. Spatial resampling
    5. Patch extraction for deep-learning pipelines

Scientific basis:
    - MicaSense calibration guide (2020)
    - Chavez 1988 DOS atmospheric correction
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from src.core.utils import normalize_band, safe_divide


# ──────────────────────────────────────────────────────────────
# Radiometric calibration
# ──────────────────────────────────────────────────────────────

def dn_to_reflectance(dn_stack: np.ndarray,
                      panel_reflectance: float = 0.50,
                      panel_mean_dn: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Convert raw DN to at-surface reflectance.

    Parameters
    ----------
    dn_stack          : (C, H, W) uint16 or float32 raw DN
    panel_reflectance : known reflectance of calibration panel (default 50 %)
    panel_mean_dn     : (C,) mean DN from panel ROI; band max used if None

    Returns
    -------
    np.ndarray  (C, H, W)  float32  in [0, 1]
    """
    dn = dn_stack.astype(np.float32)
    C = dn.shape[0]
    if panel_mean_dn is None:
        panel_mean_dn = np.array([dn[c].max() for c in range(C)], dtype=np.float32)

    pmdn = np.asarray(panel_mean_dn, dtype=np.float32).reshape(C, 1, 1)
    return np.clip((dn / pmdn) * panel_reflectance, 0.0, 1.0).astype(np.float32)


# ──────────────────────────────────────────────────────────────
# Dark Object Subtraction
# ──────────────────────────────────────────────────────────────

def dos_correction(reflectance: np.ndarray, dark_percentile: float = 1.0) -> np.ndarray:
    """Subtract per-band dark-object haze offset (Chavez 1988)."""
    corrected = np.zeros_like(reflectance)
    for c in range(reflectance.shape[0]):
        dark_val = np.percentile(reflectance[c], dark_percentile)
        corrected[c] = np.clip(reflectance[c] - dark_val, 0.0, 1.0)
    return corrected.astype(np.float32)


# ──────────────────────────────────────────────────────────────
# Normalisation helpers
# ──────────────────────────────────────────────────────────────

def per_band_normalise(stack: np.ndarray) -> np.ndarray:
    """Min-max normalise each band to [0, 1]."""
    out = np.zeros_like(stack, dtype=np.float32)
    for c in range(stack.shape[0]):
        out[c] = normalize_band(stack[c])
    return out


def per_band_standardise(stack: np.ndarray,
                          mean: Optional[Sequence[float]] = None,
                          std:  Optional[Sequence[float]] = None
                          ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Z-score standardise each band: (x - μ) / σ."""
    C = stack.shape[0]
    if mean is None:
        mean = np.array([stack[c].mean() for c in range(C)], dtype=np.float32)
    if std is None:
        std  = np.array([stack[c].std()  for c in range(C)], dtype=np.float32)
    mu  = np.asarray(mean, dtype=np.float32).reshape(C, 1, 1)
    sig = np.asarray(std,  dtype=np.float32).reshape(C, 1, 1)
    sig = np.where(sig < 1e-8, 1.0, sig)
    return (stack - mu) / sig, np.asarray(mean), np.asarray(std)


# ──────────────────────────────────────────────────────────────
# Spatial / masking helpers
# ──────────────────────────────────────────────────────────────

def resize_stack(stack: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Resize (C, H, W) stack to (C, target_h, target_w) via bilinear interp."""
    try:
        import cv2
        C = stack.shape[0]
        out = np.zeros((C, target_h, target_w), dtype=np.float32)
        for c in range(C):
            out[c] = cv2.resize(stack[c], (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        return out
    except ImportError:
        C, H, W = stack.shape
        out = np.zeros((C, target_h, target_w), dtype=np.float32)
        h_s, w_s = slice(0, min(H, target_h)), slice(0, min(W, target_w))
        out[:, h_s, w_s] = stack[:, h_s, w_s]
        return out


def create_valid_mask(stack: np.ndarray, nodata_value: float = 0.0) -> np.ndarray:
    """Return (H, W) bool mask — True = valid pixel across all bands."""
    valid = np.ones(stack.shape[1:], dtype=bool)
    for c in range(stack.shape[0]):
        valid &= (stack[c] != nodata_value)
        valid &= (stack[c] <= 1.0)
    return valid


def extract_patches(stack: np.ndarray, patch_size: int = 224,
                    stride: int = 112) -> list[tuple[np.ndarray, tuple[int, int]]]:
    """Sliding-window patch extraction → list of ((C, P, P) array, (y, x) origin)."""
    C, H, W = stack.shape
    patches = []
    for y in range(0, H - patch_size + 1, stride):
        for x in range(0, W - patch_size + 1, stride):
            patch = stack[:, y:y+patch_size, x:x+patch_size].astype(np.float32)
            patches.append((patch, (y, x)))
    return patches


def reconstruct_from_patches(patches: list[np.ndarray], origins: list[tuple[int, int]],
                             target_shape: tuple[int, int], patch_size: int = 224) -> np.ndarray:
    """Reconstruct full image from overlapping patches using linear blending."""
    H, W = target_shape
    reconstructed = np.zeros((H, W), dtype=np.float32)
    weights = np.zeros((H, W), dtype=np.float32)
    
    for patch, (y, x) in zip(patches, origins):
        if patch.ndim == 3:
            patch = patch[0]
        reconstructed[y:y+patch_size, x:x+patch_size] += patch
        weights[y:y+patch_size, x:x+patch_size] += 1.0
        
    weights = np.where(weights == 0, 1.0, weights)
    return reconstructed / weights


# ──────────────────────────────────────────────────────────────
# Full pipeline
# ──────────────────────────────────────────────────────────────

def preprocess_stack(dn_stack: np.ndarray,
                     panel_reflectance: float = 0.50,
                     panel_mean_dn: Optional[np.ndarray] = None,
                     apply_dos: bool = True,
                     target_size: Optional[tuple[int, int]] = None) -> np.ndarray:
    """
    DN → calibrated reflectance → DOS corrected → normalised (→ resized).

    Returns
    -------
    np.ndarray  (C, H, W)  float32  ready for index computation or DL models
    """
    ref = dn_to_reflectance(dn_stack, panel_reflectance, panel_mean_dn)
    if apply_dos:
        ref = dos_correction(ref)
    ref = per_band_normalise(ref)
    if target_size is not None:
        ref = resize_stack(ref, *target_size)
    return ref

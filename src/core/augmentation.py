"""
augmentation.py
---------------
Multispectral data augmentation for training deep-learning models.

Supports 4-channel (Green, Red, RedEdge, NIR) stacks.
All transforms are applied consistently across channels.

Augmentations:
    - Random horizontal / vertical flip
    - Random rotation (90° multiples)
    - Random brightness / contrast jitter
    - Gaussian noise injection
    - Random crop and resize
    - Spectral jitter (per-band gain / offset)
    - CutOut regularisation
"""

from __future__ import annotations

import random
from typing import Optional

import numpy as np


# ──────────────────────────────────────────────────────────────
# Base transform protocol
# ──────────────────────────────────────────────────────────────

class Transform:
    """Base class — subclasses implement __call__(stack) → stack."""
    def __call__(self, stack: np.ndarray) -> np.ndarray:
        raise NotImplementedError


# ──────────────────────────────────────────────────────────────
# Geometric transforms  (applied identically to all channels)
# ──────────────────────────────────────────────────────────────

class RandomHorizontalFlip(Transform):
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, stack: np.ndarray) -> np.ndarray:
        if random.random() < self.p:
            return stack[:, :, ::-1].copy()
        return stack


class RandomVerticalFlip(Transform):
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, stack: np.ndarray) -> np.ndarray:
        if random.random() < self.p:
            return stack[:, ::-1, :].copy()
        return stack


class RandomRotate90(Transform):
    """Randomly rotate by 0, 90, 180, or 270 degrees."""
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, stack: np.ndarray) -> np.ndarray:
        if random.random() < self.p:
            k = random.randint(1, 3)
            return np.rot90(stack, k=k, axes=(1, 2)).copy()
        return stack


class RandomCropResize(Transform):
    """
    Random crop a sub-window then resize back to original size.

    Parameters
    ----------
    scale : (min_frac, max_frac) of original area to crop
    """
    def __init__(self, scale: tuple[float, float] = (0.7, 1.0), p: float = 0.5):
        self.scale = scale
        self.p = p

    def __call__(self, stack: np.ndarray) -> np.ndarray:
        if random.random() >= self.p:
            return stack
        _, H, W = stack.shape
        frac = random.uniform(*self.scale)
        ch, cw = int(H * frac), int(W * frac)
        y0 = random.randint(0, H - ch)
        x0 = random.randint(0, W - cw)
        cropped = stack[:, y0:y0+ch, x0:x0+cw]
        try:
            import cv2
            out = np.zeros_like(stack)
            for c in range(stack.shape[0]):
                out[c] = cv2.resize(cropped[c], (W, H), interpolation=cv2.INTER_LINEAR)
            return out
        except ImportError:
            return stack  # skip if cv2 unavailable


# ──────────────────────────────────────────────────────────────
# Radiometric / spectral transforms
# ──────────────────────────────────────────────────────────────

class RandomBrightnessContrast(Transform):
    """
    Adjust brightness (offset) and contrast (gain) independently per-image.

    out = clip(gain * x + offset, 0, 1)
    """
    def __init__(self, brightness_limit: float = 0.15,
                 contrast_limit: float = 0.15, p: float = 0.5):
        self.brightness_limit = brightness_limit
        self.contrast_limit   = contrast_limit
        self.p = p

    def __call__(self, stack: np.ndarray) -> np.ndarray:
        if random.random() >= self.p:
            return stack
        gain   = 1.0 + random.uniform(-self.contrast_limit, self.contrast_limit)
        offset = random.uniform(-self.brightness_limit, self.brightness_limit)
        return np.clip(stack * gain + offset, 0.0, 1.0).astype(np.float32)


class SpectralJitter(Transform):
    """
    Per-band random gain and offset to simulate sensor variability.

    Simulates inter-flight calibration differences between UAV surveys.
    """
    def __init__(self, gain_std: float = 0.05, offset_std: float = 0.02, p: float = 0.5):
        self.gain_std   = gain_std
        self.offset_std = offset_std
        self.p = p

    def __call__(self, stack: np.ndarray) -> np.ndarray:
        if random.random() >= self.p:
            return stack
        C = stack.shape[0]
        gain   = 1.0 + np.random.normal(0, self.gain_std, (C, 1, 1)).astype(np.float32)
        offset = np.random.normal(0, self.offset_std, (C, 1, 1)).astype(np.float32)
        return np.clip(stack * gain + offset, 0.0, 1.0).astype(np.float32)


class GaussianNoise(Transform):
    """Add per-pixel Gaussian noise to simulate sensor read noise."""
    def __init__(self, sigma: float = 0.01, p: float = 0.3):
        self.sigma = sigma
        self.p = p

    def __call__(self, stack: np.ndarray) -> np.ndarray:
        if random.random() >= self.p:
            return stack
        noise = np.random.normal(0, self.sigma, stack.shape).astype(np.float32)
        return np.clip(stack + noise, 0.0, 1.0).astype(np.float32)


class CutOut(Transform):
    """
    CutOut regularisation — zero out a random rectangular patch.

    DeVries & Taylor (2017): https://arxiv.org/abs/1708.04552
    """
    def __init__(self, max_holes: int = 4, max_size_frac: float = 0.15, p: float = 0.5):
        self.max_holes    = max_holes
        self.max_size_frac = max_size_frac
        self.p = p

    def __call__(self, stack: np.ndarray) -> np.ndarray:
        if random.random() >= self.p:
            return stack
        _, H, W = stack.shape
        out = stack.copy()
        n_holes = random.randint(1, self.max_holes)
        for _ in range(n_holes):
            ph = random.randint(1, max(1, int(H * self.max_size_frac)))
            pw = random.randint(1, max(1, int(W * self.max_size_frac)))
            y  = random.randint(0, H - ph)
            x  = random.randint(0, W - pw)
            out[:, y:y+ph, x:x+pw] = 0.0
        return out


# ──────────────────────────────────────────────────────────────
# Compose
# ──────────────────────────────────────────────────────────────

class Compose(Transform):
    """Apply a list of transforms in sequence."""
    def __init__(self, transforms: list[Transform]):
        self.transforms = transforms

    def __call__(self, stack: np.ndarray) -> np.ndarray:
        for t in self.transforms:
            stack = t(stack)
        return stack


# ──────────────────────────────────────────────────────────────
# Preset pipelines
# ──────────────────────────────────────────────────────────────

def get_train_augmentation(strong: bool = False) -> Compose:
    """
    Return a Compose pipeline suitable for training.

    Parameters
    ----------
    strong : enable stronger augmentation (CutOut + SpectralJitter)
    """
    transforms: list[Transform] = [
        RandomHorizontalFlip(p=0.5),
        RandomVerticalFlip(p=0.5),
        RandomRotate90(p=0.5),
        RandomBrightnessContrast(p=0.5),
        GaussianNoise(sigma=0.01, p=0.3),
    ]
    if strong:
        transforms += [
            SpectralJitter(p=0.4),
            CutOut(max_holes=4, p=0.4),
            RandomCropResize(scale=(0.75, 1.0), p=0.3),
        ]
    return Compose(transforms)


def get_val_augmentation() -> Compose:
    """No-op pipeline for validation (identity transform)."""
    return Compose([])

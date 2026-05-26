"""
utils.py
--------
General-purpose utility helpers shared across the UAV Crop Stress
Intelligence codebase.

Includes:
    - Logging setup (loguru)
    - Band normalization
    - Image saving / loading helpers
    - Timer context manager
    - Colour-map application
    - Safe numpy operations
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Union

import numpy as np

try:
    from loguru import logger
    _HAS_LOGURU = True
except ImportError:
    import logging as _logging
    logger = _logging.getLogger(__name__)  # type: ignore[assignment]
    _HAS_LOGURU = False


# ──────────────────────────────────────────────────────────────
# Logging helpers
# ──────────────────────────────────────────────────────────────

def setup_logger(log_dir: Optional[Union[str, Path]] = None,
                 level: str = "INFO") -> None:
    """
    Configure loguru to write to stderr + rotating file.

    Parameters
    ----------
    log_dir : directory for log files; skipped if None
    level   : minimum log level string
    """
    if not _HAS_LOGURU:
        return

    import sys
    logger.remove()
    logger.add(sys.stderr, level=level,
               format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}")

    if log_dir:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        logger.add(
            log_path / "crop_stress_{time:YYYY-MM-DD}.log",
            level=level,
            rotation="10 MB",
            retention="14 days",
            compression="zip",
        )
        logger.info(f"Log file → {log_path}")


# ──────────────────────────────────────────────────────────────
# Timer context manager
# ──────────────────────────────────────────────────────────────

@contextmanager
def timer(label: str = ""):
    """Simple wall-clock timer context manager."""
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        logger.debug(f"[TIMER] {label}: {elapsed:.3f}s")


# ──────────────────────────────────────────────────────────────
# Array helpers
# ──────────────────────────────────────────────────────────────

def safe_divide(numerator: np.ndarray, denominator: np.ndarray,
                fill: float = 0.0) -> np.ndarray:
    """Element-wise division, replacing 0/0 with `fill`."""
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(denominator != 0,
                          numerator / denominator,
                          fill).astype(np.float32)
    return result


def normalize_band(band: np.ndarray, lo: float = 0.0,
                   hi: float = 1.0) -> np.ndarray:
    """
    Min-max normalise a 2-D band array to [lo, hi].

    Returns float32.
    """
    band = band.astype(np.float32)
    bmin, bmax = band.min(), band.max()
    if bmax - bmin < 1e-8:
        return np.full_like(band, lo, dtype=np.float32)
    return (lo + (band - bmin) / (bmax - bmin) * (hi - lo)).astype(np.float32)


def clip_index(index: np.ndarray, lo: float = -1.0,
               hi: float = 1.0) -> np.ndarray:
    """Clip a vegetation index to [lo, hi] and return float32."""
    return np.clip(index, lo, hi).astype(np.float32)


def stack_bands(*bands: np.ndarray) -> np.ndarray:
    """
    Stack 2-D band arrays along a new channel axis.

    Returns
    -------
    np.ndarray  (C, H, W)  float32
    """
    return np.stack([b.astype(np.float32) for b in bands], axis=0)


# ──────────────────────────────────────────────────────────────
# Colour-map helper
# ──────────────────────────────────────────────────────────────

def apply_colormap(array: np.ndarray, cmap: str = "RdYlGn",
                   vmin: float = 0.0, vmax: float = 1.0) -> np.ndarray:
    """
    Apply a matplotlib colormap to a 2-D float array.

    Returns
    -------
    np.ndarray  (H, W, 3)  uint8 RGB
    """
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors

    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap_fn = cm.get_cmap(cmap)
    rgba = cmap_fn(norm(array))            # (H, W, 4)  float [0,1]
    rgb  = (rgba[:, :, :3] * 255).astype(np.uint8)
    return rgb


# ──────────────────────────────────────────────────────────────
# Image I/O helpers
# ──────────────────────────────────────────────────────────────

def save_array_as_png(array: np.ndarray, path: Union[str, Path],
                      cmap: Optional[str] = None) -> Path:
    """
    Save a 2-D float or 3-D uint8 array as PNG.

    Parameters
    ----------
    array : (H, W) float or (H, W, 3) uint8
    path  : output file path
    cmap  : if provided, apply colormap to 2-D float array first
    """
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if array.ndim == 2 and cmap:
        array = apply_colormap(array, cmap=cmap)

    if array.dtype != np.uint8:
        array = (normalize_band(array) * 255).astype(np.uint8)

    plt.imsave(str(path), array)
    logger.info(f"Saved array → {path}")
    return path


def compute_percentile_stretch(band: np.ndarray,
                               low: float = 2.0, high: float = 98.0) -> np.ndarray:
    """
    Percentile stretch for display: clip to [low, high] percentiles then
    normalise to [0, 1] float32.
    """
    lo_val = np.percentile(band, low)
    hi_val = np.percentile(band, high)
    stretched = np.clip(band, lo_val, hi_val)
    return normalize_band(stretched)


# ──────────────────────────────────────────────────────────────
# Metadata helpers
# ──────────────────────────────────────────────────────────────

def flatten_dict(d: dict, parent_key: str = "", sep: str = ".") -> dict:
    """Recursively flatten a nested dict with dotted keys."""
    items: list = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def seed_everything(seed: int = 42) -> None:
    """Seed random number generators for reproducibility."""
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device_str: str = "cpu") -> torch.device:
    """Get the torch device based on availability and request."""
    import torch
    if device_str == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    elif device_str == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int, metric: float, path: Union[str, Path]) -> None:
    """Save model checkpoint."""
    import torch
    from pathlib import Path
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "metric": metric,
    }
    torch.save(state, path)
    logger.info(f"Checkpoint saved to {path} (epoch {epoch}, metric {metric:.4f})")


def format_metrics(metrics: dict, prefix: str = "") -> str:
    """Format dictionary metrics as a log string."""
    return " | ".join(f"{prefix}{k}={v:.4f}" if isinstance(v, float) else f"{prefix}{k}={v}" for k, v in metrics.items())

"""
multispectral_loader.py
-----------------------
Loads UAV multispectral GeoTIFF files and produces band arrays,
false-color composites, and georeferencing metadata.

Dataset band layout (1-indexed as stored in TIFF):
    Band 1 = Green
    Band 2 = Red
    Band 3 = Red Edge
    Band 4 = NIR

Requires: rasterio, numpy
Optional: geopandas (for GIS export)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import rasterio
    from rasterio.transform import Affine
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False
    print("[WARNING] rasterio not installed. GeoTIFF georeferencing disabled.")

try:
    import tifffile
    HAS_TIFFFILE = True
except ImportError:
    HAS_TIFFFILE = False


# ──────────────────────────────────────────────────────────────
# Band names & order
# ──────────────────────────────────────────────────────────────

BAND_ORDER = ["green", "red", "red_edge", "nir"]

BAND_DESCRIPTIONS = {
    "green":    "Green (550 nm) — chlorophyll / water body detection",
    "red":      "Red (670 nm)   — photosynthesis, NDVI denominator",
    "red_edge": "Red Edge (720 nm) — chlorophyll stress, NDRE",
    "nir":      "NIR (840 nm)   — biomass, vegetation vigour",
}


# ──────────────────────────────────────────────────────────────
# Core loader
# ──────────────────────────────────────────────────────────────

class MultispectralImage:
    """Container for a loaded multispectral UAV image."""

    def __init__(
        self,
        bands: dict[str, np.ndarray],
        crs: Optional[str] = None,
        transform: Optional[object] = None,
        source_path: Optional[str] = None,
    ):
        self.bands = bands                  # {name: float32 (H,W)}
        self.crs = crs
        self.transform = transform
        self.source_path = source_path
        h, w = next(iter(bands.values())).shape
        self.height = h
        self.width = w

    def __repr__(self):
        return (f"MultispectralImage(size={self.height}×{self.width}, "
                f"bands={list(self.bands.keys())}, crs={self.crs})")

    # ── band accessors ──────────────────────────────────────

    @property
    def green(self) -> np.ndarray:
        return self.bands["green"]

    @property
    def red(self) -> np.ndarray:
        return self.bands["red"]

    @property
    def red_edge(self) -> np.ndarray:
        return self.bands["red_edge"]

    @property
    def nir(self) -> np.ndarray:
        return self.bands["nir"]

    @property
    def data(self) -> np.ndarray:
        """Return stacked 4-band array (4, H, W)."""
        return np.stack([self.green, self.red, self.red_edge, self.nir], axis=0)

    # ── false-color composites ──────────────────────────────

    def false_color_cir(self) -> np.ndarray:
        """
        Color-Infrared (CIR) composite:
            R ← NIR  (reveals vegetation vigour)
            G ← Red
            B ← Green
        Returns uint8 (H, W, 3).
        Healthy vegetation appears bright red.
        """
        return self._stack_rgb(self.nir, self.red, self.green)

    def false_color_vegetation(self) -> np.ndarray:
        """
        Vegetation-stress composite:
            R ← NIR
            G ← Red Edge
            B ← Green
        Returns uint8 (H, W, 3).
        Stressed areas appear blue/cyan; healthy areas red.
        """
        return self._stack_rgb(self.nir, self.red_edge, self.green)

    def false_color_redge_emphasis(self) -> np.ndarray:
        """
        Red Edge emphasis composite:
            R ← Red Edge
            G ← NIR
            B ← Red
        Returns uint8 (H, W, 3).
        Useful for early chlorophyll stress detection.
        """
        return self._stack_rgb(self.red_edge, self.nir, self.red)

    @staticmethod
    def _stack_rgb(r: np.ndarray, g: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Normalise three bands to [0,255] uint8 and stack."""
        def norm(arr):
            arr = arr.astype(np.float32)
            lo, hi = np.percentile(arr, 2), np.percentile(arr, 98)
            if hi == lo:
                return np.zeros_like(arr, dtype=np.uint8)
            return np.clip((arr - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)

        return np.stack([norm(r), norm(g), norm(b)], axis=-1)

    def rgb_preview(self) -> np.ndarray:
        """Pseudo-RGB from Red / Green (no blue channel). Returns uint8 (H,W,3)."""
        return self._stack_rgb(self.red, self.green, self.green)


# ──────────────────────────────────────────────────────────────
# Loaders
# ──────────────────────────────────────────────────────────────

def load_multispectral_tiff(path: str | Path) -> MultispectralImage:
    """
    Load a 4-band multispectral GeoTIFF.
    Tries rasterio first (preserves CRS/geotransform), falls back to tifffile.

    Parameters
    ----------
    path : path to .tif file

    Returns
    -------
    MultispectralImage
    """
    path = str(path)

    if HAS_RASTERIO:
        return _load_with_rasterio(path)
    elif HAS_TIFFFILE:
        return _load_with_tifffile(path)
    else:
        raise ImportError("Install rasterio or tifffile: pip install rasterio tifffile")


def _load_with_rasterio(path: str) -> MultispectralImage:
    with rasterio.open(path) as src:
        n_bands = src.count
        if n_bands < 4:
            raise ValueError(f"Expected ≥4 bands, got {n_bands} in {path}")

        data = src.read().astype(np.float32)  # (B, H, W)
        # Percentile-clip reflectance artefacts
        for i in range(data.shape[0]):
            p2, p98 = np.percentile(data[i], 2), np.percentile(data[i], 98)
            data[i] = np.clip(data[i], p2, p98)
            if p98 > p2:
                data[i] = (data[i] - p2) / (p98 - p2)

        bands = {
            "green":    data[0],
            "red":      data[1],
            "red_edge": data[2],
            "nir":      data[3],
        }
        crs = str(src.crs) if src.crs else None
        transform = src.transform

    return MultispectralImage(bands, crs=crs, transform=transform, source_path=path)


def _load_with_tifffile(path: str) -> MultispectralImage:
    import tifffile
    data = tifffile.imread(path).astype(np.float32)

    # Handle (H,W,B) or (B,H,W)
    if data.ndim == 3 and data.shape[2] <= 8:
        data = data.transpose(2, 0, 1)  # → (B, H, W)
    if data.ndim == 2:
        raise ValueError("Single-band image; expected 4-band multispectral.")

    for i in range(data.shape[0]):
        p2, p98 = np.percentile(data[i], 2), np.percentile(data[i], 98)
        if p98 > p2:
            data[i] = np.clip((data[i] - p2) / (p98 - p2), 0, 1)

    bands = {
        "green":    data[0],
        "red":      data[1],
        "red_edge": data[2],
        "nir":      data[3],
    }
    return MultispectralImage(bands, source_path=path)


# ──────────────────────────────────────────────────────────────
# Multi-stage dataset loader
# ──────────────────────────────────────────────────────────────

CROP_STAGES = ["Nursery", "Vegetative", "Flowering", "Mature"]


def load_temporal_dataset(root_dir: str | Path) -> dict[str, list[MultispectralImage]]:
    """
    Load the multi-stage paddy dataset.

    Expected folder structure:
        root_dir/
            Nursery/      *.tif
            Vegetative/   *.tif
            Flowering/    *.tif
            Mature/       *.tif

    Returns
    -------
    dict mapping stage name → list of MultispectralImage
    """
    root_dir = Path(root_dir)
    dataset: dict[str, list[MultispectralImage]] = {}

    for stage in CROP_STAGES:
        stage_dir = root_dir / stage
        if not stage_dir.exists():
            print(f"[INFO] Stage directory not found: {stage_dir}")
            dataset[stage] = []
            continue

        tiffs = sorted(stage_dir.glob("*.tif")) + sorted(stage_dir.glob("*.tiff"))
        images = []
        for tif in tiffs:
            try:
                img = load_multispectral_tiff(tif)
                images.append(img)
            except Exception as e:
                print(f"[WARN] Skipping {tif.name}: {e}")
        dataset[stage] = images
        print(f"[INFO] Loaded {len(images)} images for stage: {stage}")

    return dataset
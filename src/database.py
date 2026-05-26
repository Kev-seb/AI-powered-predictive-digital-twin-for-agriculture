"""
dataset.py
----------
PyTorch Dataset for the UAV Multispectral & RGB Multi-Stage Paddy Dataset.

Supports:
    - MultispectralStageDataset  : stage classification (Nursery/Vegetative/Flowering/Mature)
    - MultispectralStressDataset : stress level classification (computed from indices)
    - TemporalPatchDataset       : paired patches across time for temporal modelling

Usage:
    ds = MultispectralStageDataset(root="data/paddy", split="train")
    loader = DataLoader(ds, batch_size=8, shuffle=True, num_workers=4)

Expected directory structure:
    data/paddy/
        Nursery/
            img_001.tif   (4-band multispectral)
            img_002.tif
            ...
        Vegetative/
        Flowering/
        Mature/
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Optional, Callable

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split

from src.core.multispectral_loader import load_multispectral_tiff, CROP_STAGES
from src.indices.indices import compute_all_indices


# ──────────────────────────────────────────────────────────────
# Label maps
# ──────────────────────────────────────────────────────────────

STAGE_TO_IDX = {s: i for i, s in enumerate(CROP_STAGES)}
IDX_TO_STAGE = {i: s for s, i in STAGE_TO_IDX.items()}

# Stress labels derived from index thresholds
STRESS_TO_IDX = {"No Stress": 0, "Low": 1, "Moderate": 2, "High": 3}
IDX_TO_STRESS = {i: s for s, i in STRESS_TO_IDX.items()}


def _ndvi_to_stress_label(ndvi_mean: float) -> int:
    """Convert field-level mean NDVI to stress class index."""
    if ndvi_mean >= 0.60:
        return 0   # No stress
    elif ndvi_mean >= 0.40:
        return 1   # Low
    elif ndvi_mean >= 0.20:
        return 2   # Moderate
    else:
        return 3   # High


# ──────────────────────────────────────────────────────────────
# Transforms
# ──────────────────────────────────────────────────────────────

def random_flip(tensor: torch.Tensor) -> torch.Tensor:
    if random.random() > 0.5:
        tensor = torch.flip(tensor, dims=[2])   # horizontal flip
    if random.random() > 0.5:
        tensor = torch.flip(tensor, dims=[1])   # vertical flip
    return tensor


def random_rotate90(tensor: torch.Tensor) -> torch.Tensor:
    k = random.randint(0, 3)
    return torch.rot90(tensor, k=k, dims=[1, 2])


def add_gaussian_noise(tensor: torch.Tensor, std: float = 0.02) -> torch.Tensor:
    return tensor + torch.randn_like(tensor) * std


def random_band_dropout(tensor: torch.Tensor, p: float = 0.1) -> torch.Tensor:
    """Randomly zero-out one band to improve robustness."""
    if random.random() < p:
        band = random.randint(0, tensor.shape[0] - 1)
        tensor = tensor.clone()
        tensor[band] = 0
    return tensor


DEFAULT_AUGMENTATIONS = [random_flip, random_rotate90]


# ──────────────────────────────────────────────────────────────
# Base multispectral dataset
# ──────────────────────────────────────────────────────────────

class _BaseMultispectralDataset(Dataset):
    """
    Base class: scans stage directories, loads TIFFs on-demand,
    converts to 4-channel float32 tensor (C, H, W).
    """

    def __init__(
        self,
        root: str | Path,
        patch_size: int = 224,
        add_indices: bool = False,
        augment: bool = True,
        transforms: Optional[list[Callable]] = None,
    ):
        self.root = Path(root)
        self.patch_size = patch_size
        self.add_indices = add_indices
        self.augment = augment
        self.transforms = transforms or (DEFAULT_AUGMENTATIONS if augment else [])

        self.samples: list[tuple[Path, int]] = []   # (tif_path, label)
        self._scan()

    def _scan(self):
        raise NotImplementedError

    def __len__(self):
        return len(self.samples)

    def _load_tensor(self, tif_path: Path) -> torch.Tensor:
        import cv2
        ms = load_multispectral_tiff(tif_path)

        if self.add_indices:
            idx = compute_all_indices(ms.bands)
            stack = [
                ms.bands["green"], ms.bands["red"],
                ms.bands["red_edge"], ms.bands["nir"],
                idx["ndvi"], idx["ndre"], idx["ndwi"],
            ]
        else:
            stack = [ms.bands["green"], ms.bands["red"],
                     ms.bands["red_edge"], ms.bands["nir"]]

        arr = np.stack(stack, axis=0).astype(np.float32)  # (C, H, W)

        # Resize to patch_size
        arr_resized = np.stack([
            cv2.resize(arr[c], (self.patch_size, self.patch_size),
                       interpolation=cv2.INTER_LINEAR)
            for c in range(arr.shape[0])
        ], axis=0)

        tensor = torch.from_numpy(arr_resized)
        for t in self.transforms:
            tensor = t(tensor)
        return tensor

    def __getitem__(self, idx_):
        path, label = self.samples[idx_]
        tensor = self._load_tensor(path)
        return tensor, label


# ──────────────────────────────────────────────────────────────
# Stage classification dataset
# ──────────────────────────────────────────────────────────────

class MultispectralStageDataset(_BaseMultispectralDataset):
    """
    Classifies each image by crop growth stage.
    Labels: Nursery=0, Vegetative=1, Flowering=2, Mature=3
    """

    def _scan(self):
        for stage in CROP_STAGES:
            stage_dir = self.root / stage
            if not stage_dir.exists():
                continue
            for tif in sorted(stage_dir.glob("*.tif")) + sorted(stage_dir.glob("*.tiff")):
                self.samples.append((tif, STAGE_TO_IDX[stage]))
        print(f"[StageDataset] {len(self.samples)} images across {len(CROP_STAGES)} stages.")


# ──────────────────────────────────────────────────────────────
# Stress level dataset
# ──────────────────────────────────────────────────────────────

class MultispectralStressDataset(_BaseMultispectralDataset):
    """
    Derives stress labels from NDVI computed per image.
    No manual annotations required — self-supervised from remote sensing.

    Labels:
        0 = No Stress  (NDVI ≥ 0.60)
        1 = Low        (0.40 – 0.60)
        2 = Moderate   (0.20 – 0.40)
        3 = High       (< 0.20)
    """

    def _scan(self):
        from src.indices.indices import compute_ndvi
        for stage in CROP_STAGES:
            stage_dir = self.root / stage
            if not stage_dir.exists():
                continue
            for tif in sorted(stage_dir.glob("*.tif")) + sorted(stage_dir.glob("*.tiff")):
                try:
                    ms = load_multispectral_tiff(tif)
                    ndvi = compute_ndvi(ms.nir, ms.red)
                    label = _ndvi_to_stress_label(float(ndvi.mean()))
                    self.samples.append((tif, label))
                except Exception as e:
                    print(f"[WARN] Skipping {tif.name}: {e}")
        print(f"[StressDataset] {len(self.samples)} labelled images.")


# ──────────────────────────────────────────────────────────────
# Segmentation dataset
# ──────────────────────────────────────────────────────────────

class MultispectralSegmentationDataset(Dataset):
    """
    Segmentation dataset using rule-based pseudo-labels from vegetation indices.
    No manual pixel annotations required.

    Returns:
        tensor   : (C, H, W) float32 multispectral input
        mask     : (H, W) int64 class mask (0–4)
    """

    def __init__(self, root: str | Path, patch_size: int = 512, augment: bool = True):
        self.root = Path(root)
        self.patch_size = patch_size
        self.augment = augment
        self.tif_paths: list[Path] = []

        for stage in CROP_STAGES:
            stage_dir = self.root / stage
            if stage_dir.exists():
                self.tif_paths.extend(
                    sorted(stage_dir.glob("*.tif")) + sorted(stage_dir.glob("*.tiff"))
                )
        print(f"[SegDataset] {len(self.tif_paths)} images found.")

    def __len__(self):
        return len(self.tif_paths)

    def __getitem__(self, i):
        import cv2
        from stress_segmentation import rule_based_stress_segmentation

        ms = load_multispectral_tiff(self.tif_paths[i])

        idx = compute_all_indices(ms.bands)
        stack = [ms.bands["green"], ms.bands["red"],
                 ms.bands["red_edge"], ms.bands["nir"],
                 idx["ndvi"], idx["ndre"], idx["ndwi"]]
        arr = np.stack(stack, axis=0).astype(np.float32)

        H, W = self.patch_size, self.patch_size
        arr_r = np.stack([
            cv2.resize(arr[c], (W, H), interpolation=cv2.INTER_LINEAR)
            for c in range(arr.shape[0])
        ], axis=0)

        mask = rule_based_stress_segmentation(ms.bands)
        mask_r = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)

        tensor = torch.from_numpy(arr_r)
        mask_t = torch.from_numpy(mask_r).long()

        if self.augment:
            for t in DEFAULT_AUGMENTATIONS:
                tensor = t(tensor)
                mask_t = t(mask_t.unsqueeze(0)).squeeze(0)

        return tensor, mask_t


# ──────────────────────────────────────────────────────────────
# DataLoader factory
# ──────────────────────────────────────────────────────────────

def get_loaders(
    dataset: Dataset,
    batch_size: int = 8,
    train_ratio: float = 0.80,
    num_workers: int = 4,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader]:
    """Split dataset and return train/val DataLoaders."""
    n_train = int(len(dataset) * train_ratio)
    n_val   = len(dataset) - n_train
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(seed)
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader
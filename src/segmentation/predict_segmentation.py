"""
Full-image inference for the segmentation model.
Handles arbitrary-size GeoTIFFs via sliding-window patching + weighted reconstruction.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import rasterio
from rasterio.transform import Affine
from loguru import logger

from src.config.config import settings
from src.core.multispectral_loader import load_multispectral_tiff, MultispectralImage
from src.core.preprocessing import extract_patches, reconstruct_from_patches
from src.segmentation.deeplabv3_model import MultispectralDeepLabV3Plus


class SegmentationPredictor:

    def __init__(self, model_path: str | Path, device: str | None = None):
        self.device    = device or settings.device
        self.cfg       = settings.segmentation
        self.model     = MultispectralDeepLabV3Plus.load(
            model_path,
            num_classes=self.cfg.num_classes,
            encoder_weights=None,
            device=self.device,
        )
        logger.info(f"SegmentationPredictor ready | device={self.device}")

    # ── Public ────────────────────────────────────────────────────────────────

    def predict_file(self, tiff_path: str | Path) -> np.ndarray:
        """
        Run segmentation on a GeoTIFF.
        Returns (H, W) int8 class map.
        """
        image = load_multispectral_tiff(tiff_path)
        return self.predict_image(image)

    def predict_image(self, image: MultispectralImage) -> np.ndarray:
        _, H, W = image.data.shape
        patch_size = self.cfg.patch_size
        stride     = patch_size // 2

        patches, origins = [], []
        for patch, origin in extract_patches(image.data, patch_size, stride):
            patches.append(patch)
            origins.append(origin)

        prob_maps = self._batch_predict(patches)

        # Reconstruct per-class probability maps
        full_probs = np.zeros(
            (self.cfg.num_classes, H, W), dtype=np.float32
        )
        for cls in range(self.cfg.num_classes):
            class_patches = [pm[cls] for pm in prob_maps]
            full_probs[cls] = reconstruct_from_patches(
                class_patches, origins, (H, W), patch_size
            )

        return full_probs.argmax(axis=0).astype(np.int8)

    def save_prediction(
        self,
        class_map:   np.ndarray,
        reference:   MultispectralImage,
        output_path: str | Path,
    ) -> None:
        """Save class map as single-band GeoTIFF with original georeferencing."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        profile = reference.profile.copy()
        profile.update(count=1, dtype="int8", nodata=-1)

        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(class_map[np.newaxis, :, :])

        logger.info(f"Segmentation mask saved → {output_path}")

    # ── Private ───────────────────────────────────────────────────────────────

    def _batch_predict(self, patches: list[np.ndarray]) -> list[np.ndarray]:
        """
        Run inference on a list of patches in mini-batches.
        Returns list of (num_classes, H, W) probability arrays.
        """
        all_probs = []
        bs = self.cfg.batch_size

        for i in range(0, len(patches), bs):
            batch = np.stack(patches[i:i+bs])          # (B, C, H, W)
            tensor = torch.from_numpy(batch).float().to(self.device)
            with torch.no_grad():
                logits = self.model(tensor)
                probs  = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.extend(list(probs))

        return all_probs
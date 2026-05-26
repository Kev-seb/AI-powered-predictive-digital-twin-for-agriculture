"""
stress_segmentation.py
-----------------------
DeepLabV3+ based semantic segmentation for crop stress mapping.

Segments each UAV image pixel into one of:
    0  background / water / soil
    1  healthy canopy      (NDVI > 0.6, NDRE > 0.4)
    2  mild stress         (NDVI 0.3–0.6)
    3  moderate stress     (NDVI 0.1–0.3)
    4  severe stress       (NDVI < 0.1 over vegetation)

Model input: 4-channel multispectral tensor (Green, Red, RedEdge, NIR)
             OR 7-channel (+ NDVI, NDRE, NDWI as engineered features)

GradCAM-based explainability is included via explain_prediction().
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.segmentation import deeplabv3_resnet50


# ──────────────────────────────────────────────────────────────
# Class definitions
# ──────────────────────────────────────────────────────────────

NUM_CLASSES = 5

CLASS_LABELS = {
    0: "Background / Non-vegetation",
    1: "Healthy Canopy",
    2: "Mild Stress",
    3: "Moderate Stress",
    4: "Severe Stress",
}

# Colour palette for overlay rendering (RGBA uint8)
CLASS_COLORS = np.array([
    [128, 128, 128, 160],   # 0 Background  – grey
    [ 34, 139,  34, 160],   # 1 Healthy     – forest green
    [255, 255,   0, 160],   # 2 Mild        – yellow
    [255, 140,   0, 160],   # 3 Moderate    – orange
    [220,  20,  60, 160],   # 4 Severe      – crimson
], dtype=np.uint8)


# ──────────────────────────────────────────────────────────────
# Model architecture
# ──────────────────────────────────────────────────────────────

class MultispectralDeepLabV3(nn.Module):
    """
    DeepLabV3+ adapted for multispectral input.

    in_channels: 4 (raw bands) or 7 (bands + computed indices)
    """

    def __init__(self, in_channels: int = 4, num_classes: int = NUM_CLASSES):
        super().__init__()
        # Load pretrained DeepLabV3+ backbone
        self.backbone = deeplabv3_resnet50(weights=None, num_classes=num_classes)

        # Replace first conv to accept in_channels (not 3)
        old_conv = self.backbone.backbone.conv1
        self.backbone.backbone.conv1 = nn.Conv2d(
            in_channels, old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )
        nn.init.kaiming_normal_(self.backbone.backbone.conv1.weight, mode="fan_out")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.backbone(x)["out"]          # (B, num_classes, H, W)
        return out


# ──────────────────────────────────────────────────────────────
# Preprocessing
# ──────────────────────────────────────────────────────────────

def bands_to_tensor(bands: dict, add_indices: bool = True,
                    size: tuple[int, int] = (512, 512)) -> torch.Tensor:
    """
    Convert band dict → model-ready float32 tensor (1, C, H, W).

    Parameters
    ----------
    bands       : dict with keys green, red, red_edge, nir
    add_indices : if True, appends NDVI/NDRE/NDWI as channels (C=7)
    size        : (H, W) to resize to

    Returns
    -------
    torch.Tensor (1, C, H, W)
    """
    from src.indices.indices import compute_all_indices
    import cv2

    stack = [bands["green"], bands["red"], bands["red_edge"], bands["nir"]]

    if add_indices:
        idx = compute_all_indices(bands)
        stack += [idx["ndvi"], idx["ndre"], idx["ndwi"]]

    arr = np.stack(stack, axis=0).astype(np.float32)  # (C, H, W)

    # Resize each channel
    resized = np.stack([
        cv2.resize(arr[c], (size[1], size[0]), interpolation=cv2.INTER_LINEAR)
        for c in range(arr.shape[0])
    ], axis=0)

    return torch.from_numpy(resized).unsqueeze(0)   # (1, C, H, W)


# ──────────────────────────────────────────────────────────────
# Index-based rule segmentation (no-model fallback)
# ──────────────────────────────────────────────────────────────

def rule_based_stress_segmentation(bands: dict) -> np.ndarray:
    """
    Physics-informed segmentation using vegetation index thresholds.
    No trained model required — scientifically grounded fallback.

    Returns integer mask (H, W) with class ids 0–4.

    Thresholds based on:
        Liu et al. 2012 — paddy NDVI thresholds
        Gitelson 2003   — Red Edge stress indicators
    """
    from src.indices.indices import compute_ndvi, compute_ndwi, compute_ndre

    nir = bands["nir"]
    red = bands["red"]
    re  = bands["red_edge"]
    grn = bands["green"]

    ndvi = compute_ndvi(nir, red)
    ndre = compute_ndre(nir, re)

    h, w = ndvi.shape
    mask = np.zeros((h, w), dtype=np.uint8)   # default: background

    # Vegetated pixels: NDVI threshold
    veg = ndvi > 0.15

    # Classify vegetated pixels
    mask[veg & (ndvi >= 0.6) & (ndre >= 0.3)] = 1   # Healthy
    mask[veg & (ndvi >= 0.3) & (ndvi < 0.6)]  = 2   # Mild stress
    mask[veg & (ndvi >= 0.1) & (ndvi < 0.3)]  = 3   # Moderate stress
    mask[veg & (ndvi < 0.1)]                   = 4   # Severe stress

    return mask


# ──────────────────────────────────────────────────────────────
# Colour overlay
# ──────────────────────────────────────────────────────────────

def mask_to_overlay(mask: np.ndarray, base_rgb: np.ndarray,
                    alpha: float = 0.5) -> np.ndarray:
    """
    Blend a class mask with an RGB image for visualisation.

    Parameters
    ----------
    mask     : int (H, W) class mask
    base_rgb : uint8 (H, W, 3) background image
    alpha    : blend ratio for class colours

    Returns
    -------
    uint8 (H, W, 3) blended image
    """
    overlay = base_rgb.copy().astype(np.float32)
    for cls_id, color in enumerate(CLASS_COLORS):
        where = mask == cls_id
        for c in range(3):
            overlay[where, c] = (
                alpha * color[c] + (1 - alpha) * base_rgb[where, c]
            )
    return np.clip(overlay, 0, 255).astype(np.uint8)


# ──────────────────────────────────────────────────────────────
# GradCAM explainability
# ──────────────────────────────────────────────────────────────

class GradCAM:
    """
    GradCAM attention map for the final convolutional layer
    of the DeepLabV3+ classifier head.

    Usage:
        cam = GradCAM(model)
        heatmap = cam(tensor_input, target_class=1)  # class 1 = healthy
    """

    def __init__(self, model: MultispectralDeepLabV3):
        self.model = model
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None
        self._register_hooks()

    def _register_hooks(self):
        target_layer = self.model.backbone.backbone.layer4[-1]

        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        target_layer.register_forward_hook(forward_hook)
        target_layer.register_full_backward_hook(backward_hook)

    def __call__(self, x: torch.Tensor, target_class: int = 1) -> np.ndarray:
        """
        Compute GradCAM heatmap.

        Returns
        -------
        np.ndarray (H, W) float32 in [0, 1]
        """
        self.model.eval()
        x = x.requires_grad_(True)

        logits = self.model(x)                              # (1, C, H, W)
        score = logits[0, target_class].sum()               # scalar for target class
        self.model.zero_grad()
        score.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(x.shape[2], x.shape[3]),
                            mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        # Normalise to [0, 1]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam.astype(np.float32)


# ──────────────────────────────────────────────────────────────
# Evaluation metrics
# ──────────────────────────────────────────────────────────────

def compute_iou(pred: np.ndarray, target: np.ndarray,
                num_classes: int = NUM_CLASSES) -> dict:
    """Per-class IoU and mean IoU."""
    iou_per_class = {}
    for cls in range(num_classes):
        tp = ((pred == cls) & (target == cls)).sum()
        fp = ((pred == cls) & (target != cls)).sum()
        fn = ((pred != cls) & (target == cls)).sum()
        denom = tp + fp + fn
        iou_per_class[CLASS_LABELS[cls]] = float(tp / denom) if denom > 0 else float("nan")
    valid = [v for v in iou_per_class.values() if not np.isnan(v)]
    iou_per_class["mIoU"] = float(np.mean(valid)) if valid else float("nan")
    return iou_per_class


def compute_dice(pred: np.ndarray, target: np.ndarray,
                 num_classes: int = NUM_CLASSES) -> dict:
    """Per-class Dice score."""
    dice_per_class = {}
    for cls in range(num_classes):
        tp = ((pred == cls) & (target == cls)).sum()
        fp = ((pred == cls) & (target != cls)).sum()
        fn = ((pred != cls) & (target == cls)).sum()
        denom = 2 * tp + fp + fn
        dice_per_class[CLASS_LABELS[cls]] = float(2 * tp / denom) if denom > 0 else float("nan")
    return dice_per_class
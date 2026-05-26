"""
gradcam_segmentation.py
------------------------
GradCAM and feature-level saliency maps for the DeepLabV3+ crop stress
segmentation model.

Provides:
    - SegmentationGradCAM  — class-specific GradCAM for semantic seg
    - integrated_gradients()  — integrated gradients attribution
    - overlay_segmentation_cam() — fuse CAM with predicted mask
    - class_activation_map()  — lightweight CAM without gradients
      (uses backbone feature norms as proxy)

Reference: Selvaraju et al. 2017 Grad-CAM, extended for dense prediction.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ──────────────────────────────────────────────────────────────
# Segmentation GradCAM
# ──────────────────────────────────────────────────────────────

class SegmentationGradCAM:
    """
    GradCAM adapted for semantic segmentation.

    Instead of a single classification score, we aggregate the gradient
    signal over pixels predicted as `target_class`.

    Parameters
    ----------
    model        : segmentation model with a `backbone` attribute exposing
                   intermediate feature layers
    target_layer : nn.Module to hook (e.g. model.backbone.layer4)
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model       = model
        self.activations: Optional[torch.Tensor] = None
        self.gradients:   Optional[torch.Tensor] = None
        self._hooks = [
            target_layer.register_forward_hook(self._save_activations),
            target_layer.register_full_backward_hook(self._save_gradients),
        ]

    def _save_activations(self, _m, _i, output):
        self.activations = output.detach()

    def _save_gradients(self, _m, _gi, grad_out):
        self.gradients = grad_out[0].detach()

    def generate(self, tensor: torch.Tensor,
                 target_class: int) -> np.ndarray:
        """
        Compute GradCAM for a specific class prediction.

        Parameters
        ----------
        tensor       : (1, C, H, W) input tensor
        target_class : segmentation class to explain

        Returns
        -------
        np.ndarray (H, W) float32 in [0, 1]
        """
        self.model.eval()
        tensor = tensor.clone().requires_grad_(True)
        logits = self.model(tensor)                    # (1, num_classes, H, W)

        # Aggregate over predicted target-class pixels
        class_score = logits[0, target_class].mean()
        self.model.zero_grad()
        class_score.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(tensor.shape[2], tensor.shape[3]),
                            mode="bilinear", align_corners=False)

        cam_np = cam.squeeze().cpu().numpy()
        cam_np = (cam_np - cam_np.min()) / (cam_np.max() - cam_np.min() + 1e-8)
        return cam_np.astype(np.float32)

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def __del__(self):
        self.remove_hooks()


# ──────────────────────────────────────────────────────────────
# Feature norm CAM  (no-gradient proxy — fast, memory efficient)
# ──────────────────────────────────────────────────────────────

@torch.no_grad()
def class_activation_map(model: nn.Module,
                          tensor: torch.Tensor,
                          feature_extractor_attr: str = "backbone") -> np.ndarray:
    """
    Lightweight proxy CAM using L2 norm of backbone feature maps.

    Does not require gradient computation — suitable for dashboards
    and real-time visualisation.

    Parameters
    ----------
    model                  : segmentation model
    tensor                 : (1, C, H, W) input
    feature_extractor_attr : attribute name that returns intermediate features

    Returns
    -------
    np.ndarray (H, W) float32 in [0, 1]
    """
    feats_list = []

    def _hook(_m, _i, output):
        feats_list.append(output.detach())

    # Hook the first conv-stage output for speed
    backbone = getattr(model, feature_extractor_attr, model)
    handle = None
    for module in backbone.modules():
        if isinstance(module, nn.Conv2d):
            handle = module.register_forward_hook(_hook)
            break

    _ = model(tensor)
    if handle:
        handle.remove()

    if not feats_list:
        return np.zeros((tensor.shape[2], tensor.shape[3]), dtype=np.float32)

    feat = feats_list[0]                             # (1, C_feat, h, w)
    cam  = feat.norm(dim=1, keepdim=True)            # (1, 1, h, w)
    cam  = F.interpolate(cam, size=(tensor.shape[2], tensor.shape[3]),
                         mode="bilinear", align_corners=False)
    cam_np = cam.squeeze().cpu().numpy()
    cam_np = (cam_np - cam_np.min()) / (cam_np.max() - cam_np.min() + 1e-8)
    return cam_np.astype(np.float32)


# ──────────────────────────────────────────────────────────────
# Overlay helpers
# ──────────────────────────────────────────────────────────────

def overlay_segmentation_cam(rgb_image: np.ndarray,
                              cam: np.ndarray,
                              seg_mask: Optional[np.ndarray] = None,
                              alpha: float = 0.40,
                              colormap: str = "plasma") -> np.ndarray:
    """
    Overlay GradCAM heatmap (and optionally segmentation contours) on RGB.

    Parameters
    ----------
    rgb_image : (H, W, 3) uint8
    cam       : (H, W) float32 in [0, 1]
    seg_mask  : (H, W) int — predicted segmentation mask (optional)
    alpha     : heatmap opacity
    colormap  : matplotlib colormap for the CAM

    Returns
    -------
    np.ndarray (H, W, 3) uint8
    """
    import matplotlib.cm as cm_module

    cmap_fn   = cm_module.get_cmap(colormap)
    heatmap   = (cmap_fn(cam)[:, :, :3] * 255).astype(np.uint8)

    rgb_f  = rgb_image.astype(np.float32) / 255.0
    heat_f = heatmap.astype(np.float32)  / 255.0
    overlay = np.clip(alpha * heat_f + (1 - alpha) * rgb_f, 0, 1)
    out     = (overlay * 255).astype(np.uint8)

    # Draw segmentation class boundaries in white
    if seg_mask is not None:
        try:
            from scipy import ndimage as ndi
            boundaries = np.zeros_like(seg_mask, dtype=bool)
            for cls_id in np.unique(seg_mask):
                cls_binary = (seg_mask == cls_id).astype(np.uint8)
                eroded     = ndi.binary_erosion(cls_binary)
                boundaries |= (cls_binary.astype(bool) & ~eroded)
            out[boundaries] = [255, 255, 255]
        except ImportError:
            pass   # skip boundary drawing if scipy unavailable

    return out


# ──────────────────────────────────────────────────────────────
# Integrated gradients (attribution-based explainability)
# ──────────────────────────────────────────────────────────────

def integrated_gradients(model: nn.Module,
                          tensor: torch.Tensor,
                          target_class: int,
                          steps: int = 50,
                          baseline: Optional[torch.Tensor] = None) -> np.ndarray:
    """
    Integrated Gradients pixel attribution for a segmentation class.

    Sundararajan et al. (2017) — "Axiomatic Attribution for Deep Networks"
    https://arxiv.org/abs/1703.01365

    Parameters
    ----------
    tensor       : (1, C, H, W) input image
    target_class : class index to explain
    steps        : number of interpolation steps (higher = more accurate)
    baseline     : reference image (zeros if None)

    Returns
    -------
    np.ndarray (H, W) float32  — attribution magnitude per pixel
    """
    if baseline is None:
        baseline = torch.zeros_like(tensor)

    interpolated = [
        baseline + float(k) / steps * (tensor - baseline)
        for k in range(steps + 1)
    ]
    interpolated = torch.cat(interpolated, dim=0).requires_grad_(True)   # (S+1, C, H, W)

    model.eval()
    logits = model(interpolated)                                           # (S+1, cls, H, W)
    score  = logits[:, target_class, :, :].mean()
    model.zero_grad()
    score.backward()

    grads = interpolated.grad.detach()                                     # (S+1, C, H, W)
    avg_grads = grads.mean(dim=0)                                          # (C, H, W)
    ig = (tensor.squeeze(0) - baseline.squeeze(0)) * avg_grads            # (C, H, W)
    ig_map = ig.abs().sum(dim=0).cpu().numpy()                             # (H, W)
    ig_map = (ig_map - ig_map.min()) / (ig_map.max() - ig_map.min() + 1e-8)
    return ig_map.astype(np.float32)

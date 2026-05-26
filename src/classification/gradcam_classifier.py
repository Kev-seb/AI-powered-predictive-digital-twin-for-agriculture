"""
gradcam_classifier.py
----------------------
Gradient-weighted Class Activation Mapping (GradCAM) for the
EfficientNet-based multispectral crop stress / stage classifiers.

Provides:
    - GradCAM class (hook-based, works on EfficientNet last MBConv block)
    - overlay_cam()  — superimpose heatmap on an RGB composite
    - batch_gradcam() — generate CAM for multiple samples

Reference: Selvaraju et al. (2017) "Grad-CAM: Visual Explanations from
           Deep Networks via Gradient-based Localization"
           https://arxiv.org/abs/1610.02391
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.classification.stress_classifier import MultispectralEfficientNet


# ──────────────────────────────────────────────────────────────
# GradCAM
# ──────────────────────────────────────────────────────────────

class GradCAM:
    """
    GradCAM implementation for MultispectralEfficientNet.

    Hooks are attached to the last convolutional block of the EfficientNet
    backbone (`features[-1]`).

    Usage
    -----
    >>> cam_gen = GradCAM(model)
    >>> cam = cam_gen.generate(tensor, target_class=2)   # (H, W) float32 [0,1]
    """

    def __init__(self, model: MultispectralEfficientNet):
        self.model       = model
        self.activations: Optional[torch.Tensor] = None
        self.gradients:   Optional[torch.Tensor] = None
        self._hooks: list = []

        target_layer = model.backbone.features[-1]
        self._hooks.append(
            target_layer.register_forward_hook(self._save_activations)
        )
        self._hooks.append(
            target_layer.register_full_backward_hook(self._save_gradients)
        )

    def _save_activations(self, _module, _inp, output):
        self.activations = output.detach()

    def _save_gradients(self, _module, _grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def generate(self, tensor: torch.Tensor,
                 target_class: Optional[int] = None) -> np.ndarray:
        """
        Compute GradCAM heatmap for a given input tensor.

        Parameters
        ----------
        tensor       : (1, C, H, W) float32 on same device as model
        target_class : class index to explain; predicted class used if None

        Returns
        -------
        np.ndarray  (H, W)  float32  in [0, 1]
        """
        self.model.eval()
        tensor = tensor.clone().requires_grad_(True)
        logits = self.model(tensor)

        if target_class is None:
            target_class = int(logits.argmax(dim=1).item())

        self.model.zero_grad()
        logits[0, target_class].backward()

        # Weights = global average pooled gradients
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)   # (1, C, 1, 1)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, h, w)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(tensor.shape[2], tensor.shape[3]),
                            mode="bilinear", align_corners=False)

        cam_np = cam.squeeze().cpu().numpy()
        cam_min, cam_max = cam_np.min(), cam_np.max()
        cam_np = (cam_np - cam_min) / (cam_max - cam_min + 1e-8)
        return cam_np.astype(np.float32)

    def remove_hooks(self) -> None:
        """Remove all registered forward/backward hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def __del__(self):
        self.remove_hooks()


# ──────────────────────────────────────────────────────────────
# Overlay helper
# ──────────────────────────────────────────────────────────────

def overlay_cam(rgb_image: np.ndarray, cam: np.ndarray,
                alpha: float = 0.45, colormap: str = "jet") -> np.ndarray:
    """
    Overlay a GradCAM heatmap on an RGB image.

    Parameters
    ----------
    rgb_image : (H, W, 3) uint8 RGB image (e.g. false-colour composite)
    cam       : (H, W) float32 in [0, 1]
    alpha     : blending weight for the heatmap overlay
    colormap  : matplotlib colormap name

    Returns
    -------
    np.ndarray  (H, W, 3)  uint8
    """
    import matplotlib.cm as cm_module
    import matplotlib.colors as mcolors

    cmap_fn = cm_module.get_cmap(colormap)
    heatmap = cmap_fn(cam)[:, :, :3]                          # (H, W, 3) float [0,1]
    heatmap_u8 = (heatmap * 255).astype(np.uint8)

    rgb_f    = rgb_image.astype(np.float32) / 255.0
    heat_f   = heatmap_u8.astype(np.float32) / 255.0

    overlay  = (alpha * heat_f + (1 - alpha) * rgb_f)
    return np.clip(overlay * 255, 0, 255).astype(np.uint8)


# ──────────────────────────────────────────────────────────────
# Batch helper
# ──────────────────────────────────────────────────────────────

def batch_gradcam(model: MultispectralEfficientNet,
                   tensors: list[torch.Tensor],
                   target_classes: Optional[list[int]] = None,
                   device: str = "cpu") -> list[np.ndarray]:
    """
    Generate GradCAM heatmaps for a list of input tensors.

    Parameters
    ----------
    model          : trained MultispectralEfficientNet
    tensors        : list of (1, C, H, W) float32 tensors
    target_classes : optional per-sample target class; None = use predictions
    device         : torch device string

    Returns
    -------
    list of (H, W) float32 CAM arrays
    """
    cam_gen = GradCAM(model.to(device))
    cams = []
    for i, t in enumerate(tensors):
        t = t.to(device)
        tc = target_classes[i] if target_classes else None
        cams.append(cam_gen.generate(t, target_class=tc))
    cam_gen.remove_hooks()
    return cams

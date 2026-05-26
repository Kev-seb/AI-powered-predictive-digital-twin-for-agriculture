"""
efficientnet_model.py
----------------------
Factory functions and weight utilities for EfficientNet-based multispectral
classifiers used in the UAV Crop Stress Intelligence pipeline.

Provides:
    - build_model()          — construct and optionally load weights
    - save_checkpoint()      — save full training checkpoint
    - load_checkpoint()      — restore model + optimizer state
    - count_parameters()     — human-readable parameter count
    - freeze_backbone()      — lock all layers except classifier head
    - unfreeze_backbone()    — unlock full model for fine-tuning
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn

from src.classification.stress_classifier import (
    MultispectralEfficientNet,
    STAGE_CLASSES,
    STRESS_CLASSES,
)


# ──────────────────────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────────────────────

def build_model(
    task:        str  = "stress",     # "stress" | "stage"
    in_channels: int  = 4,
    pretrained:  bool = True,
    weights_path: Optional[Union[str, Path]] = None,
    device:      str  = "cpu",
) -> MultispectralEfficientNet:
    """
    Build a MultispectralEfficientNet for the specified task.

    Parameters
    ----------
    task         : "stress" → 4-class stress level;  "stage" → 4-class growth stage
    in_channels  : number of input spectral bands (default 4)
    pretrained   : initialise backbone from ImageNet weights
    weights_path : optional path to a saved checkpoint (.pt)
    device       : torch device string

    Returns
    -------
    MultispectralEfficientNet  on the requested device
    """
    num_classes = len(STRESS_CLASSES) if task == "stress" else len(STAGE_CLASSES)
    model = MultispectralEfficientNet(
        num_classes=num_classes,
        in_channels=in_channels,
        pretrained=pretrained,
    )

    if weights_path is not None:
        ckpt = torch.load(str(weights_path), map_location="cpu")
        state_dict = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state_dict, strict=False)

    return model.to(device)


# ──────────────────────────────────────────────────────────────
# Checkpoint helpers
# ──────────────────────────────────────────────────────────────

def save_checkpoint(model: nn.Module,
                    optimizer: torch.optim.Optimizer,
                    epoch: int,
                    val_acc: float,
                    path: Union[str, Path]) -> None:
    """Save full training checkpoint (model + optimizer + metadata)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch":             epoch,
        "val_acc":           val_acc,
        "model_state_dict":  model.state_dict(),
        "optim_state_dict":  optimizer.state_dict(),
    }, str(path))


def load_checkpoint(model: nn.Module,
                    optimizer: Optional[torch.optim.Optimizer],
                    path: Union[str, Path],
                    device: str = "cpu") -> dict:
    """
    Restore model (and optionally optimizer) from a checkpoint.

    Returns
    -------
    dict with keys: epoch, val_acc
    """
    ckpt = torch.load(str(path), map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optim_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optim_state_dict"])
    return {"epoch": ckpt.get("epoch", 0), "val_acc": ckpt.get("val_acc", 0.0)}


# ──────────────────────────────────────────────────────────────
# Layer freeze / unfreeze
# ──────────────────────────────────────────────────────────────

def freeze_backbone(model: MultispectralEfficientNet) -> None:
    """Freeze all layers except the classifier head (transfer learning phase 1)."""
    for name, param in model.named_parameters():
        if "backbone.classifier" not in name:
            param.requires_grad_(False)


def unfreeze_backbone(model: MultispectralEfficientNet) -> None:
    """Unfreeze all layers for end-to-end fine-tuning (phase 2)."""
    for param in model.parameters():
        param.requires_grad_(True)


# ──────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────

def count_parameters(model: nn.Module) -> str:
    """Return human-readable trainable parameter count string."""
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return (
        f"Total params: {total:,}  |  "
        f"Trainable: {trainable:,}  ({100*trainable/max(total,1):.1f}%)"
    )


def get_device(prefer_gpu: bool = True) -> str:
    """Return the best available device string."""
    if prefer_gpu:
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    return "cpu"

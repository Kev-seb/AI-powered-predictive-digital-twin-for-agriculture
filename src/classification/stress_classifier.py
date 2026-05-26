"""
stress_classifier.py
---------------------
EfficientNet-B0 based crop stage and stress level classifier.

Two tasks:
    1. Crop Stage Classification  — Nursery / Vegetative / Flowering / Mature
    2. Stress Level Classification — None / Low / Medium / High

Both models accept 4-channel multispectral input (Green, Red, RedEdge, NIR).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


# ──────────────────────────────────────────────────────────────
# Label maps
# ──────────────────────────────────────────────────────────────

STAGE_CLASSES = {0: "Nursery", 1: "Vegetative", 2: "Flowering", 3: "Mature"}
STRESS_CLASSES = {0: "No Stress", 1: "Low Stress", 2: "Moderate Stress", 3: "High Stress"}


# ──────────────────────────────────────────────────────────────
# Architecture
# ──────────────────────────────────────────────────────────────

class MultispectralEfficientNet(nn.Module):
    """
    EfficientNet-B0 adapted for 4-channel multispectral input.

    Strategy: average the pretrained RGB weights across the 4 input channels
    (weight inflation / band expansion) — preserves ImageNet features.
    """

    def __init__(self, num_classes: int, in_channels: int = 4,
                 pretrained: bool = True):
        super().__init__()
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        self.backbone = efficientnet_b0(weights=weights)

        # Replace first stem conv
        old_stem = self.backbone.features[0][0]
        new_stem = nn.Conv2d(
            in_channels, old_stem.out_channels,
            kernel_size=old_stem.kernel_size,
            stride=old_stem.stride,
            padding=old_stem.padding,
            bias=False,
        )
        if pretrained:
            # Inflate weights: average pretrained RGB weights across new channels
            with torch.no_grad():
                new_stem.weight[:] = old_stem.weight.mean(dim=1, keepdim=True).repeat(
                    1, in_channels, 1, 1
                ) / in_channels * 3
        self.backbone.features[0][0] = new_stem

        # Replace classifier head
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


# ──────────────────────────────────────────────────────────────
# Training helpers
# ──────────────────────────────────────────────────────────────

def get_stage_classifier(pretrained: bool = True) -> MultispectralEfficientNet:
    return MultispectralEfficientNet(num_classes=len(STAGE_CLASSES), pretrained=pretrained)


def get_stress_classifier(pretrained: bool = True) -> MultispectralEfficientNet:
    return MultispectralEfficientNet(num_classes=len(STRESS_CLASSES), pretrained=pretrained)


def build_optimizer(model: nn.Module, lr: float = 1e-4, weight_decay: float = 1e-4):
    """AdamW with differential LR: lower LR for backbone, higher for head."""
    backbone_params = [p for name, p in model.named_parameters() if "backbone.classifier" not in name]
    head_params     = [p for name, p in model.named_parameters() if "backbone.classifier"     in name]
    return torch.optim.AdamW([
        {"params": backbone_params, "lr": lr * 0.1},
        {"params": head_params,     "lr": lr},
    ], weight_decay=weight_decay)


def build_loss():
    """Focal-style loss for imbalanced crop stage data."""
    return nn.CrossEntropyLoss(label_smoothing=0.1)


# ──────────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        correct    += (logits.argmax(1) == y).sum().item()
        total      += x.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
        correct    += (logits.argmax(1) == y).sum().item()
        total      += x.size(0)
    return total_loss / total, correct / total


# ──────────────────────────────────────────────────────────────
# Inference
# ──────────────────────────────────────────────────────────────

@torch.no_grad()
def predict_stage(model: MultispectralEfficientNet,
                  tensor: torch.Tensor,
                  device: str = "cpu") -> dict:
    """
    Predict crop growth stage probabilities.

    Parameters
    ----------
    tensor : (1, 4, H, W) float32 tensor from bands_to_tensor()

    Returns
    -------
    dict with keys: 'stage', 'confidence', 'probabilities'
    """
    model.eval()
    tensor = tensor.to(device)
    logits = model(tensor)
    probs  = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
    pred   = int(probs.argmax())
    return {
        "stage":        STAGE_CLASSES[pred],
        "confidence":   float(probs[pred]),
        "probabilities": {STAGE_CLASSES[i]: float(p) for i, p in enumerate(probs)},
    }


@torch.no_grad()
def predict_stress(model: MultispectralEfficientNet,
                   tensor: torch.Tensor,
                   device: str = "cpu") -> dict:
    """
    Predict crop stress level probabilities.
    """
    model.eval()
    tensor = tensor.to(device)
    logits = model(tensor)
    probs  = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
    pred   = int(probs.argmax())
    return {
        "stress_level": STRESS_CLASSES[pred],
        "confidence":   float(probs[pred]),
        "probabilities": {STRESS_CLASSES[i]: float(p) for i, p in enumerate(probs)},
    }


# ──────────────────────────────────────────────────────────────
# GradCAM for classifiers
# ──────────────────────────────────────────────────────────────

class ClassifierGradCAM:
    """
    GradCAM for EfficientNet classifier.
    Attaches to the last MBConv block's activation.
    """

    def __init__(self, model: MultispectralEfficientNet):
        self.model = model
        self.activations = None
        self.gradients = None

        # Hook into the last convolutional block
        target = model.backbone.features[-1]
        target.register_forward_hook(self._save_activation)
        target.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def generate(self, tensor: torch.Tensor, target_class: int) -> 'np.ndarray':
        import torch.nn.functional as F
        import numpy as np

        self.model.eval()
        tensor = tensor.requires_grad_(True)
        logits = self.model(tensor)
        score  = logits[0, target_class]
        self.model.zero_grad()
        score.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(tensor.shape[2], tensor.shape[3]),
                            mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam.astype(np.float32)
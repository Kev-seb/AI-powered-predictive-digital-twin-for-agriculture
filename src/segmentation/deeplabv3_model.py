"""
DeepLabV3+ semantic segmentation model for multispectral crop stress mapping.
Uses segmentation-models-pytorch with a custom 5-band input adapter.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from loguru import logger

from src.config.config import settings


class MultispectralDeepLabV3Plus(nn.Module):
    """
    DeepLabV3+ adapted for 5-band multispectral input.

    Architecture:
        Input Adapter (5 → 3 channels, learnable)
        → DeepLabV3+ (ResNet-50 encoder)
        → Segmentation head (num_classes channels)
    """

    def __init__(
        self,
        encoder_name:    str = "resnet50",
        encoder_weights: str = "imagenet",
        num_classes:     int = 5,
        in_channels:     int = 5,
    ):
        super().__init__()

        # Learnable band adapter: project 5 MS bands to 3 RGB-compatible channels
        self.band_adapter = nn.Sequential(
            nn.Conv2d(in_channels, 3, kernel_size=1, bias=False),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True),
        )

        self.backbone = smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=num_classes,
            activation=None,        # raw logits
        )

        self.num_classes = num_classes
        self._init_adapter()
        logger.info(
            f"MultispectralDeepLabV3Plus | encoder={encoder_name} | classes={num_classes}"
        )

    def _init_adapter(self) -> None:
        """Initialise band adapter so RGB channels map to standard Red/Green/Blue."""
        with torch.no_grad():
            weight = torch.zeros(3, 5, 1, 1)
            weight[0, 2] = 1.0   # R ← Red band
            weight[1, 1] = 1.0   # G ← Green band
            weight[2, 0] = 1.0   # B ← Blue band
            self.band_adapter[0].weight.copy_(weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, 5, H, W) float32 tensor

        Returns
        -------
        logits : (B, num_classes, H, W)
        """
        x = self.band_adapter(x)
        return self.backbone(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return softmax probabilities (B, num_classes, H, W)."""
        return torch.softmax(self.forward(x), dim=1)

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return argmax class map (B, H, W) in eval mode."""
        self.eval()
        return self.forward(x).argmax(dim=1)

    # ── Serialisation ─────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)
        logger.info(f"Segmentation model saved → {path}")

    @classmethod
    def load(
        cls,
        path:            str | Path,
        num_classes:     int = 5,
        encoder_name:    str = "resnet50",
        encoder_weights: Optional[str] = None,   # None = no pretrained weights
        device:          str = "cpu",
    ) -> "MultispectralDeepLabV3Plus":
        model = cls(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            num_classes=num_classes,
        )
        state = torch.load(str(path), map_location=device)
        model.load_state_dict(state)
        model.to(device)
        model.eval()
        logger.info(f"Segmentation model loaded ← {path}")
        return model


def build_model(cfg=None) -> MultispectralDeepLabV3Plus:
    """Build model from settings config."""
    cfg = cfg or settings.segmentation
    return MultispectralDeepLabV3Plus(
        encoder_name=cfg.encoder,
        encoder_weights=cfg.encoder_weights,
        num_classes=cfg.num_classes,
    )
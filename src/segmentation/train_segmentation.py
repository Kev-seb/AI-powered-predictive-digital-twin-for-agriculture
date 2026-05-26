"""
Training loop for DeepLabV3+ multispectral segmentation.
Supports mixed precision, cosine LR scheduling, and early stopping.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torch.cuda.amp import GradScaler, autocast
from loguru import logger
from tqdm import tqdm

from src.config.config import settings
from src.core.utils import seed_everything, get_device, save_checkpoint, format_metrics
from src.segmentation.deeplabv3_model import build_model
from src.segmentation.segmentation_metrics import SegmentationMetrics


# ─── Dataset stub ─────────────────────────────────────────────────────────────

class MultispectralSegDataset(Dataset):
    """
    Expects image patches as (C, H, W) float32 .npy files
    and corresponding mask files as (H, W) int64 .npy files.
    Both must share the same filename stem.
    """

    def __init__(self, image_dir: str | Path, mask_dir: str | Path, transform=None):
        self.image_paths = sorted(Path(image_dir).glob("*.npy"))
        self.mask_dir    = Path(mask_dir)
        self.transform   = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        import numpy as np
        img_path  = self.image_paths[idx]
        mask_path = self.mask_dir / img_path.name

        image = torch.from_numpy(np.load(img_path)).float()
        mask  = torch.from_numpy(np.load(mask_path)).long()

        if self.transform:
            # Apply any tensor-compatible transforms
            image, mask = self.transform(image, mask)

        return image, mask


# ─── Loss ─────────────────────────────────────────────────────────────────────

class CombinedLoss(nn.Module):
    """Cross-entropy + Dice loss combination."""

    def __init__(self, num_classes: int, alpha: float = 0.5):
        super().__init__()
        self.ce    = nn.CrossEntropyLoss(ignore_index=255)
        self.alpha = alpha
        self.num_classes = num_classes

    def dice_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred    = torch.softmax(pred, dim=1)
        one_hot = nn.functional.one_hot(target.clamp(0), self.num_classes)
        one_hot = one_hot.permute(0, 3, 1, 2).float()

        inter   = (pred * one_hot).sum(dim=(2, 3))
        union   = pred.sum(dim=(2, 3)) + one_hot.sum(dim=(2, 3))
        dice    = (2 * inter + 1e-6) / (union + 1e-6)
        return 1 - dice.mean()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce_l   = self.ce(pred, target)
        dice_l = self.dice_loss(pred, target)
        return self.alpha * ce_l + (1 - self.alpha) * dice_l


# ─── Trainer ──────────────────────────────────────────────────────────────────

class SegmentationTrainer:

    def __init__(
        self,
        image_dir:    str | Path,
        mask_dir:     str | Path,
        output_dir:   str | Path = settings.segmentation.checkpoint_dir,
        val_fraction: float = 0.15,
    ):
        seed_everything(settings.seed)
        self.device    = get_device(settings.device)
        self.cfg       = settings.segmentation
        self.out_dir   = Path(output_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        # Data
        dataset = MultispectralSegDataset(image_dir, mask_dir)
        val_n   = int(len(dataset) * val_fraction)
        train_n = len(dataset) - val_n
        train_ds, val_ds = random_split(dataset, [train_n, val_n])

        self.train_loader = DataLoader(
            train_ds, batch_size=self.cfg.batch_size,
            shuffle=True, num_workers=4, pin_memory=True,
        )
        self.val_loader = DataLoader(
            val_ds, batch_size=self.cfg.batch_size,
            shuffle=False, num_workers=4, pin_memory=True,
        )

        # Model, optimiser, scheduler, scaler
        self.model     = build_model().to(self.device)
        self.criterion = CombinedLoss(self.cfg.num_classes).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.cfg.learning_rate, weight_decay=1e-4
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.cfg.epochs, eta_min=1e-6
        )
        self.scaler    = GradScaler()
        self.metrics   = SegmentationMetrics(self.cfg.num_classes)
        self.best_iou  = 0.0

    def train(self) -> None:
        logger.info(f"Training segmentation for {self.cfg.epochs} epochs")
        for epoch in range(1, self.cfg.epochs + 1):
            train_loss = self._train_epoch(epoch)
            val_metrics = self._val_epoch(epoch)

            self.scheduler.step()

            logger.info(
                f"Epoch {epoch:03d}/{self.cfg.epochs} | "
                f"train_loss={train_loss:.4f} | "
                + format_metrics(val_metrics, prefix="val_")
            )

            if val_metrics["mean_iou"] > self.best_iou:
                self.best_iou = val_metrics["mean_iou"]
                save_checkpoint(
                    self.model, self.optimizer, epoch,
                    self.best_iou,
                    self.out_dir / "best_segmentation.pth",
                )

        logger.success(f"Training complete. Best mIoU: {self.best_iou:.4f}")

    def _train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        for images, masks in tqdm(self.train_loader, desc=f"Train {epoch}", leave=False):
            images = images.to(self.device)
            masks  = masks.to(self.device)

            self.optimizer.zero_grad()
            with autocast():
                logits = self.model(images)
                loss   = self.criterion(logits, masks)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def _val_epoch(self, epoch: int) -> dict:
        self.model.eval()
        self.metrics.reset()
        for images, masks in tqdm(self.val_loader, desc=f"Val   {epoch}", leave=False):
            images = images.to(self.device)
            masks  = masks.to(self.device)
            preds  = self.model(images).argmax(dim=1)
            self.metrics.update(preds.cpu(), masks.cpu())
        return self.metrics.compute()


# ─── CLI entry ────────────────────────────────────────────────────────────────

def main(image_dir: str, mask_dir: str, output_dir: str = "models/segmentation/checkpoints") -> None:
    trainer = SegmentationTrainer(image_dir, mask_dir, output_dir)
    trainer.train()


if __name__ == "__main__":
    import sys
    main(sys.argv[1], sys.argv[2])
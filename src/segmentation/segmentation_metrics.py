"""
Segmentation evaluation metrics: IoU, Dice, pixel accuracy.
"""

from __future__ import annotations

import numpy as np
import torch


class SegmentationMetrics:
    """Accumulates confusion matrix over batches, computes IoU/Dice/accuracy."""

    def __init__(self, num_classes: int, ignore_index: int = 255):
        self.num_classes  = num_classes
        self.ignore_index = ignore_index
        self.reset()

    def reset(self) -> None:
        self.confusion = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        pred   = pred.cpu().numpy().flatten()
        target = target.cpu().numpy().flatten()

        mask   = target != self.ignore_index
        pred   = pred[mask]
        target = target[mask]

        for t, p in zip(target, pred):
            if 0 <= t < self.num_classes and 0 <= p < self.num_classes:
                self.confusion[t, p] += 1

    def compute(self) -> dict[str, float]:
        cm = self.confusion.astype(np.float64)

        # Per-class IoU
        tp  = np.diag(cm)
        fn  = cm.sum(axis=1) - tp
        fp  = cm.sum(axis=0) - tp
        iou = np.where((tp + fn + fp) > 0, tp / (tp + fn + fp), np.nan)

        # Per-class Dice
        dice = np.where(
            (2 * tp + fn + fp) > 0,
            2 * tp / (2 * tp + fn + fp),
            np.nan,
        )

        # Pixel accuracy
        total   = cm.sum()
        correct = tp.sum()
        pixel_acc = correct / total if total > 0 else 0.0

        # Mean metrics (ignore NaN classes)
        mean_iou  = float(np.nanmean(iou))
        mean_dice = float(np.nanmean(dice))

        metrics = {
            "mean_iou":  mean_iou,
            "mean_dice": mean_dice,
            "pixel_acc": float(pixel_acc),
        }
        for c in range(self.num_classes):
            metrics[f"iou_class_{c}"]  = float(iou[c])  if not np.isnan(iou[c])  else 0.0
            metrics[f"dice_class_{c}"] = float(dice[c]) if not np.isnan(dice[c]) else 0.0

        return metrics
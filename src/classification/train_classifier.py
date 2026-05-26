"""
train_classifier.py
--------------------
Training script for the EfficientNet-B0 crop stress / growth-stage classifiers.

Features:
    - Two-phase training: freeze backbone → fine-tune full model
    - Learning-rate scheduler (CosineAnnealingLR)
    - Early stopping with best-model checkpointing
    - Live metric logging (loguru)
    - Optional TensorBoard / CSV logging

Usage (CLI)
-----------
    python -m src.classification.train_classifier \
        --task stress \
        --data_dir data/processed/classification \
        --model_dir models/classifiers \
        --epochs 50 \
        --device cuda
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure absolute imports resolve when running the script directly
project_root = str(Path(__file__).resolve().parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

try:
    from loguru import logger
except ImportError:
    import logging as logger  # type: ignore[assignment]

from src.classification.stress_classifier import (
    MultispectralEfficientNet,
    STAGE_CLASSES,
    STRESS_CLASSES,
    build_optimizer,
    build_loss,
    train_epoch,
    eval_epoch,
)
from src.classification.efficientnet_model import (
    build_model,
    save_checkpoint,
    freeze_backbone,
    unfreeze_backbone,
    count_parameters,
)
from src.classification.classifier_metrics import compute_metrics, plot_training_curves
from src.core.augmentation import get_train_augmentation, get_val_augmentation


# ──────────────────────────────────────────────────────────────
# Synthetic dataset (used when no real data directory is given)
# ──────────────────────────────────────────────────────────────

class SyntheticMultispectralDataset(Dataset):
    """
    Synthetic dataset for smoke-test training without real UAV imagery.

    Generates random (4, 224, 224) float32 patches with integer labels.
    """

    def __init__(self, n_samples: int, num_classes: int,
                 in_channels: int = 4, img_size: int = 224,
                 augment: bool = False):
        self.n       = n_samples
        self.C       = in_channels
        self.H       = img_size
        self.nc      = num_classes
        self.augment = augment
        self.aug_fn  = get_train_augmentation() if augment else get_val_augmentation()

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
        img   = np.random.rand(self.C, self.H, self.H).astype(np.float32)
        label = idx % self.nc
        img   = self.aug_fn(img)
        return torch.from_numpy(img), torch.tensor(label, dtype=torch.long)


class RealMultispectralDataset(Dataset):
    """Loads pre-extracted .npy patches from disk."""
    def __init__(self, data_dir: str, class_map: dict[int, str], augment: bool = False):
        self.data_dir = Path(data_dir)
        self.samples = []
        self.augment = augment
        self.aug_fn = get_train_augmentation() if augment else get_val_augmentation()

        # Map string labels to int
        rev_map = {v: k for k, v in class_map.items()}

        for class_idx, class_name in class_map.items():
            class_path = self.data_dir / class_name
            if not class_path.exists():
                continue
            for npy_file in class_path.glob("*.npy"):
                self.samples.append((str(npy_file), class_idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = np.load(path).astype(np.float32)
        img = self.aug_fn(img)
        return torch.from_numpy(img), torch.tensor(label, dtype=torch.long)


# ──────────────────────────────────────────────────────────────
# Training pipeline
# ──────────────────────────────────────────────────────────────

def train(
    task:         str   = "stress",
    data_dir:     Optional[str] = None,
    model_dir:    str   = "models/classifiers",
    epochs:       int   = 50,
    batch_size:   int   = 16,
    lr:           float = 1e-4,
    device:       str   = "cpu",
    freeze_epochs: int  = 10,
    pretrained:   bool  = True,
    n_synthetic:  int   = 256,
) -> MultispectralEfficientNet:
    """
    Full two-phase training pipeline.

    Phase 1 (freeze_epochs): backbone frozen, only head trained.
    Phase 2 (remaining):      full model fine-tuned at lower LR.

    Parameters
    ----------
    task          : "stress" or "stage"
    data_dir      : path to dataset root; uses synthetic data if None
    model_dir     : directory to save checkpoints
    epochs        : total training epochs
    batch_size    : samples per batch
    lr            : peak learning rate for classifier head
    device        : torch device
    freeze_epochs : number of head-only epochs before unfreezing
    pretrained    : use ImageNet backbone initialisation
    n_synthetic   : samples per class for synthetic dataset smoke test

    Returns
    -------
    Best-performing MultispectralEfficientNet
    """
    num_classes  = len(STRESS_CLASSES) if task == "stress" else len(STAGE_CLASSES)
    class_names  = list(STRESS_CLASSES.values()) if task == "stress" else list(STAGE_CLASSES.values())
    model_path   = Path(model_dir) / f"efficientnet_{task}_best.pt"

    logger.info(f"Training {task} classifier  |  classes={num_classes}  |  device={device}")

    # ── Dataset ─────────────────────────────────────────────
    if data_dir is None:
        logger.warning("No data_dir provided — using synthetic dataset for smoke test.")
        full_ds   = SyntheticMultispectralDataset(n_synthetic * num_classes, num_classes, augment=True)
    else:
        logger.info(f"Loading real dataset from {data_dir}")
        class_map = STRESS_CLASSES if task == "stress" else STAGE_CLASSES
        full_ds = RealMultispectralDataset(data_dir, class_map, augment=True)
        
    val_size  = max(1, int(len(full_ds) * 0.2))
    train_size = len(full_ds) - val_size
    train_ds, val_ds = random_split(full_ds, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=(device != "cpu"))
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)

    # ── Model ─────────────────────────────────────────────
    model = build_model(task=task, pretrained=pretrained, device=device)
    logger.info(count_parameters(model))

    criterion   = build_loss()
    optimizer   = build_optimizer(model, lr=lr)
    scheduler   = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # ── Phase 1: freeze backbone ──────────────────────────
    freeze_backbone(model)
    logger.info(f"Phase 1: head-only training for {freeze_epochs} epochs")

    best_val_acc  = 0.0
    train_losses, val_losses = [], []
    train_accs,   val_accs   = [], []

    for epoch in range(1, epochs + 1):
        # Phase 2 transition
        if epoch == freeze_epochs + 1:
            unfreeze_backbone(model)
            # Reset optimizer with lower LR for fine-tuning
            optimizer = build_optimizer(model, lr=lr * 0.1)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs - freeze_epochs)
            logger.info("Phase 2: full model fine-tuning")

        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        vl_loss, vl_acc = eval_epoch(model, val_loader,   criterion,             device)
        scheduler.step()

        train_losses.append(tr_loss); val_losses.append(vl_loss)
        train_accs.append(tr_acc);   val_accs.append(vl_acc)

        lr_now = optimizer.param_groups[-1]["lr"]
        logger.info(
            f"Epoch {epoch:03d}/{epochs}  "
            f"train_loss={tr_loss:.4f}  train_acc={tr_acc:.3f}  "
            f"val_loss={vl_loss:.4f}  val_acc={vl_acc:.3f}  lr={lr_now:.2e}"
        )

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            save_checkpoint(model, optimizer, epoch, vl_acc, model_path)
            logger.info(f"  Success: Checkpoint saved  (val_acc={vl_acc:.4f})")

    logger.info(f"Training complete. Best val_acc = {best_val_acc:.4f}")

    # ── Plot curves ───────────────────────────────────────
    try:
        fig = plot_training_curves(train_losses, val_losses, train_accs, val_accs,
                                   title=f"{task.capitalize()} Classifier Training")
        curve_path = Path(model_dir) / f"{task}_training_curves.png"
        curve_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(curve_path), dpi=150, bbox_inches="tight")
        logger.info(f"Training curve → {curve_path}")
    except Exception as e:
        logger.warning(f"Could not save training curves: {e}")

    return model


# ──────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────

def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="Train crop stress / stage classifier")
    p.add_argument("--task",          default="stress",  choices=["stress", "stage"])
    p.add_argument("--data_dir",      default=None,      help="Dataset root directory")
    p.add_argument("--model_dir",     default="models/classifiers")
    p.add_argument("--epochs",        type=int,   default=50)
    p.add_argument("--batch_size",    type=int,   default=16)
    p.add_argument("--lr",            type=float, default=1e-4)
    p.add_argument("--freeze_epochs", type=int,   default=10)
    p.add_argument("--device",        default="cpu")
    p.add_argument("--no_pretrained", action="store_true")
    return p.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    train(
        task=args.task,
        data_dir=args.data_dir,
        model_dir=args.model_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        freeze_epochs=args.freeze_epochs,
        pretrained=not args.no_pretrained,
    )

"""
predict_classifier.py
----------------------
Inference utilities for the trained EfficientNet crop stress and
growth-stage classifiers.

Provides:
    - predict_single()   — classify one multispectral patch
    - predict_batch()    — classify a list of patches efficiently
    - predict_from_path()— load image file and classify
    - sliding_window_predict() — full-field sliding-window inference
    - build_prediction_overlay() — visualise predictions on the field image
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import torch.nn.functional as F

from src.classification.stress_classifier import (
    MultispectralEfficientNet,
    STAGE_CLASSES,
    STRESS_CLASSES,
    predict_stage,
    predict_stress,
)
from src.classification.efficientnet_model import build_model, load_checkpoint
from src.core.preprocessing import preprocess_stack


# ──────────────────────────────────────────────────────────────
# Single-patch inference
# ──────────────────────────────────────────────────────────────

def predict_single(model: MultispectralEfficientNet,
                   patch: np.ndarray,
                   task: str = "stress",
                   device: str = "cpu") -> dict:
    """
    Classify a single multispectral patch.

    Parameters
    ----------
    model  : trained MultispectralEfficientNet
    patch  : (C, H, W) float32 normalised array
    task   : "stress" or "stage"
    device : torch device string

    Returns
    -------
    dict with keys: label, confidence, probabilities
    """
    tensor = torch.from_numpy(patch[np.newaxis]).to(device)   # (1, C, H, W)
    if task == "stress":
        return predict_stress(model, tensor, device=device)
    return predict_stage(model, tensor, device=device)


# ──────────────────────────────────────────────────────────────
# Batch inference
# ──────────────────────────────────────────────────────────────

@torch.no_grad()
def predict_batch(model: MultispectralEfficientNet,
                  patches: list[np.ndarray],
                  task: str = "stress",
                  batch_size: int = 32,
                  device: str = "cpu") -> list[dict]:
    """
    Run inference on a list of patches in mini-batches.

    Parameters
    ----------
    patches    : list of (C, H, W) float32 arrays
    task       : "stress" or "stage"
    batch_size : number of patches per forward pass

    Returns
    -------
    list of prediction dicts (one per patch)
    """
    class_map = STRESS_CLASSES if task == "stress" else STAGE_CLASSES
    model.eval()
    model.to(device)
    results = []

    for start in range(0, len(patches), batch_size):
        batch = patches[start:start + batch_size]
        tensor = torch.from_numpy(np.stack(batch, axis=0)).to(device)   # (B, C, H, W)
        logits = model(tensor)
        probs  = F.softmax(logits, dim=1).cpu().numpy()                 # (B, num_classes)

        for prob_row in probs:
            pred = int(prob_row.argmax())
            results.append({
                "label":        class_map[pred],
                "confidence":   float(prob_row[pred]),
                "probabilities": {class_map[i]: float(p)
                                   for i, p in enumerate(prob_row)},
            })

    return results


# ──────────────────────────────────────────────────────────────
# Sliding-window field inference
# ──────────────────────────────────────────────────────────────

@torch.no_grad()
def sliding_window_predict(model: MultispectralEfficientNet,
                            field_stack: np.ndarray,
                            task: str = "stress",
                            patch_size: int = 224,
                            stride: int = 112,
                            device: str = "cpu") -> np.ndarray:
    """
    Classify the entire field by sliding-window inference.

    Returns a probability-averaged label map (H, W) of integer class indices.

    Parameters
    ----------
    field_stack : (C, H, W) float32 full-field multispectral stack
    patch_size  : sliding window size in pixels
    stride      : step between windows

    Returns
    -------
    np.ndarray  (H, W)  int32  predicted class per pixel
    """
    from src.core.preprocessing import extract_patches

    C, H, W = field_stack.shape
    num_classes = len(STRESS_CLASSES) if task == "stress" else len(STAGE_CLASSES)
    class_map = STRESS_CLASSES if task == "stress" else STAGE_CLASSES

    # Accumulate probability scores
    score_map  = np.zeros((num_classes, H, W), dtype=np.float32)
    count_map  = np.zeros((H, W), dtype=np.float32)

    model.eval()
    model.to(device)

    positions, patches = [], []
    for y in range(0, H - patch_size + 1, stride):
        for x in range(0, W - patch_size + 1, stride):
            positions.append((y, x))
            patches.append(field_stack[:, y:y+patch_size, x:x+patch_size])

    batch_size = 16
    for b_start in range(0, len(patches), batch_size):
        b_patches = patches[b_start:b_start+batch_size]
        b_pos     = positions[b_start:b_start+batch_size]
        tensor = torch.from_numpy(np.stack(b_patches, axis=0)).to(device)
        probs  = F.softmax(model(tensor), dim=1).cpu().numpy()           # (B, C)

        for (y, x), prob_row in zip(b_pos, probs):
            score_map[:, y:y+patch_size, x:x+patch_size] += prob_row[:, None, None]
            count_map[y:y+patch_size, x:x+patch_size]    += 1.0

    count_map = np.where(count_map == 0, 1.0, count_map)
    score_map /= count_map
    label_map  = score_map.argmax(axis=0).astype(np.int32)
    return label_map


# ──────────────────────────────────────────────────────────────
# Prediction overlay visualisation
# ──────────────────────────────────────────────────────────────

def build_prediction_overlay(label_map: np.ndarray,
                              task: str = "stress",
                              alpha: float = 0.55) -> np.ndarray:
    """
    Convert an integer label map to an RGB colour overlay image.

    Parameters
    ----------
    label_map : (H, W) int32 class indices
    task      : "stress" or "stage"
    alpha     : opacity of colour layer

    Returns
    -------
    np.ndarray  (H, W, 3)  uint8
    """
    from src.config.constants import STRESS_PALETTE, STAGE_PALETTE
    import matplotlib.colors as mcolors

    palette = STRESS_PALETTE if task == "stress" else STAGE_PALETTE
    class_map = STRESS_CLASSES if task == "stress" else STAGE_CLASSES

    H, W = label_map.shape
    canvas = np.zeros((H, W, 3), dtype=np.uint8)

    for idx, name in class_map.items():
        color = palette.get(name, "#95A5A6")
        rgb   = tuple(int(c * 255) for c in mcolors.to_rgb(color))
        canvas[label_map == idx] = rgb

    return canvas


# ──────────────────────────────────────────────────────────────
# CLI — load model & run inference on a single stack
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run classifier inference")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--task",       default="stress", choices=["stress", "stage"])
    parser.add_argument("--device",     default="cpu")
    args = parser.parse_args()

    model = build_model(task=args.task, pretrained=False, device=args.device)
    load_checkpoint(model, None, args.model_path, device=args.device)

    # Smoke test with random patch
    dummy = np.random.rand(4, 224, 224).astype(np.float32)
    result = predict_single(model, dummy, task=args.task, device=args.device)
    print(result)

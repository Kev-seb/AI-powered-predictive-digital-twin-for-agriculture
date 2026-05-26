"""
classifier_metrics.py
----------------------
Evaluation metrics for the crop stress and growth-stage classifiers.

Computes:
    - Accuracy, Top-2 accuracy
    - Per-class Precision, Recall, F1
    - Weighted macro F1
    - Confusion matrix (absolute + normalised)
    - ROC-AUC (one-vs-rest, macro average)
    - Classification report (text)
    - Metric history plotting for training curves
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

try:
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
        precision_recall_fscore_support,
        roc_auc_score,
    )
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ──────────────────────────────────────────────────────────────
# Core metric computation
# ──────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray,
                    y_pred: np.ndarray,
                    class_names: Optional[Sequence[str]] = None) -> dict:
    """
    Compute a comprehensive set of classification metrics.

    Parameters
    ----------
    y_true       : (N,) integer class labels
    y_pred       : (N,) predicted class labels
    class_names  : optional list of class name strings

    Returns
    -------
    dict with keys: accuracy, f1_macro, f1_weighted, per_class, confusion_matrix
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if not HAS_SKLEARN:
        # Minimal numpy fallback
        acc = float((y_true == y_pred).mean())
        return {"accuracy": acc, "f1_macro": None, "f1_weighted": None,
                "per_class": {}, "confusion_matrix": None}

    acc       = float(accuracy_score(y_true, y_pred))
    f1_macro  = float(f1_score(y_true, y_pred, average="macro",  zero_division=0))
    f1_weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    classes = class_names or [str(i) for i in range(len(prec))]
    per_class = {
        cls: {"precision": float(p), "recall": float(r), "f1": float(f)}
        for cls, p, r, f in zip(classes, prec, rec, f1)
    }

    cm = confusion_matrix(y_true, y_pred)

    return {
        "accuracy":        acc,
        "f1_macro":        f1_macro,
        "f1_weighted":     f1_weighted,
        "per_class":       per_class,
        "confusion_matrix": cm,
    }


def compute_roc_auc(y_true: np.ndarray,
                    y_prob: np.ndarray) -> float:
    """
    Compute one-vs-rest macro-average ROC-AUC.

    Parameters
    ----------
    y_true : (N,) integer class labels
    y_prob : (N, C) softmax probabilities

    Returns
    -------
    float  ROC-AUC score (0.5 = random, 1.0 = perfect)
    """
    if not HAS_SKLEARN:
        return 0.0
    try:
        return float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"))
    except ValueError:
        return 0.0


def top_k_accuracy(y_true: np.ndarray, y_prob: np.ndarray, k: int = 2) -> float:
    """Top-k accuracy: correct if true label in top-k predicted probabilities."""
    top_k = np.argsort(y_prob, axis=1)[:, -k:]
    correct = sum(y_true[i] in top_k[i] for i in range(len(y_true)))
    return float(correct / len(y_true))


def print_classification_report(y_true: np.ndarray, y_pred: np.ndarray,
                                  class_names: Optional[Sequence[str]] = None) -> str:
    """Return sklearn classification report string."""
    if not HAS_SKLEARN:
        return "sklearn not installed."
    return classification_report(y_true, y_pred, target_names=class_names, zero_division=0)


# ──────────────────────────────────────────────────────────────
# Confusion matrix visualisation
# ──────────────────────────────────────────────────────────────

def plot_confusion_matrix(cm: np.ndarray,
                           class_names: Sequence[str],
                           normalise: bool = True,
                           title: str = "Confusion Matrix") -> 'plt.Figure':
    """
    Plot a confusion matrix heatmap.

    Parameters
    ----------
    cm          : (C, C) integer confusion matrix
    class_names : list of class labels
    normalise   : if True, show row-normalised values (recall per class)
    """
    if not HAS_MPL:
        raise ImportError("matplotlib required for plot_confusion_matrix()")

    if normalise:
        cm_display = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
        fmt = ".2f"
    else:
        cm_display = cm
        fmt = "d"

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm_display, interpolation="nearest", cmap="Blues",
                   vmin=0, vmax=1 if normalise else None)
    plt.colorbar(im, ax=ax)

    tick_marks = range(len(class_names))
    ax.set_xticks(tick_marks); ax.set_xticklabels(class_names, rotation=30, ha="right")
    ax.set_yticks(tick_marks); ax.set_yticklabels(class_names)

    thresh = cm_display.max() / 2.0
    for i in range(cm_display.shape[0]):
        for j in range(cm_display.shape[1]):
            val = f"{cm_display[i,j]:{fmt}}"
            ax.text(j, i, val, ha="center", va="center",
                    color="white" if cm_display[i,j] > thresh else "black", fontsize=9)

    ax.set_ylabel("True label", fontsize=11)
    ax.set_xlabel("Predicted label", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────
# Training curve visualisation
# ──────────────────────────────────────────────────────────────

def plot_training_curves(train_losses: list[float],
                          val_losses:   list[float],
                          train_accs:   list[float],
                          val_accs:     list[float],
                          title: str = "Training Curves") -> 'plt.Figure':
    """
    Plot loss and accuracy curves for a training run.

    Parameters
    ----------
    train_losses, val_losses : per-epoch loss values
    train_accs,   val_accs   : per-epoch accuracy values (0-1)

    Returns
    -------
    matplotlib Figure
    """
    if not HAS_MPL:
        raise ImportError("matplotlib required for plot_training_curves()")

    epochs = range(1, len(train_losses) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, train_losses, "b-o", label="Train", markersize=4)
    axes[0].plot(epochs, val_losses,   "r-o", label="Val",   markersize=4)
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss Curve"); axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, [a*100 for a in train_accs], "b-o", label="Train", markersize=4)
    axes[1].plot(epochs, [a*100 for a in val_accs],   "r-o", label="Val",   markersize=4)
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title("Accuracy Curve"); axes[1].legend(); axes[1].grid(alpha=0.3)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig

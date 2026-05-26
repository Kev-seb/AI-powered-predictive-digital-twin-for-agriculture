"""
temporal_analytics.py
----------------------
Temporal crop intelligence: compare vegetation indices and stress levels
across paddy growth stages (Nursery → Vegetative → Flowering → Mature).

Outputs:
    - per-stage index statistics (mean, std, p10, p25, p75, p90)
    - stress progression time-series
    - canopy cover estimates
    - temporal change detection
    - matplotlib figures for the Streamlit dashboard
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec

from src.indices.indices import compute_all_indices

warnings.filterwarnings("ignore")

STAGES = ["Nursery", "Vegetative", "Flowering", "Mature"]
STAGE_COLORS = {
    "Nursery":    "#90EE90",
    "Vegetative": "#32CD32",
    "Flowering":  "#FFD700",
    "Mature":     "#FF8C00",
}


# ──────────────────────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────────────────────

@dataclass
class StageStats:
    stage: str
    n_images: int
    ndvi_mean: float
    ndvi_std: float
    ndvi_p10: float
    ndvi_p90: float
    ndre_mean: float
    ndwi_mean: float
    gndvi_mean: float
    stress_mean: float
    stress_std: float
    canopy_cover_pct: float   # % pixels with NDVI > 0.3
    stress_area_pct: float    # % pixels with stress_score > 0.5


@dataclass
class TemporalReport:
    stage_stats: list[StageStats]
    summary_df: pd.DataFrame
    stress_progression: list[float]   # stress_mean per stage
    ndvi_progression:   list[float]


# ──────────────────────────────────────────────────────────────
# Core analytics
# ──────────────────────────────────────────────────────────────

def analyse_stage(stage: str, images: list) -> Optional[StageStats]:
    """
    Compute aggregate index statistics across all images in one stage.

    Parameters
    ----------
    stage  : stage name string
    images : list of MultispectralImage objects
    """
    if not images:
        return None

    ndvi_vals, ndre_vals, ndwi_vals, gndvi_vals, stress_vals = [], [], [], [], []

    for img in images:
        idx = compute_all_indices(img.bands)
        ndvi_vals.append(idx["ndvi"].ravel())
        ndre_vals.append(idx["ndre"].ravel())
        ndwi_vals.append(idx["ndwi"].ravel())
        gndvi_vals.append(idx["gndvi"].ravel())
        stress_vals.append(idx["stress_score"].ravel())

    ndvi_all   = np.concatenate(ndvi_vals)
    ndre_all   = np.concatenate(ndre_vals)
    ndwi_all   = np.concatenate(ndwi_vals)
    gndvi_all  = np.concatenate(gndvi_vals)
    stress_all = np.concatenate(stress_vals)

    return StageStats(
        stage=stage,
        n_images=len(images),
        ndvi_mean=float(ndvi_all.mean()),
        ndvi_std=float(ndvi_all.std()),
        ndvi_p10=float(np.percentile(ndvi_all, 10)),
        ndvi_p90=float(np.percentile(ndvi_all, 90)),
        ndre_mean=float(ndre_all.mean()),
        ndwi_mean=float(ndwi_all.mean()),
        gndvi_mean=float(gndvi_all.mean()),
        stress_mean=float(stress_all.mean()),
        stress_std=float(stress_all.std()),
        canopy_cover_pct=float((ndvi_all > 0.3).mean() * 100),
        stress_area_pct=float((stress_all > 0.5).mean() * 100),
    )


def build_temporal_report(dataset: dict[str, list]) -> TemporalReport:
    """
    Run analysis across all crop stages.

    Parameters
    ----------
    dataset : {stage_name: [MultispectralImage, ...]}

    Returns
    -------
    TemporalReport
    """
    stats_list = []
    for stage in STAGES:
        imgs = dataset.get(stage, [])
        s = analyse_stage(stage, imgs)
        if s is not None:
            stats_list.append(s)

    rows = [{
        "Stage":           s.stage,
        "N Images":        s.n_images,
        "NDVI Mean":       round(s.ndvi_mean, 4),
        "NDVI Std":        round(s.ndvi_std, 4),
        "NDRE Mean":       round(s.ndre_mean, 4),
        "NDWI Mean":       round(s.ndwi_mean, 4),
        "GNDVI Mean":      round(s.gndvi_mean, 4),
        "Stress Mean":     round(s.stress_mean, 4),
        "Canopy Cover %":  round(s.canopy_cover_pct, 2),
        "Stress Area %":   round(s.stress_area_pct, 2),
    } for s in stats_list]

    df = pd.DataFrame(rows)

    return TemporalReport(
        stage_stats=stats_list,
        summary_df=df,
        stress_progression=[s.stress_mean for s in stats_list],
        ndvi_progression=[s.ndvi_mean for s in stats_list],
    )


# ──────────────────────────────────────────────────────────────
# Plotting helpers
# ──────────────────────────────────────────────────────────────

def plot_ndvi_progression(report: TemporalReport) -> plt.Figure:
    """Line + shaded band chart of NDVI across growth stages."""
    stats = report.stage_stats
    stages = [s.stage for s in stats]
    means  = [s.ndvi_mean for s in stats]
    p10    = [s.ndvi_p10  for s in stats]
    p90    = [s.ndvi_p90  for s in stats]
    x = np.arange(len(stages))

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(x, means, "o-", color="#2ECC71", linewidth=2.5, markersize=8, label="NDVI Mean")
    ax.fill_between(x, p10, p90, alpha=0.2, color="#2ECC71", label="P10–P90 range")
    ax.axhline(0.3, linestyle="--", color="#E74C3C", alpha=0.6, label="Canopy threshold (0.3)")
    ax.axhline(0.6, linestyle="--", color="#27AE60", alpha=0.6, label="Healthy canopy (0.6)")
    ax.set_xticks(x)
    ax.set_xticklabels(stages, fontsize=11)
    ax.set_ylabel("NDVI", fontsize=12)
    ax.set_title("Temporal NDVI Progression — Paddy Growth Stages", fontsize=13, fontweight="bold")
    ax.set_ylim(-0.1, 1.05)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_stress_progression(report: TemporalReport) -> plt.Figure:
    """Bar chart of composite stress score across stages."""
    stats = report.stage_stats
    stages = [s.stage for s in stats]
    stress = [s.stress_mean for s in stats]
    colors = [STAGE_COLORS.get(s, "#888") for s in stages]

    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.bar(stages, stress, color=colors, edgecolor="white", linewidth=1.2)
    for bar, val in zip(bars, stress):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("Composite Stress Score [0–1]", fontsize=12)
    ax.set_title("Crop Stress Progression Across Growth Stages", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_multi_index_radar(report: TemporalReport) -> plt.Figure:
    """Radar / spider chart comparing index profiles across stages."""
    indices_keys = ["NDVI Mean", "NDRE Mean", "GNDVI Mean"]
    stats = report.stage_stats
    stages = [s.stage for s in stats]

    data = np.array([
        [s.ndvi_mean, s.ndre_mean, s.gndvi_mean]
        for s in stats
    ])
    # Normalize to [0,1] per index column for radar shape
    data_norm = (data - data.min(0)) / (data.max(0) - data.min(0) + 1e-8)

    angles = np.linspace(0, 2 * np.pi, len(indices_keys), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    for i, (stage, row) in enumerate(zip(stages, data_norm)):
        values = row.tolist() + row[:1].tolist()
        ax.plot(angles, values, "o-", label=stage, linewidth=2)
        ax.fill(angles, values, alpha=0.08)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(indices_keys, fontsize=11)
    ax.set_title("Multi-Index Vegetation Profile", fontsize=13, fontweight="bold", pad=15)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=9)
    fig.tight_layout()
    return fig


def plot_canopy_stress_area(report: TemporalReport) -> plt.Figure:
    """Grouped bar: canopy cover vs stressed area per stage."""
    stats = report.stage_stats
    stages = [s.stage for s in stats]
    canopy = [s.canopy_cover_pct for s in stats]
    stress_area = [s.stress_area_pct for s in stats]

    x = np.arange(len(stages))
    w = 0.35

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(x - w / 2, canopy, w, label="Canopy Cover %", color="#27AE60", alpha=0.85)
    ax.bar(x + w / 2, stress_area, w, label="Stressed Area %", color="#E74C3C", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(stages, fontsize=11)
    ax.set_ylabel("% of Image Area", fontsize=12)
    ax.set_title("Canopy Cover vs Stressed Area per Growth Stage", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_ylim(0, 110)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_index_heatmap(index_arr: np.ndarray, title: str,
                       cmap: str = "RdYlGn", vmin: float = -1, vmax: float = 1) -> plt.Figure:
    """
    Render a single vegetation index as a colour-mapped heatmap.
    Can be used for NDVI, NDWI, NDRE, stress_score, etc.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(index_arr, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.axis("off")
    fig.tight_layout()
    return fig
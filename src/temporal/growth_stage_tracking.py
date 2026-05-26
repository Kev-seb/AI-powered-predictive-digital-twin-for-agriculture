"""
growth_stage_tracking.py
-------------------------
Track paddy rice growth stages across multiple UAV survey dates.

Capabilities:
    - Map NDVI / RedEdge time-series to canonical phenological stages
    - Flag anomalous stage progressions (e.g. premature senescence)
    - Estimate days-to-harvest from current stage and crop calendar
    - Generate crop-calendar timeline charts

Scientific basis:
    - Paddy rice phenology: IRRI (2013) crop calendar guide
    - NDVI–phenology mapping: Liu et al. (2019) Remote Sensing
    - GDD accumulation model: McMaster & Wilhelm (1997)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, date, timedelta
from typing import Optional

import numpy as np


# ──────────────────────────────────────────────────────────────
# Phenological stage definitions
# ──────────────────────────────────────────────────────────────

# Canonical duration in days per stage (wet season tropics)
STAGE_DURATION_DAYS = {
    "Nursery":    21,
    "Vegetative": 35,
    "Flowering":  14,
    "Mature":     21,
}

# NDVI range for each stage (Liu et al. 2019)
STAGE_NDVI_RANGE = {
    "Nursery":    (0.10, 0.30),
    "Vegetative": (0.30, 0.65),
    "Flowering":  (0.55, 0.80),
    "Mature":     (0.25, 0.55),
}

TOTAL_SEASON_DAYS = sum(STAGE_DURATION_DAYS.values())   # ~91 days


# ──────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────

@dataclass
class StageRecord:
    survey_date:    date
    days_after_transplanting: int
    ndvi_mean:      float
    ndre_mean:      float
    predicted_stage: str
    confidence:     float         # 0–1
    anomaly_flag:   bool
    anomaly_reason: str


# ──────────────────────────────────────────────────────────────
# NDVI → stage mapping
# ──────────────────────────────────────────────────────────────

def ndvi_to_stage(ndvi_mean: float, days_after_transplanting: int) -> tuple[str, float]:
    """
    Estimate growth stage from mean NDVI and days after transplanting (DAT).

    Uses both NDVI range overlap and expected phenological window.

    Returns
    -------
    (stage_name, confidence)  — confidence = overlap score [0, 1]
    """
    # Expected stage from DAT calendar
    cumulative = 0
    calendar_stage = "Mature"
    for stage, dur in STAGE_DURATION_DAYS.items():
        cumulative += dur
        if days_after_transplanting <= cumulative:
            calendar_stage = stage
            break

    # NDVI-based stage
    ndvi_stage = "Unknown"
    best_overlap = 0.0
    for stage, (lo, hi) in STAGE_NDVI_RANGE.items():
        if lo <= ndvi_mean <= hi:
            # Overlap ratio
            overlap = (min(ndvi_mean, hi) - lo) / max(hi - lo, 1e-5)
            if overlap > best_overlap:
                best_overlap = overlap
                ndvi_stage   = stage

    if ndvi_stage == "Unknown":
        # Fall back to calendar stage
        return calendar_stage, 0.40

    # Confidence boosted if calendar and NDVI agree
    confidence = 0.5 + 0.5 * best_overlap
    if ndvi_stage == calendar_stage:
        confidence = min(1.0, confidence + 0.25)

    return ndvi_stage, float(confidence)


# ──────────────────────────────────────────────────────────────
# Anomaly detection
# ──────────────────────────────────────────────────────────────

def detect_stage_anomaly(records: list[StageRecord]) -> list[StageRecord]:
    """
    Flag anomalous stage progressions in a chronological list of records.

    Checks:
        - Backward stage regression (e.g. Mature → Vegetative)
        - NDVI drop > 0.15 between consecutive surveys (possible lodging / disease)
        - Stage skipped (jumped more than one stage forward)

    Mutates records in place; returns the same list.
    """
    STAGE_ORDER = ["Nursery", "Vegetative", "Flowering", "Mature"]

    for i in range(1, len(records)):
        prev, curr = records[i - 1], records[i]
        prev_idx = STAGE_ORDER.index(prev.predicted_stage) if prev.predicted_stage in STAGE_ORDER else -1
        curr_idx = STAGE_ORDER.index(curr.predicted_stage) if curr.predicted_stage in STAGE_ORDER else -1

        reasons = []
        if curr_idx < prev_idx:
            reasons.append(f"Stage regression: {prev.predicted_stage} → {curr.predicted_stage}")
        if curr_idx - prev_idx > 1:
            reasons.append(f"Stage skipped: {prev.predicted_stage} → {curr.predicted_stage}")
        ndvi_delta = curr.ndvi_mean - prev.ndvi_mean
        if ndvi_delta < -0.15:
            reasons.append(f"Rapid NDVI decline ({ndvi_delta:+.3f})")

        if reasons:
            curr.anomaly_flag   = True
            curr.anomaly_reason = "; ".join(reasons)

    return records


# ──────────────────────────────────────────────────────────────
# Days-to-harvest estimation
# ──────────────────────────────────────────────────────────────

def estimate_days_to_harvest(days_after_transplanting: int) -> int:
    """Return estimated remaining days until harvest (minimum 0)."""
    return max(0, TOTAL_SEASON_DAYS - days_after_transplanting)


# ──────────────────────────────────────────────────────────────
# Time-series tracking
# ──────────────────────────────────────────────────────────────

def build_stage_timeline(
    survey_dates: list[date],
    ndvi_series:  list[float],
    ndre_series:  list[float],
    transplant_date: Optional[date] = None,
) -> list[StageRecord]:
    """
    Build a chronological list of StageRecord objects from survey time-series.

    Parameters
    ----------
    survey_dates     : list of survey dates (ascending order)
    ndvi_series      : mean NDVI per survey date
    ndre_series      : mean NDRE per survey date
    transplant_date  : field transplanting date; first survey date used if None

    Returns
    -------
    list[StageRecord]  sorted by survey_date
    """
    assert len(survey_dates) == len(ndvi_series) == len(ndre_series), \
        "survey_dates, ndvi_series, ndre_series must have the same length"

    if transplant_date is None:
        transplant_date = survey_dates[0]

    records = []
    for survey_date, ndvi, ndre in zip(survey_dates, ndvi_series, ndre_series):
        dat   = (survey_date - transplant_date).days
        stage, conf = ndvi_to_stage(ndvi, dat)
        records.append(StageRecord(
            survey_date=survey_date,
            days_after_transplanting=dat,
            ndvi_mean=float(ndvi),
            ndre_mean=float(ndre),
            predicted_stage=stage,
            confidence=conf,
            anomaly_flag=False,
            anomaly_reason="",
        ))

    detect_stage_anomaly(records)
    return records


# ──────────────────────────────────────────────────────────────
# Visualisation
# ──────────────────────────────────────────────────────────────

def plot_growth_timeline(records: list[StageRecord]) -> 'plt.Figure':
    """
    Plot NDVI time-series overlaid with stage predictions and anomaly flags.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        raise ImportError("matplotlib required for plot_growth_timeline()")

    from src.config.constants import STAGE_PALETTE

    dates  = [r.survey_date for r in records]
    ndvis  = [r.ndvi_mean   for r in records]
    ndres  = [r.ndre_mean   for r in records]

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(dates, ndvis, "g-o", linewidth=2, markersize=7, label="NDVI", zorder=5)
    ax.plot(dates, ndres, "b--s", linewidth=1.5, markersize=5, label="NDRE", alpha=0.8)

    # Shade stage background bands
    STAGE_ORDER = ["Nursery", "Vegetative", "Flowering", "Mature"]
    cumulative_days = 0
    if dates:
        t0 = dates[0]
        for stage in STAGE_ORDER:
            dur = STAGE_DURATION_DAYS[stage]
            x0  = t0 + timedelta(days=cumulative_days)
            x1  = t0 + timedelta(days=cumulative_days + dur)
            color = STAGE_PALETTE.get(stage, "#AAAAAA")
            ax.axvspan(x0, x1, alpha=0.12, color=color, label=f"_{stage}")
            ax.text((x0 + (x1 - x0) / 2).replace(tzinfo=None) if hasattr(x0, "replace") else x0,
                    0.95, stage, transform=ax.get_xaxis_transform(),
                    ha="center", fontsize=8, color="dimgray")
            cumulative_days += dur

    # Anomaly markers
    for r in records:
        if r.anomaly_flag:
            ax.plot(r.survey_date, r.ndvi_mean, "rv", markersize=12,
                    label="Anomaly", zorder=10, markeredgecolor="darkred")

    ax.set_xlabel("Survey Date", fontsize=11)
    ax.set_ylabel("Index Value", fontsize=11)
    ax.set_title("Crop Growth Stage Timeline", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig

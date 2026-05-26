"""
prescription_maps.py
---------------------
Generate site-specific prescription maps for precision agriculture.

A prescription map translates management zone information and stress analysis
into actionable variable-rate application (VRA) recommendations:
    - Nitrogen fertiliser rate (kg/ha)
    - Irrigation volume (mm)
    - Pesticide / fungicide application flag
    - Overall intervention priority score

Scientific basis:
    - IRRI precision nutrient management guidelines (2019)
    - FAO crop water requirements for irrigated rice (CROPWAT model defaults)
    - Paddy nitrogen response curves (Dobermann & Fairhurst 2000)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ──────────────────────────────────────────────────────────────
# Prescription data structures
# ──────────────────────────────────────────────────────────────

@dataclass
class ZonePrescription:
    zone_name:          str
    ndvi_mean:          float
    stress_mean:        float
    # VRA recommendations
    nitrogen_kg_ha:     float          # N application rate
    irrigation_mm:      float          # water application
    fungicide_flag:     bool           # True = apply fungicide
    priority_score:     float          # 0 (low) → 1 (high) urgency
    action_notes:       str
    color:              str            # hex for map rendering


@dataclass
class PrescriptionReport:
    timestamp:           str
    total_area_ha:       float
    zones:               list[ZonePrescription]
    field_avg_ndvi:      float
    field_avg_stress:    float
    overall_priority:    str           # "Normal" / "Attention" / "Urgent"
    summary_text:        str


# ──────────────────────────────────────────────────────────────
# VRA calculation logic
# ──────────────────────────────────────────────────────────────

# Nitrogen response look-up  {zone_name: kg/ha}
_BASE_N_RATES: dict[str, float] = {
    "High Productivity":   40.0,
    "Medium Productivity": 60.0,
    "Low Productivity":    80.0,
    "Stressed / Bare":    100.0,
}

# Irrigation look-up  {zone_name: mm}
_BASE_IRRIG: dict[str, float] = {
    "High Productivity":    0.0,
    "Medium Productivity": 10.0,
    "Low Productivity":    20.0,
    "Stressed / Bare":     30.0,
}

# Priority scores  {zone_name: 0-1}
_PRIORITY: dict[str, float] = {
    "High Productivity":   0.1,
    "Medium Productivity": 0.4,
    "Low Productivity":    0.7,
    "Stressed / Bare":     1.0,
}

_ZONE_COLORS: dict[str, str] = {
    "High Productivity":   "#2ECC71",
    "Medium Productivity": "#F39C12",
    "Low Productivity":    "#E67E22",
    "Stressed / Bare":     "#E74C3C",
}


def compute_zone_prescription(zone_name: str,
                               ndvi_mean: float,
                               stress_mean: float,
                               humidity_pct: float = 70.0) -> ZonePrescription:
    """
    Compute variable-rate prescription for a single management zone.

    Parameters
    ----------
    zone_name    : e.g. "Low Productivity"
    ndvi_mean    : mean NDVI of zone
    stress_mean  : mean composite stress score [0, 1]
    humidity_pct : ambient relative humidity (%) — affects fungicide flag

    Returns
    -------
    ZonePrescription
    """
    # Base rates from lookup, adjusted by actual stress level
    stress_factor = 1.0 + (stress_mean - 0.3) * 0.5   # up to +35% at stress=1
    stress_factor = max(0.5, min(2.0, stress_factor))

    nitrogen_kg_ha  = round(_BASE_N_RATES.get(zone_name, 60.0) * stress_factor, 1)
    irrigation_mm   = round(_BASE_IRRIG.get(zone_name, 10.0)   * stress_factor, 1)
    fungicide_flag  = (humidity_pct > 80.0 and stress_mean > 0.50)
    priority_score  = _PRIORITY.get(zone_name, 0.5)

    if priority_score >= 0.9:
        notes = "URGENT: field scouting required. Consider replanting bare patches."
    elif priority_score >= 0.6:
        notes = "Apply recommended N + irrigation. Monitor for disease progression."
    elif priority_score >= 0.3:
        notes = "Routine input application. Check soil moisture midseason."
    else:
        notes = "Maintain current inputs. Minimal intervention required."

    if fungicide_flag:
        notes += " High humidity detected — fungicide application advised."

    return ZonePrescription(
        zone_name=zone_name,
        ndvi_mean=ndvi_mean,
        stress_mean=stress_mean,
        nitrogen_kg_ha=nitrogen_kg_ha,
        irrigation_mm=irrigation_mm,
        fungicide_flag=fungicide_flag,
        priority_score=priority_score,
        action_notes=notes,
        color=_ZONE_COLORS.get(zone_name, "#95A5A6"),
    )


# ──────────────────────────────────────────────────────────────
# Report generation
# ──────────────────────────────────────────────────────────────

def build_prescription_report(zone_prescriptions: list[ZonePrescription],
                               total_area_ha: float = 1.0,
                               timestamp: Optional[str] = None) -> PrescriptionReport:
    """
    Aggregate per-zone prescriptions into a field-level report.

    Parameters
    ----------
    zone_prescriptions : output of compute_zone_prescription() per zone
    total_area_ha      : total field area in hectares
    timestamp          : ISO datetime string; auto-generated if None

    Returns
    -------
    PrescriptionReport
    """
    import datetime
    ts = timestamp or datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z"

    all_ndvi    = [z.ndvi_mean    for z in zone_prescriptions]
    all_stress  = [z.stress_mean  for z in zone_prescriptions]
    all_priority= [z.priority_score for z in zone_prescriptions]

    avg_ndvi   = float(np.mean(all_ndvi))
    avg_stress = float(np.mean(all_stress))
    max_pri    = max(all_priority) if all_priority else 0.0

    if max_pri >= 0.9:
        overall = "Urgent"
    elif max_pri >= 0.5:
        overall = "Attention"
    else:
        overall = "Normal"

    n_urgent = sum(1 for p in all_priority if p >= 0.9)
    summary  = (
        f"Field assessment ({ts}): {len(zone_prescriptions)} management zones analysed. "
        f"Average NDVI: {avg_ndvi:.3f}, Average Stress: {avg_stress:.3f}. "
        f"Overall status: {overall}. "
        f"{n_urgent} zone(s) require urgent intervention."
    )

    return PrescriptionReport(
        timestamp=ts,
        total_area_ha=total_area_ha,
        zones=zone_prescriptions,
        field_avg_ndvi=avg_ndvi,
        field_avg_stress=avg_stress,
        overall_priority=overall,
        summary_text=summary,
    )


# ──────────────────────────────────────────────────────────────
# Visualisation
# ──────────────────────────────────────────────────────────────

def plot_prescription_map(report: PrescriptionReport,
                          zone_masks: dict[str, np.ndarray],
                          shape: tuple[int, int]) -> 'plt.Figure':
    """
    Render a prescription priority map as an RGB image.

    Parameters
    ----------
    report     : PrescriptionReport
    zone_masks : {zone_name: bool (H, W)} pixel masks
    shape      : (H, W) of output image

    Returns
    -------
    matplotlib Figure
    """
    if not HAS_MPL:
        raise ImportError("matplotlib required for plot_prescription_map()")

    import matplotlib.colors as mcolors
    import matplotlib.patches as mpatches

    canvas = np.zeros((*shape, 3), dtype=np.uint8)
    for zp in report.zones:
        mask = zone_masks.get(zp.zone_name)
        if mask is None:
            continue
        rgb = tuple(int(c * 255) for c in mcolors.to_rgb(zp.color))
        canvas[mask] = rgb

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Prescription colour map
    axes[0].imshow(canvas)
    axes[0].set_title("Prescription Zone Map", fontsize=13, fontweight="bold")
    axes[0].axis("off")
    patches = [mpatches.Patch(color=zp.color, label=zp.zone_name)
               for zp in report.zones]
    axes[0].legend(handles=patches, loc="lower right", fontsize=8)

    # Bar chart — N application rate per zone
    names    = [zp.zone_name.replace(" ", "\n") for zp in report.zones]
    n_rates  = [zp.nitrogen_kg_ha              for zp in report.zones]
    colors   = [zp.color                       for zp in report.zones]
    axes[1].barh(names, n_rates, color=colors, edgecolor="black", linewidth=0.5)
    axes[1].set_xlabel("N Application Rate (kg/ha)", fontsize=11)
    axes[1].set_title("Variable-Rate Nitrogen Map", fontsize=13, fontweight="bold")
    axes[1].invert_yaxis()

    fig.suptitle(f"Precision Agriculture Prescription — {report.timestamp}",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig


def export_prescription_csv(report: PrescriptionReport, path: str) -> None:
    """Write zone prescriptions to a CSV file for VRA machinery upload."""
    import csv
    from pathlib import Path

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    headers = ["zone_name", "ndvi_mean", "stress_mean", "nitrogen_kg_ha",
               "irrigation_mm", "fungicide_flag", "priority_score", "action_notes"]
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for zp in report.zones:
            writer.writerow({
                "zone_name":       zp.zone_name,
                "ndvi_mean":       f"{zp.ndvi_mean:.4f}",
                "stress_mean":     f"{zp.stress_mean:.4f}",
                "nitrogen_kg_ha":  zp.nitrogen_kg_ha,
                "irrigation_mm":   zp.irrigation_mm,
                "fungicide_flag":  int(zp.fungicide_flag),
                "priority_score":  f"{zp.priority_score:.2f}",
                "action_notes":    zp.action_notes,
            })

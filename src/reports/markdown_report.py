"""
markdown_report.py
------------------
Generate automated Markdown reports for crop stress and field zoning.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.gis.prescription_maps import PrescriptionReport

def generate_markdown_report(report: PrescriptionReport, path: str | Path, image_paths: dict[str, str] | None = None) -> Path:
    """
    Generate a markdown report from a PrescriptionReport.
    
    Parameters
    ----------
    report      : PrescriptionReport object
    path        : Output path for the markdown file
    image_paths : Dict mapping image names to relative file paths to embed
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    
    lines = [
        f"# UAV Crop Stress Intelligence Report",
        f"**Date:** {report.timestamp}",
        f"**Total Area Analysed:** {report.total_area_ha:.2f} ha",
        f"**Overall Priority:** {report.overall_priority}",
        "",
        "## Summary",
        report.summary_text,
        "",
        "## Management Zones & Prescriptions",
        "| Zone Name | NDVI | Stress | N (kg/ha) | Irrigation (mm) | Fungicide | Priority | Notes |",
        "|---|---|---|---|---|---|---|---|"
    ]
    
    for z in report.zones:
        fung = "Yes" if z.fungicide_flag else "No"
        lines.append(
            f"| {z.zone_name} | {z.ndvi_mean:.3f} | {z.stress_mean:.3f} | {z.nitrogen_kg_ha} | {z.irrigation_mm} | {fung} | {z.priority_score:.2f} | {z.action_notes} |"
        )
        
    lines.append("")
    
    if image_paths:
        lines.append("## Visualisations")
        for title, img_path in image_paths.items():
            lines.append(f"### {title}")
            lines.append(f"![{title}]({img_path})")
            lines.append("")
            
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        
    return out

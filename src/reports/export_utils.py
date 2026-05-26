"""
export_utils.py
---------------
Utility functions for exporting reports and data tables to disk.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Sequence

def export_json(data: dict | list, path: str | Path) -> Path:
    """Export dictionary or list to JSON."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    return out

def export_csv(data: Sequence[dict[str, Any]], path: str | Path, fieldnames: list[str] | None = None) -> Path:
    """Export a list of dictionaries to CSV."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not data:
        out.touch()
        return out
    
    if fieldnames is None:
        fieldnames = list(data[0].keys())
        
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in data:
            writer.writerow(row)
            
    return out

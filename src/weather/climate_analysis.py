"""
climate_analysis.py
-------------------
Aggregates and analyses historical weather/climate data to establish baselines.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from src.weather.openmeteo_client import compute_gdd, fill_missing

def calculate_climate_normals(historical_weather: dict) -> dict[str, Any]:
    """
    Calculate climate normal baselines from historical weather data.
    
    Parameters
    ----------
    historical_weather : Dict returned by openmeteo_client.fetch_historical_daily
    
    Returns
    -------
    dict of baseline statistics
    """
    tmax = np.array(fill_missing(historical_weather.get("temperature_2m_max", [])))
    tmin = np.array(fill_missing(historical_weather.get("temperature_2m_min", [])))
    precip = np.array(fill_missing(historical_weather.get("precipitation_sum", [])))
    
    return {
        "avg_tmax": float(tmax.mean()) if len(tmax) > 0 else 0.0,
        "avg_tmin": float(tmin.mean()) if len(tmin) > 0 else 0.0,
        "total_precip": float(precip.sum()),
        "max_tmax": float(tmax.max()) if len(tmax) > 0 else 0.0,
        "min_tmin": float(tmin.min()) if len(tmin) > 0 else 0.0,
        "days_precip": int((precip > 0.1).sum()),
    }

def gdd_accumulation(historical_weather: dict, t_base: float = 10.0, t_ceiling: float = 35.0) -> list[float]:
    """
    Calculate cumulative GDD over the historical period.
    """
    tmax = historical_weather.get("temperature_2m_max", [])
    tmin = historical_weather.get("temperature_2m_min", [])
    
    daily_gdd = compute_gdd(tmax, tmin, t_base, t_ceiling)
    
    cumulative = []
    current = 0.0
    for g in daily_gdd:
        current += g
        cumulative.append(current)
        
    return cumulative

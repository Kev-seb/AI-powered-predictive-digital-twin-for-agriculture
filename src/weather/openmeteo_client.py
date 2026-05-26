"""
openmeteo_client.py
--------------------
HTTP client for the Open-Meteo free weather API.

Fetches:
    - Hourly / daily meteorological variables for a lat/lon
    - Historical re-analysis data (ERA5 backend)
    - Short-range forecast (7 days)

API reference: https://open-meteo.com/en/docs

No API key required for basic use.
"""

from __future__ import annotations

import time
from datetime import date, datetime, timedelta
from typing import Optional

import numpy as np

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

try:
    from loguru import logger
except ImportError:
    import logging as logger  # type: ignore[assignment]


# ──────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────

BASE_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

DAILY_VARS = [
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "windspeed_10m_max",
    "et0_fao_evapotranspiration",
    "relative_humidity_2m_max",
    "relative_humidity_2m_min",
    "sunshine_duration",
]

HOURLY_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "windspeed_10m",
    "shortwave_radiation",
    "vapour_pressure_deficit",
]

DEFAULT_TIMEOUT = 15   # seconds


# ──────────────────────────────────────────────────────────────
# Request helpers
# ──────────────────────────────────────────────────────────────

def _get(url: str, params: dict, retries: int = 3, backoff: float = 1.5) -> dict:
    """HTTP GET with retry / exponential back-off."""
    if not HAS_REQUESTS:
        raise ImportError("requests package is required: pip install requests")

    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, params=params, timeout=DEFAULT_TIMEOUT)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as exc:
            logger.warning(f"[OpenMeteo] Attempt {attempt}/{retries} failed: {exc}")
            if attempt < retries:
                time.sleep(backoff ** attempt)
            else:
                raise


# ──────────────────────────────────────────────────────────────
# Forecast fetcher
# ──────────────────────────────────────────────────────────────

def fetch_daily_forecast(lat: float, lon: float,
                          forecast_days: int = 7,
                          timezone: str = "auto") -> dict:
    """
    Fetch daily forecast weather data from Open-Meteo.

    Parameters
    ----------
    lat           : latitude (decimal degrees)
    lon           : longitude (decimal degrees)
    forecast_days : number of forecast days (1–16)
    timezone      : IANA timezone string or "auto"

    Returns
    -------
    dict with keys: dates (list[str]), and one list per DAILY_VARS variable.
    """
    params = {
        "latitude":      lat,
        "longitude":     lon,
        "daily":         ",".join(DAILY_VARS),
        "forecast_days": min(16, max(1, forecast_days)),
        "timezone":      timezone,
    }
    raw   = _get(BASE_URL, params)
    daily = raw.get("daily", {})

    out: dict = {"dates": daily.get("time", [])}
    for var in DAILY_VARS:
        out[var] = daily.get(var, [])
    return out


def fetch_hourly_forecast(lat: float, lon: float,
                           forecast_days: int = 3,
                           timezone: str = "auto") -> dict:
    """
    Fetch hourly forecast weather data.

    Returns
    -------
    dict with key "time" (ISO datetimes) and one list per HOURLY_VARS variable.
    """
    params = {
        "latitude":      lat,
        "longitude":     lon,
        "hourly":        ",".join(HOURLY_VARS),
        "forecast_days": min(16, max(1, forecast_days)),
        "timezone":      timezone,
    }
    raw    = _get(BASE_URL, params)
    hourly = raw.get("hourly", {})

    out: dict = {"time": hourly.get("time", [])}
    for var in HOURLY_VARS:
        out[var] = hourly.get(var, [])
    return out


# ──────────────────────────────────────────────────────────────
# Historical re-analysis fetcher
# ──────────────────────────────────────────────────────────────

def fetch_historical_daily(lat: float, lon: float,
                            start_date: date,
                            end_date: date,
                            timezone: str = "auto") -> dict:
    """
    Fetch historical daily weather from ERA5 re-analysis via Open-Meteo archive API.

    Parameters
    ----------
    start_date, end_date : inclusive date range

    Returns
    -------
    dict  (same schema as fetch_daily_forecast output)
    """
    params = {
        "latitude":   lat,
        "longitude":  lon,
        "start_date": start_date.isoformat(),
        "end_date":   end_date.isoformat(),
        "daily":      ",".join(DAILY_VARS),
        "timezone":   timezone,
    }
    raw   = _get(ARCHIVE_URL, params)
    daily = raw.get("daily", {})

    out: dict = {"dates": daily.get("time", [])}
    for var in DAILY_VARS:
        out[var] = daily.get(var, [])
    return out


# ──────────────────────────────────────────────────────────────
# Growing Degree Days helper
# ──────────────────────────────────────────────────────────────

def compute_gdd(tmax_series: list[float], tmin_series: list[float],
                t_base: float = 10.0, t_ceiling: float = 35.0) -> list[float]:
    """
    Compute daily Growing Degree Days (GDD) from Tmax/Tmin series.

    GDD = ((Tmax + Tmin) / 2) − T_base,  capped at t_ceiling, floored at 0.

    Parameters
    ----------
    tmax_series, tmin_series : lists of daily max/min temperatures (°C)
    t_base    : base temperature below which crop growth stops
    t_ceiling : temperature ceiling (heat stress cap)

    Returns
    -------
    list[float]  daily GDD values
    """
    gdd = []
    for tmax, tmin in zip(tmax_series, tmin_series):
        if tmax is None or tmin is None:
            gdd.append(0.0)
            continue
        tmax_eff = min(float(tmax), t_ceiling)
        tmin_eff = min(float(tmin), t_ceiling)
        tmean    = (tmax_eff + tmin_eff) / 2.0
        gdd.append(max(0.0, tmean - t_base))
    return gdd


# ──────────────────────────────────────────────────────────────
# Data-quality helpers
# ──────────────────────────────────────────────────────────────

def fill_missing(series: list, fill_value: float = 0.0) -> list[float]:
    """Replace None values in a weather series with `fill_value`."""
    return [fill_value if v is None else float(v) for v in series]


def series_to_array(weather_dict: dict, variable: str) -> np.ndarray:
    """Extract a variable from a weather dict as a numpy float32 array."""
    return np.array(fill_missing(weather_dict.get(variable, [])), dtype=np.float32)

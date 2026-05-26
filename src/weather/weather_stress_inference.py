"""
weather_stress_inference.py
----------------------------
Integrates weather API data with vegetation index analysis to produce
weather-aware crop stress risk assessments.

Stress risk factors:
    - Heat stress    : Tmax > 35°C during flowering (IRRI threshold)
    - Cold stress    : Tmin < 15°C at nursery / vegetative stage
    - Water stress   : < 5 mm rain over 7 days + NDWI < -0.1
    - Waterlogging   : > 100 mm rain over 3 days + NDWI > 0.3
    - Humidity risk  : RH > 85% for 3+ days (blast disease susceptibility)

API: Open-Meteo (free, no key required)
     https://open-meteo.com/en/docs
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

import numpy as np


# ──────────────────────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────────────────────

@dataclass
class WeatherData:
    latitude: float
    longitude: float
    temperature_max: list[float]    # daily Tmax (°C) for last 7 days
    temperature_min: list[float]    # daily Tmin (°C)
    precipitation:   list[float]    # daily rain (mm)
    humidity:        list[float]    # daily mean relative humidity (%)
    wind_speed:      list[float]    # daily mean wind speed (km/h)
    current_temp:    float
    current_humidity: float
    current_precip:  float


@dataclass
class StressRiskFactor:
    name: str
    active: bool
    severity: str           # "Low" / "Medium" / "High"
    description: str
    recommendation: str


@dataclass
class WeatherStressAssessment:
    weather: WeatherData
    risk_factors: list[StressRiskFactor]
    overall_risk: str                   # "Low" / "Medium" / "High" / "Critical"
    composite_risk_score: float         # 0–1
    ai_recommendation: str
    stage_specific_warnings: list[str]


# ──────────────────────────────────────────────────────────────
# Weather API
# ──────────────────────────────────────────────────────────────

def fetch_weather(lat: float, lon: float, past_days: int = 7) -> Optional[WeatherData]:
    """
    Fetch historical + current weather from Open-Meteo (free, no key).

    Parameters
    ----------
    lat, lon   : field coordinates
    past_days  : how many past days to retrieve (max 92)
    """
    if not HAS_REQUESTS:
        print("[WARN] requests not installed. Run: pip install requests")
        return None

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude":  lat,
        "longitude": lon,
        "daily": [
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_sum",
            "relative_humidity_2m_mean",
            "wind_speed_10m_max",
        ],
        "current": [
            "temperature_2m",
            "relative_humidity_2m",
            "precipitation",
        ],
        "past_days": past_days,
        "timezone": "auto",
    }

    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        daily = data.get("daily", {})
        current = data.get("current", {})

        return WeatherData(
            latitude=lat,
            longitude=lon,
            temperature_max=daily.get("temperature_2m_max", []),
            temperature_min=daily.get("temperature_2m_min", []),
            precipitation=daily.get("precipitation_sum", []),
            humidity=daily.get("relative_humidity_2m_mean", []),
            wind_speed=daily.get("wind_speed_10m_max", []),
            current_temp=current.get("temperature_2m", float("nan")),
            current_humidity=current.get("relative_humidity_2m", float("nan")),
            current_precip=current.get("precipitation", 0.0),
        )
    except Exception as e:
        print(f"[ERROR] Weather API failed: {e}")
        return None


# ──────────────────────────────────────────────────────────────
# Stress rule engine
# ──────────────────────────────────────────────────────────────

def assess_weather_stress(weather: WeatherData,
                          crop_stage: str,
                          ndvi_mean: float,
                          ndwi_mean: float) -> WeatherStressAssessment:
    """
    Combine weather data with spectral indices to infer stress risk.

    Parameters
    ----------
    weather     : WeatherData from fetch_weather()
    crop_stage  : one of Nursery / Vegetative / Flowering / Mature
    ndvi_mean   : field-level mean NDVI from remote sensing
    ndwi_mean   : field-level mean NDWI from remote sensing

    Returns
    -------
    WeatherStressAssessment
    """
    risk_factors = []
    risk_score = 0.0

    tmax = weather.temperature_max
    tmin = weather.temperature_min
    rain = weather.precipitation
    rh   = weather.humidity

    # ── Heat stress ────────────────────────────────────────
    recent_tmax = tmax[-3:] if len(tmax) >= 3 else tmax
    heat_days = sum(1 for t in recent_tmax if t > 35)
    if heat_days > 0:
        sev = "High" if heat_days >= 2 else "Medium"
        risk_factors.append(StressRiskFactor(
            name="Heat Stress",
            active=True,
            severity=sev,
            description=f"Temperature exceeded 35°C on {heat_days} of last 3 days.",
            recommendation="Ensure adequate irrigation. Avoid mid-day operations." +
                           (" Critical during flowering — spikelet sterility risk." if crop_stage == "Flowering" else ""),
        ))
        risk_score += 0.35 if sev == "High" else 0.15

    # ── Cold stress ────────────────────────────────────────
    recent_tmin = tmin[-5:] if len(tmin) >= 5 else tmin
    cold_days = sum(1 for t in recent_tmin if t < 15)
    if cold_days >= 2 and crop_stage in ("Nursery", "Vegetative"):
        risk_factors.append(StressRiskFactor(
            name="Cold / Chilling Stress",
            active=True,
            severity="Medium",
            description=f"Temperatures below 15°C for {cold_days} days.",
            recommendation="Increase water depth to 5 cm for thermal buffering.",
        ))
        risk_score += 0.20

    # ── Drought / Water stress ─────────────────────────────
    rain_7d = sum(rain[-7:]) if len(rain) >= 7 else sum(rain)
    if rain_7d < 5 and ndwi_mean < -0.1:
        risk_factors.append(StressRiskFactor(
            name="Water / Drought Stress",
            active=True,
            severity="High" if ndwi_mean < -0.3 else "Medium",
            description=f"Only {rain_7d:.1f} mm rain in last 7 days. NDWI = {ndwi_mean:.3f}.",
            recommendation="Immediate supplemental irrigation recommended.",
        ))
        risk_score += 0.30

    # ── Waterlogging ───────────────────────────────────────
    rain_3d = sum(rain[-3:]) if len(rain) >= 3 else sum(rain)
    if rain_3d > 100 and ndwi_mean > 0.30:
        risk_factors.append(StressRiskFactor(
            name="Waterlogging / Flood Risk",
            active=True,
            severity="High",
            description=f"{rain_3d:.1f} mm rain in last 3 days. NDWI = {ndwi_mean:.3f}.",
            recommendation="Open drainage channels. Monitor for root oxygen stress.",
        ))
        risk_score += 0.25

    # ── Blast disease humidity ─────────────────────────────
    high_rh_days = sum(1 for h in rh[-5:] if h > 85) if rh else 0
    if high_rh_days >= 3:
        risk_factors.append(StressRiskFactor(
            name="Disease Susceptibility (High Humidity)",
            active=True,
            severity="Medium",
            description=f"RH > 85% for {high_rh_days} days — favourable for blast / sheath blight.",
            recommendation="Scout for fungal lesions. Consider preventive fungicide application.",
        ))
        risk_score += 0.15

    # ── Low NDVI anomaly ───────────────────────────────────
    expected_ndvi = {"Nursery": 0.25, "Vegetative": 0.60, "Flowering": 0.65, "Mature": 0.55}
    exp = expected_ndvi.get(crop_stage, 0.5)
    if ndvi_mean < exp - 0.15:
        risk_factors.append(StressRiskFactor(
            name="Below-Expected Canopy Vigour",
            active=True,
            severity="Medium",
            description=f"NDVI {ndvi_mean:.3f} is {exp - ndvi_mean:.3f} below expected for {crop_stage} stage.",
            recommendation="Tissue test for nutrient deficiency. Review fertilisation schedule.",
        ))
        risk_score += 0.15

    # ── Stage-specific warnings ───────────────────────────
    stage_warnings = _get_stage_warnings(crop_stage, weather)

    # ── Overall risk ───────────────────────────────────────
    risk_score = min(risk_score, 1.0)
    if risk_score >= 0.6:
        overall = "Critical"
    elif risk_score >= 0.35:
        overall = "High"
    elif risk_score >= 0.15:
        overall = "Medium"
    else:
        overall = "Low"

    ai_rec = _generate_ai_recommendation(overall, risk_factors, crop_stage)

    return WeatherStressAssessment(
        weather=weather,
        risk_factors=risk_factors,
        overall_risk=overall,
        composite_risk_score=float(risk_score),
        ai_recommendation=ai_rec,
        stage_specific_warnings=stage_warnings,
    )


def _get_stage_warnings(stage: str, weather: WeatherData) -> list[str]:
    warnings = []
    tmax = weather.temperature_max[-3:] if weather.temperature_max else []

    if stage == "Flowering":
        if any(t > 35 for t in tmax):
            warnings.append("Flowering stage + heat stress: risk of spikelet sterility (>10% yield loss possible).")
        if any(t > 33 for t in tmax):
            warnings.append("Pollination window vulnerable. Temperatures above 33°C reduce pollen viability.")

    if stage == "Nursery":
        warnings.append("Nursery: monitor seedling density and uniform germination via NDVI maps.")

    if stage == "Vegetative":
        warnings.append("Vegetative: NDRE most sensitive to N-deficiency at this stage.")

    if stage == "Mature":
        warnings.append("Harvest readiness: monitor NDVI decline + canopy browning for optimal timing.")

    return warnings


def _generate_ai_recommendation(overall: str, factors: list[StressRiskFactor], stage: str) -> str:
    if not factors:
        return (f"No significant weather stress detected for {stage} stage. "
                "Continue standard monitoring. Next UAV survey recommended in 7–10 days.")

    priority = factors[0]
    base = f"[{overall} Risk — {stage} Stage] Priority concern: {priority.name}. {priority.recommendation}"
    if len(factors) > 1:
        others = ", ".join(f.name for f in factors[1:])
        base += f" Secondary factors to monitor: {others}."
    return base
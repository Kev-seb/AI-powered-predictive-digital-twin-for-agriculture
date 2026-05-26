"""
weather_risk_engine.py
----------------------
Evaluates crop risks (disease, drought, lodging) based on weather forecasts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

@dataclass
class WeatherRisk:
    risk_type: str
    risk_level: str  # "Low", "Medium", "High"
    probability: float # 0-1
    trigger_conditions: str
    recommendation: str

def evaluate_disease_risk(humidity: Sequence[float], temp: Sequence[float], rainfall: Sequence[float]) -> WeatherRisk:
    """
    Evaluate fungal disease risk based on humidity and temperature.
    High risk if humidity > 85% and temp is between 25-32C for prolonged periods.
    """
    high_hum_days = sum(1 for h in humidity if h is not None and h > 85.0)
    warm_days = sum(1 for t in temp if t is not None and 25.0 <= t <= 32.0)
    
    prob = min(1.0, (high_hum_days * 0.1) + (warm_days * 0.05))
    
    if prob > 0.7:
        level = "High"
        rec = "Preventative fungicide application recommended."
    elif prob > 0.4:
        level = "Medium"
        rec = "Monitor canopy for fungal lesions."
    else:
        level = "Low"
        rec = "No immediate action required."
        
    return WeatherRisk(
        risk_type="Fungal Disease",
        risk_level=level,
        probability=prob,
        trigger_conditions=f"{high_hum_days} days high humidity, {warm_days} optimal temp days",
        recommendation=rec
    )

def evaluate_drought_risk(rainfall: Sequence[float], et0: Sequence[float]) -> WeatherRisk:
    """
    Evaluate drought stress risk using rainfall deficit (Precipitation - ET0).
    """
    deficit = 0.0
    for r, e in zip(rainfall, et0):
        if r is not None and e is not None:
            deficit += (e - r)
            
    prob = min(1.0, max(0.0, deficit / 50.0))
    
    if prob > 0.7:
        level = "High"
        rec = "Immediate supplemental irrigation required."
    elif prob > 0.4:
        level = "Medium"
        rec = "Plan for irrigation in the next 3-5 days."
    else:
        level = "Low"
        rec = "Soil moisture adequate."
        
    return WeatherRisk(
        risk_type="Drought / Water Stress",
        risk_level=level,
        probability=prob,
        trigger_conditions=f"Cumulative deficit: {deficit:.1f} mm",
        recommendation=rec
    )

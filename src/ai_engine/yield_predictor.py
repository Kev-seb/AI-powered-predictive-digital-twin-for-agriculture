"""
yield_predictor.py
------------------
Advanced AI-Driven Crop Yield Prediction, Biomass Estimation, and Harvest Forecasting Engine.
Uses multispectral indices (NDVI, NDRE), temporal progression, weather data, and growth stages.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


@dataclass
class HarvestForecast:
    predicted_harvest_date: datetime.date
    days_to_harvest: int
    optimal_window_start: datetime.date
    optimal_window_end: datetime.date
    harvest_readiness_pct: float       # 0% to 100%
    average_yield_t_ha: float
    total_production_t: float
    estimated_biomass_t_ha: float
    limiting_factors: List[str]
    harvest_recommendations: List[str]


class CropYieldPredictor:
    def __init__(self, crop_type: str = "Paddy Rice", base_harvest_index: float = 0.48):
        """
        Initialize the predictor with crop-specific agronomic parameters.
        Parameters:
            crop_type: "Paddy Rice", "Corn", "Wheat", etc.
            base_harvest_index: Ratio of grain yield to total above-ground biomass (0.35 - 0.55).
        """
        self.crop_type = crop_type
        self.base_harvest_index = base_harvest_index
        
        # Growing Degree Days (GDD) thresholds from transplanting/planting to maturity
        # Base temperature (T_base) below which crop growth ceases
        self.agronomic_params = {
            "Paddy Rice": {
                "t_base": 10.0,
                "t_max": 35.0,
                "gdd_required_total": 1350.0,  # Degree-days from transplanting to harvest
                "peak_agb_multiplier": 14.5,    # Peak above-ground biomass in t/ha for healthy crop
            },
            "Corn": {
                "t_base": 10.0,
                "t_max": 30.0,
                "gdd_required_total": 1450.0,
                "peak_agb_multiplier": 18.0,
            },
            "Wheat": {
                "t_base": 4.0,
                "t_max": 25.0,
                "gdd_required_total": 1200.0,
                "peak_agb_multiplier": 12.0,
            }
        }
        
        # Load params for specified crop, fallback to Rice
        self.params = self.agronomic_params.get(self.crop_type, self.agronomic_params["Paddy Rice"])

    def compute_growing_degree_days(self, daily_temps: List[float]) -> float:
        """
        Compute accumulated Growing Degree Days (GDD) for a list of daily average temperatures.
        Formula: GDD = max((T_max + T_min)/2 - T_base, 0)
        Using daily average temperature as a standard proxy: max(T_avg - T_base, 0)
        """
        t_base = self.params["t_base"]
        gdd = 0.0
        for temp in daily_temps:
            gdd += max(0.0, temp - t_base)
        return gdd

    def estimate_biomass(
        self,
        ndvi: np.ndarray,
        ndre: np.ndarray,
        growth_stage: str,
        canopy_height: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Estimate Above-Ground Biomass (AGB) in tonnes per hectare (t/ha) at pixel level.
        Uses NDVI (canopy density proxy), NDRE (leaf chlorophyll/nitrogen proxy),
        and growth stage scaling factor.
        """
        # Define growth stage multiplier (AGB accumulates over time)
        stage_multipliers = {
            "Nursery": 0.08,
            "Vegetative": 0.45,
            "Flowering": 0.85,
            "Mature": 1.00
        }
        multiplier = stage_multipliers.get(growth_stage, 0.50)
        
        # Vectorized AGB estimation: combination of NDVI (canopy size) and NDRE (canopy density/chlorophyll)
        # Standard healthy max biomass multiplier
        peak_biomass = self.params["peak_agb_multiplier"]
        
        # NDVI * NDRE gives a robust proxy for leaf volume + activity
        # Clip indices to normal ranges
        ndvi_clipped = np.clip(ndvi, 0.0, 1.0)
        ndre_clipped = np.clip(ndre, 0.0, 1.0)
        
        # Biomass estimation formula combining light interception (NDVI) and leaf activity (NDRE)
        # Using square root of NDVI * NDRE to model saturation effects
        vegetation_proxy = np.sqrt(ndvi_clipped * ndre_clipped + 1e-6)
        biomass_map = vegetation_proxy * peak_biomass * multiplier
        
        # Integrate canopy height if DSM/CHM data is available
        if canopy_height is not None:
            # Scale biomass based on crop height (standardized relative to a healthy peak height of 0.8m for rice)
            height_factor = np.clip(canopy_height / 0.8, 0.5, 1.5)
            biomass_map *= height_factor
            
        return np.clip(biomass_map, 0.0, peak_biomass * 1.5)

    def predict_yield(
        self,
        biomass_map: np.ndarray,
        stress_score: np.ndarray,
        weather: Dict[str, Any],
        growth_stage: str
    ) -> np.ndarray:
        """
        Predict crop yield map in tonnes per hectare (t/ha).
        Formula: Yield = Biomass * Harvest Index * Stress Penalty
        Adjusts the harvest index dynamically based on weather stresses (e.g. heat during flowering).
        """
        # Dynamic Harvest Index calculation
        hi = self.base_harvest_index
        
        temp_c = weather.get("temperature", 25.0)
        if isinstance(temp_c, list) and len(temp_c) > 0:
            temp_c = float(temp_c[0])
        elif isinstance(temp_c, (int, float)):
            temp_c = float(temp_c)
        else:
            temp_c = 25.0
            
        # Spikelet sterility check: If temp > 35C during Flowering, Harvest Index drops significantly
        if growth_stage == "Flowering" and temp_c > 35.0:
            heat_deficit = temp_c - 35.0
            hi_penalty = min(0.25, heat_deficit * 0.05) # up to 25% drop in grain filling
            hi -= hi_penalty
            
        # Local stress penalty map: pixels with high stress have reduced grain-filling/panicle formation
        # Stressed pixels can lose up to 40% of potential yield relative to their biomass
        stress_penalty = 1.0 - (0.40 * np.clip(stress_score, 0.0, 1.0))
        
        yield_map = biomass_map * hi * stress_penalty
        return np.clip(yield_map, 0.0, biomass_map * 0.7)

    def generate_harvest_forecast(
        self,
        yield_map: np.ndarray,
        biomass_map: np.ndarray,
        current_gdd_accumulated: float,
        weather_forecast: List[Dict[str, Any]],
        growth_stage: str,
        days_after_transplanting: int,
        field_area_ha: float = 1.0
    ) -> HarvestForecast:
        """
        Forecast harvest date, readiness level, optimal windows, and limiting factors.
        """
        today = datetime.date.today()
        gdd_req = self.params["gdd_required_total"]
        
        # Estimate average daily GDD from 7-day weather forecast
        forecast_temps = [w.get("temperature", 25.0) for w in weather_forecast]
        avg_daily_gdd = self.compute_growing_degree_days(forecast_temps) / max(1, len(forecast_temps))
        if avg_daily_gdd < 1.0:
            avg_daily_gdd = 15.0 # fallback default daily accumulated GDD for tropics
            
        gdd_remaining = max(0.0, gdd_req - current_gdd_accumulated)
        
        # Adjust remaining GDD depending on growth stage
        # If in mature stage, we are very close to harvest
        if growth_stage == "Mature":
            days_to_harvest = max(3, int(gdd_remaining / avg_daily_gdd) if gdd_remaining > 0 else 7)
            # Max mature phase duration is ~21 days
            days_to_harvest = min(days_to_harvest, 15)
        elif growth_stage == "Flowering":
            days_to_harvest = max(18, int(gdd_remaining / avg_daily_gdd))
            days_to_harvest = min(days_to_harvest, 35)
        elif growth_stage == "Vegetative":
            days_to_harvest = max(45, int(gdd_remaining / avg_daily_gdd))
        else: # Nursery
            days_to_harvest = max(70, int(gdd_remaining / avg_daily_gdd))
            
        predicted_harvest_date = today + datetime.timedelta(days=days_to_harvest)
        
        # Optimal window is a 5-day window centered around the predicted date
        optimal_window_start = predicted_harvest_date - datetime.timedelta(days=2)
        optimal_window_end = predicted_harvest_date + datetime.timedelta(days=2)
        
        # Harvest readiness percentage
        total_cycle_days = 91 if self.crop_type == "Paddy Rice" else 110
        readiness_pct = min(100.0, (days_after_transplanting / total_cycle_days) * 100.0)
        if growth_stage == "Mature":
            readiness_pct = max(80.0, readiness_pct)
        elif growth_stage == "Flowering":
            readiness_pct = min(79.0, max(60.0, readiness_pct))
            
        avg_yield = float(yield_map.mean())
        total_prod = avg_yield * field_area_ha
        avg_biomass = float(biomass_map.mean())
        
        # Identify limiting factors
        limiting_factors = []
        recommendations = []
        
        # 1. Weather constraints
        rain_prob = max([w.get("precipitation_probability", 0.0) for w in weather_forecast])
        if rain_prob > 70.0:
            limiting_factors.append("High Pre-Harvest Precipitation Risk")
            recommendations.append("High rain risk detected near harvest. Monitor field soil dryness. If wet weather persists, consider pre-emptive harvest to prevent grain lodging and head-rice spoilage.")
            
        # 2. Yield/Stress factors
        avg_stress = (biomass_map * 0.48 - yield_map).mean() / max(0.1, avg_biomass)
        if avg_stress > 0.15:
            limiting_factors.append("Localized Crop Stress Penalty")
            recommendations.append("Severe stress hotspots are lowering potential grain yields. Focus post-harvest residue recycling in stressed regions to rebuild soil carbon.")
            
        # 3. Weather temperature deficit
        avg_temp = np.mean(forecast_temps)
        if avg_temp < self.params["t_base"] + 2.0:
            limiting_factors.append("Low Temperature Growth Retardation")
            recommendations.append("Low thermal accumulation (low GDD). Grain ripening will be delayed. Extend observation phase.")
            
        if not limiting_factors:
            limiting_factors.append("None (Optimal Climatic & Crop Conditions)")
            recommendations.append("Crop is progressing optimally. Schedule standard machinery operation for the projected window.")
            
        # Normal recommendations
        recommendations.append("Maintain drainage channel clearance on final week to allow soil drying for harvester weight load.")
        
        return HarvestForecast(
            predicted_harvest_date=predicted_harvest_date,
            days_to_harvest=days_to_harvest,
            optimal_window_start=optimal_window_start,
            optimal_window_end=optimal_window_end,
            harvest_readiness_pct=float(round(readiness_pct, 1)),
            average_yield_t_ha=float(round(avg_yield, 2)),
            total_production_t=float(round(total_prod, 2)),
            estimated_biomass_t_ha=float(round(avg_biomass, 2)),
            limiting_factors=limiting_factors,
            harvest_recommendations=recommendations
        )

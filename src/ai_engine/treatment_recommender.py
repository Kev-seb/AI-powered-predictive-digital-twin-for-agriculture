"""
treatment_recommender.py
-------------------------
Production-grade AI Treatment Recommendation & Fungicide Optimization Engine.
Includes nutrient clustering, disease risk probability mapping, and VRA GIS export.
"""

from __future__ import annotations

import datetime
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class TreatmentRecommendation:
    category: str               # "Nutrient", "Irrigation", "Fungicide", "Pesticide"
    severity_score: float       # 0.0 to 1.0
    confidence_score: float     # 0.0 to 1.0
    dosage_rate: str            # e.g., "120 kg/ha N", "15 mm", "1.5 L/ha"
    action_note: str
    urgency: str                # "Low", "Medium", "High", "Critical"
    spray_window: str           # e.g., "Next 24-48h (Dry conditions)"


@dataclass
class ZonePrescription:
    zone_id: int
    zone_name: str
    area_pct: float
    ndvi_mean: float
    ndre_mean: float
    cire_mean: float
    ndwi_mean: float
    n_deficiency: str           # "None", "Mild", "Moderate", "Severe"
    fungal_risk_prob: float     # 0.0 to 1.0
    recommendations: List[TreatmentRecommendation]


class AITreatmentRecommender:
    def __init__(self, settings: Optional[Any] = None):
        self.settings = settings

    def run_nutrient_clustering(
        self,
        ndvi: np.ndarray,
        ndre: np.ndarray,
        cire: np.ndarray,
        k: int = 4
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Segment the field into k homogeneous nutrient zones using a lightweight
        NumPy-based KMeans implementation. Avoids scikit-learn dependency issues.
        """
        H, W = ndvi.shape
        # Stack indices and flatten to (N, 3)
        features = np.stack([ndvi.ravel(), ndre.ravel(), cire.ravel()], axis=1)
        
        # Subsample points for fast clustering
        n_samples = min(5000, features.shape[0])
        idx = np.random.choice(features.shape[0], n_samples, replace=False)
        sub_features = features[idx]
        
        # Initialize centroids randomly
        c_idx = np.random.choice(n_samples, k, replace=False)
        centroids = sub_features[c_idx]
        
        for _ in range(50):
            # Distance computation: (n_samples, k)
            dists = np.linalg.norm(sub_features[:, np.newaxis] - centroids, axis=2)
            labels = np.argmin(dists, axis=1)
            
            # Update centroids
            new_centroids = np.zeros_like(centroids)
            for j in range(k):
                mask = labels == j
                if np.any(mask):
                    new_centroids[j] = sub_features[mask].mean(axis=0)
                else:
                    new_centroids[j] = centroids[j]
                    
            if np.allclose(centroids, new_centroids, atol=1e-4):
                break
            centroids = new_centroids
            
        # Assign all pixels to closest centroid
        all_dists = np.linalg.norm(features[:, np.newaxis] - centroids, axis=2)
        all_labels = np.argmin(all_dists, axis=1).reshape(H, W)
        
        # Sort centroids by NDVI mean so that label 0 is lowest productivity (stress/bare)
        # and label k-1 is highest productivity
        ndvi_means = centroids[:, 0]
        sort_idx = np.argsort(ndvi_means)
        
        sorted_labels = np.zeros_like(all_labels)
        sorted_centroids = np.zeros_like(centroids)
        for new_label, old_label in enumerate(sort_idx):
            sorted_labels[all_labels == old_label] = new_label
            sorted_centroids[new_label] = centroids[old_label]
            
        return sorted_labels, sorted_centroids

    def predict_fungal_risk(
        self,
        ndwi: np.ndarray,
        stress_score: np.ndarray,
        temp_c: float,
        humidity_pct: float,
        precip_prob: float
    ) -> Tuple[np.ndarray, float]:
        """
        Predict localized fungal outbreak probability using climate variables,
        water index (moisture), and spatial stress zones.
        
        Fungal pathogens thrive in warm, highly humid, and water-logged areas.
        """
        # Climate risk: high humidity (> 75%) and warm temp (18 - 30 C) favor growth
        temp_factor = np.clip(1.0 - abs(temp_c - 24.0) / 10.0, 0.0, 1.0) # peak at 24C
        humidity_factor = np.clip((humidity_pct - 50.0) / 40.0, 0.0, 1.0) # rises above 50%
        climate_risk = temp_factor * humidity_factor
        
        # Localized risk map based on NDWI (moisture) and composite stress
        ndwi_norm = np.clip(ndwi, 0, 1) # positive NDWI = moisture/water logging
        pixel_risk = 0.6 * ndwi_norm + 0.4 * stress_score
        
        # Combined risk probability map
        risk_map = np.clip(pixel_risk * (0.3 + 0.7 * climate_risk), 0.0, 1.0)
        mean_field_risk = float(risk_map.mean())
        
        return risk_map, mean_field_risk

    def estimate_nitrogen_deficiency(
        self,
        ndre_mean: float,
        cire_mean: float
    ) -> Tuple[str, float]:
        """
        Estimate Nitrogen deficiency severity using NDRE and CIre.
        Chlorophyll Index (CIre) is directly proportional to leaf nitrogen.
        """
        # Typical healthy CIre range: 1.5 - 4.5
        # Typical healthy NDRE range: 0.35 - 0.70
        if cire_mean < 0.8 or ndre_mean < 0.20:
            return "Severe", 0.90
        elif cire_mean < 1.5 or ndre_mean < 0.35:
            return "Moderate", 0.70
        elif cire_mean < 2.5 or ndre_mean < 0.45:
            return "Mild", 0.40
        else:
            return "None", 0.10

    def generate_zone_prescriptions(
        self,
        indices: Dict[str, np.ndarray],
        weather: Dict[str, Any],
        crop_stage: str,
        n_zones: int = 4
    ) -> Tuple[List[ZonePrescription], np.ndarray]:
        """
        Generate Variable-Rate prescriptions across the clustered field zones.
        """
        ndvi = indices["ndvi"]
        ndre = indices["ndre"]
        cire = indices["cire"]
        ndwi = indices["ndwi"]
        stress = indices["stress_score"]
        
        # 1. Cluster field into homogeneous zones
        labels, centroids = self.run_nutrient_clustering(ndvi, ndre, cire, k=n_zones)
        
        temp_c = weather.get("temperature", 25.0)
        humidity_pct = weather.get("humidity", 75.0)
        precip_prob = weather.get("precipitation_probability", 20.0)
        wind_speed = weather.get("wind_speed", 10.0)
        
        # Defensive conversion if lists are passed
        if isinstance(temp_c, list):
            temp_c = float(temp_c[0]) if len(temp_c) > 0 else 25.0
        if isinstance(humidity_pct, list):
            humidity_pct = float(humidity_pct[0]) if len(humidity_pct) > 0 else 75.0
        if isinstance(precip_prob, list):
            precip_prob = float(precip_prob[0]) if len(precip_prob) > 0 else 20.0
        if isinstance(wind_speed, list):
            wind_speed = float(wind_speed[0]) if len(wind_speed) > 0 else 10.0
            
        temp_c = float(temp_c)
        humidity_pct = float(humidity_pct)
        precip_prob = float(precip_prob)
        wind_speed = float(wind_speed)
        
        # 3. Compute fungal risk map
        risk_map, _ = self.predict_fungal_risk(ndwi, stress, temp_c, humidity_pct, precip_prob)
        
        total_pixels = ndvi.size
        prescriptions = []
        
        zone_names = {
            0: "Zone A: Severe Stress / Low Vigour",
            1: "Zone B: Moderate Stress / Under-performing",
            2: "Zone C: Optimal / Transitioning",
            3: "Zone D: High Productivity / Dense Canopy"
        }
        if n_zones != 4:
            zone_names = {i: f"Zone {chr(65+i)}: Performance Level {i+1}" for i in range(n_zones)}
            
        for i in range(n_zones):
            mask = labels == i
            if not np.any(mask):
                continue
                
            area_pct = float(mask.sum() / total_pixels * 100)
            ndvi_m = float(ndvi[mask].mean())
            ndre_m = float(ndre[mask].mean())
            cire_m = float(cire[mask].mean())
            ndwi_m = float(ndwi[mask].mean())
            stress_m = float(stress[mask].mean())
            fungal_risk_m = float(risk_map[mask].mean())
            
            # Nitrogen deficiency estimation
            n_def, n_sev = self.estimate_nitrogen_deficiency(ndre_m, cire_m)
            
            recs = []
            
            # A. Variable Rate Fertilizer N-P-K recommendation
            # Higher deficiency -> more N needed. Dense canopy -> maintain maintenance dose.
            if n_def == "Severe":
                n_dose = "120 kg/ha N"
                p_dose = "45 kg/ha P2O5"
                k_dose = "60 kg/ha K2O"
                n_rec_text = "Severe chlorophyll depletion. Apply high N correction immediately."
                n_sev_val = 0.9
            elif n_def == "Moderate":
                n_dose = "80 kg/ha N"
                p_dose = "30 kg/ha P2O5"
                k_dose = "45 kg/ha K2O"
                n_rec_text = "Moderate nitrogen deficit. Variable rate booster advised."
                n_sev_val = 0.6
            elif n_def == "Mild":
                n_dose = "40 kg/ha N"
                p_dose = "20 kg/ha P2O5"
                k_dose = "30 kg/ha K2O"
                n_rec_text = "Mild nutrient lag. Standard maintenance application."
                n_sev_val = 0.3
            else:
                n_dose = "15 kg/ha N (Spoon feeding)"
                p_dose = "0 kg/ha P2O5"
                k_dose = "15 kg/ha K2O"
                n_rec_text = "Canopy saturated. Skip large inputs to prevent lodging or disease."
                n_sev_val = 0.05
                
            recs.append(TreatmentRecommendation(
                category="Nutrient",
                severity_score=n_sev_val,
                confidence_score=0.85,
                dosage_rate=f"{n_dose} | {p_dose} | {k_dose}",
                action_note=n_rec_text,
                urgency="High" if n_def in ["Severe", "Moderate"] else "Low",
                spray_window="Next 4 days (Clear weather)"
            ))
            
            # B. Irrigation Recommendation
            # Under water stress: positive NDWI. FAO reference crop requirement.
            if ndwi_m > 0.15 or stress_m > 0.60:
                irrig_mm = "25 mm (Deep wetting)"
                irrig_rec = "Soil moisture deficit detected. Deep irrigation cycle needed."
                irrig_urg = "High" if ndwi_m > 0.30 else "Medium"
                irrig_sev = min(1.0, float(ndwi_m * 2))
            elif ndwi_m < -0.10:
                irrig_mm = "0 mm"
                irrig_rec = "Adequate soil/canopy water. Suspend irrigation."
                irrig_urg = "Low"
                irrig_sev = 0.0
            else:
                irrig_mm = "10 mm (Maintenance)"
                irrig_rec = "Normal moisture. Maintain light irrigation cycle."
                irrig_urg = "Low"
                irrig_sev = 0.2
                
            recs.append(TreatmentRecommendation(
                category="Irrigation",
                severity_score=irrig_sev,
                confidence_score=0.80,
                dosage_rate=irrig_mm,
                action_note=irrig_rec,
                urgency=irrig_urg,
                spray_window="Next 24h"
            ))
            
            # C. Fungicide Recommendation
            # Combines local moisture, stress, and weather (humidity)
            if fungal_risk_m > 0.70:
                fung_dose = "2.0 L/ha (Systemic Triazole)"
                fung_rec = "CRITICAL fungal risk: High moisture & temperature. Immediate spray required."
                fung_urg = "Critical"
                spray_win = "Immediate (Wind < 12 km/h, No rain expected)"
            elif fungal_risk_m > 0.45:
                fung_dose = "1.2 L/ha (Strobilurin protective)"
                fung_rec = "Moderate risk: Humid microclimate. Prophylactic application recommended."
                fung_urg = "Medium"
                spray_win = "Next 48h (Dry window)"
            else:
                fung_dose = "0.0 L/ha"
                fung_rec = "Fungal pressure low. Regular scouting is sufficient."
                fung_urg = "Low"
                spray_win = "Not required"
                
            recs.append(TreatmentRecommendation(
                category="Fungicide",
                severity_score=fungal_risk_m,
                confidence_score=0.88,
                dosage_rate=fung_dose,
                action_note=fung_rec,
                urgency=fung_urg,
                spray_window=spray_win
            ))
            
            prescriptions.append(ZonePrescription(
                zone_id=i,
                zone_name=zone_names[i],
                area_pct=area_pct,
                ndvi_mean=ndvi_m,
                ndre_mean=ndre_m,
                cire_mean=cire_m,
                ndwi_mean=ndwi_m,
                n_deficiency=n_def,
                fungal_risk_prob=fungal_risk_m,
                recommendations=recs
            ))
            
        return prescriptions, labels

    def export_gis_geojson(
        self,
        prescriptions: List[ZonePrescription],
        zone_labels: np.ndarray,
        center_lat: float,
        center_lon: float,
        output_file: str,
        gsd: float = 0.05
    ) -> None:
        """
        Build and export a GIS GeoJSON map of VRA treatment rates for precision controllers.
        """
        import shapely.geometry
        import geopandas as gpd
        
        H, W = zone_labels.shape
        lat_deg_per_meter = 1.0 / 111120.0
        lon_deg_per_meter = 1.0 / (111120.0 * np.cos(np.radians(center_lat)))
        
        polygons = []
        properties = []
        
        # Grid aggregation to make geometry clean and fast
        grid_size = 16
        for r in range(0, H, grid_size):
            for c in range(0, W, grid_size):
                r_end = min(H, r + grid_size)
                c_end = min(W, c + grid_size)
                
                # Major label in grid cell
                patch = zone_labels[r:r_end, c:c_end]
                if patch.size == 0:
                    continue
                z_id = int(np.bincount(patch.ravel()).argmax())
                
                # Coordinate vertices
                y0 = center_lat + (H/2 - r) * gsd * lat_deg_per_meter
                y1 = center_lat + (H/2 - r_end) * gsd * lat_deg_per_meter
                x0 = center_lon + (c - W/2) * gsd * lon_deg_per_meter
                x1 = center_lon + (c_end - W/2) * gsd * lon_deg_per_meter
                
                poly = shapely.geometry.box(x0, y1, x1, y0)
                polygons.append(poly)
                
                # Fetch prescriptions for this zone
                zp = next((p for p in prescriptions if p.zone_id == z_id), None)
                if zp:
                    n_rec = next((r for r in zp.recommendations if r.category == "Nutrient"), None)
                    irrig_rec = next((r for r in zp.recommendations if r.category == "Irrigation"), None)
                    fung_rec = next((r for r in zp.recommendations if r.category == "Fungicide"), None)
                    
                    properties.append({
                        "zone_id": z_id,
                        "zone_name": zp.zone_name,
                        "n_dose": n_rec.dosage_rate if n_rec else "0",
                        "irrig_mm": irrig_rec.dosage_rate if irrig_rec else "0",
                        "fung_dose": fung_rec.dosage_rate if fung_rec else "0",
                        "n_deficit": zp.n_deficiency,
                        "fung_risk": float(round(zp.fungal_risk_prob, 3))
                    })
                else:
                    properties.append({"zone_id": z_id, "zone_name": f"Zone {z_id}"})
                    
        gdf = gpd.GeoDataFrame(properties, geometry=polygons, crs="EPSG:4326")
        out_path = Path(output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        gdf.to_file(str(out_path), driver="GeoJSON")

    def generate_uav_inspection_mission(
        self,
        anomaly_coords_list: List[Tuple[float, float]],
        base_altitude: float = 30.0
    ) -> Dict[str, Any]:
        """
        Takes GPS coordinates of anomalies detected by the Sentinel-2 Satellite
        and generates a targeted QGroundControl (QGC) waypoint mission for the UAV.
        """
        if not anomaly_coords_list:
            return {"error": "No anomalies provided."}
            
        waypoints = []
        for i, (lon, lat) in enumerate(anomaly_coords_list):
            waypoints.append({
                "command": 16, # MAV_CMD_NAV_WAYPOINT
                "coordinate": [lat, lon, base_altitude],
                "autoContinue": True,
                "param1": 0, # Hold time
                "param2": 2, # Acceptance radius
                "param3": 0,
                "param4": 0,
                "type": "missionItem"
            })
            
            # Add a loiter/camera trigger at the anomaly
            waypoints.append({
                "command": 2000, # MAV_CMD_IMAGE_START_CAPTURE
                "coordinate": [lat, lon, base_altitude],
                "autoContinue": True,
                "param1": 0,
                "param2": 0,
                "param3": 1,
                "param4": 0,
                "type": "missionItem"
            })
            
        mission = {
            "fileType": "Plan",
            "geoFence": {"polygon": [], "version": 2},
            "groundStation": "AI Digital Twin",
            "mission": {
                "cruiseSpeed": 5.0,
                "hoverSpeed": 2.0,
                "items": waypoints,
                "plannedHomePosition": [
                    anomaly_coords_list[0][1],
                    anomaly_coords_list[0][0],
                    0.0
                ],
                "vehicleType": 2,
                "version": 2
            },
            "rallyPoints": {"points": [], "version": 2},
            "version": 1
        }
        
        return mission


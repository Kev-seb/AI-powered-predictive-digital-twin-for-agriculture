"""
simulator.py
------------
Vectorized multi-layer physical/agronomic simulator for predictive digital twin playback.
Simulates crop health recovery, soil moisture depletion, lateral nutrient diffusion,
fungal pathogen spreading, and generates MAVLink/QGroundControl flight missions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np

from src.ai_engine.epidemiology import EpidemiologyForecaster


class AgronomicDigitalTwinSimulator:
    def __init__(
        self,
        ndvi: np.ndarray,
        stress: np.ndarray,
        ndwi: np.ndarray,
        cire: np.ndarray,
        zone_labels: np.ndarray,
        zone_names: Dict[int, str]
    ):
        self.H, self.W = ndvi.shape
        self.zone_labels = zone_labels
        self.zone_names = zone_names

        # Initialize multi-layer simulation grids
        self.ndvi = np.copy(ndvi).astype(np.float32)
        self.stress = np.copy(stress).astype(np.float32)
        
        # Map NDWI [-1, 1] to Soil Moisture [0, 1]
        self.moisture = np.clip((ndwi + 1.0) / 2.0, 0.05, 0.95).astype(np.float32)
        
        # Map CIre [0, 5] to Soil Nitrogen [0, 1]
        self.nitrogen = np.clip(cire / 5.0, 0.1, 1.0).astype(np.float32)
        
        # Initialize Fungal Pressure based on stress score and moisture
        self.fungus = np.clip(stress * 0.4 + np.clip(ndwi, 0, 1) * 0.3, 0.0, 1.0).astype(np.float32)

        # Initialize epidemiological layers
        self.fungus_velocity = np.zeros_like(self.ndvi, dtype=np.float32)
        self.fungus_direction = np.zeros((2, self.H, self.W), dtype=np.float32)
        self.fungus_urgency = np.zeros_like(self.ndvi, dtype=np.float32)
        self.fungus_boundaries = np.zeros_like(self.ndvi, dtype=np.float32)
        self.forecaster = EpidemiologyForecaster()

    def simulate_day(
        self,
        weather: Dict[str, Any],
        interventions: List[Dict[str, Any]],
        propagation_model: str = "pde",
        growth_stage: str = "Vegetative"
    ) -> List[str]:
        """
        Advance the spatial physical simulator by 1 day.
        Returns a list of text logs detailing physical events that occurred.
        """
        logs = []
        
        # Extract climate parameters
        temp_c = float(weather.get("temperature", 24.5))
        humidity = float(weather.get("humidity", 75.0))
        rain = float(weather.get("precipitation", 0.0))
        wind = float(weather.get("wind_speed", 10.0))

        # ----------------- 1. IRRIGATION & MOISTURE UPDATES -----------------
        # Evapotranspiration depletion (higher with temp, wind, and vegetation density)
        et = 0.03 * (temp_c / 25.0) * (1.0 + wind / 35.0) * (0.4 + 0.6 * np.clip(self.ndvi, 0.0, 1.0))
        self.moisture = np.clip(self.moisture - et, 0.0, 1.0)
        
        # Rain replenishment
        if rain > 0.0:
            rain_replenish = rain * 0.06
            self.moisture = np.clip(self.moisture + rain_replenish, 0.0, 1.0)
            logs.append(f"Rainfall event of {rain:.1f} mm replenished soil moisture across the field.")

        # Process irrigation interventions
        irrig_zones = [int(intv["zone_id"]) for intv in interventions if intv["type"] == "Precision Irrigation"]
        for z_id in irrig_zones:
            mask = self.zone_labels == z_id
            if np.any(mask):
                self.moisture[mask] = np.clip(self.moisture[mask] + 0.35, 0.0, 1.0)
                logs.append(f"Irrigation applied to {self.zone_names.get(z_id, f'Zone {z_id}')}, replenishing soil moisture.")

        # ----------------- 2. FERTILIZER & NITROGEN UPDATES -----------------
        # Nitrogen depletion via crop absorption and growth
        growth_depletion = 0.015 * np.clip(self.ndvi, 0.0, 1.0) * (self.moisture > 0.25).astype(np.float32)
        self.nitrogen = np.clip(self.nitrogen - growth_depletion, 0.0, 1.0)

        # Soil Nitrogen lateral diffusion (lateral flow of nutrients in soil moisture)
        # Implement lateral soil flow by averaging neighbor shifts
        shifted = np.zeros_like(self.nitrogen)
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                shifted += np.roll(np.roll(self.nitrogen, dx, axis=1), dy, axis=0)
        # Merge diffused nitrogen slightly (5% lateral diffusion rate)
        self.nitrogen = 0.95 * self.nitrogen + 0.05 * (shifted / 9.0)

        # Process fertilization interventions
        fert_zones = [int(intv["zone_id"]) for intv in interventions if intv["type"] == "Nutrient Top-Dress"]
        for z_id in fert_zones:
            mask = self.zone_labels == z_id
            if np.any(mask):
                self.nitrogen[mask] = np.clip(self.nitrogen[mask] + 0.40, 0.0, 1.0)
                logs.append(f"Nitrogen fertilizer applied to {self.zone_names.get(z_id, f'Zone {z_id}')}.")

        # ----------------- 3. FUNGICIDE & FUNGAL SPREAD UPDATES -----------------
        # Build fungicide spray coverage map
        fungicide_mask = np.zeros_like(self.fungus)
        fung_zones = [int(intv["zone_id"]) for intv in interventions if intv["type"] == "Fungicide Spray"]
        for z_id in fung_zones:
            mask = self.zone_labels == z_id
            if np.any(mask):
                fungicide_mask[mask] = 1.0
                logs.append(f"Fungicide spray scheduled for {self.zone_names.get(z_id, f'Zone {z_id}')}.")

        # Growth stage susceptibility factor mapping
        stage_susceptibility = {"Emergence": 0.4, "Vegetative": 0.7, "Flowering": 1.0, "Senescence": 0.8}
        stage_mult = stage_susceptibility.get(growth_stage, 0.7)

        # Select spatiotemporal propagation model
        if propagation_model == "baseline":
            # Baseline rolling average diffusion
            temp_factor = np.clip(1.0 - abs(temp_c - 24.0) / 10.0, 0.0, 1.0)
            humidity_factor = np.clip((humidity - 50.0) / 45.0, 0.0, 1.0)
            climate_suitability = temp_factor * humidity_factor
            fungal_growth = 0.04 * climate_suitability * (self.moisture > 0.48).astype(np.float32)
            self.fungus = np.clip(self.fungus + fungal_growth * stage_mult * (1.0 - 0.85 * fungicide_mask), 0.0, 1.0)
            
            shifted_fungus = np.zeros_like(self.fungus)
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    shifted_fungus += np.roll(np.roll(self.fungus, dx, axis=1), dy, axis=0)
            self.fungus = 0.92 * self.fungus + 0.08 * (shifted_fungus / 9.0)
            # Subtract direct spray
            self.fungus = np.clip(self.fungus - 0.75 * fungicide_mask, 0.0, 1.0)
            
            # Simple velocity/direction fallback
            self.fungus_velocity = np.clip(fungal_growth * (1.0 - fungicide_mask), 0.0, 1.0)
            self.fungus_direction = np.zeros((2, self.H, self.W), dtype=np.float32)
            
        elif propagation_model == "pde" or propagation_model == "hybrid":
            # Advanced Fisher-Kolmogorov reaction-diffusion PDE
            P_curr = np.copy(self.fungus)
            dt_step = 0.1
            for _ in range(10):
                P_curr, vel, direct = self.forecaster.simulate_pde_step(
                    pathogen=P_curr,
                    ndvi=self.ndvi,
                    weather=weather,
                    growth_stage_susceptibility=stage_mult,
                    fungicide=fungicide_mask,
                    dt=dt_step
                )
            
            # Hybrid model: add graph message passing spore clouds on top of PDE diffusion
            if propagation_model == "hybrid":
                # Compute zone properties
                zone_pressures = {}
                zone_centers = {}
                ndvi_means = {}
                fungicide_suppression = {}
                for z_id in self.zone_names:
                    mask = self.zone_labels == z_id
                    if np.any(mask):
                        zone_pressures[z_id] = float(P_curr[mask].mean())
                        rows, cols = np.where(mask)
                        zone_centers[z_id] = (float(rows.mean()), float(cols.mean()))
                        ndvi_means[z_id] = float(self.ndvi[mask].mean())
                        fungicide_suppression[z_id] = float(fungicide_mask[mask].mean())
                    else:
                        zone_pressures[z_id] = 0.0
                        zone_centers[z_id] = (0.0, 0.0)
                        ndvi_means[z_id] = 0.5
                        fungicide_suppression[z_id] = 0.0
                        
                _, t_probs = self.forecaster.simulate_gnn_step(
                    zone_pressures=zone_pressures,
                    zone_centers=zone_centers,
                    ndvi_means=ndvi_means,
                    weather=weather,
                    susceptibility_mult=stage_mult,
                    fungicide_suppression=fungicide_suppression
                )
                
                # Add GNN spore depositions
                for key, prob in t_probs.items():
                    if prob > 0.05:
                        parts = key.split("->")
                        src_id = int(parts[0])
                        dst_id = int(parts[1])
                        if zone_pressures[src_id] > 0.2:
                            dst_mask = self.zone_labels == dst_id
                            P_curr[dst_mask] = np.clip(P_curr[dst_mask] + 0.08 * prob * zone_pressures[src_id], 0.0, 1.0)
            
            self.fungus = P_curr
            self.fungus_velocity, _, self.fungus_direction = self.forecaster.simulate_pde_step(
                pathogen=self.fungus,
                ndvi=self.ndvi,
                weather=weather,
                growth_stage_susceptibility=stage_mult,
                fungicide=fungicide_mask,
                dt=1.0
            )
            
        elif propagation_model == "gnn":
            zone_pressures = {}
            zone_centers = {}
            ndvi_means = {}
            fungicide_suppression = {}
            for z_id in self.zone_names:
                mask = self.zone_labels == z_id
                if np.any(mask):
                    zone_pressures[z_id] = float(self.fungus[mask].mean())
                    rows, cols = np.where(mask)
                    zone_centers[z_id] = (float(rows.mean()), float(cols.mean()))
                    ndvi_means[z_id] = float(self.ndvi[mask].mean())
                    fungicide_suppression[z_id] = float(fungicide_mask[mask].mean())
                else:
                    zone_pressures[z_id] = 0.0
                    zone_centers[z_id] = (0.0, 0.0)
                    ndvi_means[z_id] = 0.5
                    fungicide_suppression[z_id] = 0.0
            
            next_pressures, _ = self.forecaster.simulate_gnn_step(
                zone_pressures=zone_pressures,
                zone_centers=zone_centers,
                ndvi_means=ndvi_means,
                weather=weather,
                susceptibility_mult=stage_mult,
                fungicide_suppression=fungicide_suppression
            )
            
            prev_fungus = np.copy(self.fungus)
            for z_id, pressure in next_pressures.items():
                mask = self.zone_labels == z_id
                self.fungus[mask] = pressure
                
            self.fungus_velocity = np.clip(self.fungus - prev_fungus, 0.0, 1.0)
            
            # Direction gradients from updated zones
            direction_y, direction_x = np.gradient(self.fungus)
            magnitude = np.sqrt(direction_x**2 + direction_y**2) + 1e-8
            direction_x = np.where(magnitude > 0.02, direction_x / magnitude, 0.0)
            direction_y = np.where(magnitude > 0.02, direction_y / magnitude, 0.0)
            self.fungus_direction = np.stack([direction_x, direction_y], axis=0)

        # Generate Intervention Urgency
        self.fungus_urgency = self.forecaster.generate_intervention_urgency(
            pathogen=self.fungus,
            velocity=self.fungus_velocity,
            fungicide=fungicide_mask
        )

        # Generate Probabilistic Boundaries
        boundary_dict = self.forecaster.generate_probabilistic_boundaries(self.fungus)
        self.fungus_boundaries = (
            boundary_dict[0.50] * 0.5 +
            boundary_dict[0.75] * 0.75 +
            boundary_dict[0.90] * 0.90
        ).astype(np.float32)

        # Process fungicide suppression logging (matching previous suppression outputs)
        for z_id in fung_zones:
            mask = self.zone_labels == z_id
            if np.any(mask):
                logs.append(f"Applied Fungicide Spray to {self.zone_names.get(z_id, f'Zone {z_id}')}, suppressing fungal pathogen load.")

        # Trigger outbreak warning logs
        for z_id in self.zone_names:
            mask = self.zone_labels == z_id
            if np.any(mask) and float(self.fungus[mask].mean()) > 0.50:
                logs.append(f"WARNING: High Fungal pathogen infestation detected in {self.zone_names[z_id]}! Pathogen load: {float(self.fungus[mask].mean())*100:.1f}%.")

        # ----------------- 4. CROP VIGOR (NDVI) & STRESS STAGE UPDATES -----------------
        # Calculate growth vs decline
        # A. Growth conditions: optimal moisture (30% to 80%), good nitrogen, low fungus
        growth_cond = (self.moisture >= 0.3) & (self.moisture <= 0.8) & (self.fungus < 0.25)
        growth_val = 0.02 * self.nitrogen * (self.moisture - 0.1) * growth_cond.astype(np.float32)
        
        # B. Decline conditions: fungus pressure, drought (moisture < 0.25), waterlogging (moisture > 0.85)
        decline_fungus = 0.045 * self.fungus
        decline_drought = 0.03 * np.clip(0.25 - self.moisture, 0, 1)
        decline_wet = 0.02 * np.clip(self.moisture - 0.85, 0, 1)
        decline_temp = 0.025 if (temp_c > 35.0 or temp_c < 6.0) else 0.0
        
        self.ndvi = np.clip(self.ndvi + growth_val - decline_fungus - decline_drought - decline_wet - decline_temp, -1.0, 1.0)
        
        # Sync stress score inversely to health + fungus influence
        self.stress = np.clip((1.0 - self.ndvi) / 2.0 + 0.35 * self.fungus, 0.0, 1.0)
        
        return logs

    def get_state_maps(self) -> Dict[str, np.ndarray]:
        """
        Return maps of the active simulation layers.
        """
        return {
            "ndvi": np.copy(self.ndvi),
            "moisture": np.copy(self.moisture),
            "nitrogen": np.copy(self.nitrogen),
            "fungus": np.copy(self.fungus),
            "stress": np.copy(self.stress),
            "fungus_velocity": np.copy(self.fungus_velocity),
            "fungus_direction": np.copy(self.fungus_direction),
            "fungus_urgency": np.copy(self.fungus_urgency),
            "fungus_boundaries": np.copy(self.fungus_boundaries)
        }


class QGCMissionGenerator:
    @staticmethod
    def generate_precision_flight_plan(
        prescriptions: List[Any],
        zone_labels: np.ndarray,
        center_lat: float,
        center_lon: float,
        gsd: float = 0.05,
        target_altitude_m: float = 5.0
    ) -> Dict[str, Any]:
        """
        Compile spatial VRA target zones into a standard waypoint grid flight mission in QGroundControl .mission format.
        Injects spray valve actuator settings (MAV_CMD_DO_SET_SERVO) when entering/exiting stress zones.
        """
        H, W = zone_labels.shape
        lat_deg_per_meter = 1.0 / 111320.0
        lon_deg_per_meter = 1.0 / (111320.0 * np.cos(center_lat * np.pi / 180.0))

        # Find which zones require active precision chemical application
        active_zones = []
        for p in prescriptions:
            needs_spray = False
            for rec in p.recommendations:
                if rec.category in ["Fungicide", "Nutrient"] and "0" not in rec.dosage_rate:
                    needs_spray = True
            if needs_spray:
                active_zones.append(p.zone_id)

        # Generate a grid flight path over the bounding boxes of active zones
        waypoints = []
        spray_triggers = [] # True/False spray command mapped to each waypoint

        # Add home position as entry
        waypoints.append((center_lat, center_lon, target_altitude_m))
        spray_triggers.append(False)

        for z_id in active_zones:
            # Find boundary coordinates in pixels
            rows, cols = np.where(zone_labels == z_id)
            if len(rows) == 0:
                continue
                
            r_min, r_max = int(rows.min()), int(rows.max())
            c_min, c_max = int(cols.min()), int(cols.max())

            # Convert to lat/lon bounding box
            # local coords: (0,0) is center of image
            y_min = center_lat + (H/2 - r_max) * gsd * lat_deg_per_meter
            y_max = center_lat + (H/2 - r_min) * gsd * lat_deg_per_meter
            x_min = center_lon + (c_min - W/2) * gsd * lon_deg_per_meter
            x_max = center_lon + (c_max - W/2) * gsd * lon_deg_per_meter

            # 4-point grid sweeping flight path inside the zone box
            waypoints.append((y_min, x_min, target_altitude_m))
            spray_triggers.append(True) # Start spray upon entering zone bounding box

            waypoints.append((y_min, x_max, target_altitude_m))
            spray_triggers.append(True)

            waypoints.append((y_max, x_max, target_altitude_m))
            spray_triggers.append(True)

            waypoints.append((y_max, x_min, target_altitude_m))
            spray_triggers.append(True)

            # Exit command
            waypoints.append((y_max, x_min, target_altitude_m))
            spray_triggers.append(False) # Turn spray off upon leaving zone

        # Final Return to Launch (RTL)
        waypoints.append((center_lat, center_lon, target_altitude_m))
        spray_triggers.append(False)

        # Construct QGroundControl JSON items list
        items = []
        
        # 1. Takeoff command
        items.append({
            "autoContinue": True,
            "command": 22, # MAV_CMD_NAV_TAKEOFF
            "doJumpId": 1,
            "frame": 3, # MAV_FRAME_GLOBAL_RELATIVE_ALT
            "params": [15, 0, 0, 0, 0, 0, target_altitude_m],
            "type": "SimpleItem"
        })

        # 2. Waypoints and Spray Trigger Servo items
        jump_id = 2
        for idx, (lat, lon, alt) in enumerate(waypoints):
            # Check if spray trigger changed from previous state
            is_spraying = spray_triggers[idx]
            prev_spraying = spray_triggers[idx - 1] if idx > 0 else False

            if is_spraying != prev_spraying:
                # Inject spray command: MAV_CMD_DO_SET_SERVO (servo channel 5, PWM value)
                pwm_val = 2000 if is_spraying else 1000
                items.append({
                    "autoContinue": True,
                    "command": 183, # MAV_CMD_DO_SET_SERVO
                    "doJumpId": jump_id,
                    "frame": 2, # MAV_FRAME_MISSION
                    "params": [5, pwm_val, 0, 0, 0, 0, 0],
                    "type": "SimpleItem"
                })
                jump_id += 1

            # Standard Navigate to Waypoint item
            items.append({
                "autoContinue": True,
                "command": 16, # MAV_CMD_NAV_WAYPOINT
                "doJumpId": jump_id,
                "frame": 3, # MAV_FRAME_GLOBAL_RELATIVE_ALT
                "params": [0, 0, 0, 0, lat, lon, alt],
                "type": "SimpleItem"
            })
            jump_id += 1

        # 3. Return to Launch item
        items.append({
            "autoContinue": True,
            "command": 20, # MAV_CMD_NAV_RETURN_TO_LAUNCH
            "doJumpId": jump_id,
            "frame": 2,
            "params": [0, 0, 0, 0, 0, 0, 0],
            "type": "SimpleItem"
        })

        # Compile Plan
        mission_plan = {
            "fileType": "Plan",
            "version": 1,
            "groundStation": "QGroundControl",
            "mission": {
                "plannedHomePosition": [center_lat, center_lon, 0.0],
                "hoverSpeed": 4.0,
                "flightSpeed": 6.5,
                "items": items,
                "vehicleType": 2 # Rotary wing
            }
        }
        
        return mission_plan

"""
epidemiology.py
---------------
Epidemiological disease spread forecasting engine for predictive crop digital twin.
Integrates weather dynamics, crop growth stages, and microclimates using:
1. Fisher-Kolmogorov PDE (anisotropic wind-skewed reaction-diffusion)
2. Directed Graph Neural Network (GNN) spore dispersal modeling
3. Temporal Transformer sequence forecasting
4. Spatiotemporal boundary, direction, and urgency estimation
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class TemporalEpidemiologyTransformer(nn.Module):
    """
    Lightweight Temporal Attention/Transformer model to forecast regional disease intensity
    based on sequence data of historical weather risks, growth stage, and previous stress values.
    """
    def __init__(self, input_dim: int = 5, model_dim: int = 16, num_heads: int = 2, num_layers: int = 1):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, model_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=32,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_head = nn.Sequential(
            nn.Linear(model_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, SeqLen, InputDim)
        projected = self.input_projection(x)
        features = self.transformer(projected)
        # Take the output of the last sequence step
        last_step = features[:, -1, :]
        out = self.output_head(last_step)
        return out


class EpidemiologyForecaster:
    def __init__(self, model_path: Optional[str] = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transformer = TemporalEpidemiologyTransformer(input_dim=5, model_dim=16)
        self.transformer.to(self.device)
        self.transformer.eval()
        
        if model_path and os.path.exists(model_path):
            try:
                self.transformer.load_state_dict(torch.load(model_path, map_location=self.device))
            except Exception as e:
                print(f"[WARN] Failed to load Temporal Transformer: {e}")

    @staticmethod
    def calculate_climate_suitability(temperature: float, humidity: float) -> float:
        """
        Calculate fungal pathogen suitability index based on temperature and relative humidity.
        Optimal temp range for fungal growth is 20-28 C. High humidity is required.
        """
        # Gaussian temperature suitability (peak at 24 C)
        temp_factor = np.exp(-((temperature - 24.0) ** 2) / (2 * (5.0 ** 2)))
        # Humidity factor (highly favorable above 70% RH)
        humidity_factor = np.clip((humidity - 50.0) / 40.0, 0.0, 1.0)
        return float(temp_factor * humidity_factor)

    @staticmethod
    def calculate_canopy_wetness(ndvi: np.ndarray, humidity: float, temperature: float) -> np.ndarray:
        """
        Estimate spatial canopy wetness based on local vegetation density (NDVI)
        and ambient atmospheric humidity and temperature.
        """
        # More leaves = higher transpiration and dew retention
        ndvi_clamped = np.clip(ndvi, 0.0, 1.0)
        rh_factor = np.clip((humidity - 40.0) / 50.0, 0.0, 1.0)
        # Lower temp at night/early morning causes dew condensation
        dew_factor = np.clip(1.0 - (temperature - 15.0) / 20.0, 0.1, 1.0)
        return ndvi_clamped * rh_factor * dew_factor

    def simulate_pde_step(
        self,
        pathogen: np.ndarray,
        ndvi: np.ndarray,
        weather: Dict[str, Any],
        growth_stage_susceptibility: float,
        fungicide: np.ndarray,
        dt: float = 0.1,
        dx: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Fisher-Kolmogorov PDE reaction-diffusion equation solver with anisotropic wind drift.
        
        dP/dt = D * Laplacian(P) - beta * (V_wind . grad(P)) + r * P * (1 - P/K) - u * P
        """
        H, W = pathogen.shape
        P = np.copy(pathogen).astype(np.float64)
        K = np.clip(ndvi, 0.01, 1.0).astype(np.float64) # Carrying capacity
        
        temp = float(weather.get("temperature", 24.0))
        humidity = float(weather.get("humidity", 75.0))
        wind_speed = float(weather.get("wind_speed", 10.0))
        wind_dir_deg = float(weather.get("wind_direction", 0.0))
        
        suitability = self.calculate_climate_suitability(temp, humidity)
        canopy_wetness = self.calculate_canopy_wetness(ndvi, humidity, temp)
        
        # Growth rate suppressed by fungicide
        r = 0.12 * suitability * canopy_wetness * growth_stage_susceptibility * (1.0 - 0.85 * fungicide)
        
        # Wind-blown spatial expansion
        D = 0.02 * (1.0 + 0.15 * wind_speed)
        
        theta = np.radians(wind_dir_deg)
        vx = np.sin(theta)
        vy = -np.cos(theta)
        
        grad_x = np.zeros_like(P)
        grad_y = np.zeros_like(P)
        grad_x[:, 1:-1] = (P[:, 2:] - P[:, :-2]) / (2 * dx)
        grad_y[1:-1, :] = (P[2:, :] - P[:-2, :]) / (2 * dx)
        advection = -(vx * grad_x + vy * grad_y) * wind_speed * 0.012
        
        laplacian = np.zeros_like(P)
        laplacian[1:-1, 1:-1] = (
            P[1:-1, 2:] + P[1:-1, :-2] + P[2:, 1:-1] + P[:-2, 1:-1] - 4 * P[1:-1, 1:-1]
        ) / (dx ** 2)
        
        u = 0.25 * fungicide
        reaction = r * P * (1.0 - P / K) - u * P
        
        dP_dt = D * laplacian + advection + reaction
        P_next = np.clip(P + dP_dt * dt, 0.0, 1.0)
        
        velocity = np.clip(P_next - P, 0.0, 1.0) / dt
        
        # Wave-front expansion direction (gradient direction)
        direction_y, direction_x = np.gradient(P)
        magnitude = np.sqrt(direction_x**2 + direction_y**2) + 1e-8
        direction_x = np.where(magnitude > 0.02, direction_x / magnitude, 0.0)
        direction_y = np.where(magnitude > 0.02, direction_y / magnitude, 0.0)
        direction_vectors = np.stack([direction_x, direction_y], axis=0)
        
        return P_next.astype(np.float32), velocity.astype(np.float32), direction_vectors.astype(np.float32)

    def simulate_gnn_step(
        self,
        zone_pressures: Dict[int, float],
        zone_centers: Dict[int, Tuple[float, float]],
        ndvi_means: Dict[int, float],
        weather: Dict[str, Any],
        susceptibility_mult: float,
        fungicide_suppression: Dict[int, float]
    ) -> Tuple[Dict[int, float], Dict[int, float]]:
        """
        Directed Graph Neural Network message passing simulator.
        Models spore transport between field zone nodes based on distance and wind vectors.
        """
        temp = float(weather.get("temperature", 24.0))
        humidity = float(weather.get("humidity", 75.0))
        wind_speed = float(weather.get("wind_speed", 10.0))
        wind_dir_deg = float(weather.get("wind_direction", 0.0))
        
        suitability = self.calculate_climate_suitability(temp, humidity)
        theta_wind = np.radians(wind_dir_deg)
        
        ids = list(zone_pressures.keys())
        n = len(ids)
        
        A = np.zeros((n, n))
        for i in range(n):
            z_i = ids[i]
            x_i, y_i = zone_centers[z_i]
            for j in range(n):
                if i == j:
                    continue
                z_j = ids[j]
                x_j, y_j = zone_centers[z_j]
                
                dx = x_j - x_i
                dy = y_j - y_i
                dist = np.sqrt(dx**2 + dy**2) + 1e-8
                theta_edge = np.arctan2(dx, -dy)
                
                wind_align = max(0.0, np.cos(theta_edge - theta_wind))
                decay_length = 50.0 + 3.0 * wind_speed
                spore_dispersal = np.exp(-dist / decay_length) * (0.2 + 0.8 * wind_align)
                A[i, j] = spore_dispersal
                
        transmission_probs = {}
        next_pressures = {}
        
        for j in range(n):
            z_j = ids[j]
            P_j = zone_pressures[z_j]
            suppressed = fungicide_suppression.get(z_j, 0.0)
            
            local_grow = 0.05 * suitability * ndvi_means[z_j] * susceptibility_mult * (1.0 - 0.8 * suppressed)
            
            spore_inflow = 0.0
            for i in range(n):
                z_i = ids[i]
                P_i = zone_pressures[z_i]
                t_prob = A[i, j] * P_i * suitability * (1.0 - 0.7 * suppressed)
                spore_inflow += t_prob * P_i
                transmission_probs[f"{z_i}->{z_j}"] = float(t_prob)
                
            P_j_next = P_j + (local_grow * P_j * (1.0 - P_j) + spore_inflow * (1.0 - P_j) - 0.15 * suppressed * P_j)
            next_pressures[z_j] = float(np.clip(P_j_next, 0.0, 1.0))
            
        return next_pressures, transmission_probs

    def run_transformer_forecast(
        self,
        historical_seq: List[Dict[str, Any]],
        future_weather: List[Dict[str, Any]],
        growth_stage: str
    ) -> float:
        """
        Execute Transformer sequence forecasting to predict overall field infection risk trend.
        """
        stage_map = {"Emergence": 0.1, "Vegetative": 0.3, "Flowering": 0.8, "Senescence": 0.5}
        stage_val = stage_map.get(growth_stage, 0.3)
        
        seq_len = len(historical_seq)
        if seq_len < 1:
            return 0.1
            
        features = []
        for state in historical_seq:
            features.append([
                float(state.get("mean_ndvi", 0.65)),
                float(state.get("mean_stress", 0.1)),
                float(state.get("temp", 24.0)),
                float(state.get("humidity", 75.0)),
                stage_val
            ])
            
        while len(features) < 4:
            features.insert(0, features[0])
            
        input_tensor = torch.tensor([features[-8:]], dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            try:
                pred = self.transformer(input_tensor)
                return float(pred[0, 0].cpu().item())
            except Exception:
                last_stress = features[-1][1]
                suitability = self.calculate_climate_suitability(future_weather[0]["temperature"], future_weather[0]["humidity"])
                trend = (features[-1][1] - features[0][1]) / max(1, len(features)-1)
                return float(np.clip(last_stress + trend + 0.05 * suitability * stage_val, 0.0, 1.0))

    @staticmethod
    def generate_intervention_urgency(
        pathogen: np.ndarray,
        velocity: np.ndarray,
        fungicide: np.ndarray
    ) -> np.ndarray:
        """
        Generate spatial Urgency index maps: Urgency = Pathogen * 0.4 + Velocity * 0.6.
        """
        urgency = pathogen * 0.4 + velocity * 0.6
        urgency = urgency * (1.0 - 0.9 * fungicide)
        return np.clip(urgency, 0.0, 1.0).astype(np.float32)

    @staticmethod
    def generate_probabilistic_boundaries(
        pathogen: np.ndarray,
        levels: List[float] = [0.50, 0.75, 0.90]
    ) -> Dict[float, np.ndarray]:
        """
        Isolate contagion front contours at target density thresholds.
        """
        boundaries = {}
        for lvl in levels:
            binary_mask = pathogen >= lvl
            edges = np.zeros_like(binary_mask, dtype=bool)
            edges[1:-1, 1:-1] = binary_mask[1:-1, 1:-1] & ~(
                binary_mask[1:-1, 2:] & binary_mask[1:-1, :-2] & binary_mask[2:, 1:-1] & binary_mask[:-2, 1:-1]
            )
            boundaries[lvl] = edges.astype(np.float32)
        return boundaries

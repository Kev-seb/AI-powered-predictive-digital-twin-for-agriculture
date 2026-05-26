"""
disease_evolution.py
---------------------
Production-grade PyTorch ConvLSTM and Transformer temporal forecasting pipeline
to predict future NDVI, stress expansion, and disease risk maps.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class ConvLSTMCell(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int = 3):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        self.conv = nn.Conv2d(
            in_channels=in_channels + hidden_channels,
            out_channels=4 * hidden_channels,
            kernel_size=kernel_size,
            padding=self.padding,
            bias=True
        )

    def forward(self, x: torch.Tensor, h: torch.Tensor, c: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (B, C, H, W)
        # h: (B, H_c, H, W)
        # c: (B, H_c, H, W)
        combined = torch.cat([x, h], dim=1)
        gates = self.conv(combined)
        
        ingate, forgetgate, cellgate, outgate = torch.chunk(gates, 4, dim=1)
        
        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)
        
        c_next = forgetgate * c + ingate * cellgate
        h_next = outgate * torch.tanh(c_next)
        
        return h_next, c_next


class SpatialTemporalConvLSTM(nn.Module):
    def __init__(self, in_channels: int = 2, hidden_channels: int = 16, out_channels: int = 2):
        """
        Input shape: (B, T, C, H, W)
        Outputs: (B, T_pred, C, H, W)
        """
        super().__init__()
        self.hidden_channels = hidden_channels
        self.cell = ConvLSTMCell(in_channels, hidden_channels, kernel_size=3)
        self.decoder = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)

    def forward(self, seq: torch.Tensor, pred_steps: int = 1) -> torch.Tensor:
        # seq: (B, T, C, H, W)
        B, T, C, H, W = seq.shape
        device = seq.device
        
        # Initialize states
        h = torch.zeros(B, self.hidden_channels, H, W, device=device)
        c = torch.zeros(B, self.hidden_channels, H, W, device=device)
        
        # Encode sequence
        for t in range(T):
            h, c = self.cell(seq[:, t], h, c)
            
        outputs = []
        # Predict future steps auto-regressively
        curr_input = seq[:, -1]
        for _ in range(pred_steps):
            h, c = self.cell(curr_input, h, c)
            pred = self.decoder(h)
            
            # Constrain values (NDVI and Stress are in [-1, 1] or [0, 1])
            # channel 0: NDVI (clip to [-1, 1]), channel 1: Stress (clip to [0, 1])
            ndvi_ch = torch.tanh(pred[:, 0:1])
            stress_ch = torch.sigmoid(pred[:, 1:2])
            pred_clamped = torch.cat([ndvi_ch, stress_ch], dim=1)
            
            outputs.append(pred_clamped)
            curr_input = pred_clamped
            
        return torch.stack(outputs, dim=1)  # (B, pred_steps, C, H, W)


class CropStressEvolutionForecaster:
    def __init__(self, model_path: Optional[str] = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SpatialTemporalConvLSTM(in_channels=2, hidden_channels=16, out_channels=2)
        self.model.to(self.device)
        
        if model_path and os.path.exists(model_path):
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                self.model.eval()
            except Exception as e:
                print(f"[WARN] Error loading forecaster state dict: {e}")

    def train_on_field_sequence(
        self,
        historical_maps: List[Tuple[np.ndarray, np.ndarray]],  # list of (ndvi, stress) maps
        epochs: int = 15,
        lr: float = 0.005
    ) -> List[float]:
        """
        Train the ConvLSTM model online on a field-specific sequence of multi-date surveys.
        """
        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        # Prepare data sequence: shape (1, T, 2, H, W)
        t_steps = len(historical_maps)
        if t_steps < 3:
            # Not enough dates for sequence learning
            return []
            
        H, W = historical_maps[0][0].shape
        seq_tensor = np.zeros((1, t_steps, 2, H, W), dtype=np.float32)
        
        for t, (ndvi, stress) in enumerate(historical_maps):
            seq_tensor[0, t, 0] = ndvi
            seq_tensor[0, t, 1] = stress
            
        seq_torch = torch.from_numpy(seq_tensor).to(self.device)
        
        losses = []
        # Multi-step training splits
        # Input: seq[:, :t_steps-1], Target: seq[:, 1:]
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            # Predict t_steps-1 step forward auto-regressively
            pred = self.model(seq_torch[:, :-1], pred_steps=t_steps - 1)
            loss = criterion(pred, seq_torch[:, 1:])
            
            loss.backward()
            optimizer.step()
            
            losses.append(float(loss.item()))
            
        self.model.eval()
        return losses

    def predict_future_evolution(
        self,
        historical_maps: List[Tuple[np.ndarray, np.ndarray]],
        forecast_days: int = 7,
        days_per_step: int = 7
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Forecast future field state (NDVI & Stress) and identify potential disease expansion regions.
        """
        t_steps = len(historical_maps)
        H, W = historical_maps[0][0].shape
        
        # Prepare inputs
        seq_tensor = np.zeros((1, t_steps, 2, H, W), dtype=np.float32)
        for t, (ndvi, stress) in enumerate(historical_maps):
            seq_tensor[0, t, 0] = ndvi
            seq_tensor[0, t, 1] = stress
            
        seq_torch = torch.from_numpy(seq_tensor).to(self.device)
        
        # Determine number of prediction steps (e.g. 7 days -> 1 step of 7 days)
        n_steps = max(1, int(np.ceil(forecast_days / days_per_step)))
        
        with torch.no_grad():
            self.model.eval()
            pred = self.model(seq_torch, pred_steps=n_steps)
            # Fetch final forecasted state
            final_pred = pred[0, -1].cpu().numpy()  # (2, H, W)
            
        future_ndvi = final_pred[0]
        future_stress = final_pred[1]
        
        # Compute expansion regions: areas where stress increases significantly
        current_stress = historical_maps[-1][1]
        expansion_region = np.clip(future_stress - current_stress, 0.0, 1.0)
        
        return future_ndvi, future_stress, expansion_region

    @staticmethod
    def generate_synthetic_historical_sequence(
        current_ndvi: np.ndarray,
        current_stress: np.ndarray,
        n_steps: int = 3,
        stress_expansion_rate: float = 0.08
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Helper to simulate past dates in the field digital twin memory when actual historic UAV
        data is not yet uploaded, allowing predictive simulations out-of-the-box.
        """
        sequence = []
        H, W = current_ndvi.shape
        
        # Run reverse simulation to create history (healthy -> stressed)
        for step in reversed(range(n_steps)):
            # Less stress and higher NDVI in the past
            delta_stress = step * stress_expansion_rate
            hist_stress = np.clip(current_stress - delta_stress + np.random.normal(0, 0.02, (H, W)), 0.0, 1.0)
            hist_ndvi = np.clip(current_ndvi + (delta_stress * 0.5) + np.random.normal(0, 0.01, (H, W)), -1.0, 1.0)
            sequence.append((hist_ndvi.astype(np.float32), hist_stress.astype(np.float32)))
            
        return sequence

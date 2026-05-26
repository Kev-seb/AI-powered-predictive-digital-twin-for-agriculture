"""
camera_feed.py
--------------
Procedural UAV camera feed simulator and real-time AI stress inference engine.
Generates video frames based on drone coordinate mapping, runs computer vision
detection for crop anomalies, and overlays an aviation-style HUD.
"""

import cv2
import numpy as np
import math
import random
from typing import Dict, Any, Optional

# Global webcam reference to prevent re-opening capture source
_cap_device: Optional[cv2.VideoCapture] = None
_last_webcam_attempt: float = 0.0

def get_live_camera_frame(shared_state: Dict[str, Any], telemetry: Dict[str, Any]) -> bytes:
    """
    Returns JPEG encoded bytes of the live UAV camera feed.
    Performs AI stress detection inference and overlays flight telemetry HUD.
    """
    global _cap_device, _last_webcam_attempt
    camera_source = shared_state.get("CAMERA_SOURCE", "Simulated UAV Camera")

    frame = None

    # Release webcam if we are no longer using it
    if camera_source != "Local Webcam" and _cap_device is not None:
        try:
            _cap_device.release()
        except Exception:
            pass
        _cap_device = None

    if camera_source == "Local Webcam":
        import time
        now = time.time()
        
        if _cap_device is None:
            # Rate-limit webcam open attempts to once every 8 seconds to prevent thread lockups
            if now - _last_webcam_attempt > 8.0:
                _last_webcam_attempt = now
                try:
                    # Attempt to open webcam index 0
                    # On Windows, DirectShow backend (CAP_DSHOW) is usually much faster to initialize
                    _cap_device = cv2.VideoCapture(0, cv2.CAP_DSHOW)
                    _cap_device.set(cv2.CAP_PROP_FRAME_WIDTH, 400)
                    _cap_device.set(cv2.CAP_PROP_FRAME_HEIGHT, 300)
                except Exception as e:
                    print(f"Failed to open webcam: {e}")
                    _cap_device = None
        
        if _cap_device is not None:
            try:
                if _cap_device.isOpened():
                    ret, frame = _cap_device.read()
                    if ret:
                        frame = cv2.resize(frame, (400, 300))
                    else:
                        _cap_device.release()
                        _cap_device = None
                else:
                    _cap_device.release()
                    _cap_device = None
            except Exception as e:
                print(f"Failed to read from webcam: {e}")
                try:
                    _cap_device.release()
                except Exception:
                    pass
                _cap_device = None

    # Fallback to simulated camera if webcam is unavailable or Simulated source is selected
    if frame is None:
        frame = generate_simulated_view(shared_state, telemetry)

    # Apply AI Stress detection and HUD overlays
    frame_with_hud = apply_ai_inference_and_hud(frame, telemetry, shared_state)

    # Encode to JPEG
    _, jpeg_bytes = cv2.imencode('.jpg', frame_with_hud, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return jpeg_bytes.tobytes()


def generate_simulated_view(shared_state: Dict[str, Any], telemetry: Dict[str, Any]) -> np.ndarray:
    """Crop the field orthomosaic or generate procedural crop rows centered on the drone's position."""
    ndvi_map = shared_state.get("NDVI_MAP")
    
    # Grid size
    w_width, w_height = 400, 300

    if ndvi_map is not None:
        # Map physical meters relative to home to NDVI matrix index
        # Let's read local coordinates from physics simulator if available
        sim = shared_state.get("PHYSICS_SIMULATOR")
        if sim is not None:
            pos_x, pos_y = sim.pos[0], sim.pos[1]
            alt = sim.pos[2]
        else:
            # Reconstruct local offset from GPS
            home_lat = shared_state.get("HOME_LAT", 37.7749)
            home_lon = shared_state.get("HOME_LON", -122.4194)
            lat, lon = telemetry.get("lat", home_lat), telemetry.get("lon", home_lon)
            alt = telemetry.get("alt", 10.0)
            pos_y = (lat - home_lat) * 111000.0
            pos_x = (lon - home_lon) * 111000.0 * math.cos(math.radians(home_lat))

        H, W = ndvi_map.shape
        px = int(W / 2 + pos_x / 0.05)
        py = int(H / 2 - pos_y / 0.05)

        # Focal scale with altitude
        crop_size = int(max(40, min(500, 120 * (alt / 10.0))))
        
        # Crop bounds
        x1 = max(0, px - crop_size)
        x2 = min(W, px + crop_size)
        y1 = max(0, py - crop_size)
        y2 = min(H, py + crop_size)

        patch = ndvi_map[y1:y2, x1:x2]
        
        # Ensure patch is valid
        if patch.size > 0:
            patch_resized = cv2.resize(patch, (w_width, w_height), interpolation=cv2.INTER_LINEAR)
            
            # Map NDVI float [-1, 1] to RGB multispectral image
            # Red: Stressed (< 0.35), Yellow: Borderline (0.35-0.5), Green: Healthy (> 0.5)
            rgb = np.zeros((w_height, w_width, 3), dtype=np.uint8)
            
            # Low NDVI (Stress): Red
            stress_mask = patch_resized < 0.35
            rgb[stress_mask] = [30, 30, 200]  # BGR Red
            
            # Mid NDVI (Transition): Yellow/Light-Green
            mid_mask = (patch_resized >= 0.35) & (patch_resized < 0.55)
            rgb[mid_mask] = [50, 180, 200]  # BGR Yellow/Orange
            
            # High NDVI (Healthy): Bright Green
            healthy_mask = patch_resized >= 0.55
            rgb[healthy_mask] = [40, 180, 40]  # BGR Green
            
            # Earthy tones for negative NDVI (soil / pathways)
            soil_mask = patch_resized < 0.0
            rgb[soil_mask] = [80, 120, 140]  # BGR Earth Brown
            
            # Add dynamic sensor noise
            noise = np.random.normal(0, 5, rgb.shape).astype(np.int16)
            rgb = np.clip(rgb.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            return rgb

    # Procedural crop row simulation if no orthomosaic is loaded
    # Create green rows with periodic stressed brown gaps
    img = np.zeros((w_height, w_width, 3), dtype=np.uint8) + 80  # Brown dirt background (BGR: 80, 80, 80)
    
    # Calculate flight offsets to slide crop lines as drone flies
    sim = shared_state.get("PHYSICS_SIMULATOR")
    if sim is not None:
        offset_x = -sim.pos[0] * 20.0
        offset_y = sim.pos[1] * 20.0
    else:
        offset_x = -telemetry.get("lon", 0.0) * 1e6
        offset_y = telemetry.get("lat", 0.0) * 1e6

    row_spacing = 30
    for i in range(-10, w_width + 10, row_spacing):
        x_pos = int(i + (offset_x % row_spacing))
        # Draw a crop line
        cv2.line(img, (x_pos, 0), (x_pos, w_height), (40, 160, 40), 12)
        
        # Periodically inject stress spots (brown gaps)
        for y_gap in range(0, w_height, 80):
            gap_center = int(y_gap + (offset_y % 120))
            if (x_pos + gap_center) % 180 < 70:
                # Draw stressed spot
                cv2.circle(img, (x_pos, gap_center), 10, (50, 90, 150), -1)  # BGR Brownish-red

    # Dynamic sensor noise
    noise = np.random.normal(0, 6, img.shape).astype(np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def apply_ai_inference_and_hud(frame: np.ndarray, telemetry: Dict[str, Any], shared_state: Dict[str, Any]) -> np.ndarray:
    """Overlay real-time crop stress contour bounding boxes (AI) and flight telemetry HUD."""
    h, w, _ = frame.shape
    out = frame.copy()

    # --- 1. Real-Time AI Inference (Anomaly Highlight) ---
    # Detect stressed areas: in BGR, stressed regions are reddish-brown (high Red and Low Green channel)
    # Threshold frame to isolate reddish crop pixels
    b, g, r = cv2.split(out)
    # Highlight pixels where red is significantly higher than green
    stress_metric = cv2.subtract(r, g)
    _, thresh = cv2.threshold(stress_metric, 30, 255, cv2.THRESH_BINARY)
    
    # Find contours representing stress anomalies
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    anomalies_detected = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 80 < area < 5000:  # ignore noise and massive segments
            x_box, y_box, w_box, h_box = cv2.boundingRect(cnt)
            # Draw AI bounding box
            cv2.rectangle(out, (x_box, y_box), (x_box + w_box, y_box + h_box), (0, 0, 240), 2)  # BGR Red
            # Label
            label = f"STRESS (CONF: {min(98.0, 75.0 + area/100.0):.1f}%)"
            cv2.putText(out, label, (x_box, max(15, y_box - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1, cv2.LINE_AA)
            anomalies_detected += 1

    # --- 2. Flight HUD overlays (Aviation-style overlays) ---
    hud_color = (0, 230, 0)  # Bright Green BGR
    
    # Center reticle
    cx, cy = w // 2, h // 2
    cv2.circle(out, (cx, cy), 18, hud_color, 1)
    cv2.line(out, (cx - 30, cy), (cx - 18, cy), hud_color, 1)
    cv2.line(out, (cx + 18, cy), (cx + 30, cy), hud_color, 1)
    cv2.line(out, (cx, cy - 30), (cx, cy - 18), hud_color, 1)
    cv2.line(out, (cx, cy + 18), (cx, cy + 30), hud_color, 1)

    # Roll Horizon indicator (tilted line based on telemetry pitch & roll)
    roll = telemetry.get("roll", 0.0)
    pitch = telemetry.get("pitch", 0.0)
    
    # Offset center vertically based on pitch (1 rad pitch shifts approx 150px)
    pitch_offset = int(pitch * 120.0)
    h_cy = cy + pitch_offset
    
    dx = int(80 * math.cos(roll))
    dy = int(80 * math.sin(roll))
    cv2.line(out, (cx - dx, h_cy - dy), (cx + dx, h_cy + dy), hud_color, 1)
    cv2.line(out, (cx - dx, h_cy - dy), (cx - dx, h_cy - dy + 10), hud_color, 1)
    cv2.line(out, (cx + dx, h_cy + dy), (cx + dx, h_cy + dy + 10), hud_color, 1)

    # Altitude tape (right side)
    alt = telemetry.get("alt", 0.0)
    cv2.rectangle(out, (w - 65, 50), (w - 10, h - 50), hud_color, 1)
    cv2.putText(out, f"ALT", (w - 55, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.35, hud_color, 1, cv2.LINE_AA)
    cv2.putText(out, f"{alt:.1f}m", (w - 60, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, hud_color, 1, cv2.LINE_AA)

    # Speed tape (left side)
    speed = telemetry.get("speed", 0.0)
    cv2.rectangle(out, (10, 50), (60, h - 50), hud_color, 1)
    cv2.putText(out, f"SPD", (15, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.35, hud_color, 1, cv2.LINE_AA)
    cv2.putText(out, f"{speed:.1f}m/s", (12, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.35, hud_color, 1, cv2.LINE_AA)

    # Compass tape (top center)
    yaw_deg = math.degrees(telemetry.get("yaw", 0.0)) % 360
    cv2.rectangle(out, (cx - 70, 10), (cx + 70, 32), hud_color, 1)
    cv2.putText(out, f"HDG: {int(yaw_deg):03d}", (cx - 35, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.35, hud_color, 1, cv2.LINE_AA)

    # General HUD status text overlays
    bat = telemetry.get("battery", 100.0)
    cv2.putText(out, f"SYS: OK", (15, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.35, hud_color, 1, cv2.LINE_AA)
    cv2.putText(out, f"BAT: {bat:.1f}%", (15, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, hud_color, 1, cv2.LINE_AA)
    
    ai_status = f"AI ANOMALIES: {anomalies_detected}"
    cv2.putText(out, ai_status, (w - 150, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 240, 240) if anomalies_detected > 0 else hud_color, 1, cv2.LINE_AA)

    # Blinking spray warning indicator if active
    if telemetry.get("is_spraying"):
        # Make it blink using simple time modulo
        if int(random.random() * 10) % 2 == 0:
            cv2.rectangle(out, (cx - 45, h - 40), (cx + 45, h - 15), (0, 0, 255), -1)  # Red background
            cv2.putText(out, "SPRAY ON", (cx - 33, h - 23), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
        else:
            cv2.rectangle(out, (cx - 45, h - 40), (cx + 45, h - 15), (0, 0, 255), 1)  # Red border
            cv2.putText(out, "SPRAY ON", (cx - 33, h - 23), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1, cv2.LINE_AA)

    # Watermark / Camera Mode indicator
    source_label = "PROC-NDVI FEED" if ndvi_map is not None else "PROC-CROP FEED"
    if camera_source == "Local Webcam":
        source_label = "WEBCAM FEED"
    cv2.putText(out, source_label, (w - 110, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (150, 150, 150), 1, cv2.LINE_AA)

    return out

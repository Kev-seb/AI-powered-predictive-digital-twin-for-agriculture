"""
flight_physics.py
-----------------
Aerospace-grade 6-DOF flight physics simulator for UAV digital twin operations.
Simulates translation, rotation, aerodynamic forces (thrust, drag, lift),
Dryden-like wind turbulence, terrain-following, obstacle avoidance, and battery-aware routing.
"""

import math
import random
import numpy as np
from typing import Dict, Any, Tuple, Optional, List

class UAVFlightDynamicsSimulator:
    def __init__(self, shared_state: Dict[str, Any], home_lat: float = 37.7749, home_lon: float = -122.4194, drone_id: str = "drone_alpha"):
        self.shared_state = shared_state
        self.home_lat = home_lat
        self.home_lon = home_lon
        self.drone_id = drone_id

        # 6-DOF State variables (SI units)
        self.pos = np.array([0.0, 0.0, 10.0])  # [x, y, z] relative to home (z is altitude)
        self.vel = np.array([0.0, 0.0, 0.0])  # [vx, vy, vz] (m/s)
        self.attitude = np.array([0.0, 0.0, 0.0])  # [roll, pitch, yaw] (radians)
        self.omega = np.array([0.0, 0.0, 0.0])  # [p, q, r] (rad/s)

        # UAV Physical constants
        self.dry_mass = 8.0  # kg
        self.max_payload_mass = 10.0  # kg (liquid spray tank capacity)
        self.payload_mass = 10.0  # current liquid payload
        self.g = 9.81  # m/s^2

        # Aerodynamic params
        self.Cd_trans = 0.15  # translational drag coefficient
        self.Cd_rot = 0.05   # rotational drag coefficient
        self.lift_coeff = 0.03  # lift coefficient scaling with airspeed

        # Battery management
        self.battery = 100.0  # %
        self.discharge_rate_factor = 0.035
        self.is_spraying = False
        
        # Actuators
        self.target_motor_speeds = np.zeros(4)
        self.motor_speeds = np.zeros(4)
        self.motor_tau = 0.08  # motor response time constant (seconds)

        # Autopilot state
        self.target_pos = np.array([0.0, 0.0, 10.0])
        self.target_yaw = 0.0
        self.autopilot_mode = "stabilized"  # stabilized, terrain_follow, rtl, landing, manual
        self.clearance_target = 5.0  # target clearance above canopy for terrain-following

        # Fault injections
        self.fault_rotor_failure = False
        self.fault_active_gust = False
        self.gust_timer = 0.0
        self.gust_vector = np.zeros(3)

        # Dryden wind turbulence states
        self.turb_x = 0.0
        self.turb_y = 0.0
        self.turb_z = 0.0

    def get_mass(self) -> float:
        return self.dry_mass + self.payload_mass

    def get_terrain_height(self, x: float, y: float) -> float:
        """Query canopy height map (CHM) from shared state at local (x, y) coordinates."""
        chm = self.shared_state.get("CHM")
        if chm is None:
            return 0.0
        
        # Dimensions and pixel bounds mapping
        H, W = chm.shape
        # Resolution: 0.05m per pixel. Home (0,0) is in the center of the grid.
        px = int(W / 2 + x / 0.05)
        py = int(H / 2 - y / 0.05)
        
        if 0 <= px < W and 0 <= py < H:
            return float(chm[py, px])
        return 0.0

    def update_sensors_and_wind(self, dt: float) -> np.ndarray:
        """Compute base wind vector plus stochastic Dryden-like turbulence."""
        env = self.shared_state.get("CURRENT_ENV", {})
        base_wind_speed = float(env.get("wind_speed", 0.0))
        base_wind_direction = math.radians(float(env.get("wind_direction", 0.0)))

        # Base wind vector in local coordinates
        # wind_direction represents the heading the wind is coming FROM
        w_dir = base_wind_direction + math.pi
        base_wind = np.array([
            math.cos(w_dir) * base_wind_speed,
            math.sin(w_dir) * base_wind_speed,
            0.0
        ])

        # Dryden-like turbulence using a low-pass filtered random walk
        alpha = 0.95  # filtering coefficient
        self.turb_x = alpha * self.turb_x + (1.0 - alpha) * random.gauss(0.0, 0.4 * base_wind_speed + 0.1)
        self.turb_y = alpha * self.turb_y + (1.0 - alpha) * random.gauss(0.0, 0.4 * base_wind_speed + 0.1)
        self.turb_z = alpha * self.turb_z + (1.0 - alpha) * random.gauss(0.0, 0.1 * base_wind_speed + 0.05)
        turbulence = np.array([self.turb_x, self.turb_y, self.turb_z])

        # Fault Gust Injection
        if self.fault_active_gust:
            self.gust_timer += dt
            if self.gust_timer < 3.0:
                gust = self.gust_vector
            else:
                gust = np.zeros(3)
                self.fault_active_gust = False
                self.gust_timer = 0.0
        else:
            gust = np.zeros(3)

        return base_wind + turbulence + gust

    def compute_forces_and_moments(self, wind: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """Compute 6-DOF forces (Thrust, Drag, Lift, Gravity) and Moments (Attitude torque)."""
        m = self.get_mass()
        
        # Motor speeds transient dynamics
        # dw/dt = (w_target - w) / tau
        self.motor_speeds += (self.target_motor_speeds - self.motor_speeds) * (dt / self.motor_tau)
        
        # Inject Rotor Failure Fault (loses lift on motor 3)
        if self.fault_rotor_failure:
            self.motor_speeds[2] = 0.0

        # Propeller Thrust forces (collective thrust directed along body z-axis)
        # For simulation, individual motor speeds determine total thrust and attitude moments
        w_sq = self.motor_speeds ** 2
        # Lift constant (derived from hovering condition)
        kf = (m * self.g) / (4.0 * (150.0 ** 2))  # hover at ~150 rad/s motor speed
        individual_thrusts = kf * w_sq
        total_thrust = np.sum(individual_thrusts)

        # Body orientation rotations
        phi, theta, psi = self.attitude
        # Rotation matrix from Body to World
        R_x = np.array([
            [1, 0, 0],
            [0, math.cos(phi), -math.sin(phi)],
            [0, math.sin(phi), math.cos(phi)]
        ])
        R_y = np.array([
            [math.cos(theta), 0, math.sin(theta)],
            [0, 1, 0],
            [-math.sin(theta), 0, math.cos(theta)]
        ])
        R_z = np.array([
            [math.cos(psi), -math.sin(psi), 0],
            [math.sin(psi), math.cos(psi), 0],
            [0, 0, 1]
        ])
        R = R_z @ R_y @ R_x

        # Thrust vector in World frame
        thrust_force = R @ np.array([0.0, 0.0, total_thrust])

        # Airspeed vector (Relative velocity to wind)
        airspeed = self.vel - wind
        speed = np.linalg.norm(airspeed)

        # Aerodynamic Drag force
        drag_force = -self.Cd_trans * speed * airspeed

        # Aerodynamic Lift force (proportional to airspeed squared)
        lift_magnitude = self.lift_coeff * (speed ** 1.8)
        # Lift acts vertically upwards
        lift_force = np.array([0.0, 0.0, lift_magnitude])

        # Gravity
        gravity_force = np.array([0.0, 0.0, -m * self.g])

        # Total force in World Coordinates
        total_force = thrust_force + drag_force + lift_force + gravity_force

        # Moments (Torque) calculation
        # Arm length: 0.35m
        L = 0.35
        km = 0.015 * kf  # drag torque constant
        
        # Roll, Pitch, and Yaw torques in Body coordinates
        # Motor layout: 1: front-left (CCW), 2: front-right (CW), 3: rear-right (CCW), 4: rear-left (CW)
        tau_x = L * (individual_thrusts[1] + individual_thrusts[2] - individual_thrusts[0] - individual_thrusts[3])
        tau_y = L * (individual_thrusts[2] + individual_thrusts[3] - individual_thrusts[0] - individual_thrusts[1])
        tau_z = km * (-w_sq[0] + w_sq[1] - w_sq[2] + w_sq[3])

        # Rotational drag moment
        moments = np.array([tau_x, tau_y, tau_z]) - self.Cd_rot * self.omega

        # Instability effects: add high-speed turbulence torque
        if speed > 10.0:
            instability_torque = np.random.normal(0.0, 0.05 * (speed - 10.0), size=3)
            moments += instability_torque

        return total_force, moments

    def run_autopilot_guidance(self, dt: float):
        """Autopilot stabilization, waypoint tracking, and obstacle avoidance."""
        m = self.get_mass()
        
        # --- 1. Obstacle Avoidance Scan ---
        # Scan 1.5 seconds ahead in flight path
        scan_dist = np.linalg.norm(self.vel[:2]) * 1.5
        if scan_dist > 1.0:
            dir_vec = self.vel[:2] / np.linalg.norm(self.vel[:2])
            scan_x = self.pos[0] + dir_vec[0] * scan_dist
            scan_y = self.pos[1] + dir_vec[1] * scan_dist
            canopy_height_ahead = self.get_terrain_height(scan_x, scan_y)
            
            # If obstacle ahead exceeds current altitude, command an automatic height gain
            if canopy_height_ahead + 2.0 > self.pos[2]:
                # Temporary climbing boost
                self.target_pos[2] = max(self.target_pos[2], canopy_height_ahead + 5.0)

        # --- 2. Terrain-Following Guidance ---
        if self.autopilot_mode == "terrain_follow":
            terrain_z = self.get_terrain_height(self.pos[0], self.pos[1])
            self.target_pos[2] = terrain_z + self.clearance_target

        # --- 3. Waypoint Tracking PID (Position Controller) ---
        # Output: Target pitch/roll angles and total thrust command
        pos_err = self.target_pos - self.pos
        
        # Proportional position loops to get velocity targets
        Kp_pos = 0.8
        target_vel = pos_err * Kp_pos
        # Cap max speeds
        target_vel[:2] = np.clip(target_vel[:2], -8.0, 8.0)
        target_vel[2] = np.clip(target_vel[2], -2.5, 2.5)

        # Velocity error tracking
        vel_err = target_vel - self.vel
        Kd_vel = 4.0
        acc_cmd = vel_err * Kd_vel

        # --- 3.5 Potential Field Collision Avoidance ---
        repulsive_acc = np.zeros(3)
        simulators = self.shared_state.get("PHYSICS_SIMULATORS", {})
        safety_distance = 6.0  # meters Proximity limit
        k_repulsive = 35.0     # Repulsive potential gain factor
        
        for other_id, other_sim in simulators.items():
            if other_id != self.drone_id and other_sim is not None:
                diff = self.pos - other_sim.pos
                dist = np.linalg.norm(diff)
                if dist < safety_distance and dist > 0.05:
                    # Direction vector pointing away from other drone
                    direction = diff / dist
                    # Repulsive force logic: inversely proportional to distance squared
                    force = k_repulsive * (1.0 / dist - 1.0 / safety_distance) * (1.0 / (dist ** 2))
                    repulsive_acc += direction * force
                    
                    # Log collision warning in shared state for UI display
                    warning_log = f"Proximity warning: {self.drone_id} and {other_id} within safety radius! Repulsion vector applied."
                    if "SWARM_WARNINGS" in self.shared_state:
                        if warning_log not in self.shared_state["SWARM_WARNINGS"]:
                            self.shared_state["SWARM_WARNINGS"].append(warning_log)
                            
        # Combine collision avoidance acceleration
        acc_cmd += repulsive_acc

        # Target pitch and roll (small angle approximation)
        yaw = self.attitude[2]
        pitch_cmd = (acc_cmd[0] * math.cos(yaw) + acc_cmd[1] * math.sin(yaw)) / self.g
        roll_cmd = (acc_cmd[0] * math.sin(yaw) - acc_cmd[1] * math.cos(yaw)) / self.g
        
        # Limit attitude commands to 30 degrees
        pitch_cmd = np.clip(pitch_cmd, -0.5, 0.5)
        roll_cmd = np.clip(roll_cmd, -0.5, 0.5)

        # Altitude thrust stabilization
        thrust_base = m * self.g
        thrust_stabilize = m * acc_cmd[2]
        total_thrust_cmd = thrust_base + thrust_stabilize
        # Cap collective thrust command
        total_thrust_cmd = np.clip(total_thrust_cmd, 0.2 * m * self.g, 1.8 * m * self.g)

        # --- 4. Attitude Loop PID Controller ---
        # Drive attitude Euler angles to commands using motor speeds differential mixing
        target_roll = roll_cmd
        target_pitch = pitch_cmd
        target_yaw = self.target_yaw

        if self.autopilot_mode == "landing":
            target_roll = 0.0
            target_pitch = 0.0
            total_thrust_cmd = m * self.g * 0.85  # controlled descent

        # Attitude error
        att_err = np.array([
            target_roll - self.attitude[0],
            target_pitch - self.attitude[1],
            self.angle_diff(target_yaw, self.attitude[2])
        ])

        # Angular rate target (rad/s)
        Kp_att = 2.5
        target_omega = att_err * Kp_att
        target_omega = np.clip(target_omega, -4.0, 4.0)

        # Rate loop error to mix motor outputs
        omega_err = target_omega - self.omega
        Kd_rate = 2.0
        moment_cmd = omega_err * Kd_rate

        # Mix motors: convert collective thrust and moment commands to motor speeds
        # Motor layout: w1, w2, w3, w4
        # We solve for w^2 commands
        hover_w_sq = total_thrust_cmd / (4.0 * ( (m * self.g) / (4.0 * (150.0 ** 2)) ))
        
        # Mix commands (r, p, y torque adjustments)
        d_roll = moment_cmd[0] * 400.0
        d_pitch = moment_cmd[1] * 400.0
        d_yaw = moment_cmd[2] * 400.0

        w_sq_cmds = np.array([
            hover_w_sq - d_roll - d_pitch - d_yaw,  # FL (0)
            hover_w_sq + d_roll - d_pitch + d_yaw,  # FR (1)
            hover_w_sq + d_roll + d_pitch - d_yaw,  # RR (2)
            hover_w_sq - d_roll + d_pitch + d_yaw   # RL (3)
        ])

        # Clean target motor speeds (rad/s)
        self.target_motor_speeds = np.sqrt(np.clip(w_sq_cmds, 10.0**2, 300.0**2))

    def step(self, dt: float):
        """Advance the physics simulation by dt seconds."""
        m = self.get_mass()

        # Update spray depletion
        if self.is_spraying and self.payload_mass > 0.0:
            deplete = 0.12 * dt  # kg/s
            self.payload_mass = max(0.0, self.payload_mass - deplete)

        # Get environmental forces
        wind = self.update_sensors_and_wind(dt)

        # Autopilot update
        if self.autopilot_mode != "manual":
            self.run_autopilot_guidance(dt)

        # Integrate Forces & Moments (6-DOF Translation and Rotation)
        total_force, moments = self.compute_forces_and_moments(wind, dt)

        # 1. Translation integration
        accel = total_force / m
        self.vel += accel * dt
        self.pos += self.vel * dt

        # Terrain ground check (stop falling if ground hit)
        ground_z = self.get_terrain_height(self.pos[0], self.pos[1])
        if self.pos[2] < ground_z:
            self.pos[2] = ground_z
            self.vel = np.array([0.0, 0.0, 0.0])
            self.attitude[:2] = 0.0  # level out
            self.omega = np.array([0.0, 0.0, 0.0])

        # 2. Rotation integration
        # Inertia moments (scaled roughly for quadcopter)
        I_diag = np.array([0.08, 0.08, 0.15])
        omega_dot = moments / I_diag
        self.omega += omega_dot * dt
        
        # Integrate Euler angles (direct rates for stability)
        self.attitude += self.omega * dt
        self.attitude = (self.attitude + math.pi) % (2.0 * math.pi) - math.pi

        # 3. Battery depletion
        thrust_ratio = np.sum(self.motor_speeds ** 2) / (4.0 * (150.0**2))
        spray_factor = 0.045 if self.is_spraying else 0.0
        battery_drain = dt * (0.04 + 0.065 * thrust_ratio + spray_factor)
        self.battery = max(0.0, self.battery - battery_drain)

        # --- 5. Battery-Aware Safety & Landing Rules ---
        if self.battery < 15.0 and self.autopilot_mode not in ["landing", "manual"]:
            self.trigger_emergency_landing()
        elif self.battery < 30.0 and self.autopilot_mode not in ["rtl", "landing", "manual"]:
            # Evaluate if return to home is required
            self.evaluate_battery_rtl()

    def evaluate_battery_rtl(self):
        """Predict headwind return cost and trigger RTL if necessary."""
        dist = np.linalg.norm(self.pos[:2])  # distance to home (0,0)
        env = self.shared_state.get("CURRENT_ENV", {})
        base_wind_speed = float(env.get("wind_speed", 0.0))
        base_wind_direction = math.radians(float(env.get("wind_direction", 0.0)))
        
        # Vector from drone to home
        return_dir = -self.pos[:2] / (np.linalg.norm(self.pos[:2]) + 1e-6)
        
        # Wind vector
        w_dir = base_wind_direction + math.pi
        wind_vec = np.array([math.cos(w_dir), math.sin(w_dir)]) * base_wind_speed
        
        # Headwind component
        headwind = np.dot(wind_vec, -return_dir)
        
        return_ground_speed = max(2.0, 6.5 - headwind)
        time_to_return = dist / return_ground_speed
        
        # Required battery (approx 0.12% per second return drain + 10% safety margin)
        estimated_drain = time_to_return * 0.12
        required_battery = estimated_drain + 10.0
        
        if self.battery <= required_battery:
            self.autopilot_mode = "rtl"
            self.target_pos = np.array([0.0, 0.0, 12.0])  # fly home at 12m safe height
            self.target_yaw = math.atan2(return_dir[1], return_dir[0])

    def trigger_emergency_landing(self):
        """Enter emergency descent, searching locally for safe flat terrain."""
        self.autopilot_mode = "landing"
        chm = self.shared_state.get("CHM")
        
        if chm is not None:
            # Search 10x10 meters surrounding zone for flat pixel index (canopy height near 0)
            H, W = chm.shape
            cur_px = int(W / 2 + self.pos[0] / 0.05)
            cur_py = int(H / 2 - self.pos[1] / 0.05)
            
            best_val = 999.0
            best_coords = (self.pos[0], self.pos[1])
            
            # Check a 30x30 pixel grid (1.5m radius)
            for dx in range(-15, 16):
                for dy in range(-15, 16):
                    px = cur_px + dx
                    py = cur_py + dy
                    if 0 <= px < W and 0 <= py < H:
                        height = chm[py, px]
                        if height < best_val:
                            best_val = height
                            # convert back to local meters
                            best_coords = ((px - W/2)*0.05, (H/2 - py)*0.05)
            
            self.target_pos[0] = best_coords[0]
            self.target_pos[1] = best_coords[1]
        else:
            self.target_pos[0] = self.pos[0]
            self.target_pos[1] = self.pos[1]
            
        self.target_pos[2] = 0.0  # land on ground
        self.target_yaw = self.attitude[2]

    def inject_wind_gust(self):
        """Inject a sudden wind gust of 18 m/s in a random direction."""
        angle = random.uniform(0, 2.0 * math.pi)
        self.gust_vector = np.array([
            math.cos(angle) * 18.0,
            math.sin(angle) * 18.0,
            random.uniform(-4.0, 4.0)
        ])
        self.fault_active_gust = True
        self.gust_timer = 0.0

    @staticmethod
    def angle_diff(target: float, current: float) -> float:
        """Find signed smallest difference between two angles in radians."""
        diff = target - current
        while diff > math.pi: diff -= 2.0 * math.pi
        while diff < -math.pi: diff += 2.0 * math.pi
        return diff

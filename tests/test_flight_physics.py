import numpy as np
import pytest
import math
from src.digital_twin.flight_physics import UAVFlightDynamicsSimulator

def test_simulator_initialization():
    shared_state = {}
    sim = UAVFlightDynamicsSimulator(shared_state, home_lat=37.7749, home_lon=-122.4194)
    
    assert np.allclose(sim.pos, np.array([0.0, 0.0, 10.0]))
    assert np.allclose(sim.vel, np.array([0.0, 0.0, 0.0]))
    assert np.allclose(sim.attitude, np.array([0.0, 0.0, 0.0]))
    assert sim.battery == 100.0
    assert sim.is_spraying is False

def test_simulator_step_stabilization():
    shared_state = {}
    sim = UAVFlightDynamicsSimulator(shared_state, home_lat=37.7749, home_lon=-122.4194)
    
    # Run simulation for a brief period to stabilize
    sim.target_pos = np.array([5.0, -5.0, 12.0])
    sim.autopilot_mode = "stabilized"
    
    for _ in range(20):
        sim.step(dt=0.05)
        
    # Check that position has moved towards target (5.0, -5.0, 12.0)
    assert sim.pos[0] > 0.0
    assert sim.pos[1] < 0.0
    assert sim.pos[2] > 10.0
    assert sim.battery < 100.0  # Battery consumption occurred

def test_simulator_low_battery_rtl_and_landing():
    shared_state = {}
    sim = UAVFlightDynamicsSimulator(shared_state, home_lat=37.7749, home_lon=-122.4194)
    
    # 1. Test battery under 15% triggers emergency landing
    sim.pos = np.array([10.0, 10.0, 15.0])
    sim.battery = 14.5  # Low battery
    sim.autopilot_mode = "stabilized"
    
    sim.step(dt=0.05)
    assert sim.autopilot_mode == "landing"
    
    # 2. Test battery between 15% and 30% with a long distance triggers RTL
    sim.autopilot_mode = "stabilized"
    sim.battery = 17.5
    sim.pos = np.array([300.0, 300.0, 15.0])
    
    sim.step(dt=0.05)
    assert sim.autopilot_mode == "rtl"
    assert np.allclose(sim.target_pos, np.array([0.0, 0.0, 12.0])) # RTL height

def test_wind_gust_perturbation():
    shared_state = {}
    sim = UAVFlightDynamicsSimulator(shared_state, home_lat=37.7749, home_lon=-122.4194)
    
    # Inject wind gust
    sim.inject_wind_gust()
    assert np.linalg.norm(sim.vel) == 0.0 # Instant velocity doesn't change, but acceleration/wind force does
    
    sim.step(dt=0.05)
    # Velocity should now reflect gust acceleration
    assert np.linalg.norm(sim.vel) > 0.0

def test_rotor_failure():
    shared_state = {}
    sim = UAVFlightDynamicsSimulator(shared_state, home_lat=37.7749, home_lon=-122.4194)
    
    sim.fault_rotor_failure = True
    sim.autopilot_mode = "stabilized"
    
    for _ in range(5):
        sim.step(dt=0.05)
        
    # Jammed rotor should lead to stabilization instability (non-zero roll/pitch deviation)
    assert np.linalg.norm(sim.attitude[:2]) > 0.01

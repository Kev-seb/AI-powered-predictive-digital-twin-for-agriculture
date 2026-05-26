import numpy as np
import pytest
import math
from src.digital_twin.flight_physics import UAVFlightDynamicsSimulator

def test_swarm_initialization():
    shared_state = {"PHYSICS_SIMULATORS": {}, "SWARM_WARNINGS": []}
    sim_alpha = UAVFlightDynamicsSimulator(shared_state, drone_id="drone_alpha")
    sim_beta = UAVFlightDynamicsSimulator(shared_state, drone_id="drone_beta")
    
    shared_state["PHYSICS_SIMULATORS"]["drone_alpha"] = sim_alpha
    shared_state["PHYSICS_SIMULATORS"]["drone_beta"] = sim_beta
    
    assert sim_alpha.drone_id == "drone_alpha"
    assert sim_beta.drone_id == "drone_beta"
    assert len(shared_state["PHYSICS_SIMULATORS"]) == 2

def test_potential_field_repulsion():
    # Setup two drones close to each other
    shared_state = {"PHYSICS_SIMULATORS": {}, "SWARM_WARNINGS": []}
    sim_alpha = UAVFlightDynamicsSimulator(shared_state, drone_id="drone_alpha")
    sim_beta = UAVFlightDynamicsSimulator(shared_state, drone_id="drone_beta")
    
    shared_state["PHYSICS_SIMULATORS"]["drone_alpha"] = sim_alpha
    shared_state["PHYSICS_SIMULATORS"]["drone_beta"] = sim_beta
    
    # Position them at a distance of 3 meters (safety threshold is 6 meters)
    sim_alpha.pos = np.array([0.0, 0.0, 10.0])
    sim_beta.pos = np.array([3.0, 0.0, 10.0])
    
    # Target at the same position — so the only force driving X movement is repulsion
    sim_alpha.target_pos = np.array([0.0, 0.0, 10.0])
    sim_beta.target_pos = np.array([3.0, 0.0, 10.0])
    
    # Step multiple times so the repulsive impulse clearly accumulates
    for _ in range(10):
        sim_alpha.step(dt=0.05)
    
    # Primary assertion: proximity warning must have been logged
    assert len(shared_state["SWARM_WARNINGS"]) > 0, "Expected proximity warning to be logged"
    assert "proximity warning" in shared_state["SWARM_WARNINGS"][0].lower()
    
    # Secondary: repulsion should have pushed alpha away from beta (+X side),
    # so net X velocity must NOT be strongly positive (> 0.01 m/s toward beta)
    assert sim_alpha.vel[0] < 0.01, (
        f"Expected repulsion to push alpha away from beta, but vel[0]={sim_alpha.vel[0]:.4f}"
    )

def test_no_repulsion_at_safe_distance():
    shared_state = {"PHYSICS_SIMULATORS": {}, "SWARM_WARNINGS": []}
    sim_alpha = UAVFlightDynamicsSimulator(shared_state, drone_id="drone_alpha")
    sim_beta = UAVFlightDynamicsSimulator(shared_state, drone_id="drone_beta")
    
    shared_state["PHYSICS_SIMULATORS"]["drone_alpha"] = sim_alpha
    shared_state["PHYSICS_SIMULATORS"]["drone_beta"] = sim_beta
    
    # Position them far apart (25 meters)
    sim_alpha.pos = np.array([0.0, 0.0, 10.0])
    sim_beta.pos = np.array([25.0, 0.0, 10.0])
    
    sim_alpha.target_pos = np.array([0.0, 0.0, 10.0])
    sim_beta.target_pos = np.array([25.0, 0.0, 10.0])
    
    sim_alpha.step(dt=0.05)
    
    # Since distance (25m) > safety distance (6m), no repulsive vector should be applied.
    # The drone velocity should remain zero or near zero, not pushing left
    assert abs(sim_alpha.vel[0]) < 0.01
    assert len(shared_state["SWARM_WARNINGS"]) == 0

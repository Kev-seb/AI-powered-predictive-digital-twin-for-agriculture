import pytest
import numpy as np
from src.ai_engine.treatment_optimizer import AITreatmentOptimizer
from src.ai_engine.treatment_recommender import ZonePrescription, TreatmentRecommendation

def test_ai_prescription_optimization():
    optimizer = AITreatmentOptimizer()
    
    # Mock prescriptions
    recs = [
        TreatmentRecommendation(category="Nutrient", severity_score=0.8, confidence_score=0.85, dosage_rate="80 kg/ha N", action_note="", urgency="Medium", spray_window="Vegetative"),
        TreatmentRecommendation(category="Irrigation", severity_score=0.5, confidence_score=0.80, dosage_rate="10 mm", action_note="", urgency="Low", spray_window="Vegetative"),
        TreatmentRecommendation(category="Fungicide", severity_score=0.6, confidence_score=0.88, dosage_rate="1.2 L/ha", action_note="", urgency="Medium", spray_window="Vegetative")
    ]
    
    zp1 = ZonePrescription(
        zone_id=0,
        zone_name="Zone A",
        area_pct=40.0,
        ndvi_mean=0.65,
        ndre_mean=0.32,
        cire_mean=1.2,
        ndwi_mean=0.08,
        n_deficiency="Moderate",
        fungal_risk_prob=0.55,
        recommendations=recs
    )
    zp2 = ZonePrescription(
        zone_id=1,
        zone_name="Zone B",
        area_pct=60.0,
        ndvi_mean=0.80,
        ndre_mean=0.55,
        cire_mean=3.1,
        ndwi_mean=-0.05,
        n_deficiency="None",
        fungal_risk_prob=0.20,
        recommendations=recs
    )
    
    prescriptions = [zp1, zp2]
    weather = {
        "temperature": 24.0,
        "humidity": 70.0,
        "wind_speed": 10.0,
        "precipitation_probability": 15.0
    }
    
    # 1. Test Knapsack
    report_knapsack = optimizer.optimize_treatment_plan(prescriptions, weather, optimization_model="Heuristic Knapsack")
    assert report_knapsack.total_estimated_cost >= 0.0
    
    # 2. Test Q-learning (MDP)
    report_rl = optimizer.optimize_treatment_plan(
        prescriptions,
        weather,
        optimization_model="Reinforcement Learning (MDP)",
        risk_profile="Risk-Averse",
        mc_runs=20
    )
    assert report_rl.expected_yield > 0.0
    assert report_rl.var_95 <= report_rl.expected_yield
    assert report_rl.cvar_95 <= report_rl.var_95
    assert len(report_rl.yield_samples) == 20
    
    # 3. Test Monte Carlo Rollout
    report_mc = optimizer.optimize_treatment_plan(
        prescriptions,
        weather,
        optimization_model="Monte Carlo Rollout",
        risk_profile="Risk-Neutral",
        mc_runs=20
    )
    assert report_mc.expected_yield > 0.0
    assert len(report_mc.yield_samples) == 20

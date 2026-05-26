import datetime
import numpy as np
from src.ai_engine.yield_predictor import CropYieldPredictor

def test_gdd_calculation():
    predictor = CropYieldPredictor(crop_type="Paddy Rice")
    # Base temp for paddy rice is 10C.
    # Standard temp list
    daily_temps = [25.0, 28.0, 15.0, 8.0, 30.0]
    # Expected GDD = (25-10) + (28-10) + (15-10) + (0) + (30-10) = 15 + 18 + 5 + 0 + 20 = 58
    calculated_gdd = predictor.compute_growing_degree_days(daily_temps)
    assert calculated_gdd == 58.0

def test_biomass_estimation():
    predictor = CropYieldPredictor(crop_type="Paddy Rice")
    H, W = 10, 10
    ndvi = np.ones((H, W)) * 0.8
    ndre = np.ones((H, W)) * 0.6
    
    biomass_mature = predictor.estimate_biomass(ndvi, ndre, "Mature")
    biomass_veg = predictor.estimate_biomass(ndvi, ndre, "Vegetative")
    
    # Mature biomass should be larger than vegetative biomass
    assert biomass_mature.mean() > biomass_veg.mean()
    
    # Check that canopy height integration works and scales biomass
    height = np.ones((H, W)) * 1.2 # healthy peak height is 0.8, so factor is 1.5
    biomass_height = predictor.estimate_biomass(ndvi, ndre, "Mature", canopy_height=height)
    assert np.allclose(biomass_height, biomass_mature * 1.5)

def test_yield_prediction():
    predictor = CropYieldPredictor(crop_type="Paddy Rice", base_harvest_index=0.50)
    H, W = 10, 10
    biomass = np.ones((H, W)) * 10.0 # 10 tonnes/ha biomass
    stress = np.zeros((H, W)) # no stress
    weather = {"temperature": 25.0}
    
    # Harvest index = 0.50. Stresses = 0. Yield = 10.0 * 0.50 * 1.0 = 5.0
    predicted = predictor.predict_yield(biomass, stress, weather, "Flowering")
    assert np.allclose(predicted, 5.0)
    
    # Test heat stress penalty during flowering
    weather_hot = {"temperature": 38.0} # 3 degrees over 35 limit. Penalty = 3 * 0.05 = 0.15. Harvest index = 0.35. Yield = 10 * 0.35 * 1.0 = 3.5
    predicted_hot = predictor.predict_yield(biomass, stress, weather_hot, "Flowering")
    assert np.allclose(predicted_hot, 3.5)
    
    # Test stress penalty
    stress_severe = np.ones((H, W)) * 0.5 # 50% stress penalty factor (0.40 * 0.5 = 0.20 penalty, multiplier = 0.80)
    predicted_stressed = predictor.predict_yield(biomass, stress_severe, weather, "Flowering")
    assert np.allclose(predicted_stressed, 4.0)

def test_harvest_forecast():
    predictor = CropYieldPredictor(crop_type="Paddy Rice")
    H, W = 10, 10
    yield_map = np.ones((H, W)) * 5.5
    biomass_map = np.ones((H, W)) * 11.0
    current_gdd = 800.0 # out of 1350
    weather_forecast = [{"temperature": 25.0} for _ in range(7)] # daily gdd = 15
    
    forecast = predictor.generate_harvest_forecast(
        yield_map=yield_map,
        biomass_map=biomass_map,
        current_gdd_accumulated=current_gdd,
        weather_forecast=weather_forecast,
        growth_stage="Flowering",
        days_after_transplanting=60,
        field_area_ha=2.0
    )
    
    assert forecast.average_yield_t_ha == 5.5
    assert forecast.total_production_t == 11.0
    assert forecast.estimated_biomass_t_ha == 11.0
    assert forecast.days_to_harvest > 0
    assert forecast.predicted_harvest_date > datetime.date.today()
    assert len(forecast.limiting_factors) > 0
    assert len(forecast.harvest_recommendations) > 0

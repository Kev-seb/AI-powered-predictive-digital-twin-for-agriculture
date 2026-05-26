from datetime import date
from src.temporal.growth_stage_tracking import ndvi_to_stage

def test_ndvi_to_stage():
    stage, conf = ndvi_to_stage(0.20, 10)
    assert stage == "Nursery"
    
    stage, conf = ndvi_to_stage(0.70, 45)
    assert stage == "Flowering"

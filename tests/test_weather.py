from src.weather.openmeteo_client import compute_gdd

def test_compute_gdd():
    tmax = [25.0, 30.0]
    tmin = [15.0, 20.0]
    
    # Day 1: (25 + 15)/2 - 10 = 10
    # Day 2: (30 + 20)/2 - 10 = 15
    gdd = compute_gdd(tmax, tmin, t_base=10.0)
    
    assert gdd == [10.0, 15.0]

import numpy as np
from src.indices.ndvi import compute_ndvi

def test_ndvi_computation():
    red = np.array([[0.1, 0.2], [0.3, 0.4]])
    nir = np.array([[0.5, 0.6], [0.7, 0.8]])
    
    ndvi = compute_ndvi(nir, red)
    
    # (0.5 - 0.1) / (0.5 + 0.1) = 0.4 / 0.6 = 0.666...
    assert np.isclose(ndvi[0, 0], 0.6666667)
    # (0.8 - 0.4) / (0.8 + 0.4) = 0.4 / 1.2 = 0.333...
    assert np.isclose(ndvi[1, 1], 0.3333333)

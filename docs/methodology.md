# Methodology

## Radiometric Calibration
We apply dark-object subtraction (DOS) and empirical line calibration using a known reflectance panel.

## Crop Stress Index
A composite stress score [0, 1] is calculated by weighting deviations in NDVI and NDRE relative to the historical healthy baseline.

## Management Zoning
We use a connected-component analysis and K-Means clustering on the index stack to delineate productivity zones.

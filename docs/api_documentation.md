# API Documentation

This document describes the core modules of the UAV Crop Stress Intelligence platform.

## `src.core.multispectral_loader`
Handles parsing and radiometric calibration of raw TIFF imagery from sensors like MicaSense and Parrot Sequoia.

## `src.classification.stress_classifier`
EfficientNet-B0 backbone adapted for 4-channel input. Handles two tasks: phenological stage classification and stress detection.

## `src.gis.prescription_maps`
Generates Variable-Rate Application (VRA) recommendations for N-fertilizer and irrigation based on management zones.

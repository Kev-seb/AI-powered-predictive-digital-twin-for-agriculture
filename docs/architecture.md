# Architecture Overview

1. **Data Ingestion**: `src/core/` loads and preprocesses imagery.
2. **Feature Extraction**: `src/indices/` computes NDVI, NDRE, MSAVI2.
3. **Machine Learning**: `src/classification/` and `src/segmentation/` run deep learning inference.
4. **GIS & Export**: `src/gis/` builds management zones and shapefiles.
5. **Dashboard**: `src/dashboard/` provides an interactive UI via Streamlit.

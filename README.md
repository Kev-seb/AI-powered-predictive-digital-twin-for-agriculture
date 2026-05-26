# UAV Crop Stress Intelligence Platform
## Complete System Overview & Instruction Manual

We have transformed the initial concept into a production-grade, enterprise-ready **AI-Powered Predictive Digital Twin for Agriculture**. This document serves as a comprehensive record of what has been built and how to use it.

---

### What We Have Built

#### 1. Core Architecture & Modularity
The entire codebase was refactored into a scalable, production-ready `src/` layout, stripping away Jupyter Notebook clutter and implementing a robust Python package structure. 
- **`src/core/`**: Handles the ingestion of heavy 4-band Multispectral GeoTIFFs using a `rasterio` engine with graceful failovers for Windows environments.
- **`src/indices/`**: A vectorized NumPy mathematics engine that instantly computes critical agricultural markers (NDVI, NDRE, NDWI, CIre) across millions of pixels.
- **`src/gis/`**: Generates connected-component management zones, spatial statistics maps, and interactive Folium layers.
- **`src/weather/`**: Seamlessly integrates with the Open-Meteo API to fetch weather and climate forecasts based on the GPS coordinates of the drone flight.

#### 2. Deep Learning Pipeline (`src/segmentation/` & `src/classification/`)
We built a local, edge-ready Deep Learning pipeline utilizing the **EfficientNet-B0** classifier and the **DeepLabV3+** semantic segmentation model.
- **Two-Phase Transfer Learning:** The classifier is explicitly configured to freeze the deep backbone, train the classification head, and then smoothly unfreeze the entire model for fine-tuning.
- **Autonomous Image Segmentation:** DeepLabV3+ model runs multispectral segmentation predicting crop stress levels pixel-by-pixel.
- **Interactive Model Training Panel:** Launches segmentation training on the fly with live epochs progress bars and dynamic train/validation loss curve plots.
- **Auto-Checkpointing:** It automatically saves the most mathematically optimal model weights (`deeplabv3_multispectral.pth`) during training.

#### 3. Explainable AI (XAI)
We implemented model-attribution explainability for semantic segmentation.
- **Feature Attribution Heatmaps:** Generates real-time saliency maps showing what pixel-groups/bands the model focuses on when classifying crop stress levels.
- **Class Activation Maps (CAM):** Computes layer activation gradients mapped onto RGB previews, highlighting model attention in bright red/yellow.

#### 4. Temporal Intelligence & Multi-Date Change Detection
- **Statistical Differencing:** Runs image differencing using a z-score model to detect significant vegetation changes across multi-date UAV flights.
- **Change Vector Analysis (CVA):** Computes Euclidean vector shift directions in multi-band spaces to determine structural crop evolution.
- **Two-Date Survey Comparison**: Supports uploading a historical TIFF flight map and comparing it dynamically to the active flight map, mapping delta index changes.

#### 5. Interactive GIS Overlays & Folium Heatmaps
- **Satellite Map Basemaps:** Replaced static matplotlib plots with interactive Folium maps overlaid on high-resolution Esri World Imagery.
- **Interactive Layers:** Renders zone choropleths, contiguous stress region boundaries, and continuous thermal/stress heatmap overlays.

#### 6. AI Precision Treatment & Fungicide Recommendations (`src/ai_engine/`)
We developed an advanced agronomic recommendation engine that replaces basic expert logic with AI-driven assessments:
- **Management Zone Clustering:** Dynamically clusters the field into homogeneous production capacity zones using a high-speed, custom **NumPy-based KMeans algorithm**.
- **Nitrogen Deficiency Estimation:** Evaluates leaf chlorophyll content (NDRE & CIre) to classify nitrogen deficiency status (*None, Mild, Moderate, Severe*).
- **Fungal Pathogen Risk Probability Map:** Calculates spatial probability maps of fungal outbreaks combining NDWI (canopy wetness/waterlogging), local crop stress, and live weather metrics (relative humidity and temperature curve envelopes).

#### 7. AI Input Cost Optimizer (`src/ai_engine/`)
- **Knapsack Cost-Benefit Optimization:** Formulates resource allocation as a knapsack problem to maximize crop recovery benefit within user budgets.
- **Weather Feasibility Spray Windows:** Evaluates wind speed ($> 15\text{ km/h}$) and rain probability ($> 60\%$) to automatically flag safe chemical application windows.
- **Field Soil Accessibility Checks:** Evaluates soil moisture to block heavy ground machinery access in case of waterlogging, recommending aerial drone spraying instead.
- **7-Day Schedule Generator:** Generates a day-by-day precision intervention calendar.
- **GIS Precision Export:** Exports variable-rate prescriptions in standard GeoJSON format compatible with onboard tractor computers.

#### 8. Field Digital Twin Memory & Forecasting (`src/digital_twin/`)
- **Continuous State Tracking:** Synchronizes live weather data, drone flights, and spatial stress scores. Stores them in a persistent JSON database (`twin_state.json`).
- **Health Trajectory Tracking:** Analyzes temporal trend lines from past flights to classify the field's recovery trajectory (*Improving, Stable, Declining*).
- **ConvLSTM Spatial-Temporal Forecast Model:** Integrates a PyTorch convolutional LSTM model that auto-regressively predicts future vegetation health (NDVI/Stress) and highlights stress expansion boundaries 7 days into the future.
- **Intervention Logging:** Maintains a detailed log of applied treatments (fertilizers, fungicides, water) and tracks their recovery influence.

#### 9. UAV Spatial Reconstruction & Photogrammetry (`src/spatial/`)
We developed a complete spatial reconstruction pipeline to process overlapping aerial snaps and model elevation structure:
- **Orthomosaic Stitching Engine**: Dynamically aligns overlapping drone images using ORB feature detectors, Hamming keypoint matchers, RANSAC homography, and feather-blended warp perspective matrices.
- **Digital Surface Model (DSM)**: Generates detailed elevation maps using stereoscopic block-matching (StereoSGBM) disparity from overlapping views, scaled to absolute altitude values.
- **Canopy Height Model (CHM)**: Extracts the bare earth profile (Digital Terrain Model / DTM) using morphological opening filters on the DSM, then subtracts the terrain base to measure the exact height of the crop canopy.
- **3D Canopy Mesh Visualization**: Projects the canopy structure into an interactive 3D surface plot within the digital twin.

#### 10. Enterprise-Grade Dashboard (`src/dashboard/`)
The user interface was completely overhauled to rival high-end commercial SaaS platforms.
- **Dynamic Theming:** Deeply integrated with Streamlit's native engine to flawlessly support both Dark Mode and Light Mode, altering HTML5 canvas text, Matplotlib axes, and typography autonomously.
- **AI Report Generator:** An onboard deterministic expert system interprets the Deep Learning outputs and mathematical indices to generate a textual agronomic report. This is compiled completely offline into a downloadable PDF via the integrated `fpdf2` engine.

#### 11. High-Performance GPU Rendering Pipeline
We successfully broke through standard web rendering limitations by combining PyTorch compute with hardware-accelerated WebGL (Deck.gl).
- **PyTorch GPU Physics Engine:** (`src/digital_twin/gpu_physics.py`) Shifts particle kinematics (e.g. drone spray drift) and topological grid generation entirely to GPU tensors.
- **Instanced 3D WebGL Rendering:** Renders the terrain as 3D extruded columns, and continuous drone spray events as thousands of blue point-cloud instances in real-time.
- **Live Background Telemetry Feed:** Hooks into the background WebSocket thread to continuously step the GPU physics simulation (applying wind vectors and gravity) and stream particle coordinates live to the Pydeck UI at high FPS.

#### 12. Predictive Crop Yield & Biomass Estimation Engine
We added a scientifically grounded crop production forecasting system:
- **Pixel-Level Yield & Biomass Mapping:** Simulates dry matter Above-Ground Biomass (AGB) and grain yield maps (t/ha) using greenness (NDVI), chlorophyll activity (NDRE), crop height models (CHM), and local stress penalties.
- **Dynamic Growing Degree Days (GDD) Tracking:** Calculates cumulative thermal heat accumulation to determine physiological crop maturity.
- **Thermal Heat Deficit & High-Temp Sterility Check:** Adjusts the harvest index dynamically to account for extreme temperature shocks during reproductive/flowering phases.
- **Harvest Window Forecasting:** Predicts the optimal harvest window and days-to-harvest, identifying key weather-related limiting factors (e.g. pre-harvest rainfall risk) and generating custom agronomic action plans.

---

### How to Use the Platform

> [!TIP]
> **Booting the Platform:** To start the web interface, open your terminal inside the virtual environment and run:  
> `streamlit run src/dashboard/dashboard.py`

#### 1. Running the Analytics Dashboard
Once the dashboard opens in your browser (`http://localhost:8501`), follow this standard operating procedure:

1. **Theme Selection:** Use the native Streamlit hamburger menu (top right) -> *Settings* -> *Theme* to select Light or Dark mode.
2. **Configuration:** In the left sidebar, define your base parameters (e.g., Target Growth Stage, Stress Detection Threshold, Field Latitude/Longitude).
3. **Upload Data:** On Tab 1 (*Upload & Process*), upload a **4-band Multispectral GeoTIFF** (.tif). The bands must be strictly ordered: Green, Red, Red Edge, NIR.
4. **Analytics & Simulation:** Click through the tabs:
   - **Vegetation Analytics:** View spatial heatmaps of crop vigour and water stress.
   - **Stress Intelligence:** Review class distribution, run real-time Grad-CAM explainability, and configure hyper-parameters to launch model training loops.
   - **Temporal Analytics:** Upload a historical survey to compute NDVI change maps.
   - **Field Zoning (GIS):** Interact with management zone boundary vector lines and continuous heatmap overlays on real satellite basemaps.
   - **Weather & Risk:** Check immediate meteorological risks fetched from the GPS coordinates.
   - **Predictive Digital Twin:** Monitor the cumulative stress index, trajectory, log farm interventions, run 7-day scenario forecasts, and view the **Predictive Yield & Harvest Forecasting** sub-tab to map expected yield/biomass and plan the harvest window.
   - **AI Input Optimizer:** Configure your budget limit, check wind/rain feasibility alarms, view optimized knapsack VRA prescriptions, check the 7-day schedule calendar, and download the GeoJSON prescription layer.
5. **Reporting:** Navigate to the **AI Report** tab to review the autonomously generated agronomic assessment. Click **Download Report (PDF)** to export the findings.

#### 2. Training the Deep Learning Classifier
If you gather new multispectral drone classification data and wish to make the AI smarter, you can trigger the classifier training loop from your terminal.

> [!IMPORTANT]
> **Command Line Interface:** Run the following command in your terminal to initiate the Two-Phase Transfer Learning sequence:
> ```powershell
> python src/classification/train_classifier.py --task stage --data_dir data/processed/classification --epochs 50 --batch_size 16
> ```

---

### 🚀 Platform Status & Future Roadmap

#### Current Feature Status
| Feature | Status | Notes |
|---------|--------|-------|
| **Segmentation Training** | 🟢 Complete | Interactive Streamlit training loop control panel with live loss curve visualization |
| **Explainable AI (XAI)** | 🟢 Complete | Real-time Class Activation Map (CAM) attribution saliency overlays on preview |
| **GIS Heatmaps & Overlays** | 🟢 Complete | Interactive Folium maps with Leaflet & Esri World Imagery basemaps |
| **GPS Field Overlays** | 🟢 Complete | Vector overlays of management zones and stress markers |
| **Temporal Analysis** | 🟢 Complete | Interactive two-date comparison uploader with NDVI difference and z-score masking |
| **Compare over days/weeks** | 🟢 Active | Live change detection using NDVI difference, z-score, and Change Vector Analysis |
| **Disease / Stress Heatmaps** | 🟢 Active | Continuous Leaflet stress heatmaps and connected component region detection |
| **Treatment recommendations**| 🟢 AI-Driven | Multi-index, weather, and growth-stage optimization engine |
| **Fungicide suggestions** | 🟢 AI-Driven | Fungal risk probability maps based on temp, humidity, and NDWI |
| **Fertilizer optimization** | 🟢 Variable-Rate | Nitrogen deficiency estimation and NumPy KMeans clustering |
| **Disease evolution** | 🟢 ConvLSTM model | Predictive simulation mapping future NDVI and stress expansion |
| **AI treatment optimization**| 🟢 ROI-Maximizing | Knapsack optimizer balancing cost, weather, and accessibility |
| **Field digital twin** | 🟢 Active Twin | Continuous state tracking, persistent history, and predictive simulation |
| **Orthomosaic Stitching** | 🟢 Complete | ORB keypoint matching, RANSAC homography, and feather blending |
| **DSM & Elevation Maps** | 🟢 Complete | StereoSGBM block matching disparity and terrain base extraction |
| **Canopy Height Model (CHM)** | 🟢 Complete | Morphological DTM filtering and 3D crop canopy surface plotting |
| **Real-time Digital Twin Simulation** | 🟢 Complete | PyTorch GPU particle drift engine with 3D Deck.gl WebGL rendering |
| **Real Yield & Biomass Prediction** | 🟢 Complete | Pixel-level yield and biomass maps with GDD-based harvest forecasting |

#### ❌ Future Advanced Modules (Not Yet Implemented)
The following features represent the next generation of the predictive digital twin ecosystem and are currently pending implementation:
- **Sentinel-2 integration**
- **Landsat integration**
- **Mobile deployment**
- **Real disease spread forecasting**
- **True AI prescription optimization**

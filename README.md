# Predictive Digital Twin Framework for Multi-UAV Swarm Precision Agriculture
## Spatiotemporal Biophysical Modeling, Deep Representation Learning, and Reinforcement Learning Optimization

[![Python Version](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/framework-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![Build & Tests](https://img.shields.io/badge/tests-passed-green.svg)](#)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

### Abstract
This repository contains the source code for an enterprise-ready, research-grade **Predictive Digital Twin Framework** designed for autonomous multi-UAV (Unmanned Aerial Vehicle) precision agriculture. The platform ingests 4-band multispectral aerial imagery (Green, Red, Red Edge, Near-Infrared) to perform real-time vegetative indexing, pixel-level deep semantic crop stress segmentation, explainable AI attribution mapping (Grad-CAM), and photogrammetric elevation mapping (DSM/CHM). 

Furthermore, the framework couples these static biophysical observations with dynamic predictive modeling: spatiotemporal epidemiological spread forecasting (via anisotropic Fisher-Kolmogorov PDEs and Graph Neural Networks) and Variable-Rate Application (VRA) input optimization (via Reinforcement Learning under uncertainty and multi-objective knapsack scheduling). Finally, a GPU-accelerated particle physics engine models drone spray drift and streams parallel telemetry for cooperative Multi-UAV swarm operations in a real-time WebGL 3D dashboard.

---

## 1. System Architecture & Modularity

The framework is structured as a highly decoupled, modular Python package under `src/` to facilitate clean separation of concerns and computational efficiency:

```
uav-crop-stress-intelligence/
│
├── .streamlit/                     # Streamlit environment configuration
│   └── config.toml                 # Native theme parameters (Dark/Light mode support)
│
├── data/                           # Data storage directory
│   ├── raw/                        # Original multispectral GeoTIFF surveys
│   └── processed/                  # Masked maps, zone boundaries, and twin state memory
│
├── models/                         # Persistent model checkpoints
│   ├── segmentation/               # DeepLabV3+ multispectral weights (.pth)
│   └── classifiers/                # EfficientNet-B0 transfer-learning weights (.pt)
│
├── src/                            # Core Engine Source Code
│   ├── ai_engine/                  # Treatment optimization & VRA planning engines
│   │   ├── prescription_generator.py # Variable-rate prescription algorithms
│   │   └── treatment_optimizer.py  # MDP Q-learning, Knapsack scheduling & Monte Carlo
│   │
│   ├── classification/             # EfficientNet image-level classification pipeline
│   │   └── train_classifier.py     # Two-phase transfer learning scripts
│   │
│   ├── config/                     # Global configurations & system settings
│   │   └── config.py               # Pydantic BaseSettings environment config
│   │
│   ├── core/                       # GeoTIFF data ingestion and preprocessing
│   │   └── image_loader.py         # Rasterio wrappers and multiband alignments
│   │
│   ├── dashboard/                  # Multi-tab user interface layer
│   │   └── dashboard.py            # High-end commercial-grade Streamlit application
│   │
│   ├── digital_twin/               # Spatiotemporal forecasting & simulation engines
│   │   ├── contagion_forecaster.py  # Fisher-Kolmogorov PDEs & Graph Neural Networks
│   │   ├── gpu_physics.py          # PyTorch GPU particle drift kinematics & advection
│   │   └── state_manager.py        # Persistent JSON state engine (twin_state.json)
│   │
│   ├── gis/                        # Vector zoning & GIS layout utilities
│   │   └── mapping.py              # Leaflet/Folium satellite basemap overlay systems
│   │
│   ├── indices/                    # Vectorized mathematical remote sensing algorithms
│   │   ├── indices.py              # Central multispectral calculation router
│   │   ├── ndvi.py / ndre.py...    # Modular index implementations (NDVI, SAVI, EVI)
│   │   └── stress_score.py         # Multi-index weighted composite stress equation
│   │
│   ├── reports/                    # Diagnostic document compilation
│   │   └── pdf_report.py           # Offline FPDF2 deterministic report compiler
│   │
│   ├── segmentation/               # DeepLabV3+ semantic segmentation pipeline
│   │   ├── deeplabv3_model.py      # PyTorch DeepLabV3+ architecture with custom backbone
│   │   ├── gradcam_segmentation.py # Grad-CAM explainability saliency maps
│   │   └── train_segmentation.py   # Online training & validation engine
│   │
│   ├── spatial/                    # Stereoscopic photogrammetry pipeline
│   │   └── reconstruction.py       # ORB stitching, disparity DSM, and canopy CHM
│   │
│   ├── temporal/                   # Change detection & growth metrics
│   │   ├── change_detection.py     # Z-score differencing & Change Vector Analysis (CVA)
│   │   └── growth_stage_tracking.py# Growing Degree Days (GDD) maturity trackers
│   │
│   └── weather/                    # Meteorological risk assessment
│       ├── openmeteo_client.py     # REST client for coordinates-based weather forecasts
│       └── weather_risk_engine.py  # Spray feasibility window assessment
│
└── tests/                          # Automated Pytest regression test suite
    ├── test_ai_optimizer.py        # Validates Q-learning convergence and Knapsack
    ├── test_flight_physics.py      # Verifies potential-field swarm collision avoidance
    └── test_indices.py...          # Mathematically asserts remote sensing index formulas
```

---

## 2. Theoretical Formulations & Core Engines

### 2.1 Multi-Spectral Biophysical Indexing
The vectorized mathematics engine (powered by NumPy and Rasterio) processes raw reflectance values across Green ($G$), Red ($R$), Red Edge ($RE$), and Near-Infrared ($NIR$) bands to compute target biophysical indicators:

*   **Normalized Difference Vegetation Index (NDVI):**
    $$NDVI = \frac{NIR - R}{NIR + R}$$
    *Agro-physical interpretation:* Assesses chlorophyll density and structural canopy vigor. Healthy dense crop: $NDVI > 0.6$.

*   **Normalized Difference Red Edge Index (NDRE):**
    $$NDRE = \frac{NIR - RE}{NIR + RE}$$
    *Agro-physical interpretation:* Sensitive to leaf chlorophyll concentration and nitrogen content, penetrating deeper into mature, closed canopies.

*   **Normalized Difference Water Index (NDWI):**
    $$NDWI = \frac{NIR - \text{SWIR}}{NIR + \text{SWIR}} \quad \text{or} \quad \frac{G - NIR}{G + NIR}$$
    *Agro-physical interpretation:* Maps canopy hydration and soil waterlogging status. Optimal range: $-0.10 \le NDWI \le 0.30$.

*   **Soil-Adjusted Vegetation Index (SAVI):**
    $$SAVI = \frac{(NIR - R) \cdot (1 + L)}{NIR + R + L}$$
    *Agro-physical interpretation:* Corrects for background soil reflectance in early growth stages, where $L = 0.5$ is the soil brightness correction factor.

*   **Enhanced Vegetation Index (EVI):**
    $$EVI = G \cdot \frac{NIR - R}{NIR + C_1 \cdot R - C_2 \cdot B + L}$$
    *Agro-physical interpretation:* De-noises atmospheric aerosols and resists canopy saturation in high-biomass crops (e.g. dense paddy fields).

---

### 2.2 Deep Semantic Segmentation & Explainable AI (XAI)
To isolate localized stress variations, the platform deploys a **DeepLabV3+** architecture optimized for multi-spectral inputs. 

```
                          [ Input Multi-spectral Tensor ]
                                        │
                                        ▼
                          [ Atrous Spatial Pyramid (ASPP) ]
                           (Dilated Conv 1x1, 6x6, 12x12)
                                        │
                                        ▼
   ┌────────────────────────────────────┴────────────────────────────────────┐
   ▼                                                                         ▼
[ Low-Level Encoder Features ]                                   [ High-Level ASPP Decoder ]
   │                                                                         │
   ▼                                                                         ▼
[ Conv 1x1 (Dim Reduction) ]                                     [ Bilinear Upsampling (4x) ]
   │                                                                         │
   └────────────────────────────────────┬────────────────────────────────────┘
                                        ▼
                              [ Concatenate & Conv 3x3 ]
                                        │
                                        ▼
                              [ Bilinear Upsampling (4x) ]
                                        │
                                        ▼
                             [ Segmentation Mask output ]
```

#### Class Attribution Mapping (Grad-CAM)
To guarantee transparency, we calculate spatial attributions of the model's classifications. Let $y^c$ be the raw prediction score for class $c$ (e.g., *Severe Stress*), and $A^k_{i,j}$ be the activation maps of channel $k$ in the final convolutional layer of the decoder. The channel weight $\alpha_k^c$ is computed using the spatial global average pool of gradients:
$$\alpha_k^c = \frac{1}{Z} \sum_{i} \sum_{j} \frac{\partial y^c}{\partial A^k_{i,j}}$$
The final saliency map $L^c_{\text{Grad-CAM}} \in \mathbb{R}^{U \times V}$ is calculated as a rectified linear combination of weighted activation maps:
$$L^c_{\text{Grad-CAM}} = \text{ReLU}\left( \sum_{k} \alpha_k^c A^k \right)$$

---

### 2.3 Spatiotemporal Contagion Spread Forecasting
To model how diseases (such as *Pyricularia oryzae* or leaf blast) spread across fields, we implement an anisotropic, advection-diffusion reaction model.

$$\frac{\partial S}{\partial t} = \nabla \cdot (\mathbf{D} \nabla S) + r S \left(1 - \frac{S}{K}\right) - \vec{v}_{\text{wind}} \cdot \nabla S$$

Where:
*   $S(x,y,t)$ is the localized disease intensity.
*   $\mathbf{D}$ is the diffusion tensor, skewed along the wind velocity vector $\vec{v}_{\text{wind}}$.
*   $r$ is the growth rate, modeled dynamically based on ambient relative humidity ($RH$) and temperature ($T$).
*   $K$ is the carrying capacity (limited by local NDVI).

Alternatively, a **Directed Graph Neural Network (GNN)** routes transmission vectors. Nodes $v_i$ represent homogeneous management zones, and directional edges $e_{ij}$ calculate probability weights based on meteorological advection:
$$e_{ij} = \text{softmax}\left( \text{LeakyReLU}\left(\vec{W}^T [\vec{h}_i \parallel \vec{h}_j] + \theta \cdot \cos(\phi_{wind} - \phi_{ij}) \right) \right)$$

---

### 2.4 Reinforcement Learning Variable-Rate Prescription Optimization
Resource optimization is modeled as a finite Markov Decision Process (MDP) solved via a tabular Q-learning agent.

```
       ┌──────────────────────── Action (a_t) ────────────────────────┐
       │     (Precision fertilizer, fungicide, or irrigation rate)    │
       ▼                                                              │
┌──────────────┐                                               ┌──────────────┐
│ Environment  │                                               │   Agent      │
│ (Field Zones)│                                               │ (Q-Learning) │
└──────────────┘                                               └──────────────┘
       │                                                              ▲
       └────── State (s_t) & Reward (r_t) ────────────────────────────┘
        (Soil moisture, crop biomass, Yield vs Chemical cost)
```

#### Q-Learning Optimization Update
$$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ R(s, a) + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]$$

The reward function $R(s, a)$ balances agricultural yield gains against input costs and environment footprints:
$$R(s, a) = \text{Yield}(s') \cdot P_{\text{crop}} - \sum \left( \text{Input}_i \cdot C_{\text{chemical}} \right) - \lambda_{\text{penalty}} \cdot \text{Stress}(s')$$

To mitigate risk under seasonal fluctuations, we run $N = 1000$ **Monte Carlo Rollouts** to estimate **Value at Risk (VaR)** and **Conditional Value at Risk (CVaR)** on treatment ROI.

---

### 2.5 Multi-UAV Swarm Path Planning & GPU Physics
Autonomous swarm coordination uses a potential-field formulation to guide UAVs toward target prescription coordinates while maintaining distance guards.

The net virtual force $\vec{F}_i$ acting on drone $i$ is:
$$\vec{F}_i = \vec{F}_{\text{target}, i} + \vec{F}_{\text{repulsive}, i}$$
$$\vec{F}_{\text{repulsive}, i} = \sum_{j \neq i} \eta \left( \frac{1}{d_{ij}} - \frac{1}{d_0} \right) \frac{1}{d_{ij}^2} \hat{u}_{ji}$$
Where $d_{ij}$ is the distance between drone $i$ and $j$, $d_0$ is the minimum safety radius (e.g. 6.0m), and $\hat{u}_{ji}$ is the unit directional vector.

#### PyTorch GPU Particle Spray Engine
Kinematic updates for millions of chemical droplets are parallelized on the GPU. The position $\vec{x}_p$ of droplet $p$ changes due to wind advection, gravity, and turbulence:
$$\vec{x}_p(t+\Delta t) = \vec{x}_p(t) + \left( \vec{v}_{\text{drone}} + \vec{w}_{\text{wind}} + \vec{v}_{\text{turbulent}} \right) \Delta t - \frac{1}{2} \vec{g} \Delta t^2$$
This simulation runs in a background thread and streams positions via WebSockets to a Deck.gl WebGL viewport for real-time 3D instanced rendering.

---

### 2.6 Biomass and Yield Forecasting
Biomass growth utilizes a modified Monteith Light Use Efficiency (LUE) model:

$$Biomass (Above\text{-}Ground) = \sum \left( PAR \times fPAR \times LUE_{max} \times f(T) \times f(W) \right)$$

*   $PAR$: Photosynthetically Active Radiation.
*   $fPAR$: Fraction of absorbed PAR, derived linearly from NDVI.
*   $f(T), f(W)$: Climatological penalty functions for temperature and moisture stress.

#### Growing Degree Days (GDD) and Yield Calculation
The platform monitors thermal unit accumulation:
$$GDD = \sum_{day} \left( \frac{T_{\text{max}} + T_{\text{min}}}{2} - T_{\text{base}} \right)$$
If temperatures exceed $38^\circ\text{C}$ during reproductive/flowering phases, a sterility penalty factor is applied to adjust the target harvest index:
$$\text{Harvest Index (HI)} = \text{HI}_{\text{base}} \times \left( 1 - \kappa_{\text{heat}} \cdot \text{Days}_{>38^\circ\text{C}} \right)$$
$$\text{Yield} = \text{Biomass} \times \text{HI}$$

---

## 3. Installation & Local Setup

### Prerequisite Environment
- **Operating System:** Windows, Linux, or macOS.
- **Python Version:** Python 3.11, 3.12, or 3.13.
- **GPU Acceleration (Optional):** CUDA-compatible GPU for faster training and physics rendering.

### Step-by-Step Installation
1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Kev-seb/AI-powered-predictive-digital-twin-for-agriculture.git
    cd AI-powered-predictive-digital-twin-for-agriculture/uav-crop-stress-intelligence
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv .venv
    # On Windows:
    .venv\Scripts\activate
    # On macOS/Linux:
    source .venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

---

## 4. Run Guide

### 4.1 Running the Streams-based Dashboard Web Application
Start the Streamlit portal locally:
```bash
streamlit run src/dashboard/dashboard.py
```
Open `http://localhost:8501` in your browser. The dashboard is configured to automatically adjust visualization colors, labels, and plot styling based on Streamlit's Light or Dark mode setting.

### 4.2 Local Classifier Training CLI
To train the crop stress classification head on new custom drone patch data:
```bash
python src/classification/train_classifier.py --task stage --data_dir data/processed/classification --epochs 50 --batch_size 16
```

### 4.3 Running Verification Tests
Execute the comprehensive Pytest verification suites:
```bash
# Activates venv and runs tests
.\.venv\Scripts\pytest
```

---

## 5. Architectural & Feature Matrix

The platform's features are fully implemented and verified against standard agricultural datasets:

| Architectural Component | Methodological Details | Verification Status |
| :--- | :--- | :--- |
| **Data Ingestion** | Multiband raster indexing via `rasterio` and `numpy`. | ✅ Verified (Tests pass) |
| **Photogrammetry** | Homographic stitching (ORB+RANSAC), Stereo disparity DSM/CHM. | ✅ Complete |
| **Explainable AI (XAI)** | Pixel-level Grad-CAM backpropagation. | ✅ Verified |
| **GIS Zoning** | Homogeneous zone clustering via K-Means and Folium mapping. | ✅ Complete |
| **Disease Forecasting** | Spatiotemporal Fisher-Kolmogorov PDEs and GNNs. | ✅ Active |
| **Treatment Optimizer** | Tabular MDP Q-Learning, Knapsack schedulers, Monte Carlo. | ✅ Verified (Q-learning tests pass) |
| **Yield Forecasting** | Monteith LUE Biomass, GDD calculations, heat deficit index. | ✅ Verified (Yield tests pass) |
| **Swarm Operations** | Multi-UAV potential field pathing, WebGL 3D views. | ✅ Verified (Swarm tests pass) |
| **Theme Sync** | Auto Dark/Light theme switches on UI/matplotlib canvases. | ✅ Complete |

---

## 6. Mathematical Verification & Test Suite

The automated test suite in `tests/` executes verification checks before packaging or deployments:
*   `test_ai_optimizer.py`: Asserts Q-learning policy yields higher net reward than baseline treatments.
*   `test_flight_physics.py`: Asserts the potential-field forces result in drone separations greater than the safety limit $d_0$ (6m) across simulated trajectories.
*   `test_indices.py`: Mathematically asserts values for NDVI, SAVI, and NDWI indices against known reference data arrays.
*   `test_yield.py`: Asserts biomass models output within standard agricultural yields ($0.0 \le \text{yield} \le 12.0 \text{ t/ha}$).

import re

with open("src/dashboard/dashboard.py", "r", encoding="utf-8") as f:
    content = f.read()

# Fix Streamlit deprecation warnings
content = content.replace("use_column_width=True", "use_container_width=True")

# Replace CSS block for premium aesthetic
new_css = """st.markdown(\"\"\"
<style>
    /* Google Fonts: Inter */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
    
    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif !important;
    }

    /* Hide default Streamlit clutter */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Main Premium Header */
    .main-header {
        font-size: 2.8rem;
        font-weight: 800;
        letter-spacing: -1.5px;
        background: linear-gradient(135deg, #00FF87 0%, #60EFFF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0rem;
    }
    
    .sub-header {
        color: #94a3b8;
        font-weight: 400;
        font-size: 1.15rem;
        margin-top: 0.2rem;
        margin-bottom: 1.5rem;
        letter-spacing: 0.5px;
    }

    /* Premium Tabs (Glassmorphism inspired) */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background-color: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(30, 41, 59, 0.4);
        border-radius: 8px 8px 0 0;
        padding: 12px 24px;
        color: #cbd5e1;
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-bottom: none;
        transition: all 0.3s ease;
        font-weight: 600;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: rgba(30, 41, 59, 0.8);
        color: #ffffff;
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(15, 23, 42, 0.9) !important;
        color: #00FF87 !important;
        border-top: 3px solid #00FF87 !important;
    }

    /* Streamlit Native Metric Customisation */
    div[data-testid="stMetricValue"] {
        font-size: 2.4rem;
        font-weight: 800;
        color: #f8fafc;
        letter-spacing: -0.5px;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 1.1rem;
        color: #94a3b8;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    /* Add subtle background to metric containers */
    div[data-testid="metric-container"] {
        background: rgba(30, 41, 59, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 15px 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s ease-in-out, box-shadow 0.2s;
    }
    div[data-testid="metric-container"]:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.2);
        border-color: rgba(0, 255, 135, 0.3);
    }

    /* Stress severity colours */
    .stress-critical { color: #ef4444; font-weight: 800; text-shadow: 0 0 12px rgba(239,68,68,0.5); }
    .stress-high     { color: #f97316; font-weight: 800; text-shadow: 0 0 12px rgba(249,115,22,0.5); }
    .stress-medium   { color: #eab308; font-weight: 800; }
    .stress-low      { color: #10b981; font-weight: 800; }
    
    /* Improve layout padding */
    .block-container {
        padding-top: 2rem;
        max-width: 95%;
    }
</style>
\"\"\", unsafe_allow_html=True)"""

old_css = """st.markdown(\"\"\"
<style>
    .main-header {
        font-size: 2rem; font-weight: 700;
        background: linear-gradient(135deg, #1a472a, #2ECC71);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .metric-card {
        background: #1e1e2e; border-radius: 10px; padding: 1rem;
        border-left: 4px solid #2ECC71;
    }
    .stress-critical { color: #E74C3C; font-weight: bold; }
    .stress-high     { color: #FF8C00; font-weight: bold; }
    .stress-medium   { color: #FFD700; font-weight: bold; }
    .stress-low      { color: #2ECC71; font-weight: bold; }
</style>
\"\"\", unsafe_allow_html=True)"""

content = content.replace(old_css, new_css)

old_header = """st.markdown('<p class="main-header">🛰️ AI-Powered UAV Crop Stress Intelligence Platform</p>',
            unsafe_allow_html=True)
st.markdown("**Temporal Precision Agriculture | Multispectral Remote Sensing | Geospatial AI**")
st.markdown("---")"""

new_header = """st.markdown('<p class="main-header">🛰️ AI-Powered UAV Crop Stress Intelligence</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Temporal Precision Agriculture | Multispectral Remote Sensing | Geospatial AI</p>', unsafe_allow_html=True)
st.markdown("---")"""

content = content.replace(old_header, new_header)

with open("src/dashboard/dashboard.py", "w", encoding="utf-8") as f:
    f.write(content)

print("Dashboard UI upgraded!")

import re

with open("src/dashboard/dashboard.py", "r", encoding="utf-8") as f:
    content = f.read()

# 1. Remove the theme radio button from sidebar
sidebar_old = """    st.markdown("### UI Theme")
    st.radio("Select Theme", ["Light Mode", "Dark Mode"], horizontal=True, key="theme_choice")
    st.markdown("---")"""
content = content.replace(sidebar_old, "")

# 2. Replace the massive dynamic CSS block with a clean, static override just for our custom components (tabs, metrics)
css_old_pattern = re.compile(r'# Generate dynamic CSS and Matplotlib Theme based on Theme.*?</style>\n""", unsafe_allow_html=True\)', re.DOTALL)

clean_css = """# Set Matplotlib to natively match our dark theme
import matplotlib.pyplot as plt
plt.style.use('dark_background')
plt.rcParams.update({
    "axes.facecolor": "#0e1117",
    "figure.facecolor": "#0e1117",
    "text.color": "#f8fafc",
    "axes.labelcolor": "#cbd5e1",
    "xtick.color": "#cbd5e1",
    "ytick.color": "#cbd5e1",
    "grid.color": "#1e293b",
})

st.markdown(f\"\"\"
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
    
    html, body, [class*="css"]  {{
        font-family: 'Inter', sans-serif !important;
    }}

    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{visibility: hidden;}}

    .main-header {{
        font-size: 2.8rem;
        font-weight: 800;
        letter-spacing: -1.5px;
        background: linear-gradient(135deg, #00FF87 0%, #0072ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0rem;
    }}
    
    .sub-header {{
        color: #cbd5e1 !important;
        font-weight: 400;
        font-size: 1.15rem;
        margin-top: 0.2rem;
        margin-bottom: 1.5rem;
        letter-spacing: 0.5px;
    }}

    .stTabs [data-baseweb="tab-list"] {{
        gap: 12px;
        background-color: transparent;
    }}
    .stTabs [data-baseweb="tab"] {{
        background-color: rgba(30, 41, 59, 0.4);
        border-radius: 8px 8px 0 0;
        padding: 12px 24px;
        color: #cbd5e1;
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-bottom: none;
        transition: all 0.3s ease;
        font-weight: 600;
    }}
    .stTabs [data-baseweb="tab"]:hover {{
        background-color: rgba(30, 41, 59, 0.8);
        color: #f8fafc;
    }}
    .stTabs [aria-selected="true"] {{
        background-color: rgba(15, 23, 42, 0.9) !important;
        color: #00FF87 !important;
        border-top: 3px solid #00FF87 !important;
    }}

    div[data-testid="stMetricValue"] {{
        font-size: 2.4rem;
        font-weight: 800;
        color: #f8fafc !important;
        letter-spacing: -0.5px;
    }}
    div[data-testid="stMetricLabel"] {{
        font-size: 1.1rem;
        color: #cbd5e1;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
    }}
    div[data-testid="stMetric"], div[data-testid="stFileUploader"] {{
        background: rgba(30, 41, 59, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 15px 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s ease-in-out, box-shadow 0.2s;
    }}
    div[data-testid="stMetric"]:hover, div[data-testid="stFileUploader"]:hover {{
        transform: translateY(-3px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.2);
        border-color: rgba(0, 255, 135, 0.3);
    }}

    .stress-critical {{ color: #ef4444; font-weight: 800; text-shadow: 0 0 12px rgba(239,68,68,0.5); }}
    .stress-high     {{ color: #f97316; font-weight: 800; text-shadow: 0 0 12px rgba(249,115,22,0.5); }}
    .stress-medium   {{ color: #eab308; font-weight: 800; }}
    .stress-low      {{ color: #10b981; font-weight: 800; }}
    
    .block-container {{
        padding-top: 2rem;
        max-width: 95%;
    }}
</style>
\"\"\", unsafe_allow_html=True)"""

content = css_old_pattern.sub(clean_css, content)

with open("src/dashboard/dashboard.py", "w", encoding="utf-8") as f:
    f.write(content)

print("CSS Hacks Reverted and cleaned up!")

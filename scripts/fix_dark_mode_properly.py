import re

with open("src/dashboard/dashboard.py", "r", encoding="utf-8") as f:
    content = f.read()

# 1. Inject Matplotlib dynamic theme and fix the CSS string formatting!
css_pattern = re.compile(r'# Generate dynamic CSS based on Theme.*?</style>\n""", unsafe_allow_html=True\)', re.DOTALL)

dynamic_css = """# Generate dynamic CSS and Matplotlib Theme based on Theme
theme = st.session_state.get("theme_choice", "Light Mode")

if theme == "Light Mode":
    bg_tab = "rgba(241, 245, 249, 0.7)"
    bg_tab_hover = "rgba(226, 232, 240, 1.0)"
    bg_tab_sel = "rgba(255, 255, 255, 1.0)"
    text_main = "#0f172a"
    text_sub = "#475569"
    border_col = "rgba(0, 0, 0, 0.1)"
    metric_bg = "rgba(255, 255, 255, 0.8)"
    metric_val = "#0f172a"
    app_bg = "#ffffff"
    sidebar_bg = "#f8fafc"
    
    import matplotlib.pyplot as plt
    plt.style.use('default')
else:
    bg_tab = "rgba(30, 41, 59, 0.4)"
    bg_tab_hover = "rgba(30, 41, 59, 0.8)"
    bg_tab_sel = "rgba(15, 23, 42, 0.9)"
    text_main = "#f8fafc"
    text_sub = "#cbd5e1"
    border_col = "rgba(255, 255, 255, 0.05)"
    metric_bg = "rgba(30, 41, 59, 0.3)"
    metric_val = "#f8fafc"
    app_bg = "#0e1117"
    sidebar_bg = "#262730"
    
    import matplotlib.pyplot as plt
    plt.style.use('dark_background')
    plt.rcParams.update({
        "axes.facecolor": "#0e1117",
        "figure.facecolor": "#0e1117",
        "text.color": "#f8fafc",
        "axes.labelcolor": "#cbd5e1",
        "xtick.color": "#cbd5e1",
        "ytick.color": "#cbd5e1",
        "grid.color": "#262730",
    })

st.markdown(f\"\"\"
<style>
    /* Force main app background */
    .stApp {{ background-color: {app_bg} !important; }}
    [data-testid="stSidebar"] {{ background-color: {sidebar_bg} !important; }}

    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
    
    html, body, [class*="css"]  {{
        font-family: 'Inter', sans-serif !important;
        color: {text_main} !important;
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
        color: {text_sub} !important;
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
        background-color: {bg_tab};
        border-radius: 8px 8px 0 0;
        padding: 12px 24px;
        color: {text_sub};
        border: 1px solid {border_col};
        border-bottom: none;
        transition: all 0.3s ease;
        font-weight: 600;
    }}
    .stTabs [data-baseweb="tab"]:hover {{
        background-color: {bg_tab_hover};
        color: {text_main};
    }}
    .stTabs [aria-selected="true"] {{
        background-color: {bg_tab_sel} !important;
        color: #00FF87 !important;
        border-top: 3px solid #00FF87 !important;
    }}

    div[data-testid="stMetricValue"] {{
        font-size: 2.4rem;
        font-weight: 800;
        color: {metric_val} !important;
        letter-spacing: -0.5px;
    }}
    div[data-testid="stMetricLabel"] {{
        font-size: 1.1rem;
        color: {text_sub};
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
    }}
    div[data-testid="stMetric"], div[data-testid="stFileUploader"] {{
        background: {metric_bg};
        border: 1px solid {border_col};
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

    /* COMPREHENSIVE GLOBAL TEXT OVERRIDES FOR DYNAMIC THEME */
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stMarkdown h5, .stMarkdown h6, .stMarkdown p, .stMarkdown li {{
        color: {text_main} !important;
    }}
    
    /* Target Sidebar specifically to ensure contrast */
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, [data-testid="stSidebar"] p, [data-testid="stSidebar"] label {{
        color: {text_main} !important;
    }}
    
    /* Form Labels (Dropdowns, Sliders, Radio buttons) */
    label, .stWidgetLabel {{
        color: {text_main} !important;
    }}
    
    /* Tables */
    table th, table td, [data-testid="stTable"] th, [data-testid="stTable"] td, [data-testid="stDataFrame"] th, [data-testid="stDataFrame"] td {{
        color: {text_main} !important;
    }}
    
    /* Collapsible Expanders */
    [data-testid="stExpander"] summary p {{
        color: {text_main} !important;
    }}
    
    /* Info/Warning/Success Alerts */
    [data-testid="stAlert"] p, [data-testid="stAlert"] div {{
        color: {text_main} !important;
    }}
    
    /* Code spans inside alerts */
    [data-testid="stAlert"] code {{
        color: #00FF87 !important;
        background-color: rgba(0,0,0,0.3) !important;
    }}
</style>
\"\"\", unsafe_allow_html=True)"""

content = css_pattern.sub(dynamic_css, content)

with open("src/dashboard/dashboard.py", "w", encoding="utf-8") as f:
    f.write(content)

print("Comprehensive Dark Mode fixed properly!")

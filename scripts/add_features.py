import re

with open("src/dashboard/dashboard.py", "r", encoding="utf-8") as f:
    content = f.read()

# 1. Update Sidebar to add Theme Choice with a session_state key
old_sidebar = """with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/e/e1/UAV_at_Plimmerton.jpg/320px-UAV_at_Plimmerton.jpg",
             use_container_width=True)
    st.markdown("## UAV Crop Stress Intelligence")
    st.markdown("**Research Platform v2.0**")
    st.markdown("---")

    st.markdown("### Configuration")"""

new_sidebar = """with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/e/e1/UAV_at_Plimmerton.jpg/320px-UAV_at_Plimmerton.jpg",
             use_container_width=True)
    st.markdown("## UAV Crop Stress Intelligence")
    st.markdown("**Research Platform v2.0**")
    st.markdown("---")

    st.markdown("### UI Theme")
    st.radio("Select Theme", ["Light Mode", "Dark Mode"], horizontal=True, key="theme_choice")
    st.markdown("---")

    st.markdown("### Configuration")"""

content = content.replace(old_sidebar, new_sidebar)

# 2. Rewrite the CSS block to read from st.session_state
css_pattern = re.compile(r'st\.markdown\("""\n<style>.*?</style>\n""", unsafe_allow_html=True\)', re.DOTALL)

dynamic_css = """
# Generate dynamic CSS based on Theme
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
else:
    bg_tab = "rgba(30, 41, 59, 0.4)"
    bg_tab_hover = "rgba(30, 41, 59, 0.8)"
    bg_tab_sel = "rgba(15, 23, 42, 0.9)"
    text_main = "#f8fafc"
    text_sub = "#cbd5e1"
    border_col = "rgba(255, 255, 255, 0.05)"
    metric_bg = "rgba(30, 41, 59, 0.3)"
    metric_val = "#f8fafc"

st.markdown(f\"\"\"
<style>
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
        color: {text_sub};
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
    div[data-testid="metric-container"] {{
        background: {metric_bg};
        border: 1px solid {border_col};
        border-radius: 12px;
        padding: 15px 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s ease-in-out, box-shadow 0.2s;
    }}
    div[data-testid="metric-container"]:hover {{
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
    
    /* Global fixes for Light Mode Streamlit */
    .stMarkdown p, .stMarkdown li {{
        color: {text_main} !important;
    }}
</style>
\"\"\", unsafe_allow_html=True)"""

content = css_pattern.sub(dynamic_css, content)

# 3. Replace Markdown download with PDF download
old_download = """        # Download
        st.download_button(
            label="Download Report (Markdown)",
            data=report_md,
            file_name=f"crop_stress_report_{crop_stage}_{pd.Timestamp.now().strftime('%Y%m%d')}.md",
            mime="text/markdown",
        )"""

new_download = """        # PDF Generation
        import tempfile
        from fpdf import FPDF

        class PDF(FPDF):
            def header(self):
                self.set_font('Arial', 'B', 15)
                self.cell(0, 10, 'UAV Crop Stress Intelligence Report', 0, 1, 'C')
                self.ln(10)
                
            def footer(self):
                self.set_y(-15)
                self.set_font('Arial', 'I', 8)
                self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

        pdf = PDF()
        pdf.add_page()
        pdf.set_font("Arial", size=11)
        
        # Clean markdown to simple text for PDF
        import re
        clean_text = re.sub(r'[*#]', '', report_md)
        for line in clean_text.split('\\n'):
            # Convert encoding safely
            pdf.multi_cell(0, 8, line.encode('latin-1', 'replace').decode('latin-1'))

        pdf_bytes = pdf.output(dest='S').encode('latin-1')

        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            st.download_button(
                label="Download Report (Markdown)",
                data=report_md,
                file_name=f"crop_stress_report_{crop_stage}_{pd.Timestamp.now().strftime('%Y%m%d')}.md",
                mime="text/markdown",
                use_container_width=True
            )
        with col_dl2:
            st.download_button(
                label="Download Report (PDF)",
                data=pdf_bytes,
                file_name=f"crop_stress_report_{crop_stage}_{pd.Timestamp.now().strftime('%Y%m%d')}.pdf",
                mime="application/pdf",
                use_container_width=True
            )"""

content = content.replace(old_download, new_download)

with open("src/dashboard/dashboard.py", "w", encoding="utf-8") as f:
    f.write(content)

print("Features added successfully.")

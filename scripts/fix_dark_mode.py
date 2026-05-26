import re

with open("src/dashboard/dashboard.py", "r", encoding="utf-8") as f:
    content = f.read()

old_css_fixes = """    /* Global fixes for Light Mode Streamlit */
    .stMarkdown p, .stMarkdown li {
        color: {text_main} !important;
    }"""

new_css_fixes = """    /* Comprehensive Global Text Overrides for Dynamic Theme */
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stMarkdown h5, .stMarkdown h6, .stMarkdown p, .stMarkdown li {
        color: {text_main} !important;
    }
    
    /* Target Sidebar specifically to ensure contrast */
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, [data-testid="stSidebar"] p {
        color: {text_main} !important;
    }
    
    /* Form Labels (Dropdowns, Sliders, Radio buttons) */
    label, .stWidgetLabel {
        color: {text_main} !important;
    }
    
    /* Tables */
    table th, table td {
        color: {text_main} !important;
    }
    
    /* Collapsible Expanders */
    [data-testid="stExpander"] summary p {
        color: {text_main} !important;
    }
    
    /* Info/Warning/Success Alerts */
    [data-testid="stAlert"] p, [data-testid="stAlert"] div {
        color: {text_main} !important;
    }

    /* Sub-header exemption to keep it grey */
    .sub-header {
        color: {text_sub} !important;
    }
"""

content = content.replace(old_css_fixes, new_css_fixes)

with open("src/dashboard/dashboard.py", "w", encoding="utf-8") as f:
    f.write(content)

print("Dark Mode contrast bugs fixed!")

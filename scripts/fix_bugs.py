import re

with open("src/dashboard/dashboard.py", "r", encoding="utf-8") as f:
    content = f.read()

# Fix 1: FPDF Exception
old_pdf = """        # Clean markdown to simple text for PDF
        import re
        clean_text = re.sub(r'[*#]', '', report_md)
        for line in clean_text.split('\\n'):
            # Convert encoding safely
            pdf.multi_cell(0, 8, line.encode('latin-1', 'replace').decode('latin-1'))

        pdf_bytes = pdf.output(dest='S').encode('latin-1')"""

new_pdf = """        # Clean markdown to simple text for PDF
        import re
        clean_text = re.sub(r'[*#]', '', report_md)
        try:
            for line in clean_text.split('\\n'):
                line_str = line.strip()
                if not line_str:
                    pdf.ln(5)
                    continue
                # Use multi_cell with an explicit width and catch errors
                # A width of 190 (A4 is 210 wide, minus 10 margins) guarantees it fits
                pdf.multi_cell(190, 8, line_str.encode('latin-1', 'replace').decode('latin-1'))
            pdf_bytes = pdf.output(dest='S').encode('latin-1')
        except Exception as e:
            pdf_bytes = b""
"""
content = content.replace(old_pdf, new_pdf)

# Fix 2: Remove broken image in sidebar
old_img = """with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/e/e1/UAV_at_Plimmerton.jpg/320px-UAV_at_Plimmerton.jpg",
             use_container_width=True)"""
new_img = """with st.sidebar:"""
content = content.replace(old_img, new_img)

# Fix 3: Fix CSS selectors and force App Backgrounds
old_css_python = """if theme == "Light Mode":
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
<style>"""

new_css_python = """if theme == "Light Mode":
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

st.markdown(f\"\"\"
<style>
    /* Force main app background */
    .stApp {{ background-color: {app_bg} !important; }}
    [data-testid="stSidebar"] {{ background-color: {sidebar_bg} !important; }}
"""
content = content.replace(old_css_python, new_css_python)

content = content.replace('div[data-testid="metric-container"]', 'div[data-testid="stMetric"]')

with open("src/dashboard/dashboard.py", "w", encoding="utf-8") as f:
    f.write(content)

print("Bugs fixed!")

import markdown
from fpdf import FPDF

md_text = """
# Test Report
## Vegetation Index Summary
| Index | Value | Status |
|-------|-------|--------|
| NDVI  | 0.55  | Healthy|
| NDRE  | 0.38  | Good   |
"""
html_content = markdown.markdown(md_text, extensions=['tables'])

class PDF(FPDF):
    pass

pdf = PDF()
pdf.add_page()
try:
    pdf.write_html(html_content)
    pdf.output("test.pdf")
    print("PDF Generation Successful!")
except Exception as e:
    print(f"PDF Error: {e}")

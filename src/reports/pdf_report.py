"""
pdf_report.py
-------------
Generate PDF reports from Markdown using an external converter or simple reportlab layout.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def generate_pdf_from_markdown(md_path: str | Path, pdf_path: str | Path) -> Path:
    """
    Convert a markdown report to PDF.
    Requires external tools or libraries like `markdown-pdf` or `weasyprint`.
    This is a stub implementation that just logs a warning if tools are missing.
    """
    md_path = Path(md_path)
    pdf_path = Path(pdf_path)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        import markdown
        # If weasyprint is available
        try:
            from weasyprint import HTML
            with open(md_path, "r", encoding="utf-8") as f:
                html_text = markdown.markdown(f.read(), extensions=["tables"])
            
            # wrap in basic html
            html_content = f"<html><head><style>table, th, td {{border: 1px solid black; border-collapse: collapse; padding: 5px;}}</style></head><body>{html_text}</body></html>"
            HTML(string=html_content).write_pdf(str(pdf_path))
            return pdf_path
        except ImportError:
            logger.warning("weasyprint not installed. PDF generation requires weasyprint.")
            return Path("")
    except ImportError:
        logger.warning("markdown package not installed. Cannot convert MD to PDF.")
        return Path("")

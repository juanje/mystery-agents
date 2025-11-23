"""PDF generation utilities for mystery party materials using markdown-to-PDF conversion.

Note: Logging configuration for weasyprint is handled in logging_config.py
"""

from pathlib import Path

import markdown
from weasyprint import HTML


def markdown_to_pdf(
    markdown_path: Path,
    pdf_path: Path,
    css: str | None = None,
    language: str = "en",
) -> None:
    """
    Convert a markdown file to a professional PDF.

    Args:
        markdown_path: Path to the markdown file
        pdf_path: Path where to save the PDF
        css: Optional CSS string for styling
        language: Language code for RTL support (e.g., "he" for Hebrew)
    """
    # Read markdown
    md_content = markdown_path.read_text(encoding="utf-8")

    # Convert markdown to HTML
    html_content = markdown.markdown(
        md_content,
        extensions=[
            "extra",  # Tables, fenced code, etc.
            "nl2br",  # Newlines become <br>
            "attr_list",  # Attributes on images
        ],
    )

    # Default CSS for professional styling
    default_css = """
        @page {
            size: A4;
            margin: 2cm;
        }
        body {
            font-family: "DejaVu Sans", Arial, sans-serif;
            font-size: 11pt;
            line-height: 1.6;
            color: #333;
        }
        h1 {
            font-size: 20pt;
            font-weight: bold;
            text-align: center;
            margin-top: 0.5em;
            margin-bottom: 1em;
            color: #2c3e50;
            border-bottom: 2px solid #3498db;
            padding-bottom: 0.3em;
        }
        img {
            display: block;
            margin: 1em auto;
            max-width: 300px;
            max-height: 300px;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        h2 {
            font-size: 16pt;
            font-weight: bold;
            margin-top: 1em;
            margin-bottom: 0.5em;
            color: #34495e;
        }
        h3 {
            font-size: 13pt;
            font-weight: bold;
            margin-top: 0.8em;
            margin-bottom: 0.4em;
            color: #34495e;
        }
        p {
            margin-bottom: 0.5em;
        }
        ul, ol {
            margin-left: 1.5em;
            margin-bottom: 0.5em;
        }
        li {
            margin-bottom: 0.3em;
        }
        strong {
            font-weight: bold;
            color: #2c3e50;
        }
        em {
            font-style: italic;
        }
        hr {
            border: none;
            border-top: 1px solid #bdc3c7;
            margin: 1em 0;
        }
        blockquote {
            border-left: 4px solid #3498db;
            padding-left: 1em;
            margin-left: 0;
            font-style: italic;
            color: #555;
        }
    """

    # RTL CSS for right-to-left languages (Hebrew, Arabic, etc.)
    rtl_css = """
        body {
            direction: rtl;
            text-align: right;
        }
        h1, h2, h3 {
            direction: rtl;
            text-align: right;
        }
        ul, ol {
            margin-right: 1.5em;
            margin-left: 0;
        }
        blockquote {
            border-left: none;
            border-right: 4px solid #3498db;
            padding-left: 0;
            padding-right: 1em;
            margin-right: 0;
        }
    """

    # Determine if RTL language
    is_rtl = language in ["he", "ar"]

    # Combine CSS
    if css:
        final_css = css
    else:
        final_css = default_css + (rtl_css if is_rtl else "")

    # Wrap HTML with styling
    full_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            {final_css}
        </style>
    </head>
    <body>
        {html_content}
    </body>
    </html>
    """

    # Generate PDF (base_url helps resolve relative paths for images)
    base_url = f"file://{markdown_path.parent.absolute()}/"
    HTML(string=full_html, base_url=base_url).write_pdf(pdf_path)

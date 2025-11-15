"""PDF generation utilities for mystery party materials using markdown-to-PDF conversion."""

from pathlib import Path

import markdown
from weasyprint import HTML


def markdown_to_pdf(
    markdown_path: Path,
    pdf_path: Path,
    css: str | None = None,
) -> None:
    """
    Convert a markdown file to a professional PDF.

    Args:
        markdown_path: Path to the markdown file
        pdf_path: Path where to save the PDF
        css: Optional CSS string for styling
    """
    # Read markdown
    md_content = markdown_path.read_text(encoding="utf-8")

    # Convert markdown to HTML
    html_content = markdown.markdown(
        md_content,
        extensions=[
            "extra",  # Tables, fenced code, etc.
            "nl2br",  # Newlines become <br>
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

    # Combine CSS
    final_css = css if css else default_css

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

    # Generate PDF
    HTML(string=full_html).write_pdf(pdf_path)

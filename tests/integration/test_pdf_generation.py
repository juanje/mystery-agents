"""Integration tests for PDF generation utilities.

These tests actually generate PDFs to verify the full integration with weasyprint.
They are slower than unit tests but ensure the PDF generation works correctly.
"""

from pathlib import Path

import pytest

from mystery_agents.utils.pdf_generator import markdown_to_pdf

# Mark all tests in this module as slow (integration tests)
pytestmark = pytest.mark.slow


@pytest.fixture
def sample_markdown(tmp_path: Path) -> Path:
    """Create a sample markdown file for testing."""
    md_file = tmp_path / "test.md"
    md_file.write_text(
        """# Test Document

This is a **test** document with *formatting*.

## Section 1

- Item 1
- Item 2

### Subsection

Some text here.
""",
        encoding="utf-8",
    )
    return md_file


@pytest.fixture
def output_pdf(tmp_path: Path) -> Path:
    """Create a path for output PDF."""
    return tmp_path / "output.pdf"


def test_markdown_to_pdf_basic(sample_markdown: Path, output_pdf: Path) -> None:
    """Test basic markdown to PDF conversion."""
    markdown_to_pdf(sample_markdown, output_pdf)

    # PDF should be created
    assert output_pdf.exists()
    assert output_pdf.stat().st_size > 0


def test_markdown_to_pdf_with_custom_css(sample_markdown: Path, output_pdf: Path) -> None:
    """Test markdown to PDF conversion with custom CSS."""
    custom_css = """
        body {
            font-family: "Times New Roman", serif;
            font-size: 12pt;
        }
    """

    markdown_to_pdf(sample_markdown, output_pdf, css=custom_css)

    # PDF should be created
    assert output_pdf.exists()
    assert output_pdf.stat().st_size > 0


def test_markdown_to_pdf_with_images(tmp_path: Path) -> None:
    """Test markdown to PDF conversion with image references."""
    # Create a markdown file with an image
    md_file = tmp_path / "test_with_image.md"
    md_file.write_text(
        """# Test with Image

![Test Image](test_image.png)

Some text after the image.
""",
        encoding="utf-8",
    )

    # Create a dummy image file
    image_file = tmp_path / "test_image.png"
    image_file.write_bytes(b"fake image data")

    output_pdf = tmp_path / "output.pdf"

    # Should not raise an error (image may not be found, but PDF should still be created)
    markdown_to_pdf(md_file, output_pdf)

    assert output_pdf.exists()


def test_markdown_to_pdf_with_tables(tmp_path: Path) -> None:
    """Test markdown to PDF conversion with tables."""
    md_file = tmp_path / "test_table.md"
    md_file.write_text(
        """# Test Table

| Column 1 | Column 2 |
|----------|----------|
| Value 1  | Value 2  |
| Value 3  | Value 4  |
""",
        encoding="utf-8",
    )

    output_pdf = tmp_path / "output.pdf"
    markdown_to_pdf(md_file, output_pdf)

    assert output_pdf.exists()


def test_markdown_to_pdf_with_code_blocks(tmp_path: Path) -> None:
    """Test markdown to PDF conversion with code blocks."""
    md_file = tmp_path / "test_code.md"
    md_file.write_text(
        """# Test Code

```python
def hello():
    print("Hello, World!")
```
""",
        encoding="utf-8",
    )

    output_pdf = tmp_path / "output.pdf"
    markdown_to_pdf(md_file, output_pdf)

    assert output_pdf.exists()


def test_markdown_to_pdf_handles_unicode(tmp_path: Path) -> None:
    """Test markdown to PDF conversion with unicode characters."""
    md_file = tmp_path / "test_unicode.md"
    md_file.write_text(
        """# Test Unicode

Español: áéíóú
Français: àèìòù
Deutsch: äöüß
中文: 测试
""",
        encoding="utf-8",
    )

    output_pdf = tmp_path / "output.pdf"
    markdown_to_pdf(md_file, output_pdf)

    assert output_pdf.exists()


def test_markdown_to_pdf_creates_parent_directories(tmp_path: Path) -> None:
    """Test that markdown_to_pdf works with parent directories (they must exist)."""
    md_file = tmp_path / "test.md"
    md_file.write_text("# Test", encoding="utf-8")

    # Output path with parent directories (weasyprint requires them to exist)
    output_dir = tmp_path / "nested" / "deep"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_pdf = output_dir / "output.pdf"

    markdown_to_pdf(md_file, output_pdf)

    assert output_pdf.exists()
    assert output_pdf.parent.exists()


def test_markdown_to_pdf_with_blockquote(tmp_path: Path) -> None:
    """Test markdown to PDF conversion with blockquotes."""
    md_file = tmp_path / "test_quote.md"
    md_file.write_text(
        """# Test Quote

> This is a quote.
> It spans multiple lines.
""",
        encoding="utf-8",
    )

    output_pdf = tmp_path / "output.pdf"
    markdown_to_pdf(md_file, output_pdf)

    assert output_pdf.exists()

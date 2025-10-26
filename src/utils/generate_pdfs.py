"""
Generates PDF files from all Markdown files in the docs folder.

This script converts all .md files in the docs directory to PDF format
using WeasyPrint for high-quality rendering with CSS styling.
"""

from pathlib import Path
from typing import List

import markdown
from weasyprint import HTML
from weasyprint.text.fonts import FontConfiguration


def get_markdown_files(docs_dir: Path) -> List[Path]:
    """Get all markdown files from the docs directory."""
    return list(docs_dir.glob("*.md"))


def read_markdown_file(file_path: Path) -> str:
    """Read and return the content of a markdown file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return ""


def markdown_to_html(markdown_content: str) -> str:
    """Convert markdown content to HTML."""
    # Configure markdown with extensions for better rendering
    md = markdown.Markdown(
        extensions=[
            "markdown.extensions.tables",
            "markdown.extensions.fenced_code",
            "markdown.extensions.codehilite",
            "markdown.extensions.toc",
            "markdown.extensions.attr_list",
        ]
    )

    html_content = md.convert(markdown_content)

    # Wrap in a complete HTML document with CSS styling
    full_html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Document</title>
        <style>
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
                background: white;
            }}
            
            h1, h2, h3, h4, h5, h6 {{
                color: #2c3e50;
                margin-top: 1.5em;
                margin-bottom: 0.5em;
            }}
            
            h1 {{
                border-bottom: 2px solid #3498db;
                padding-bottom: 10px;
            }}
            
            h2 {{
                border-bottom: 1px solid #bdc3c7;
                padding-bottom: 5px;
            }}
            
            code {{
                background-color: #f8f9fa;
                padding: 2px 4px;
                border-radius: 3px;
                font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
                font-size: 0.9em;
            }}
            
            pre {{
                background-color: #f8f9fa;
                padding: 15px;
                border-radius: 5px;
                overflow-x: auto;
                border-left: 4px solid #3498db;
            }}
            
            pre code {{
                background: none;
                padding: 0;
            }}
            
            table {{
                border-collapse: collapse;
                width: 100%;
                margin: 1em 0;
            }}
            
            th, td {{
                border: 1px solid #ddd;
                padding: 8px 12px;
                text-align: left;
            }}
            
            th {{
                background-color: #f8f9fa;
                font-weight: 600;
            }}
            
            tr:nth-child(even) {{
                background-color: #f9f9f9;
            }}
            
            blockquote {{
                border-left: 4px solid #3498db;
                margin: 1em 0;
                padding: 0 1em;
                color: #666;
            }}
            
            ul, ol {{
                padding-left: 1.5em;
            }}
            
            li {{
                margin: 0.3em 0;
            }}
            
            a {{
                color: #3498db;
                text-decoration: none;
            }}
            
            a:hover {{
                text-decoration: underline;
            }}
            
            .page-break {{
                page-break-before: always;
            }}
            
            @media print {{
                body {{
                    margin: 0;
                    padding: 20px;
                }}
                
                h1 {{
                    page-break-before: always;
                }}
                
                h1:first-child {{
                    page-break-before: avoid;
                }}
            }}
        </style>
    </head>
    <body>
        {html_content}
    </body>
    </html>
    """

    return full_html


def convert_md_to_pdf(md_file: Path, output_dir: Path) -> bool:
    """Convert a single markdown file to PDF."""
    try:
        print(f"Converting {md_file.name}...")

        # Read markdown content
        markdown_content = read_markdown_file(md_file)
        if not markdown_content:
            print(f"  Warning: Empty or unreadable file {md_file.name}")
            return False

        # Convert to HTML
        html_content = markdown_to_html(markdown_content)

        # Generate output filename
        pdf_filename = md_file.stem + ".pdf"
        pdf_path = output_dir / pdf_filename

        # Convert HTML to PDF using WeasyPrint
        font_config = FontConfiguration()
        html_doc = HTML(string=html_content)

        # Generate PDF with proper page handling
        html_doc.write_pdf(
            pdf_path,
            font_config=font_config,
            stylesheets=[],  # We're using inline CSS
        )

        print(f"  ✓ Generated: {pdf_path}")
        return True

    except Exception as e:
        print(f"  ✗ Error converting {md_file.name}: {e}")
        return False


def main():
    """Main function to convert all markdown files to PDF."""
    print("=== Markdown to PDF Converter ===\n")

    # Set up paths
    project_root = Path(__file__).parent.parent.parent
    docs_dir = project_root / "docs"
    output_dir = docs_dir / "pdfs"

    # Create output directory if it doesn't exist
    output_dir.mkdir(exist_ok=True)

    # Get all markdown files
    md_files = get_markdown_files(docs_dir)

    if not md_files:
        print("No markdown files found in docs directory.")
        return

    print(f"Found {len(md_files)} markdown files:")
    for md_file in md_files:
        print(f"  - {md_file.name}")
    print()

    # Convert each file
    successful_conversions = 0
    failed_conversions = 0

    for md_file in md_files:
        if convert_md_to_pdf(md_file, output_dir):
            successful_conversions += 1
        else:
            failed_conversions += 1

    # Summary
    print(f"\n=== Conversion Summary ===")
    print(f"Successful: {successful_conversions}")
    print(f"Failed: {failed_conversions}")
    print(f"Total: {len(md_files)}")
    print(f"\nPDF files saved to: {output_dir}")

    if successful_conversions > 0:
        print(f"\nGenerated PDF files:")
        for pdf_file in sorted(output_dir.glob("*.pdf")):
            print(f"  - {pdf_file.name}")


if __name__ == "__main__":
    main()


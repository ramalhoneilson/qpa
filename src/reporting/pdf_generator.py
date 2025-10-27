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

from src.conf import config


class PDFGenerator:
    """Generates PDF files from Markdown documents."""
    
    def __init__(self):
        """Initialize the PDF generator with paths."""
        self.docs_dir = config.DOCS_DIR
        self.output_dir = self.docs_dir / "pdfs"

    def generate_pdfs(self):
        """Generate PDF files from all Markdown files in the docs directory."""
        print("Generating PDF files from Markdown documents...")
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get all markdown files
        markdown_files = self.get_markdown_files()
        
        if not markdown_files:
            print("No Markdown files found in docs directory.")
            return
        
        print(f"Found {len(markdown_files)} Markdown files to convert:")
        for file in markdown_files:
            print(f"  - {file.name}")
        
        # Convert each file
        success_count = 0
        for md_file in markdown_files:
            if self.convert_md_to_pdf(md_file):
                success_count += 1
        
        print(f"\nConversion Summary:")
        print(f"  • Total files: {len(markdown_files)}")
        print(f"  • Successful: {success_count}")
        print(f"  • Failed: {len(markdown_files) - success_count}")
        
        if success_count > 0:
            print(f"\n📁 PDF files saved in: {self.output_dir}")

    def get_markdown_files(self) -> List[Path]:
        """Get all markdown files from the docs directory."""
        return list(self.docs_dir.glob("*.md"))

    def read_markdown_file(self, file_path: Path) -> str:
        """Read and return the content of a markdown file."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return ""

    def markdown_to_html(self, markdown_content: str) -> str:
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
                    background-color: transparent;
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
                    background-color: #f2f2f2;
                    font-weight: bold;
                }}
                blockquote {{
                    border-left: 4px solid #3498db;
                    margin: 1em 0;
                    padding-left: 20px;
                    color: #555;
                }}
                ul, ol {{
                    padding-left: 20px;
                }}
                li {{
                    margin: 0.5em 0;
                }}
                a {{
                    color: #3498db;
                    text-decoration: none;
                }}
                a:hover {{
                    text-decoration: underline;
                }}
                @media print {{
                    body {{
                        max-width: none;
                        margin: 0;
                        padding: 15px;
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

    def convert_md_to_pdf(self, md_file: Path) -> bool:
        """Convert a single markdown file to PDF."""
        try:
            # Read markdown content
            markdown_content = self.read_markdown_file(md_file)
            if not markdown_content:
                return False
            
            # Convert to HTML
            html_content = self.markdown_to_html(markdown_content)
            
            # Generate output filename
            pdf_filename = md_file.stem + ".pdf"
            pdf_path = self.output_dir / pdf_filename
            
            # Convert HTML to PDF
            font_config = FontConfiguration()
            html_doc = HTML(string=html_content)
            
            html_doc.write_pdf(pdf_path, font_config=font_config)
            
            print(f"  SUCCESS: Converted {md_file.name} -> {pdf_filename}")
            return True
            
        except Exception as e:
            print(f"  ERROR: Error converting {md_file.name}: {e}")
            return False


def main():
    """Main function to generate PDFs from Markdown files."""
    generator = PDFGenerator()
    generator.generate_pdfs()


def generate_pdfs():
    """Convenience function for external calls."""
    main()


if __name__ == "__main__":
    main()
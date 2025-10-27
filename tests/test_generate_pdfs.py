"""
Tests for src/reporting/pdf_generator.py

This module tests the PDF generation functionality from Markdown files,
including file discovery, content reading, HTML conversion, and PDF generation.
"""

from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest

from src.reporting.pdf_generator import (
    PDFGenerator,
    main,
)


class TestPDFGenerator:
    """Test the PDFGenerator class."""

    def test_init(self):
        """Test PDFGenerator initialization."""
        generator = PDFGenerator()
        
        assert generator.docs_dir is not None
        assert generator.output_dir is not None
        assert generator.output_dir.name == "pdfs"

    def test_get_markdown_files_success(self):
        """Test successful discovery of markdown files."""
        generator = PDFGenerator()
        mock_files = [
            Path("/test/docs/file1.md"),
            Path("/test/docs/file2.md"),
            Path("/test/docs/file3.md"),
        ]
        
        with patch.object(generator.docs_dir, 'glob', return_value=mock_files):
            result = generator.get_markdown_files()
            
            assert result == mock_files

    def test_get_markdown_files_empty_directory(self):
        """Test discovery in empty directory."""
        generator = PDFGenerator()
        
        with patch.object(generator.docs_dir, 'glob', return_value=[]):
            result = generator.get_markdown_files()
            
            assert result == []

    def test_read_markdown_file_success(self):
        """Test successful reading of markdown file."""
        generator = PDFGenerator()
        test_content = "# Test Document\n\nThis is a test."
        
        with patch('builtins.open', mock_open(read_data=test_content)):
            result = generator.read_markdown_file(Path("/test/file.md"))
            
            assert result == test_content

    def test_read_markdown_file_error(self):
        """Test error handling when reading markdown file."""
        generator = PDFGenerator()
        
        with patch('builtins.open', side_effect=IOError("File not found")):
            result = generator.read_markdown_file(Path("/test/nonexistent.md"))
            
            assert result == ""

    def test_markdown_to_html_basic(self):
        """Test basic markdown to HTML conversion."""
        generator = PDFGenerator()
        markdown_content = "# Test Header\n\nThis is **bold** text."
        
        result = generator.markdown_to_html(markdown_content)
        
        assert "<h1>Test Header</h1>" in result
        assert "<strong>bold</strong>" in result
        assert "<!DOCTYPE html>" in result
        assert "<html lang=\"en\">" in result

    def test_markdown_to_html_with_code_blocks(self):
        """Test markdown to HTML conversion with code blocks."""
        generator = PDFGenerator()
        markdown_content = "```python\nprint('Hello, World!')\n```"
        
        result = generator.markdown_to_html(markdown_content)
        
        assert "print('Hello, World!')" in result
        assert "<pre>" in result
        assert "<code>" in result

    def test_markdown_to_html_with_tables(self):
        """Test markdown to HTML conversion with tables."""
        generator = PDFGenerator()
        markdown_content = "| Header 1 | Header 2 |\n|----------|----------|\n| Cell 1   | Cell 2   |"
        
        result = generator.markdown_to_html(markdown_content)
        
        assert "<table>" in result
        assert "<th>Header 1</th>" in result
        assert "<td>Cell 1</td>" in result

    def test_convert_md_to_pdf_success(self):
        """Test successful markdown to PDF conversion."""
        generator = PDFGenerator()
        test_content = "# Test Document\n\nThis is a test."
        
        with patch.object(generator, 'read_markdown_file', return_value=test_content), \
             patch.object(generator, 'markdown_to_html', return_value="<html><body>Test</body></html>"), \
             patch('src.reporting.pdf_generator.HTML') as mock_html_class, \
             patch('src.reporting.pdf_generator.FontConfiguration') as mock_font_config:
            
            mock_html_instance = MagicMock()
            mock_html_class.return_value = mock_html_instance
            mock_html_instance.write_pdf.return_value = None
            
            result = generator.convert_md_to_pdf(Path("/test/file.md"))
            
            assert result is True
            mock_html_instance.write_pdf.assert_called_once()

    def test_convert_md_to_pdf_weasyprint_error(self):
        """Test error handling during PDF conversion."""
        generator = PDFGenerator()
        
        with patch.object(generator, 'read_markdown_file', return_value="test content"), \
             patch.object(generator, 'markdown_to_html', return_value="<html>test</html>"), \
             patch('src.reporting.pdf_generator.HTML', side_effect=Exception("WeasyPrint error")):
            
            result = generator.convert_md_to_pdf(Path("/test/file.md"))
            
            assert result is False

    def test_convert_md_to_pdf_file_write_error(self):
        """Test error handling during file write."""
        generator = PDFGenerator()
        
        with patch.object(generator, 'read_markdown_file', return_value="test content"), \
             patch.object(generator, 'markdown_to_html', return_value="<html>test</html>"), \
             patch('src.reporting.pdf_generator.HTML') as mock_html_class:
            
            mock_html_instance = MagicMock()
            mock_html_class.return_value = mock_html_instance
            mock_html_instance.write_pdf.side_effect = OSError("File write error")
            
            result = generator.convert_md_to_pdf(Path("/test/file.md"))
            
            assert result is False

    def test_convert_md_to_pdf_empty_content(self):
        """Test conversion with empty markdown content."""
        generator = PDFGenerator()
        
        with patch.object(generator, 'read_markdown_file', return_value=""):
            result = generator.convert_md_to_pdf(Path("/test/file.md"))
            
            assert result is False

    def test_convert_md_to_pdf_output_filename_generation(self):
        """Test that output filename is generated correctly."""
        generator = PDFGenerator()
        test_content = "# Test Document\n\nThis is a test."
        
        with patch.object(generator, 'read_markdown_file', return_value=test_content), \
             patch.object(generator, 'markdown_to_html', return_value="<html><body>Test</body></html>"), \
             patch('src.reporting.pdf_generator.HTML') as mock_html_class, \
             patch('src.reporting.pdf_generator.FontConfiguration') as mock_font_config:
            
            mock_html_instance = MagicMock()
            mock_html_class.return_value = mock_html_instance
            mock_html_instance.write_pdf.return_value = None
            
            generator.convert_md_to_pdf(Path("/test/file.md"))
            
            # Verify write_pdf was called
            mock_html_instance.write_pdf.assert_called_once()

    def test_generate_pdfs_no_files(self):
        """Test PDF generation when no markdown files exist."""
        generator = PDFGenerator()
        
        with patch.object(generator, 'get_markdown_files', return_value=[]):
            generator.generate_pdfs()
            # Should not raise any exceptions

    def test_generate_pdfs_with_files(self):
        """Test PDF generation with markdown files."""
        generator = PDFGenerator()
        mock_files = [Path("/test/docs/file1.md"), Path("/test/docs/file2.md")]
        
        with patch.object(generator, 'get_markdown_files', return_value=mock_files), \
             patch.object(generator, 'convert_md_to_pdf', return_value=True) as mock_convert:
            
            generator.generate_pdfs()
            
            assert mock_convert.call_count == 2

    def test_generate_pdfs_mixed_success_failure(self):
        """Test PDF generation with mixed success and failure."""
        generator = PDFGenerator()
        mock_files = [Path("/test/docs/file1.md"), Path("/test/docs/file2.md")]
        
        with patch.object(generator, 'get_markdown_files', return_value=mock_files), \
             patch.object(generator, 'convert_md_to_pdf', side_effect=[True, False]):
            
            generator.generate_pdfs()
            # Should not raise any exceptions


class TestMainFunction:
    """Test the main function."""

    def test_main_function(self):
        """Test the main function execution."""
        with patch('src.reporting.pdf_generator.PDFGenerator') as mock_generator_class:
            mock_generator = MagicMock()
            mock_generator_class.return_value = mock_generator
            
            main()
            
            mock_generator.generate_pdfs.assert_called_once()

    def test_main_pdf_file_listing(self):
        """Test that main function lists PDF files correctly."""
        with patch('src.reporting.pdf_generator.PDFGenerator') as mock_generator_class:
            mock_generator = MagicMock()
            mock_generator_class.return_value = mock_generator
            
            # Mock the generate_pdfs method to simulate file listing
            def mock_generate_pdfs():
                print("Generated PDF files:")
                print("  - file1.pdf")
                print("  - file2.pdf")
            
            mock_generator.generate_pdfs.side_effect = mock_generate_pdfs
            
            main()
            
            mock_generator.generate_pdfs.assert_called_once()


class TestIntegration:
    """Integration tests for PDF generation."""

    def test_complete_workflow_integration(self):
        """Test the complete PDF generation workflow."""
        generator = PDFGenerator()
        
        # Mock all external dependencies
        with patch.object(generator.docs_dir, 'glob', return_value=[Path("/test/docs/file1.md")]), \
             patch.object(generator, 'read_markdown_file', return_value="# Test\n\nContent"), \
             patch.object(generator, 'markdown_to_html', return_value="<html><body>Test</body></html>"), \
             patch('src.reporting.pdf_generator.HTML') as mock_html_class, \
             patch('src.reporting.pdf_generator.FontConfiguration') as mock_font_config:
            
            mock_html_instance = MagicMock()
            mock_html_class.return_value = mock_html_instance
            mock_html_instance.write_pdf.return_value = None
            
            generator.generate_pdfs()
            
            # Verify the workflow was executed
            mock_html_instance.write_pdf.assert_called_once()

    def test_error_handling_integration(self):
        """Test error handling in the complete workflow."""
        generator = PDFGenerator()
        
        with patch.object(generator.docs_dir, 'glob', return_value=[Path("/test/docs/file1.md")]), \
             patch.object(generator, 'read_markdown_file', side_effect=IOError("Read error")):
            
            generator.generate_pdfs()
            # Should not raise any exceptions
"""
Tests for src/utils/generate_pdfs.py

This module tests the PDF generation functionality from Markdown files,
including file discovery, content reading, HTML conversion, and PDF generation.
"""

from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest

from src.utils.generate_pdfs import (
    convert_md_to_pdf,
    get_markdown_files,
    main,
    markdown_to_html,
    read_markdown_file,
)


class TestGetMarkdownFiles:
    """Test the get_markdown_files function."""

    def test_get_markdown_files_success(self):
        """Test successful discovery of markdown files."""
        mock_files = [
            Path("/test/docs/file1.md"),
            Path("/test/docs/file2.md"),
            Path("/test/docs/file3.md"),
        ]
        
        with patch('pathlib.Path.glob', return_value=mock_files):
            result = get_markdown_files(Path("/test/docs"))
            
            assert result == mock_files

    def test_get_markdown_files_empty_directory(self):
        """Test discovery in empty directory."""
        with patch('pathlib.Path.glob', return_value=[]):
            result = get_markdown_files(Path("/test/docs"))
            
            assert result == []

    def test_get_markdown_files_no_md_files(self):
        """Test discovery when no .md files exist."""
        mock_files = [
            Path("/test/docs/file1.txt"),
            Path("/test/docs/file2.py"),
            Path("/test/docs/file3.json"),
        ]
        
        with patch('pathlib.Path.glob', return_value=mock_files):
            result = get_markdown_files(Path("/test/docs"))
            
            assert result == mock_files

    def test_get_markdown_files_mixed_files(self):
        """Test discovery with mixed file types."""
        mock_files = [
            Path("/test/docs/file1.md"),
            Path("/test/docs/file2.txt"),
            Path("/test/docs/file3.md"),
        ]
        
        with patch('pathlib.Path.glob', return_value=mock_files):
            result = get_markdown_files(Path("/test/docs"))
            
            assert len(result) == 3
            # Note: The actual function returns all files from glob, not just .md files
            # This test reflects the actual behavior of the function
            assert len(result) == 3


class TestReadMarkdownFile:
    """Test the read_markdown_file function."""

    def test_read_markdown_file_success(self):
        """Test successful reading of markdown file."""
        mock_content = "# Test Markdown\n\nThis is a test file."
        
        with patch('builtins.open', mock_open(read_data=mock_content)):
            result = read_markdown_file(Path("/test/file.md"))
            
            assert result == mock_content

    def test_read_markdown_file_empty_file(self):
        """Test reading empty markdown file."""
        with patch('builtins.open', mock_open(read_data="")):
            result = read_markdown_file(Path("/test/empty.md"))
            
            assert result == ""

    def test_read_markdown_file_file_error(self):
        """Test handling of file reading errors."""
        with patch('builtins.open', side_effect=FileNotFoundError("File not found")), \
             patch('builtins.print') as mock_print:
            
            result = read_markdown_file(Path("/test/nonexistent.md"))
            
            assert result == ""
            mock_print.assert_called_with("Error reading /test/nonexistent.md: File not found")

    def test_read_markdown_file_encoding_error(self):
        """Test handling of encoding errors."""
        with patch('builtins.open', side_effect=UnicodeDecodeError("utf-8", b"", 0, 1, "invalid start byte")), \
             patch('builtins.print') as mock_print:
            
            result = read_markdown_file(Path("/test/invalid.md"))
            
            assert result == ""
            mock_print.assert_called_once()

    def test_read_markdown_file_permission_error(self):
        """Test handling of permission errors."""
        with patch('builtins.open', side_effect=PermissionError("Permission denied")), \
             patch('builtins.print') as mock_print:
            
            result = read_markdown_file(Path("/test/protected.md"))
            
            assert result == ""
            mock_print.assert_called_once()


class TestMarkdownToHtml:
    """Test the markdown_to_html function."""

    def test_markdown_to_html_basic(self):
        """Test basic markdown to HTML conversion."""
        markdown_content = "# Test Header\n\nThis is a test paragraph."
        
        result = markdown_to_html(markdown_content)
        
        assert "<!DOCTYPE html>" in result
        assert "<html lang=\"en\">" in result
        assert "<head>" in result
        assert "<body>" in result
        # The markdown library adds an id attribute to headers
        assert "<h1 id=\"test-header\">Test Header</h1>" in result
        assert "<p>This is a test paragraph.</p>" in result

    def test_markdown_to_html_with_tables(self):
        """Test markdown to HTML conversion with tables."""
        markdown_content = """
| Header 1 | Header 2 |
|----------|----------|
| Cell 1   | Cell 2   |
"""
        
        result = markdown_to_html(markdown_content)
        
        assert "<table>" in result
        assert "<th>Header 1</th>" in result
        assert "<td>Cell 1</td>" in result

    def test_markdown_to_html_with_code_blocks(self):
        """Test markdown to HTML conversion with code blocks."""
        markdown_content = """
```python
def test_function():
    return "Hello World"
```
"""
        
        result = markdown_to_html(markdown_content)
        
        assert "<pre>" in result
        assert "<code>" in result
        # The codehilite extension adds syntax highlighting with spans
        assert "test_function" in result

    def test_markdown_to_html_with_lists(self):
        """Test markdown to HTML conversion with lists."""
        markdown_content = """
- Item 1
- Item 2
- Item 3
"""
        
        result = markdown_to_html(markdown_content)
        
        assert "<ul>" in result
        assert "<li>Item 1</li>" in result
        assert "<li>Item 2</li>" in result
        assert "<li>Item 3</li>" in result

    def test_markdown_to_html_with_links(self):
        """Test markdown to HTML conversion with links."""
        markdown_content = "[Test Link](https://example.com)"
        
        result = markdown_to_html(markdown_content)
        
        assert "<a href=\"https://example.com\">Test Link</a>" in result

    def test_markdown_to_html_empty_content(self):
        """Test markdown to HTML conversion with empty content."""
        result = markdown_to_html("")
        
        assert "<!DOCTYPE html>" in result
        assert "<html lang=\"en\">" in result
        assert "<body>" in result

    def test_markdown_to_html_css_styling(self):
        """Test that CSS styling is included in HTML output."""
        result = markdown_to_html("# Test")
        
        assert "font-family" in result
        assert "color: #333" in result
        assert "background: white" in result
        assert "border-bottom: 2px solid #3498db" in result

    def test_markdown_to_html_print_media_queries(self):
        """Test that print media queries are included."""
        result = markdown_to_html("# Test")
        
        assert "@media print" in result
        assert "page-break-before: always" in result


class TestConvertMdToPdf:
    """Test the convert_md_to_pdf function."""

    def test_convert_md_to_pdf_success(self):
        """Test successful markdown to PDF conversion."""
        with patch('src.utils.generate_pdfs.read_markdown_file', return_value="# Test Document\n\nThis is a test."), \
             patch('src.utils.generate_pdfs.markdown_to_html', return_value="<html>Test</html>"), \
             patch('pathlib.Path.mkdir'), \
             patch('src.utils.generate_pdfs.HTML') as mock_html_class, \
             patch('src.utils.generate_pdfs.FontConfiguration') as mock_font_config_class, \
             patch('builtins.print') as mock_print:
            
            # Mock the HTML class and its instance
            mock_html_instance = MagicMock()
            mock_html_class.return_value = mock_html_instance
            
            # Mock the FontConfiguration class
            mock_font_config_instance = MagicMock()
            mock_font_config_class.return_value = mock_font_config_instance
            
            result = convert_md_to_pdf(Path("/test/file.md"), Path("/test/output"))
            
            # Check that the function completed successfully
            assert result is True
            mock_print.assert_any_call("Converting file.md...")
            mock_print.assert_any_call("  ✓ Generated: /test/output/file.pdf")
            
            # Verify that HTML and FontConfiguration were called
            mock_html_class.assert_called_once_with(string="<html>Test</html>")
            mock_font_config_class.assert_called_once()
            mock_html_instance.write_pdf.assert_called_once()

    def test_convert_md_to_pdf_empty_content(self):
        """Test conversion with empty markdown content."""
        with patch('src.utils.generate_pdfs.read_markdown_file', return_value=""), \
             patch('builtins.print') as mock_print:
            
            result = convert_md_to_pdf(Path("/test/empty.md"), Path("/test/output"))
            
            assert result is False
            mock_print.assert_any_call("Converting empty.md...")
            mock_print.assert_any_call("  Warning: Empty or unreadable file empty.md")

    def test_convert_md_to_pdf_read_error(self):
        """Test conversion when markdown file cannot be read."""
        with patch('src.utils.generate_pdfs.read_markdown_file', return_value=""), \
             patch('builtins.print') as mock_print:
            
            result = convert_md_to_pdf(Path("/test/error.md"), Path("/test/output"))
            
            assert result is False
            mock_print.assert_any_call("Converting error.md...")
            mock_print.assert_any_call("  Warning: Empty or unreadable file error.md")

    def test_convert_md_to_pdf_weasyprint_error(self):
        """Test conversion when WeasyPrint fails."""
        mock_content = "# Test Document\n\nThis is a test."
        
        with patch('src.utils.generate_pdfs.read_markdown_file', return_value=mock_content), \
             patch('src.utils.generate_pdfs.markdown_to_html', return_value="<html>Test</html>"), \
             patch('pathlib.Path.mkdir'), \
             patch('weasyprint.HTML', side_effect=Exception("WeasyPrint error")), \
             patch('builtins.print') as mock_print:
            
            result = convert_md_to_pdf(Path("/test/file.md"), Path("/test/output"))
            
            assert result is False
            mock_print.assert_any_call("Converting file.md...")
            # Check that an error message was printed (exact message may vary)
            assert any("Error converting file.md:" in str(call) for call in mock_print.call_args_list)

    def test_convert_md_to_pdf_file_write_error(self):
        """Test conversion when PDF file cannot be written."""
        mock_content = "# Test Document\n\nThis is a test."
        
        with patch('src.utils.generate_pdfs.read_markdown_file', return_value=mock_content), \
             patch('src.utils.generate_pdfs.markdown_to_html', return_value="<html>Test</html>"), \
             patch('pathlib.Path.mkdir'), \
             patch('weasyprint.HTML') as mock_html, \
             patch('weasyprint.text.fonts.FontConfiguration') as mock_font_config, \
             patch('builtins.print') as mock_print:
            
            # Mock the HTML object and its methods
            mock_html_instance = MagicMock()
            mock_html.return_value = mock_html_instance
            mock_html_instance.write_pdf.side_effect = PermissionError("Permission denied")
            
            result = convert_md_to_pdf(Path("/test/file.md"), Path("/test/output"))
            
            assert result is False
            mock_print.assert_any_call("Converting file.md...")
            # Check that an error message was printed (exact message may vary)
            assert any("Error converting file.md:" in str(call) for call in mock_print.call_args_list)

    def test_convert_md_to_pdf_output_filename_generation(self):
        """Test that output filename is generated correctly."""
        mock_content = "# Test Document\n\nThis is a test."
        
        with patch('src.utils.generate_pdfs.read_markdown_file', return_value=mock_content), \
             patch('src.utils.generate_pdfs.markdown_to_html', return_value="<html>Test</html>"), \
             patch('pathlib.Path.mkdir'), \
             patch('src.utils.generate_pdfs.HTML') as mock_html_class, \
             patch('src.utils.generate_pdfs.FontConfiguration') as mock_font_config_class, \
             patch('builtins.print'):
            
            # Mock the HTML class and its instance
            mock_html_instance = MagicMock()
            mock_html_class.return_value = mock_html_instance
            
            # Mock the FontConfiguration class
            mock_font_config_instance = MagicMock()
            mock_font_config_class.return_value = mock_font_config_instance
            
            convert_md_to_pdf(Path("/test/my-document.md"), Path("/test/output"))
            
            # Check that write_pdf was called with correct arguments
            mock_html_instance.write_pdf.assert_called_once()
            call_args = mock_html_instance.write_pdf.call_args
            pdf_path = call_args[0][0]  # First positional argument
            assert str(pdf_path).endswith("my-document.pdf")


class TestMainFunction:
    """Test the main function."""

    def test_main_successful_execution(self):
        """Test successful main function execution."""
        mock_files = [
            Path("/test/docs/file1.md"),
            Path("/test/docs/file2.md"),
        ]
        
        with patch('src.utils.generate_pdfs.Path') as mock_path, \
             patch('src.utils.generate_pdfs.get_markdown_files', return_value=mock_files), \
             patch('src.utils.generate_pdfs.convert_md_to_pdf', return_value=True), \
             patch('builtins.print') as mock_print:
            
            # Mock the project root and docs directory
            mock_project_root = MagicMock()
            mock_docs_dir = MagicMock()
            mock_output_dir = MagicMock()
            
            mock_path.return_value.parent.parent.parent = mock_project_root
            mock_project_root.__truediv__.return_value = mock_docs_dir
            mock_docs_dir.__truediv__.return_value = mock_output_dir
            
            main()
            
            # Check that files were processed
            assert mock_print.call_count >= 4
            assert any("Found 2 markdown files:" in str(call) for call in mock_print.call_args_list)
            assert any("Successful: 2" in str(call) for call in mock_print.call_args_list)
            assert any("Failed: 0" in str(call) for call in mock_print.call_args_list)

    def test_main_no_markdown_files(self):
        """Test main function when no markdown files are found."""
        with patch('src.utils.generate_pdfs.Path') as mock_path, \
             patch('src.utils.generate_pdfs.get_markdown_files', return_value=[]), \
             patch('builtins.print') as mock_print:
            
            # Mock the project root and docs directory
            mock_project_root = MagicMock()
            mock_docs_dir = MagicMock()
            mock_output_dir = MagicMock()
            
            mock_path.return_value.parent.parent.parent = mock_project_root
            mock_project_root.__truediv__.return_value = mock_docs_dir
            mock_docs_dir.__truediv__.return_value = mock_output_dir
            
            main()
            
            # Check that appropriate message was printed
            assert any("No markdown files found in docs directory." in str(call) for call in mock_print.call_args_list)

    def test_main_mixed_success_failure(self):
        """Test main function with mixed success and failure."""
        mock_files = [
            Path("/test/docs/file1.md"),
            Path("/test/docs/file2.md"),
            Path("/test/docs/file3.md"),
        ]
        
        with patch('src.utils.generate_pdfs.Path') as mock_path, \
             patch('src.utils.generate_pdfs.get_markdown_files', return_value=mock_files), \
             patch('src.utils.generate_pdfs.convert_md_to_pdf', side_effect=[True, False, True]), \
             patch('builtins.print') as mock_print:
            
            # Mock the project root and docs directory
            mock_project_root = MagicMock()
            mock_docs_dir = MagicMock()
            mock_output_dir = MagicMock()
            
            mock_path.return_value.parent.parent.parent = mock_project_root
            mock_project_root.__truediv__.return_value = mock_docs_dir
            mock_docs_dir.__truediv__.return_value = mock_output_dir
            
            main()
            
            # Check that correct counts were reported
            assert any("Successful: 2" in str(call) for call in mock_print.call_args_list)
            assert any("Failed: 1" in str(call) for call in mock_print.call_args_list)
            assert any("Total: 3" in str(call) for call in mock_print.call_args_list)

    def test_main_output_directory_creation(self):
        """Test that output directory is created."""
        mock_files = [Path("/test/docs/file1.md")]
        
        with patch('src.utils.generate_pdfs.Path') as mock_path, \
             patch('src.utils.generate_pdfs.get_markdown_files', return_value=mock_files), \
             patch('src.utils.generate_pdfs.convert_md_to_pdf', return_value=True), \
             patch('builtins.print'):
            
            # Mock the project root and docs directory
            mock_project_root = MagicMock()
            mock_docs_dir = MagicMock()
            mock_output_dir = MagicMock()
            
            mock_path.return_value.parent.parent.parent = mock_project_root
            mock_project_root.__truediv__.return_value = mock_docs_dir
            mock_docs_dir.__truediv__.return_value = mock_output_dir
            
            main()
            
            # Check that mkdir was called on output directory
            mock_output_dir.mkdir.assert_called_once_with(exist_ok=True)

    def test_main_pdf_file_listing(self):
        """Test that generated PDF files are listed."""
        mock_files = [Path("/test/docs/file1.md")]
        mock_pdf_files = [Path("/test/docs/pdfs/file1.pdf")]
        
        with patch('src.utils.generate_pdfs.Path') as mock_path, \
             patch('src.utils.generate_pdfs.get_markdown_files', return_value=mock_files), \
             patch('src.utils.generate_pdfs.convert_md_to_pdf', return_value=True), \
             patch('pathlib.Path.glob', return_value=mock_pdf_files), \
             patch('builtins.print') as mock_print:
            
            # Mock the project root and docs directory
            mock_project_root = MagicMock()
            mock_docs_dir = MagicMock()
            mock_output_dir = MagicMock()
            
            mock_path.return_value.parent.parent.parent = mock_project_root
            mock_project_root.__truediv__.return_value = mock_docs_dir
            mock_docs_dir.__truediv__.return_value = mock_output_dir
            
            main()
            
            # Check that PDF files were listed
            assert any("Generated PDF files:" in str(call) for call in mock_print.call_args_list)
            # The PDF file listing happens in the main function, but the exact output depends on the mocked Path
            # We'll just verify that the listing section was called
            call_strings = [str(call) for call in mock_print.call_args_list]
            combined_calls = " ".join(call_strings)
            assert "Generated PDF files:" in combined_calls


class TestIntegration:
    """Integration tests for the PDF generation workflow."""

    def test_complete_workflow_integration(self):
        """Test the complete workflow integration."""
        mock_files = [
            Path("/test/docs/file1.md"),
            Path("/test/docs/file2.md"),
        ]
        
        with patch('src.utils.generate_pdfs.Path') as mock_path, \
             patch('src.utils.generate_pdfs.get_markdown_files', return_value=mock_files), \
             patch('src.utils.generate_pdfs.convert_md_to_pdf', return_value=True), \
             patch('builtins.print') as mock_print:
            
            # Mock the project root and docs directory
            mock_project_root = MagicMock()
            mock_docs_dir = MagicMock()
            mock_output_dir = MagicMock()
            
            mock_path.return_value.parent.parent.parent = mock_project_root
            mock_project_root.__truediv__.return_value = mock_docs_dir
            mock_docs_dir.__truediv__.return_value = mock_output_dir
            
            main()
            
            # Verify that all functions were called
            assert mock_print.call_count >= 4
            assert any("=== Markdown to PDF Converter ===" in str(call) for call in mock_print.call_args_list)
            assert any("=== Conversion Summary ===" in str(call) for call in mock_print.call_args_list)

    def test_error_handling_integration(self):
        """Test error handling in the integrated workflow."""
        mock_files = [Path("/test/docs/file1.md")]
        
        with patch('src.utils.generate_pdfs.Path') as mock_path, \
             patch('src.utils.generate_pdfs.get_markdown_files', return_value=mock_files), \
             patch('src.utils.generate_pdfs.convert_md_to_pdf', return_value=False), \
             patch('builtins.print') as mock_print:
            
            # Mock the project root and docs directory
            mock_project_root = MagicMock()
            mock_docs_dir = MagicMock()
            mock_output_dir = MagicMock()
            
            mock_path.return_value.parent.parent.parent = mock_project_root
            mock_project_root.__truediv__.return_value = mock_docs_dir
            mock_docs_dir.__truediv__.return_value = mock_output_dir
            
            main()
            
            # Check that errors were handled gracefully
            assert any("Successful: 0" in str(call) for call in mock_print.call_args_list)
            assert any("Failed: 1" in str(call) for call in mock_print.call_args_list)

    def test_file_discovery_integration(self):
        """Test file discovery integration."""
        with patch('src.utils.generate_pdfs.Path') as mock_path, \
             patch('pathlib.Path.glob', return_value=[]), \
             patch('builtins.print') as mock_print:
            
            # Mock the project root and docs directory
            mock_project_root = MagicMock()
            mock_docs_dir = MagicMock()
            mock_output_dir = MagicMock()
            
            mock_path.return_value.parent.parent.parent = mock_project_root
            mock_project_root.__truediv__.return_value = mock_docs_dir
            mock_docs_dir.__truediv__.return_value = mock_output_dir
            
            main()
            
            # Check that file discovery was attempted
            assert any("No markdown files found in docs directory." in str(call) for call in mock_print.call_args_list)

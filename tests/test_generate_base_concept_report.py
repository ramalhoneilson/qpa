"""
Test suite for src/reporting/base_concept_report.py
"""

import csv
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest

from src.reporting.base_concept_report import (
    BaseConceptReportGenerator,
    main,
)


class TestBaseConceptReportGenerator:
    """Test the BaseConceptReportGenerator class."""

    def setup_method(self):
        """Set up test data."""
        self.generator = BaseConceptReportGenerator()

    def test_init(self):
        """Test BaseConceptReportGenerator initialization."""
        assert self.generator.data_dir is not None
        assert self.generator.report_dir is not None
        assert self.generator.output_file is not None
        assert len(self.generator.input_files) == 3
        assert "Qiskit" in self.generator.input_files
        assert "PennyLane" in self.generator.input_files
        assert "Classiq" in self.generator.input_files

    def test_generate_report_success(self):
        """Test successful report generation."""
        with patch.object(self.generator, '_generate_header', return_value=["# Header"]), \
        patch.object(self.generator, '_generate_framework_tables', return_value=["## Framework"]), \
        patch.object(self.generator, '_generate_pattern_coverage_section', return_value=["## Patterns"]), \
        patch.object(self.generator.report_dir, 'mkdir'), \
        patch('builtins.open', mock_open()) as mock_file:
            
            self.generator.generate_report()
            
            # Verify file was opened for writing
            mock_file.assert_called_once()

    def test_generate_header(self):
        """Test header generation."""
        result = self.generator._generate_header()
        
        assert isinstance(result, list)
        assert len(result) > 0
        assert "# Extracted Quantum Concepts Summary" in result[0]
        assert "## Overview" in result

    def test_generate_framework_tables_success(self):
        """Test framework tables generation with existing files."""
        mock_concepts = [
            {"name": "test.concept1", "summary": "Summary 1"},
            {"name": "test.concept2", "summary": "Summary 2"}
        ]
        
        with patch.object(self.generator, '_generate_framework_table', return_value=["## Test Framework"]):
            result = self.generator._generate_framework_tables()
            
            assert isinstance(result, list)
            assert len(result) > 0

    def test_generate_framework_tables_missing_files(self):
        """Test framework tables generation with missing files."""
        with patch.object(self.generator, '_generate_framework_table', return_value=["## Test Framework"]):
            result = self.generator._generate_framework_tables()
            
            assert isinstance(result, list)

    def test_generate_framework_table_success(self):
        """Test successful framework table generation."""
        mock_concepts = [
            {"name": "test.concept1", "summary": "Summary 1"},
            {"name": "test.concept2", "summary": "Summary 2"}
        ]
        
        with patch.object(self.generator, '_read_concepts_from_csv', return_value=mock_concepts), \
        patch.object(self.generator, '_generate_markdown_table', return_value="| Concept | Summary |"):
            
            result = self.generator._generate_framework_table("TestFramework", {
                "path": Path("/test/file.csv"),
                "delimiter": ","
            })
            
            assert isinstance(result, list)
            assert "## TestFramework Quantum Concepts" in result

    def test_generate_framework_table_file_not_found(self):
        """Test framework table generation when file doesn't exist."""
        with patch.object(self.generator, '_read_concepts_from_csv', return_value=[]):
            result = self.generator._generate_framework_table("TestFramework", {
                "path": Path("/test/nonexistent.csv"),
                "delimiter": ","
            })
            
            assert isinstance(result, list)
            assert "## TestFramework Quantum Concepts" in result

    def test_generate_framework_table_error(self):
        """Test framework table generation with error."""
        with patch.object(self.generator, '_read_concepts_from_csv', side_effect=Exception("Read error")):
            result = self.generator._generate_framework_table("TestFramework", {
                "path": Path("/test/file.csv"),
                "delimiter": ","
            })
            
            assert isinstance(result, list)
            assert "**Error**:" in result[2]

    def test_read_concepts_from_csv_success(self):
        """Test successful CSV reading."""
        mock_csv_content = "name,summary\ntest.concept1,Summary 1\ntest.concept2,Summary 2"
        
        with patch('builtins.open', mock_open(read_data=mock_csv_content)), \
        patch('csv.DictReader') as mock_reader:
            mock_reader.return_value = [
                {"name": "test.concept1", "summary": "Summary 1"},
                {"name": "test.concept2", "summary": "Summary 2"}
            ]
            
            result = self.generator._read_concepts_from_csv(Path("/test/file.csv"), ",")
            
            assert isinstance(result, list)
            assert len(result) == 2
            assert result[0]["name"] == "test.concept1"

    def test_read_concepts_from_csv_file_not_found(self):
        """Test CSV reading when file doesn't exist."""
        with patch('builtins.open', side_effect=FileNotFoundError("File not found")):
            result = self.generator._read_concepts_from_csv(Path("/test/nonexistent.csv"), ",")
            
            assert isinstance(result, list)
            assert len(result) == 0

    def test_read_concepts_from_csv_error(self):
        """Test CSV reading with error."""
        with patch('builtins.open', side_effect=IOError("Read error")):
            result = self.generator._read_concepts_from_csv(Path("/test/file.csv"), ",")
            
            assert isinstance(result, list)
            assert len(result) == 0

    def test_generate_markdown_table_success(self):
        """Test successful markdown table generation."""
        concepts = [
            {"name": "test.concept1", "summary": "Summary 1"},
            {"name": "test.concept2", "summary": "Summary 2"}
        ]
        
        result = self.generator._generate_markdown_table(concepts)
        
        assert isinstance(result, str)
        assert "| Concept Name | Summary |" in result
        assert "| `test.concept1` | Summary 1 |" in result
        assert "| `test.concept2` | Summary 2 |" in result

    def test_generate_markdown_table_empty(self):
        """Test markdown table generation with empty concepts."""
        result = self.generator._generate_markdown_table([])
        
        assert isinstance(result, str)
        assert "*No concepts were extracted for this framework.*" in result

    def test_generate_markdown_table_with_pipes(self):
        """Test markdown table generation with pipe characters in content."""
        concepts = [
            {"name": "test|concept", "summary": "Summary with | pipe"}
        ]
        
        result = self.generator._generate_markdown_table(concepts)
        
        assert isinstance(result, str)
        assert "| `test\\|concept` | Summary with \\| pipe |" in result

    def test_generate_pattern_coverage_section(self):
        """Test pattern coverage section generation."""
        result = self.generator._generate_pattern_coverage_section()
        
        assert isinstance(result, list)
        assert "## Pattern Coverage Analysis" in result
        assert "### Pattern Coverage Summary" in result
        assert "### Complete List of Patterns Found" in result
        assert "### New Patterns Created" in result


class TestMainFunction:
    """Test the main function."""

    def test_main_function(self):
        """Test the main function execution."""
        with patch('src.reporting.base_concept_report.BaseConceptReportGenerator') as mock_class:
            mock_instance = MagicMock()
            mock_class.return_value = mock_instance
            
            main()
            
            mock_instance.generate_report.assert_called_once()


class TestIntegration:
    """Integration tests for base concept report generation."""

    def test_complete_workflow_integration(self):
        """Test the complete base concept report generation workflow."""
        generator = BaseConceptReportGenerator()
        
        mock_concepts = [
            {"name": "test.concept1", "summary": "Summary 1"},
            {"name": "test.concept2", "summary": "Summary 2"}
        ]
        
        with patch.object(generator, '_generate_header', return_value=["# Header"]), \
        patch.object(generator, '_generate_framework_tables', return_value=["## Framework"]), \
        patch.object(generator, '_generate_pattern_coverage_section', return_value=["## Patterns"]), \
        patch.object(generator.report_dir, 'mkdir'), \
        patch('builtins.open', mock_open()) as mock_file:
            
            generator.generate_report()
            
            # Verify the workflow was executed
            mock_file.assert_called_once()

    def test_error_handling_integration(self):
        """Test error handling in the complete workflow."""
        generator = BaseConceptReportGenerator()
        
        with patch.object(generator.report_dir, 'mkdir', side_effect=OSError("Directory error")):
            generator.generate_report()
            # Should not raise any exceptions
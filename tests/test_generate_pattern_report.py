"""
Test suite for src/reporting/pattern_report.py
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest

from src.reporting.pattern_report import (
    PatternReportGenerator,
    main,
)


class TestPatternReportGenerator:
    """Test the PatternReportGenerator class."""

    def setup_method(self):
        """Set up test data."""
        self.generator = PatternReportGenerator()

    def test_init(self):
        """Test PatternReportGenerator initialization."""
        assert self.generator.data_dir is not None
        assert self.generator.report_dir is not None
        assert self.generator.input_file is not None
        assert self.generator.output_file is not None

    def test_generate_report_success(self):
        """Test successful report generation."""
        mock_patterns = [
            {
                "name": "Test Pattern 1",
                "alias": "TP1",
                "intent": "Test intent 1",
                "context": "Test context 1",
                "solution": "Test solution 1",
                "consequences": "Test consequences 1",
                "related_patterns": "Related pattern 1",
                "implementation_notes": "Implementation notes 1",
                "examples": "Example 1"
            },
            {
                "name": "Test Pattern 2",
                "alias": "TP2",
                "intent": "Test intent 2",
                "context": "Test context 2",
                "solution": "Test solution 2"
            }
        ]
        
        with patch('builtins.open', mock_open()) as mock_file, \
        patch.object(self.generator.input_file, 'exists', return_value=True), \
        patch.object(self.generator.report_dir, 'mkdir'), \
        patch('json.load', return_value=mock_patterns):
            
            self.generator.generate_report()
            
            # Verify file was opened for writing
            mock_file.assert_called_once()

    def test_generate_report_file_not_found(self):
        """Test report generation when input file doesn't exist."""
        with patch.object(self.generator.input_file, 'exists', return_value=False):
            self.generator.generate_report()
            # Should not raise any exceptions

    def test_generate_report_json_error(self):
        """Test report generation when JSON loading fails."""
        with patch.object(self.generator.input_file, 'exists', return_value=True), \
        patch('builtins.open', side_effect=json.JSONDecodeError("Invalid JSON", "doc", 0)):
            
            self.generator.generate_report()
            # Should not raise any exceptions

    def test_generate_header(self):
        """Test header generation."""
        result = self.generator._generate_header(5)
        
        assert isinstance(result, list)
        assert len(result) > 0
        assert "# Quantum Patterns Report" in result[0]
        assert "**Total Patterns**: 5" in result

    def test_generate_pattern_details(self):
        """Test pattern details generation."""
        mock_patterns = [
            {
                "name": "Test Pattern",
                "alias": "TP",
                "intent": "Test intent",
                "context": "Test context",
                "solution": "Test solution"
            }
        ]
        
        result = self.generator._generate_pattern_details(mock_patterns)
        
        assert isinstance(result, list)
        assert len(result) > 0
        assert "## Test Pattern" in result

    def test_generate_pattern_section_complete(self):
        """Test pattern section generation with complete data."""
        pattern = {
            "name": "Test Pattern",
            "alias": "TP",
            "intent": "Test intent",
            "context": "Test context",
            "solution": "Test solution",
            "consequences": "Test consequences",
            "related_patterns": "Related pattern",
            "implementation_notes": "Implementation notes",
            "examples": "Example"
        }
        
        result = self.generator._generate_pattern_section(pattern)
        
        assert isinstance(result, list)
        assert "## Test Pattern" in result
        assert "**Alias**: TP" in result
        assert "### Intent" in result
        assert "### Context" in result
        assert "### Solution" in result
        assert "### Consequences" in result
        assert "### Related Patterns" in result
        assert "### Implementation Notes" in result
        assert "### Examples" in result

    def test_generate_pattern_section_minimal(self):
        """Test pattern section generation with minimal data."""
        pattern = {
            "name": "Test Pattern"
        }
        
        result = self.generator._generate_pattern_section(pattern)
        
        assert isinstance(result, list)
        assert "## Test Pattern" in result
        assert "**Alias**:" not in result
        assert "### Intent" not in result

    def test_generate_pattern_section_with_alias_placeholder(self):
        """Test pattern section generation with placeholder alias."""
        pattern = {
            "name": "Test Pattern",
            "alias": "—"
        }
        
        result = self.generator._generate_pattern_section(pattern)
        
        assert isinstance(result, list)
        assert "## Test Pattern" in result
        assert "**Alias**:" not in result

    def test_generate_pattern_section_empty_fields(self):
        """Test pattern section generation with empty fields."""
        pattern = {
            "name": "Test Pattern",
            "alias": "",
            "intent": "",
            "context": "",
            "solution": ""
        }
        
        result = self.generator._generate_pattern_section(pattern)
        
        assert isinstance(result, list)
        assert "## Test Pattern" in result
        assert "**Alias**:" not in result
        assert "### Intent" not in result
        assert "### Context" not in result
        assert "### Solution" not in result


class TestMainFunction:
    """Test the main function."""

    def test_main_function(self):
        """Test the main function execution."""
        with patch('src.reporting.pattern_report.PatternReportGenerator') as mock_class:
            mock_instance = MagicMock()
            mock_class.return_value = mock_instance
            
            main()
            
            mock_instance.generate_report.assert_called_once()


class TestIntegration:
    """Integration tests for pattern report generation."""

    def test_complete_workflow_integration(self):
        """Test the complete pattern report generation workflow."""
        generator = PatternReportGenerator()
        
        mock_patterns = [
            {
                "name": "Test Pattern",
                "alias": "TP",
                "intent": "Test intent",
                "context": "Test context",
                "solution": "Test solution"
            }
        ]
        
        with patch.object(generator.input_file, 'exists', return_value=True), \
        patch.object(generator.report_dir, 'mkdir'), \
        patch('builtins.open', mock_open()) as mock_file, \
        patch('json.load', return_value=mock_patterns):
            
            generator.generate_report()
            
            # Verify the workflow was executed
            mock_file.assert_called_once()

    def test_error_handling_integration(self):
        """Test error handling in the complete workflow."""
        generator = PatternReportGenerator()
        
        with patch.object(generator.input_file, 'exists', return_value=True), \
        patch('builtins.open', side_effect=IOError("File error")):
            
            generator.generate_report()
            # Should not raise any exceptions
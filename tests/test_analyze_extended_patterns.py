"""
Tests for src/reporting/extended_pattern_analysis.py

This module tests the extended pattern analysis functionality,
including pattern loading, coverage analysis, and report generation.
"""

from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest

from src.reporting.extended_pattern_analysis import (
    ExtendedPatternAnalyzer,
    main,
)


class TestExtendedPatternAnalyzer:
    """Test the ExtendedPatternAnalyzer class."""

    def test_init(self):
        """Test ExtendedPatternAnalyzer initialization."""
        analyzer = ExtendedPatternAnalyzer()
        
        assert analyzer.data_dir is not None
        assert analyzer.report_dir is not None
        assert analyzer.output_file is not None

    def test_load_extended_patterns_success(self):
        """Test successful loading of extended patterns."""
        analyzer = ExtendedPatternAnalyzer()
        mock_csv_content = "PatternName\nPattern1\nPattern2\nPattern3"
        
        with patch('builtins.open', mock_open(read_data=mock_csv_content)), \
        patch('csv.DictReader') as mock_reader:
            mock_reader.return_value = [
                {"PatternName": "Pattern1"},
                {"PatternName": "Pattern2"},
                {"PatternName": "Pattern3"}
            ]
            
            result = analyzer.load_extended_patterns()
            
            assert isinstance(result, set)
            assert "Pattern1" in result
            assert "Pattern2" in result
            assert "Pattern3" in result

    def test_load_extended_patterns_file_not_found(self):
        """Test loading when extended patterns file doesn't exist."""
        analyzer = ExtendedPatternAnalyzer()
        
        with patch('builtins.open', side_effect=FileNotFoundError("File not found")):
            result = analyzer.load_extended_patterns()
            
            assert isinstance(result, set)
            assert len(result) == 0

    def test_load_extended_patterns_empty_file(self):
        """Test loading when extended patterns file is empty."""
        analyzer = ExtendedPatternAnalyzer()
        
        with patch('builtins.open', mock_open(read_data="")), \
        patch('csv.DictReader', return_value=[]):
            
            result = analyzer.load_extended_patterns()
            
            assert isinstance(result, set)
            assert len(result) == 0

    def test_load_framework_patterns_success(self):
        """Test successful loading of framework patterns."""
        analyzer = ExtendedPatternAnalyzer()
        
        mock_csv_content = "pattern\nPattern1\nPattern2"
        
        with patch('builtins.open', mock_open(read_data=mock_csv_content)), \
        patch('csv.DictReader') as mock_reader:
            mock_reader.return_value = [
                {"pattern": "Pattern1"},
                {"pattern": "Pattern2"}
            ]
            
            result = analyzer.load_framework_patterns()
            
            assert isinstance(result, dict)
            assert "Classiq" in result
            assert "PennyLane" in result
            assert "Qiskit" in result

    def test_load_framework_patterns_missing_files(self):
        """Test loading when some framework files are missing."""
        analyzer = ExtendedPatternAnalyzer()
        
        with patch('builtins.open', side_effect=FileNotFoundError("File not found")):
            result = analyzer.load_framework_patterns()
            
            assert isinstance(result, dict)
            assert len(result) == 3
            for framework in result.values():
                assert isinstance(framework, set)
                assert len(framework) == 0

    def test_load_target_project_patterns_success(self):
        """Test successful loading of target project patterns."""
        analyzer = ExtendedPatternAnalyzer()
        mock_csv_content = "pattern\nPattern1\nPattern2\nN/A"
        
        with patch('builtins.open', mock_open(read_data=mock_csv_content)), \
        patch('csv.DictReader') as mock_reader:
            mock_reader.return_value = [
                {"pattern": "Pattern1"},
                {"pattern": "Pattern2"},
                {"pattern": "N/A"}
            ]
            
            result = analyzer.load_target_project_patterns()
            
            assert isinstance(result, set)
            assert "Pattern1" in result
            assert "Pattern2" in result
            assert "N/A" not in result

    def test_load_target_project_patterns_file_not_found(self):
        """Test loading when target project patterns file doesn't exist."""
        analyzer = ExtendedPatternAnalyzer()
        
        with patch('builtins.open', side_effect=FileNotFoundError("File not found")):
            result = analyzer.load_target_project_patterns()
            
            assert isinstance(result, set)
            assert len(result) == 0

    def test_perform_analysis_success(self):
        """Test successful pattern coverage analysis."""
        analyzer = ExtendedPatternAnalyzer()
        
        extended_patterns = {"Pattern1", "Pattern2", "Pattern3"}
        framework_patterns = {
            "Classiq": {"Pattern1", "Pattern2"},
            "PennyLane": {"Pattern2", "Pattern3"},
            "Qiskit": {"Pattern1", "Pattern3"}
        }
        target_patterns = {"Pattern1", "Pattern3"}
        
        result = analyzer.perform_analysis(extended_patterns, framework_patterns, target_patterns)
        
        assert isinstance(result, dict)
        assert "extended_patterns" in result
        assert "framework_coverage" in result
        assert "target_coverage" in result
        assert "summary" in result
        
        # Check framework coverage
        assert result["framework_coverage"]["Classiq"]["count"] == 2
        assert result["framework_coverage"]["PennyLane"]["count"] == 2
        assert result["framework_coverage"]["Qiskit"]["count"] == 2
        
        # Check target coverage
        assert result["target_coverage"]["count"] == 2

    def test_perform_analysis_empty_patterns(self):
        """Test analysis with empty pattern sets."""
        analyzer = ExtendedPatternAnalyzer()
        
        extended_patterns = set()
        framework_patterns = {"Classiq": set(), "PennyLane": set(), "Qiskit": set()}
        target_patterns = set()
        
        result = analyzer.perform_analysis(extended_patterns, framework_patterns, target_patterns)
        
        assert isinstance(result, dict)
        assert result["summary"]["total_extended_patterns"] == 0
        assert result["summary"]["found_in_frameworks"] == 0
        assert result["summary"]["found_in_targets"] == 0

    def test_generate_report_success(self):
        """Test successful report generation."""
        analyzer = ExtendedPatternAnalyzer()
        
        mock_results = {
            "summary": {
                "total_extended_patterns": 3,
                "found_in_frameworks": 2,
                "found_in_targets": 1,
                "framework_coverage_rate": 66.7,
                "target_coverage_rate": 33.3
            },
            "framework_coverage": {
                "Classiq": {"count": 2, "total": 3, "percentage": 66.7, "found": {"Pattern1", "Pattern2"}},
                "PennyLane": {"count": 1, "total": 3, "percentage": 33.3, "found": {"Pattern1"}},
                "Qiskit": {"count": 2, "total": 3, "percentage": 66.7, "found": {"Pattern1", "Pattern3"}}
            },
            "target_coverage": {
                "count": 1,
                "total": 3,
                "percentage": 33.3,
                "found": {"Pattern1"}
            }
        }
        
        with patch('builtins.open', mock_open()) as mock_file, \
        patch.object(analyzer.report_dir, 'mkdir'):
            
            analyzer.generate_report(mock_results)
            
            # Verify file was opened for writing
            mock_file.assert_called_once()

    def test_generate_report_error_handling(self):
        """Test report generation error handling."""
        analyzer = ExtendedPatternAnalyzer()
        
        mock_results = {"summary": {"total_extended_patterns": 0}}
        
        with patch('builtins.open', side_effect=IOError("Write error")):
            analyzer.generate_report(mock_results)
            # Should not raise any exceptions

    def test_print_summary_success(self):
        """Test successful summary printing."""
        analyzer = ExtendedPatternAnalyzer()
        
        mock_results = {
            "summary": {
                "total_extended_patterns": 3,
                "found_in_frameworks": 2,
                "found_in_targets": 1,
                "framework_coverage_rate": 66.7,
                "target_coverage_rate": 33.3
            }
        }
        
        with patch('builtins.print') as mock_print:
            analyzer.print_summary(mock_results)
            
            # Verify print was called
            assert mock_print.call_count > 0

    def test_analyze_pattern_coverage_integration(self):
        """Test the complete analysis workflow."""
        analyzer = ExtendedPatternAnalyzer()
        
        with patch.object(analyzer, 'load_extended_patterns', return_value={"Pattern1", "Pattern2"}), \
        patch.object(analyzer, 'load_framework_patterns', return_value={
            "Classiq": {"Pattern1"}, "PennyLane": {"Pattern2"}, "Qiskit": {"Pattern1", "Pattern2"}
        }), \
        patch.object(analyzer, 'load_target_project_patterns', return_value={"Pattern1"}), \
        patch.object(analyzer, 'perform_analysis', return_value={"summary": {"total_extended_patterns": 2}}), \
        patch.object(analyzer, 'generate_report'), \
        patch.object(analyzer, 'print_summary'):
            
            analyzer.analyze_pattern_coverage()
            
            # Verify all methods were called
            analyzer.load_extended_patterns.assert_called_once()
            analyzer.load_framework_patterns.assert_called_once()
            analyzer.load_target_project_patterns.assert_called_once()
            analyzer.perform_analysis.assert_called_once()
            analyzer.generate_report.assert_called_once()
            analyzer.print_summary.assert_called_once()


class TestMainFunction:
    """Test the main function."""

    def test_main_function(self):
        """Test the main function execution."""
        with patch('src.reporting.extended_pattern_analysis.ExtendedPatternAnalyzer') as mock_class:
            mock_instance = MagicMock()
            mock_class.return_value = mock_instance
            
            main()
            
            mock_instance.analyze_pattern_coverage.assert_called_once()


class TestIntegration:
    """Integration tests for extended pattern analysis."""

    def test_complete_workflow_integration(self):
        """Test the complete extended pattern analysis workflow."""
        analyzer = ExtendedPatternAnalyzer()
        
        # Mock all external dependencies
        with patch.object(analyzer, 'load_extended_patterns', return_value={"Pattern1", "Pattern2"}), \
        patch.object(analyzer, 'load_framework_patterns', return_value={
            "Classiq": {"Pattern1"}, "PennyLane": {"Pattern2"}, "Qiskit": {"Pattern1", "Pattern2"}
        }), \
        patch.object(analyzer, 'load_target_project_patterns', return_value={"Pattern1"}), \
        patch.object(analyzer, 'perform_analysis', return_value={"summary": {"total_extended_patterns": 2}}), \
        patch.object(analyzer, 'generate_report'), \
        patch.object(analyzer, 'print_summary'):
            
            analyzer.analyze_pattern_coverage()
            
            # Verify the workflow was executed
            analyzer.generate_report.assert_called_once()

    def test_error_handling_integration(self):
        """Test error handling in the complete workflow."""
        analyzer = ExtendedPatternAnalyzer()
        
        with patch.object(analyzer, 'load_extended_patterns', side_effect=Exception("Load error")):
            analyzer.analyze_pattern_coverage()
            # Should not raise any exceptions
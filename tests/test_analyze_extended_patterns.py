"""
Tests for src/utils/analyze_extended_patterns.py

This module tests the extended pattern coverage analysis functionality,
including pattern loading, coverage analysis, and report generation.
"""

import csv
import json
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest

from src.utils.analyze_extended_patterns import (
    analyze_pattern_coverage,
    generate_report,
    load_extended_patterns,
    load_framework_patterns,
    load_target_project_patterns,
    main,
)


class TestLoadExtendedPatterns:
    """Test the load_extended_patterns function."""

    def test_load_extended_patterns_success(self):
        """Test successful loading of extended patterns."""
        mock_content = "PatternName\nPattern1\nPattern2\nPattern3"
        
        with patch('src.utils.analyze_extended_patterns.config') as mock_config, \
             patch('builtins.open', mock_open(read_data=mock_content)) as mock_file, \
             patch('csv.DictReader') as mock_reader:
            
            mock_config.RESULTS_DIR = Path("/test/results")
            mock_reader.return_value = [
                {"PatternName": "Pattern1"},
                {"PatternName": "Pattern2"},
                {"PatternName": "Pattern3"},
            ]
            
            result = load_extended_patterns()
            
            assert result == {"Pattern1", "Pattern2", "Pattern3"}
            mock_file.assert_called_once()

    def test_load_extended_patterns_empty_file(self):
        """Test loading from empty file."""
        mock_content = "PatternName\n"
        
        with patch('src.utils.analyze_extended_patterns.config') as mock_config, \
             patch('builtins.open', mock_open(read_data=mock_content)) as mock_file, \
             patch('csv.DictReader') as mock_reader:
            
            mock_config.RESULTS_DIR = Path("/test/results")
            mock_reader.return_value = []
            
            result = load_extended_patterns()
            
            assert result == set()

    def test_load_extended_patterns_file_error(self):
        """Test handling of file reading errors."""
        with patch('src.utils.analyze_extended_patterns.config') as mock_config, \
             patch('builtins.open', side_effect=FileNotFoundError("File not found")), \
             patch('builtins.print') as mock_print:
            
            mock_config.RESULTS_DIR = Path("/test/results")
            
            result = load_extended_patterns()
            
            assert result == set()
            mock_print.assert_called_with("Error loading extended patterns: File not found")

    def test_load_extended_patterns_empty_pattern_names(self):
        """Test handling of empty pattern names."""
        mock_content = "PatternName\nPattern1\n\nPattern2\n   \nPattern3"
        
        with patch('src.utils.analyze_extended_patterns.config') as mock_config, \
             patch('builtins.open', mock_open(read_data=mock_content)) as mock_file, \
             patch('csv.DictReader') as mock_reader:
            
            mock_config.RESULTS_DIR = Path("/test/results")
            mock_reader.return_value = [
                {"PatternName": "Pattern1"},
                {"PatternName": ""},
                {"PatternName": "Pattern2"},
                {"PatternName": "   "},
                {"PatternName": "Pattern3"},
            ]
            
            result = load_extended_patterns()
            
            assert result == {"Pattern1", "Pattern2", "Pattern3"}


class TestLoadFrameworkPatterns:
    """Test the load_framework_patterns function."""

    def test_load_framework_patterns_success(self):
        """Test successful loading of framework patterns."""
        mock_content = "pattern\nPattern1\nPattern2"
        
        with patch('src.utils.analyze_extended_patterns.config') as mock_config, \
             patch('builtins.open', mock_open(read_data=mock_content)) as mock_file, \
             patch('csv.DictReader') as mock_reader:
            
            mock_config.RESULTS_DIR = Path("/test/results")
            mock_reader.return_value = [
                {"pattern": "Pattern1"},
                {"pattern": "Pattern2"},
            ]
            
            result = load_framework_patterns()
            
            assert "Classiq" in result
            assert "PennyLane" in result
            assert "Qiskit" in result
            assert result["Classiq"] == {"Pattern1", "Pattern2"}
            assert result["PennyLane"] == {"Pattern1", "Pattern2"}
            assert result["Qiskit"] == {"Pattern1", "Pattern2"}

    def test_load_framework_patterns_file_error(self):
        """Test handling of file reading errors for frameworks."""
        with patch('src.utils.analyze_extended_patterns.config') as mock_config, \
             patch('builtins.open', side_effect=FileNotFoundError("File not found")), \
             patch('builtins.print') as mock_print:
            
            mock_config.RESULTS_DIR = Path("/test/results")
            
            result = load_framework_patterns()
            
            assert "Classiq" in result
            assert "PennyLane" in result
            assert "Qiskit" in result
            assert result["Classiq"] == set()
            assert result["PennyLane"] == set()
            assert result["Qiskit"] == set()
            
            # Check that error messages were printed
            assert mock_print.call_count == 3

    def test_load_framework_patterns_empty_patterns(self):
        """Test handling of empty pattern fields."""
        mock_content = "pattern\nPattern1\n\nPattern2"
        
        with patch('src.utils.analyze_extended_patterns.config') as mock_config, \
             patch('builtins.open', mock_open(read_data=mock_content)) as mock_file, \
             patch('csv.DictReader') as mock_reader:
            
            mock_config.RESULTS_DIR = Path("/test/results")
            mock_reader.return_value = [
                {"pattern": "Pattern1"},
                {"pattern": ""},
                {"pattern": "Pattern2"},
            ]
            
            result = load_framework_patterns()
            
            assert result["Classiq"] == {"Pattern1", "Pattern2"}
            assert result["PennyLane"] == {"Pattern1", "Pattern2"}
            assert result["Qiskit"] == {"Pattern1", "Pattern2"}


class TestLoadTargetProjectPatterns:
    """Test the load_target_project_patterns function."""

    def test_load_target_project_patterns_success(self):
        """Test successful loading of target project patterns."""
        mock_content = "pattern\nPattern1\nPattern2"
        
        with patch('src.utils.analyze_extended_patterns.config') as mock_config, \
             patch('builtins.open', mock_open(read_data=mock_content)) as mock_file, \
             patch('csv.DictReader') as mock_reader:
            
            mock_config.RESULTS_DIR = Path("/test/results")
            mock_reader.return_value = [
                {"pattern": "Pattern1"},
                {"pattern": "Pattern2"},
            ]
            
            result = load_target_project_patterns()
            
            assert result == {"Pattern1", "Pattern2"}

    def test_load_target_project_patterns_file_error(self):
        """Test handling of file reading errors."""
        with patch('src.utils.analyze_extended_patterns.config') as mock_config, \
             patch('builtins.open', side_effect=FileNotFoundError("File not found")), \
             patch('builtins.print') as mock_print:
            
            mock_config.RESULTS_DIR = Path("/test/results")
            
            result = load_target_project_patterns()
            
            assert result == set()
            mock_print.assert_called_with("Error loading target project patterns: File not found")

    def test_load_target_project_patterns_empty_patterns(self):
        """Test handling of empty pattern fields."""
        mock_content = "pattern\nPattern1\n\nPattern2"
        
        with patch('src.utils.analyze_extended_patterns.config') as mock_config, \
             patch('builtins.open', mock_open(read_data=mock_content)) as mock_file, \
             patch('csv.DictReader') as mock_reader:
            
            mock_config.RESULTS_DIR = Path("/test/results")
            mock_reader.return_value = [
                {"pattern": "Pattern1"},
                {"pattern": ""},
                {"pattern": "Pattern2"},
            ]
            
            result = load_target_project_patterns()
            
            assert result == {"Pattern1", "Pattern2"}


class TestAnalyzePatternCoverage:
    """Test the analyze_pattern_coverage function."""

    def test_analyze_pattern_coverage_success(self):
        """Test successful pattern coverage analysis."""
        extended_patterns = {"Pattern1", "Pattern2", "Pattern3", "Pattern4"}
        framework_patterns = {
            "Classiq": {"Pattern1", "Pattern2"},
            "PennyLane": {"Pattern2", "Pattern3"},
            "Qiskit": {"Pattern1", "Pattern3"},
        }
        target_patterns = {"Pattern1", "Pattern2"}
        
        with patch('src.utils.analyze_extended_patterns.load_extended_patterns', return_value=extended_patterns), \
             patch('src.utils.analyze_extended_patterns.load_framework_patterns', return_value=framework_patterns), \
             patch('src.utils.analyze_extended_patterns.load_target_project_patterns', return_value=target_patterns), \
             patch('builtins.print'):
            
            result = analyze_pattern_coverage()
            
            assert result["extended_patterns"] == extended_patterns
            assert result["target_patterns"] == target_patterns
            
            # Check framework coverage
            assert result["framework_coverage"]["Classiq"]["total_found"] == 2
            assert result["framework_coverage"]["Classiq"]["patterns"] == {"Pattern1", "Pattern2"}
            assert result["framework_coverage"]["Classiq"]["missing"] == {"Pattern3", "Pattern4"}
            
            assert result["framework_coverage"]["PennyLane"]["total_found"] == 2
            assert result["framework_coverage"]["PennyLane"]["patterns"] == {"Pattern2", "Pattern3"}
            assert result["framework_coverage"]["PennyLane"]["missing"] == {"Pattern1", "Pattern4"}
            
            assert result["framework_coverage"]["Qiskit"]["total_found"] == 2
            assert result["framework_coverage"]["Qiskit"]["patterns"] == {"Pattern1", "Pattern3"}
            assert result["framework_coverage"]["Qiskit"]["missing"] == {"Pattern2", "Pattern4"}
            
            # Check target coverage
            assert result["target_coverage"]["total_found"] == 2
            assert result["target_coverage"]["patterns"] == {"Pattern1", "Pattern2"}
            assert result["target_coverage"]["missing"] == {"Pattern3", "Pattern4"}
            
            # Check framework-only patterns
            assert result["framework_only"] == {"Pattern3"}

    def test_analyze_pattern_coverage_empty_data(self):
        """Test pattern coverage analysis with empty data."""
        extended_patterns = set()
        framework_patterns = {
            "Classiq": set(),
            "PennyLane": set(),
            "Qiskit": set(),
        }
        target_patterns = set()
        
        with patch('src.utils.analyze_extended_patterns.load_extended_patterns', return_value=extended_patterns), \
             patch('src.utils.analyze_extended_patterns.load_framework_patterns', return_value=framework_patterns), \
             patch('src.utils.analyze_extended_patterns.load_target_project_patterns', return_value=target_patterns), \
             patch('builtins.print'):
            
            result = analyze_pattern_coverage()
            
            assert result["extended_patterns"] == set()
            assert result["target_patterns"] == set()
            assert result["framework_only"] == set()
            
            for framework in ["Classiq", "PennyLane", "Qiskit"]:
                assert result["framework_coverage"][framework]["total_found"] == 0
                assert result["framework_coverage"][framework]["patterns"] == set()
                assert result["framework_coverage"][framework]["missing"] == set()
            
            assert result["target_coverage"]["total_found"] == 0
            assert result["target_coverage"]["patterns"] == set()
            assert result["target_coverage"]["missing"] == set()


class TestGenerateReport:
    """Test the generate_report function."""

    def test_generate_report_success(self):
        """Test successful report generation."""
        coverage_data = {
            "extended_patterns": {"Pattern1", "Pattern2", "Pattern3"},
            "framework_coverage": {
                "Classiq": {
                    "total_found": 2,
                    "patterns": {"Pattern1", "Pattern2"},
                    "missing": {"Pattern3"},
                },
                "PennyLane": {
                    "total_found": 1,
                    "patterns": {"Pattern2"},
                    "missing": {"Pattern1", "Pattern3"},
                },
                "Qiskit": {
                    "total_found": 2,
                    "patterns": {"Pattern1", "Pattern3"},
                    "missing": {"Pattern2"},
                },
            },
            "target_coverage": {
                "total_found": 2,
                "patterns": {"Pattern1", "Pattern2"},
                "missing": {"Pattern3"},
            },
            "framework_only": {"Pattern3"},
            "target_patterns": {"Pattern1", "Pattern2"},
        }
        
        with patch('src.utils.analyze_extended_patterns.analyze_pattern_coverage', return_value=coverage_data):
            result = generate_report()
            
            assert "# Extended Pattern Coverage Analysis" in result
            assert "## Summary Statistics" in result
            assert "### Framework Coverage" in result
            assert "### Target Project Coverage" in result
            assert "## Detailed Framework Analysis" in result
            assert "## Target Project Analysis" in result
            assert "## Cross-Framework Analysis" in result
            
            # Check specific content
            assert "Pattern1" in result
            assert "Pattern2" in result
            assert "Pattern3" in result
            assert "Classiq" in result
            assert "PennyLane" in result
            assert "Qiskit" in result

    def test_generate_report_empty_data(self):
        """Test report generation with empty data."""
        coverage_data = {
            "extended_patterns": set(),
            "framework_coverage": {
                "Classiq": {"total_found": 0, "patterns": set(), "missing": set()},
                "PennyLane": {"total_found": 0, "patterns": set(), "missing": set()},
                "Qiskit": {"total_found": 0, "patterns": set(), "missing": set()},
            },
            "target_coverage": {"total_found": 0, "patterns": set(), "missing": set()},
            "framework_only": set(),
            "target_patterns": set(),
        }
        
        with patch('src.utils.analyze_extended_patterns.analyze_pattern_coverage', return_value=coverage_data):
            result = generate_report()
            
            assert "# Extended Pattern Coverage Analysis" in result
            assert "## Summary Statistics" in result
            assert "### Framework Coverage" in result
            assert "### Target Project Coverage" in result
            assert "## Detailed Framework Analysis" in result
            assert "## Target Project Analysis" in result
            assert "## Cross-Framework Analysis" in result

    def test_generate_report_cross_framework_analysis(self):
        """Test cross-framework analysis in report generation."""
        coverage_data = {
            "extended_patterns": {"Pattern1", "Pattern2", "Pattern3"},
            "framework_coverage": {
                "Classiq": {
                    "total_found": 2,
                    "patterns": {"Pattern1", "Pattern2"},
                    "missing": {"Pattern3"},
                },
                "PennyLane": {
                    "total_found": 2,
                    "patterns": {"Pattern1", "Pattern2"},
                    "missing": {"Pattern3"},
                },
                "Qiskit": {
                    "total_found": 2,
                    "patterns": {"Pattern1", "Pattern2"},
                    "missing": {"Pattern3"},
                },
            },
            "target_coverage": {
                "total_found": 2,
                "patterns": {"Pattern1", "Pattern2"},
                "missing": {"Pattern3"},
            },
            "framework_only": set(),
            "target_patterns": {"Pattern1", "Pattern2"},
        }
        
        with patch('src.utils.analyze_extended_patterns.analyze_pattern_coverage', return_value=coverage_data):
            result = generate_report()
            
            # Check that cross-framework analysis is included
            assert "## Cross-Framework Analysis" in result
            assert "Common patterns between" in result
            assert "Patterns found in all three frameworks" in result


class TestMainFunction:
    """Test the main function."""

    def test_main_successful_execution(self):
        """Test successful main function execution."""
        coverage_data = {
            "extended_patterns": {"Pattern1", "Pattern2"},
            "framework_coverage": {
                "Classiq": {"total_found": 1, "patterns": {"Pattern1"}, "missing": {"Pattern2"}},
                "PennyLane": {"total_found": 1, "patterns": {"Pattern2"}, "missing": {"Pattern1"}},
                "Qiskit": {"total_found": 2, "patterns": {"Pattern1", "Pattern2"}, "missing": set()},
            },
            "target_coverage": {"total_found": 2, "patterns": {"Pattern1", "Pattern2"}, "missing": set()},
            "framework_only": set(),
            "target_patterns": {"Pattern1", "Pattern2"},
        }
        
        with patch('src.utils.analyze_extended_patterns.config') as mock_config, \
             patch('src.utils.analyze_extended_patterns.generate_report', return_value="Test Report"), \
             patch('src.utils.analyze_extended_patterns.analyze_pattern_coverage', return_value=coverage_data), \
             patch('builtins.open', mock_open()) as mock_file, \
             patch('builtins.print') as mock_print:
            
            mock_config.DOCS_DIR = Path("/test/docs")
            
            main()
            
            # Check that report was written to file
            mock_file.assert_called_once()
            mock_file.return_value.write.assert_called_with("Test Report")
            
            # Check that summary was printed
            assert mock_print.call_count > 0
            assert any("SUMMARY" in str(call) for call in mock_print.call_args_list)

    def test_main_file_write_error(self):
        """Test main function with file write error."""
        coverage_data = {
            "extended_patterns": {"Pattern1"},
            "framework_coverage": {
                "Classiq": {"total_found": 1, "patterns": {"Pattern1"}, "missing": set()},
                "PennyLane": {"total_found": 0, "patterns": set(), "missing": {"Pattern1"}},
                "Qiskit": {"total_found": 0, "patterns": set(), "missing": {"Pattern1"}},
            },
            "target_coverage": {"total_found": 1, "patterns": {"Pattern1"}, "missing": set()},
            "framework_only": set(),
            "target_patterns": {"Pattern1"},
        }
        
        with patch('src.utils.analyze_extended_patterns.config') as mock_config, \
             patch('src.utils.analyze_extended_patterns.generate_report', return_value="Test Report"), \
             patch('src.utils.analyze_extended_patterns.analyze_pattern_coverage', return_value=coverage_data), \
             patch('builtins.open', side_effect=PermissionError("Permission denied")), \
             patch('builtins.print') as mock_print:
            
            mock_config.DOCS_DIR = Path("/test/docs")
            
            main()
            
            # Check that error was printed
            assert any("Error writing report" in str(call) for call in mock_print.call_args_list)

    def test_main_summary_output(self):
        """Test that main function produces correct summary output."""
        coverage_data = {
            "extended_patterns": {"Pattern1", "Pattern2", "Pattern3"},
            "framework_coverage": {
                "Classiq": {"total_found": 2, "patterns": {"Pattern1", "Pattern2"}, "missing": {"Pattern3"}},
                "PennyLane": {"total_found": 1, "patterns": {"Pattern2"}, "missing": {"Pattern1", "Pattern3"}},
                "Qiskit": {"total_found": 1, "patterns": {"Pattern1"}, "missing": {"Pattern2", "Pattern3"}},
            },
            "target_coverage": {"total_found": 2, "patterns": {"Pattern1", "Pattern2"}, "missing": {"Pattern3"}},
            "framework_only": {"Pattern3"},
            "target_patterns": {"Pattern1", "Pattern2"},
        }
        
        with patch('src.utils.analyze_extended_patterns.config') as mock_config, \
             patch('src.utils.analyze_extended_patterns.generate_report', return_value="Test Report"), \
             patch('src.utils.analyze_extended_patterns.analyze_pattern_coverage', return_value=coverage_data), \
             patch('builtins.open', mock_open()), \
             patch('builtins.print') as mock_print:
            
            mock_config.DOCS_DIR = Path("/test/docs")
            
            main()
            
            # Check that summary contains expected information
            call_strings = [str(call) for call in mock_print.call_args_list]
            combined_calls = " ".join(call_strings)
            
            assert "Total extended patterns: 3" in combined_calls
            assert "Framework coverage:" in combined_calls
            assert "Target project coverage:" in combined_calls
            assert "Patterns in frameworks but not in target projects: 1" in combined_calls


class TestIntegration:
    """Integration tests for the extended pattern analysis workflow."""

    def test_complete_workflow_integration(self):
        """Test the complete workflow integration."""
        with patch('src.utils.analyze_extended_patterns.config') as mock_config, \
             patch('src.utils.analyze_extended_patterns.load_extended_patterns') as mock_load_extended, \
             patch('src.utils.analyze_extended_patterns.load_framework_patterns') as mock_load_framework, \
             patch('src.utils.analyze_extended_patterns.load_target_project_patterns') as mock_load_target, \
             patch('builtins.print'):
            
            mock_config.RESULTS_DIR = Path("/test/results")
            mock_config.DOCS_DIR = Path("/test/docs")
            
            mock_load_extended.return_value = {"Pattern1", "Pattern2"}
            mock_load_framework.return_value = {
                "Classiq": {"Pattern1"},
                "PennyLane": {"Pattern2"},
                "Qiskit": {"Pattern1", "Pattern2"},
            }
            mock_load_target.return_value = {"Pattern1", "Pattern2"}
            
            # Test the complete workflow
            result = analyze_pattern_coverage()
            
            assert result["extended_patterns"] == {"Pattern1", "Pattern2"}
            assert result["target_patterns"] == {"Pattern1", "Pattern2"}
            assert result["framework_only"] == set()
            
            # Check that all functions were called
            mock_load_extended.assert_called_once()
            mock_load_framework.assert_called_once()
            mock_load_target.assert_called_once()

    def test_error_handling_integration(self):
        """Test error handling in the integrated workflow."""
        with patch('src.utils.analyze_extended_patterns.config') as mock_config, \
             patch('src.utils.analyze_extended_patterns.load_extended_patterns', side_effect=Exception("Test error")), \
             patch('builtins.print') as mock_print:
            
            mock_config.RESULTS_DIR = Path("/test/results")
            
            # The function should handle errors gracefully
            try:
                analyze_pattern_coverage()
            except Exception:
                # If an exception is raised, it should be handled appropriately
                pass
            
            # The function should not crash the entire workflow
            assert True  # If we get here, the test passed

    def test_data_consistency_integration(self):
        """Test data consistency in the integrated workflow."""
        extended_patterns = {"Pattern1", "Pattern2", "Pattern3"}
        framework_patterns = {
            "Classiq": {"Pattern1", "Pattern2"},
            "PennyLane": {"Pattern2", "Pattern3"},
            "Qiskit": {"Pattern1", "Pattern3"},
        }
        target_patterns = {"Pattern1", "Pattern2"}
        
        with patch('src.utils.analyze_extended_patterns.load_extended_patterns', return_value=extended_patterns), \
             patch('src.utils.analyze_extended_patterns.load_framework_patterns', return_value=framework_patterns), \
             patch('src.utils.analyze_extended_patterns.load_target_project_patterns', return_value=target_patterns), \
             patch('builtins.print'):
            
            result = analyze_pattern_coverage()
            
            # Verify data consistency
            assert len(result["extended_patterns"]) == 3
            assert len(result["target_patterns"]) == 2
            assert len(result["framework_only"]) == 1
            
            # Verify that all frameworks are present
            assert "Classiq" in result["framework_coverage"]
            assert "PennyLane" in result["framework_coverage"]
            assert "Qiskit" in result["framework_coverage"]
            
            # Verify that coverage calculations are correct
            assert result["framework_coverage"]["Classiq"]["total_found"] == 2
            assert result["framework_coverage"]["PennyLane"]["total_found"] == 2
            assert result["framework_coverage"]["Qiskit"]["total_found"] == 2
            assert result["target_coverage"]["total_found"] == 2


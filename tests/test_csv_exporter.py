"""
Test suite for src/workflows/csv_exporter.py
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.workflows.csv_exporter import CSVExporter
from src.workflows.statistics_calculator import StatisticsCalculator


class TestCSVExporter:
    """Test the CSVExporter class."""

    def setup_method(self):
        """Set up test data."""
        self.sample_df = pd.DataFrame(
            {
                "concept_name": ["qiskit/circuit", "pennylane/device"],
                "file_path": ["project1/file1.py", "project2/file2.py"],
                "match_type": ["name", "semantic"],
                "similarity_score": [0.95, 0.87],
                "pattern": ["Pattern1", "Pattern2"],
                "framework": ["qiskit", "pennylane"],
                "project": ["project1", "project2"],
            }
        )
        self.sample_patterns = {"Pattern1", "Pattern2"}
        self.statistics = StatisticsCalculator(self.sample_df, self.sample_patterns)

    def test_initialization(self):
        """Test CSVExporter initialization."""
        output_dir = Path("/test/output")
        exporter = CSVExporter(output_dir)

        assert exporter.output_dir == output_dir

    def test_export_all_tables(self):
        """Test export_all_tables method."""
        output_dir = Path("/test/output")
        exporter = CSVExporter(output_dir)

        with patch("pathlib.Path.mkdir") as mock_mkdir:
            with patch("pandas.DataFrame.to_csv") as mock_to_csv:
                with patch(
                    "pathlib.Path.glob", return_value=["file1.csv", "file2.csv"]
                ):
                    with patch("builtins.print") as mock_print:
                        exporter.export_all_tables(self.statistics)

                        # Should create directory
                        mock_mkdir.assert_called_with(parents=True, exist_ok=True)

                        # Should call to_csv multiple times
                        assert mock_to_csv.call_count >= 5

                        # Should print success message
                        assert any(
                            "Successfully exported" in str(call)
                            for call in mock_print.call_args_list
                        )

    def test_export_basic_statistics(self):
        """Test _export_basic_statistics method."""
        output_dir = Path("/test/output")
        exporter = CSVExporter(output_dir)

        with patch("pandas.DataFrame.to_csv") as mock_to_csv:
            exporter._export_basic_statistics(self.statistics)

            # Should call to_csv for basic statistics
            assert mock_to_csv.call_count >= 4

    def test_export_pattern_statistics(self):
        """Test _export_pattern_statistics method."""
        output_dir = Path("/test/output")
        exporter = CSVExporter(output_dir)

        with patch("pandas.DataFrame.to_csv") as mock_to_csv:
            exporter._export_pattern_statistics(self.statistics)

            # Should call to_csv for pattern statistics
            assert mock_to_csv.call_count >= 4

    def test_export_additional_tables(self):
        """Test _export_additional_tables method."""
        output_dir = Path("/test/output")
        exporter = CSVExporter(output_dir)

        with patch("pandas.DataFrame.to_csv") as mock_to_csv:
            exporter._export_additional_tables(self.statistics)

            # Should call to_csv for additional tables
            assert mock_to_csv.call_count >= 1

    def test_export_with_empty_patterns(self):
        """Test export with empty patterns."""
        df_no_patterns = self.sample_df.copy()
        df_no_patterns["pattern"] = ""
        statistics_no_patterns = StatisticsCalculator(df_no_patterns, set())

        output_dir = Path("/test/output")
        exporter = CSVExporter(output_dir)

        with patch("pathlib.Path.mkdir"):
            with patch("pandas.DataFrame.to_csv") as mock_to_csv:
                with patch("pathlib.Path.glob", return_value=["file1.csv"]):
                    with patch("builtins.print"):
                        exporter.export_all_tables(statistics_no_patterns)

                        # Should still call to_csv for basic statistics
                        assert mock_to_csv.call_count >= 4

    def test_export_with_unmatched_patterns(self):
        """Test export with unmatched patterns."""
        # Add unmatched patterns
        self.statistics.unmatched_patterns = {"Pattern3", "Pattern4"}

        output_dir = Path("/test/output")
        exporter = CSVExporter(output_dir)

        with patch("pathlib.Path.mkdir"):
            with patch("pandas.DataFrame.to_csv") as mock_to_csv:
                with patch("pathlib.Path.glob", return_value=["file1.csv"]):
                    with patch("builtins.print"):
                        exporter.export_all_tables(self.statistics)

                        # Should call to_csv for unmatched patterns
                        assert mock_to_csv.call_count >= 6

    def test_export_directory_creation(self):
        """Test that export creates directory."""
        output_dir = Path("/test/output")
        exporter = CSVExporter(output_dir)

        with patch("pathlib.Path.mkdir") as mock_mkdir:
            with patch("pandas.DataFrame.to_csv"):
                with patch("pathlib.Path.glob", return_value=["file1.csv"]):
                    with patch("builtins.print"):
                        exporter.export_all_tables(self.statistics)

                        mock_mkdir.assert_called_with(parents=True, exist_ok=True)

    def test_export_file_naming(self):
        """Test that export uses correct file names."""
        output_dir = Path("/test/output")
        exporter = CSVExporter(output_dir)

        with patch("pathlib.Path.mkdir"):
            with patch("pandas.DataFrame.to_csv") as mock_to_csv:
                with patch("pathlib.Path.glob", return_value=["file1.csv"]):
                    with patch("builtins.print"):
                        exporter.export_all_tables(self.statistics)

                        # Check that specific file names are used
                        call_args = [call[0][0] for call in mock_to_csv.call_args_list]
                        assert any(
                            "match_type_counts.csv" in str(arg) for arg in call_args
                        )
                        assert any(
                            "avg_score_by_type.csv" in str(arg) for arg in call_args
                        )
                        assert any(
                            "matches_by_framework.csv" in str(arg) for arg in call_args
                        )
                        assert any(
                            "matches_by_project.csv" in str(arg) for arg in call_args
                        )




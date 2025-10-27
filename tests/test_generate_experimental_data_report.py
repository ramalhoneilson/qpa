"""
Test suite for src/utils/generate_experimental_data_report.py
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pandas as pd
import pytest

from src.reporting.experimental_data_report import ExperimentalDataReportGenerator


class TestExperimentalDataReportGenerator:
    """Test the ExperimentalDataReportGenerator class."""

    def setup_method(self):
        """Set up test data."""
        self.generator = ExperimentalDataReportGenerator()

    def test_initialization(self):
        """Test ExperimentalDataReportGenerator initialization."""
        assert self.generator.data_dir is not None
        assert self.generator.report_dir is not None
        assert self.generator.output_file is not None
        assert len(self.generator.concept_files) == 3
        assert len(self.generator.report_files) == 4

    def test_generate_header(self):
        """Test _generate_header method."""
        header = self.generator._generate_header()

        assert len(header) > 0
        assert "# Experimental Data" in header
        assert "## Overview" in header
        assert "Framework Concept Extractions" in " ".join(header)

    def test_generate_framework_table_success(self):
        """Test _generate_framework_table with successful data."""
        mock_df = pd.DataFrame(
            {"name": ["concept1", "concept2"], "summary": ["summary1", "summary2"]}
        )

        with patch("pandas.read_csv", return_value=mock_df):
            with patch("pathlib.Path.exists", return_value=True):
                result = self.generator._generate_framework_table(
                    "TestFramework", Path("test.csv")
                )

                assert "## TestFramework Quantum Concepts" in result
                assert "**Total Concepts**: 2" in result
                assert "concept1" in " ".join(result)
                assert "Row" in " ".join(result)  # Check for row numbers

    def test_generate_framework_table_file_not_found(self):
        """Test _generate_framework_table when file doesn't exist."""
        with patch("pathlib.Path.exists", return_value=False):
            with patch(
                "pandas.read_csv", side_effect=FileNotFoundError("No such file")
            ):
                result = self.generator._generate_framework_table(
                    "TestFramework", Path("nonexistent.csv")
                )

                assert (
                    " **Error**: Could not read TestFramework concepts file"
                    in " ".join(result)
                )

    def test_generate_framework_table_error(self):
        """Test _generate_framework_table with error."""
        with patch("pandas.read_csv", side_effect=Exception("Read error")):
            with patch("pathlib.Path.exists", return_value=True):
                result = self.generator._generate_framework_table(
                    "TestFramework", Path("test.csv")
                )

                assert (
                    " **Error**: Could not read TestFramework concepts file"
                    in " ".join(result)
                )

    def test_generate_framework_table_qiskit_delimiter(self):
        """Test _generate_framework_table with Qiskit delimiter."""
        mock_df = pd.DataFrame(
            {"name": ["qiskit.circuit"], "summary": ["Quantum circuit"]}
        )

        with patch("pandas.read_csv") as mock_read_csv:
            mock_read_csv.return_value = mock_df
            with patch("pathlib.Path.exists", return_value=True):
                result = self.generator._generate_framework_table(
                    "Qiskit", Path("test.csv")
                )

                # Should use semicolon delimiter for Qiskit
                mock_read_csv.assert_called_with(Path("test.csv"), delimiter=";")
                assert "## Qiskit Quantum Concepts" in result

    def test_generate_top_concepts_table_success(self):
        """Test _generate_top_concepts_table with successful data."""
        mock_df = pd.DataFrame(
            {
                "Framework": ["qiskit", "pennylane"],
                "Concept": ["circuit", "device"],
                "Matches": [10, 8],
            }
        )

        with patch("pandas.read_csv", return_value=mock_df):
            with patch("pathlib.Path.exists", return_value=True):
                result = self.generator._generate_top_concepts_table()

                assert "### Top Matched Quantum Concepts" in result
                assert "**Total Concepts**: 2" in result
                assert "qiskit" in " ".join(result)

    def test_generate_top_concepts_table_error(self):
        """Test _generate_top_concepts_table with error."""
        with patch("pandas.read_csv", side_effect=Exception("Read error")):
            with patch("pathlib.Path.exists", return_value=True):
                result = self.generator._generate_top_concepts_table()

                assert " **Error**: Could not read top concepts file" in " ".join(
                    result
                )

    def test_generate_match_type_table_success(self):
        """Test _generate_match_type_table with successful data."""
        mock_df = pd.DataFrame({"match_type": ["name", "semantic"], "count": [50, 30]})

        with patch("pandas.read_csv", return_value=mock_df):
            with patch("pathlib.Path.exists", return_value=True):
                result = self.generator._generate_match_type_table()

                assert "### Match Type Analysis" in result
                assert "**Total Match Types**: 2" in result
                assert "name" in " ".join(result)

    def test_generate_framework_analysis_table_success(self):
        """Test _generate_framework_analysis_table with successful data."""
        mock_df = pd.DataFrame(
            {"framework": ["qiskit", "pennylane"], "count": [100, 80]}
        )

        with patch("pandas.read_csv", return_value=mock_df):
            with patch("pathlib.Path.exists", return_value=True):
                result = self.generator._generate_framework_analysis_table()

                assert "### Framework Analysis" in result
                assert "**Frameworks**: 2" in result
                assert "qiskit" in " ".join(result)

    def test_generate_pattern_frequency_table_success(self):
        """Test _generate_pattern_frequency_table with successful data."""
        mock_df = pd.DataFrame({"pattern": ["Pattern1", "Pattern2"], "count": [15, 12]})

        with patch("pandas.read_csv", return_value=mock_df):
            with patch("pathlib.Path.exists", return_value=True):
                result = self.generator._generate_pattern_frequency_table()

                assert "### Pattern Frequency Analysis" in result
                assert "**Total Patterns**: 2" in result
                assert "Pattern1" in " ".join(result)

    def test_generate_pattern_atlas_section_success(self):
        """Test _generate_pattern_atlas_section with successful data."""
        mock_patterns = [
            {"name": "Pattern1", "alias": "P1", "intent": "Test intent for pattern 1"},
            {"name": "Pattern2", "alias": "P2", "intent": "Test intent for pattern 2"},
        ]

        with patch("pathlib.Path.exists", return_value=True):
            with patch("builtins.open", mock_open(read_data=json.dumps(mock_patterns))):
                with patch("json.load", return_value=mock_patterns):
                    result = self.generator._generate_pattern_atlas_section()

                    assert "## Quantum Patterns from PlanQK Pattern Atlas" in result
                    assert "**Total Patterns**: 2" in result
                    assert "Pattern1" in " ".join(result)

    def test_generate_pattern_atlas_section_file_not_found(self):
        """Test _generate_pattern_atlas_section when file doesn't exist."""
        with patch("pathlib.Path.exists", return_value=False):
            result = self.generator._generate_pattern_atlas_section()

            assert "**Note**: The patterns file was not found" in " ".join(result)

    def test_generate_pattern_atlas_section_error(self):
        """Test _generate_pattern_atlas_section with error."""
        with patch("pathlib.Path.exists", return_value=True):
            with patch("builtins.open", side_effect=Exception("File error")):
                result = self.generator._generate_pattern_atlas_section()

                assert " **Error**: Could not read patterns file" in " ".join(result)

    def test_generate_references(self):
        """Test _generate_references method."""
        references = self.generator._generate_references()

        assert "## References" in references
        assert "PlanQK" in " ".join(references)
        assert "Quantum Computing Patterns" in " ".join(references)

    def test_generate_report_success(self):
        """Test generate_report method with successful execution."""
        with patch("pathlib.Path.mkdir"):
            with patch("builtins.open", mock_open()) as mock_file:
                with patch.object(
                    self.generator, "_generate_header", return_value=["# Header"]
                ):
                    with patch.object(
                        self.generator,
                        "_generate_framework_tables",
                        return_value=["## Framework"],
                    ):
                        with patch.object(
                            self.generator,
                            "_generate_pattern_analysis_tables",
                            return_value=["## Pattern"],
                        ):
                            with patch.object(
                                self.generator,
                                "_generate_pattern_atlas_section",
                                return_value=["## Atlas"],
                            ):
                                with patch.object(
                                    self.generator,
                                    "_generate_references",
                                    return_value=["## References"],
                                ):
                                    with patch("builtins.print"):
                                        self.generator.generate_report()

                                        # Should have called open for writing
                                        mock_file.assert_called()

    def test_generate_report_directory_creation(self):
        """Test that generate_report creates output directory."""
        with patch("pathlib.Path.mkdir") as mock_mkdir:
            with patch("builtins.open", mock_open()):
                with patch.object(
                    self.generator, "_generate_header", return_value=["# Header"]
                ):
                    with patch.object(
                        self.generator,
                        "_generate_framework_tables",
                        return_value=["## Framework"],
                    ):
                        with patch.object(
                            self.generator,
                            "_generate_pattern_analysis_tables",
                            return_value=["## Pattern"],
                        ):
                            with patch.object(
                                self.generator,
                                "_generate_pattern_atlas_section",
                                return_value=["## Atlas"],
                            ):
                                with patch.object(
                                    self.generator,
                                    "_generate_references",
                                    return_value=["## References"],
                                ):
                                    with patch("builtins.print"):
                                        self.generator.generate_report()

                                        # Should create directory
                                        mock_mkdir.assert_called_with(
                                            parents=True, exist_ok=True
                                        )

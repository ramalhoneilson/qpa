"""
Test suite for src/workflows/data_processor.py
"""

from pathlib import Path
from unittest.mock import mock_open, patch

import pandas as pd
import pytest

from src.workflows.data_processor import DataProcessor


class TestDataProcessor:
    """Test the DataProcessor class."""

    def test_initialization(self):
        """Test DataProcessor initialization."""
        csv_file = Path("test.csv")
        pattern_files = [Path("pattern1.csv"), Path("pattern2.csv")]

        processor = DataProcessor(csv_file, pattern_files)

        assert processor.csv_file == csv_file
        assert processor.pattern_files == pattern_files

    def test_load_main_data_success(self):
        """Test successful loading of main data."""
        mock_df = pd.DataFrame(
            {
                "concept_name": ["qiskit/circuit", "pennylane/device"],
                "file_path": ["project1/file1.py", "project2/file2.py"],
                "match_type": ["name", "semantic"],
                "similarity_score": [0.95, 0.87],
            }
        )

        with patch("pathlib.Path.exists", return_value=True):
            with patch("pandas.read_csv", return_value=mock_df):
                processor = DataProcessor(Path("test.csv"), [])
                result = processor.load_main_data()

                # Check that framework and project columns were added
                assert "framework" in result.columns
                assert "project" in result.columns
                assert result["framework"].iloc[0] == "qiskit"
                assert result["project"].iloc[0] == "project1"

    def test_load_main_data_file_not_found(self):
        """Test loading when file doesn't exist."""
        with patch("pathlib.Path.exists", return_value=False):
            processor = DataProcessor(Path("nonexistent.csv"), [])

            with pytest.raises(FileNotFoundError):
                processor.load_main_data()

    def test_load_main_data_empty_file(self):
        """Test loading when file is empty."""
        with patch("pathlib.Path.exists", return_value=True):
            with patch("pandas.read_csv", side_effect=pd.errors.EmptyDataError()):
                processor = DataProcessor(Path("empty.csv"), [])

                with pytest.raises(pd.errors.EmptyDataError):
                    processor.load_main_data()

    def test_load_patterns_success(self):
        """Test successful loading of patterns."""
        mock_pattern_files = [Path("pattern1.csv"), Path("pattern2.csv")]

        # Mock CSV content
        mock_csv_content_1 = (
            "pattern,description\nPattern1,Description1\nPattern2,Description2\n"
        )
        mock_csv_content_2 = (
            "pattern,description\nPattern3,Description3\nPattern4,Description4\n"
        )

        with patch("pathlib.Path.exists", return_value=True):
            with patch("builtins.open", mock_open(read_data=mock_csv_content_1)):
                with patch("csv.DictReader") as mock_reader:
                    mock_reader.return_value = [
                        {"pattern": "Pattern1", "description": "Description1"},
                        {"pattern": "Pattern2", "description": "Description2"},
                    ]
                    processor = DataProcessor(Path("test.csv"), mock_pattern_files)
                    result = processor.load_patterns()

                    assert "Pattern1" in result
                    assert "Pattern2" in result

    def test_load_patterns_missing_file(self):
        """Test loading patterns when file is missing."""
        mock_pattern_files = [Path("missing.csv")]

        with patch("pathlib.Path.exists", return_value=False):
            processor = DataProcessor(Path("test.csv"), mock_pattern_files)
            result = processor.load_patterns()

            assert result == set()

    def test_load_patterns_error(self):
        """Test loading patterns when file has errors."""
        mock_pattern_files = [Path("error.csv")]

        with patch("pathlib.Path.exists", return_value=True):
            with patch("builtins.open", side_effect=Exception("File error")):
                with patch("builtins.print") as mock_print:
                    processor = DataProcessor(Path("test.csv"), mock_pattern_files)
                    result = processor.load_patterns()

                    assert result == set()
                    mock_print.assert_called()

    def test_extract_framework(self):
        """Test _extract_framework method."""
        # Test normal case
        assert DataProcessor._extract_framework("qiskit/circuit") == "qiskit"
        assert DataProcessor._extract_framework("pennylane/device") == "pennylane"

        # Test edge cases
        assert DataProcessor._extract_framework("") == "unknown"
        assert (
            DataProcessor._extract_framework("invalid") == "invalid"
        )  # No slash, returns as-is
        assert DataProcessor._extract_framework(None) == "unknown"

    def test_extract_project(self):
        """Test _extract_project method."""
        # Test normal case
        assert DataProcessor._extract_project("project1/file1.py") == "project1"
        assert DataProcessor._extract_project("project2/subdir/file2.py") == "project2"

        # Test edge cases
        assert DataProcessor._extract_project("") == ""
        assert DataProcessor._extract_project("file.py") == "file.py"
        assert DataProcessor._extract_project(None) == ""

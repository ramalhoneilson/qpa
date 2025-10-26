"""
Test suite for src/utils/generate_base_concept_report.py
"""

import csv
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest

from src.utils.generate_base_concept_report import (
    INPUT_FILES,
    OUTPUT_MD_PATH,
    generate_markdown_table,
    main,
    read_concepts_from_csv,
)


class TestReadConceptsFromCsv:
    """Test the read_concepts_from_csv function."""

    def test_read_concepts_success(self):
        """Test successful reading of concepts from CSV."""
        mock_csv_content = "name,summary\nconcept1,summary1\nconcept2,summary2\n"

        with patch("pathlib.Path.exists", return_value=True):
            with patch("builtins.open", mock_open(read_data=mock_csv_content)):
                with patch("csv.DictReader") as mock_reader:
                    mock_reader.return_value = [
                        {"name": "concept1", "summary": "summary1"},
                        {"name": "concept2", "summary": "summary2"},
                    ]
                    result = read_concepts_from_csv(Path("test.csv"), ",")
                    assert len(result) == 2
                    assert result[0]["name"] == "concept1"
                    assert result[1]["summary"] == "summary2"

    def test_read_concepts_file_not_found(self):
        """Test reading when file doesn't exist."""
        with patch("pathlib.Path.exists", return_value=False):
            with patch("builtins.print") as mock_print:
                result = read_concepts_from_csv(Path("nonexistent.csv"), ",")
                assert result == []
                mock_print.assert_called_with(
                    "  - Warning: Input file not found: nonexistent.csv"
                )

    def test_read_concepts_error(self):
        """Test handling errors when reading CSV."""
        with patch("pathlib.Path.exists", return_value=True):
            with patch("builtins.open", side_effect=Exception("File error")):
                with patch("builtins.print") as mock_print:
                    result = read_concepts_from_csv(Path("test.csv"), ",")
                    assert result == []
                    mock_print.assert_called_with(
                        "  - Error reading test.csv: File error"
                    )


class TestGenerateMarkdownTable:
    """Test the generate_markdown_table function."""

    def test_generate_table_with_concepts(self):
        """Test generating markdown table with concepts."""
        concepts = [
            {"name": "concept1", "summary": "summary1"},
            {"name": "concept2", "summary": "summary2"},
        ]

        result = generate_markdown_table(concepts)

        assert "| Concept Name | Summary |" in result
        assert "|--------------|---------|" in result
        assert "| `concept1` | summary1 |" in result
        assert "| `concept2` | summary2 |" in result

    def test_generate_table_empty_concepts(self):
        """Test generating table with empty concepts list."""
        result = generate_markdown_table([])
        assert result == "*No concepts were extracted for this framework.*\n"

    def test_generate_table_missing_keys(self):
        """Test generating table with missing keys in concepts."""
        concepts = [
            {"name": "concept1"},  # Missing summary
            {"summary": "summary2"},  # Missing name
        ]

        result = generate_markdown_table(concepts)

        assert "| `concept1` |  |" in result
        assert "| `N/A` | summary2 |" in result

    def test_generate_table_with_pipes(self):
        """Test generating table with pipe characters in content."""
        concepts = [{"name": "concept|with|pipes", "summary": "summary|with|pipes"}]

        result = generate_markdown_table(concepts)

        assert "| `concept\\|with\\|pipes` | summary\\|with\\|pipes |" in result

    def test_generate_table_with_newlines(self):
        """Test generating table with newlines in summary."""
        concepts = [{"name": "concept1", "summary": "summary\nwith\nnewlines"}]

        result = generate_markdown_table(concepts)

        assert "| `concept1` | summary with newlines |" in result


class TestMainFunction:
    """Test the main function."""

    def test_main_successful_execution(self):
        """Test successful main execution."""
        mock_concepts = [
            {"name": "concept1", "summary": "summary1"},
            {"name": "concept2", "summary": "summary2"},
        ]

        with patch(
            "src.utils.generate_base_concept_report.read_concepts_from_csv",
            return_value=mock_concepts,
        ):
            with patch("builtins.open", mock_open()) as mock_file:
                with patch("builtins.print"):
                    main()

                    # Should have called open for writing
                    mock_file.assert_called()

    def test_main_with_missing_files(self):
        """Test main function when some files are missing."""

        def mock_read_concepts(path, delimiter):
            if "qiskit" in str(path):
                return [{"name": "qiskit_concept", "summary": "qiskit_summary"}]
            return []  # Missing files return empty list

        with patch(
            "src.utils.generate_base_concept_report.read_concepts_from_csv",
            side_effect=mock_read_concepts,
        ):
            with patch("builtins.open", mock_open()) as mock_file:
                with patch("builtins.print"):
                    main()

                    # Should have called open for writing
                    mock_file.assert_called()

    def test_main_write_error(self):
        """Test main function when writing fails."""
        mock_concepts = [{"name": "concept1", "summary": "summary1"}]

        with patch(
            "src.utils.generate_base_concept_report.read_concepts_from_csv",
            return_value=mock_concepts,
        ):
            with patch("builtins.open", side_effect=OSError("Write error")):
                with patch("builtins.print") as mock_print:
                    main()

                    # Should have printed error message
                    assert any(
                        "Could not write to file" in str(call)
                        for call in mock_print.call_args_list
                    )


class TestConstants:
    """Test module constants."""

    def test_input_files_structure(self):
        """Test INPUT_FILES constant structure."""
        assert "Qiskit" in INPUT_FILES
        assert "PennyLane" in INPUT_FILES
        assert "Classiq" in INPUT_FILES

        for framework, details in INPUT_FILES.items():
            assert "path" in details
            assert "delimiter" in details
            assert details["delimiter"] in [",", ";"]

    def test_output_md_path(self):
        """Test OUTPUT_MD_PATH constant."""
        assert OUTPUT_MD_PATH.name == "extracted_concepts_summary.md"

    def test_qiskit_config(self):
        """Test Qiskit configuration."""
        qiskit_config = INPUT_FILES["Qiskit"]
        assert qiskit_config["delimiter"] == ";"
        assert "qiskit_quantum_concepts.csv" in str(qiskit_config["path"])

    def test_pennylane_config(self):
        """Test PennyLane configuration."""
        pennylane_config = INPUT_FILES["PennyLane"]
        assert pennylane_config["delimiter"] == ","
        assert "pennylane_quantum_concepts.csv" in str(pennylane_config["path"])

    def test_classiq_config(self):
        """Test Classiq configuration."""
        classiq_config = INPUT_FILES["Classiq"]
        assert classiq_config["delimiter"] == ","
        assert "classiq_quantum_concepts.csv" in str(classiq_config["path"])


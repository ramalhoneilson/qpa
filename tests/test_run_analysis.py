"""
Test suite for src/analysis/run_analysis.py
"""

import csv
import json
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest

from src.analysis.run_analysis import (
    CONCEPT_FILES,
    NOTEBOOKS_ROOT_DIR,
    OUTPUT_CSV_FILE,
    PATTERN_FILES,
    SIMILARITY_THRESHOLDS,
    UNCLASSIFIED_CONCEPTS_FILE,
    CodeElementVisitor,
    _save_unclassified_concepts,
    extract_comments_from_script,
    extract_short_name,
    get_code_elements_from_script,
    load_patterns_map,
    load_quantum_concepts,
    main,
)


class TestCodeElementVisitor:
    """Test the CodeElementVisitor class."""

    def test_visitor_initialization(self):
        """Test visitor initialization."""
        visitor = CodeElementVisitor()
        assert visitor.found_elements == set()

    def test_visit_call_with_name(self):
        """Test visiting function calls with names."""
        visitor = CodeElementVisitor()

        # Create a real AST node for function call
        import ast

        tree = ast.parse("test_function()")
        call_node = tree.body[0].value  # This is the Call node

        # Mock generic_visit to avoid recursion
        with patch.object(visitor, "generic_visit"):
            visitor.visit_Call(call_node)
            assert "test_function" in visitor.found_elements

    def test_visit_call_with_attribute(self):
        """Test visiting function calls with attributes."""
        visitor = CodeElementVisitor()

        # Create a real AST node for method call
        import ast

        tree = ast.parse("obj.test_method()")
        call_node = tree.body[0].value  # This is the Call node

        # Mock generic_visit to avoid recursion
        with patch.object(visitor, "generic_visit"):
            visitor.visit_Call(call_node)
            assert "test_method" in visitor.found_elements


class TestHelperFunctions:
    """Test helper functions."""

    def test_get_code_elements_from_script(self):
        """Test extracting code elements from script."""
        script_content = "def test_func():\n    return test_function()"

        with patch("ast.parse") as mock_parse:
            mock_tree = MagicMock()
            mock_parse.return_value = mock_tree

            with patch(
                "src.analysis.run_analysis.CodeElementVisitor"
            ) as mock_visitor_class:
                mock_visitor = MagicMock()
                mock_visitor.found_elements = {"test_function"}
                mock_visitor_class.return_value = mock_visitor

                result = get_code_elements_from_script(script_content)
                assert result == ["test_function"]

    def test_get_code_elements_syntax_error(self):
        """Test handling syntax errors in script parsing."""
        script_content = "invalid syntax here"

        with patch("ast.parse", side_effect=SyntaxError("Invalid syntax")):
            result = get_code_elements_from_script(script_content)
            assert result == []

    def test_extract_comments_from_script(self):
        """Test extracting comments from script."""
        mock_file_content = [
            "# This is a comment",
            "def test():",
            "# Another comment",
            "    pass",
        ]

        with patch("builtins.open", mock_open(read_data="\n".join(mock_file_content))):
            result = extract_comments_from_script(Path("test.py"))
            assert "This is a comment" in result
            assert "Another comment" in result

    def test_extract_comments_file_error(self):
        """Test handling file reading errors."""
        with patch("builtins.open", side_effect=Exception("File error")):
            with patch("builtins.print") as mock_print:
                result = extract_comments_from_script(Path("test.py"))
                assert result == ""
                mock_print.assert_called()

    def test_extract_short_name(self):
        """Test extracting short names."""
        assert extract_short_name("qiskit/circuit/QuantumCircuit") == "QuantumCircuit"
        assert extract_short_name("pennylane/quantum") == "quantum"
        assert extract_short_name("simple") == "simple"
        assert extract_short_name("") == ""

    def test_load_patterns_map(self):
        """Test loading patterns map from CSV files."""
        mock_csv_content = [
            "concept_name,summary,pattern",
            "concept1,summary1,pattern1",
            "concept2,summary2,pattern2",
        ]

        with patch("pathlib.Path.exists", return_value=True):
            with patch(
                "builtins.open", mock_open(read_data="\n".join(mock_csv_content))
            ):
                result = load_patterns_map([Path("test.csv")])
                assert result["concept1"] == "pattern1"
                assert result["concept2"] == "pattern2"

    def test_load_patterns_map_missing_file(self):
        """Test loading patterns when file doesn't exist."""
        with patch("pathlib.Path.exists", return_value=False):
            with patch("builtins.print") as mock_print:
                result = load_patterns_map([Path("nonexistent.csv")])
                assert result == {}
                mock_print.assert_called_with(
                    "Warning: Pattern file not found: nonexistent.csv"
                )

    def test_load_patterns_map_error(self):
        """Test handling errors in pattern loading."""
        with patch("pathlib.Path.exists", return_value=True):
            with patch("builtins.open", side_effect=Exception("File error")):
                with patch("builtins.print") as mock_print:
                    result = load_patterns_map([Path("test.csv")])
                    assert result == {}
                    mock_print.assert_called()

    def test_load_quantum_concepts(self):
        """Test loading quantum concepts from JSON files."""
        mock_json_data = [
            {"name": "concept1", "summary": "summary1"},
            {"name": "concept2", "summary": "summary2"},
        ]

        pattern_map = {"concept1": "pattern1"}

        with patch("pathlib.Path.exists", return_value=True):
            with patch(
                "builtins.open", mock_open(read_data=json.dumps(mock_json_data))
            ):
                result = load_quantum_concepts([Path("test.json")], pattern_map)
                assert len(result) == 2
                assert result[0]["name"] == "concept1"
                assert result[0]["pattern"] == "pattern1"
                assert result[1]["pattern"] == "N/A"

    def test_load_quantum_concepts_missing_file(self):
        """Test loading concepts when file doesn't exist."""
        with patch("pathlib.Path.exists", return_value=False):
            result = load_quantum_concepts([Path("nonexistent.json")], {})
            assert result == []

    def test_load_quantum_concepts_error(self):
        """Test handling errors in concept loading."""
        with patch("pathlib.Path.exists", return_value=True):
            with patch("builtins.open", side_effect=Exception("File error")):
                with patch("builtins.print") as mock_print:
                    result = load_quantum_concepts([Path("test.json")], {})
                    assert result == []
                    mock_print.assert_called()

    def test_save_unclassified_concepts_empty(self):
        """Test saving unclassified concepts when none exist."""
        concepts = [{"name": "concept1", "pattern": "pattern1"}]

        with patch("pathlib.Path.exists", return_value=True):
            with patch("pathlib.Path.unlink"):
                with patch("builtins.print") as mock_print:
                    _save_unclassified_concepts(concepts, Path("test.csv"))
                    mock_print.assert_called_with(
                        "All concepts are classified. No 'unclassified_concepts.csv' needed."
                    )

    def test_save_unclassified_concepts_with_data(self):
        """Test saving unclassified concepts when they exist."""
        concepts = [
            {"name": "concept1", "summary": "summary1", "pattern": "N/A"},
            {"name": "concept2", "summary": "summary2", "pattern": "pattern2"},
        ]

        with patch("builtins.open", mock_open()) as mock_file:
            with patch("builtins.print") as mock_print:
                _save_unclassified_concepts(concepts, Path("test.csv"))
                mock_file.assert_called()
                # Check that the warning message was printed
                assert any(
                    "Found 1 unclassified concepts" in str(call)
                    for call in mock_print.call_args_list
                )

    def test_save_unclassified_concepts_error(self):
        """Test handling errors in saving unclassified concepts."""
        concepts = [{"name": "concept1", "summary": "summary1", "pattern": "N/A"}]

        with patch("builtins.open", side_effect=OSError("File error")):
            with patch("builtins.print") as mock_print:
                _save_unclassified_concepts(concepts, Path("test.csv"))
                mock_print.assert_called_with(
                    "  - Error writing unclassified concepts file: File error"
                )


class TestMainFunction:
    """Test the main function."""

    def test_main_no_concepts_loaded(self):
        """Test main function when no concepts are loaded."""
        with patch(
            "src.analysis.run_analysis.load_patterns_map", return_value={}
        ):
            with patch(
                "src.analysis.run_analysis.load_quantum_concepts", return_value=[]
            ):
                with patch("builtins.print") as mock_print:
                    main()
                    mock_print.assert_called_with(
                        "No quantum concepts loaded. Exiting."
                    )

    def test_main_successful_execution(self):
        """Test successful main execution."""
        mock_concepts = [
            {
                "name": "concept1",
                "summary": "summary1",
                "short_name": "concept1",
                "pattern": "pattern1",
            }
        ]

        with patch(
            "src.analysis.run_analysis.load_patterns_map",
            return_value={"concept1": "pattern1"},
        ):
            with patch(
                "src.analysis.run_analysis.load_quantum_concepts",
                return_value=mock_concepts,
            ):
                with patch(
                    "src.analysis.run_analysis._save_unclassified_concepts"
                ):
                    with patch(
                        "src.analysis.run_analysis.SentenceTransformer"
                    ) as mock_model_class:
                        mock_model = MagicMock()
                        mock_model_class.return_value = mock_model
                        mock_model.encode.return_value = MagicMock()

                        with patch(
                            "pathlib.Path.rglob", return_value=[Path("test.py")]
                        ):
                            with patch(
                                "pathlib.Path.read_text",
                                return_value="def test(): pass",
                            ):
                                with patch("builtins.open", mock_open()) as mock_file:
                                    with patch("builtins.print"):
                                        main()

                                        # Should have processed the file
                                        mock_file.assert_called()

    def test_main_file_reading_error(self):
        """Test main function when file reading fails."""
        mock_concepts = [
            {
                "name": "concept1",
                "summary": "summary1",
                "short_name": "concept1",
                "pattern": "pattern1",
            }
        ]

        with patch(
            "src.analysis.run_analysis.load_patterns_map", return_value={}
        ):
            with patch(
                "src.analysis.run_analysis.load_quantum_concepts",
                return_value=mock_concepts,
            ):
                with patch(
                    "src.analysis.run_analysis._save_unclassified_concepts"
                ):
                    with patch(
                        "src.analysis.run_analysis.SentenceTransformer"
                    ) as mock_model_class:
                        mock_model = MagicMock()
                        mock_model_class.return_value = mock_model
                        mock_model.encode.return_value = MagicMock()

                        with patch(
                            "pathlib.Path.rglob", return_value=[Path("test.py")]
                        ):
                            with patch(
                                "pathlib.Path.read_text",
                                side_effect=Exception("Read error"),
                            ):
                                with patch("builtins.print") as mock_print:
                                    main()
                                    # Should handle the error gracefully
                                    assert any(
                                        "Could not read file" in str(call)
                                        for call in mock_print.call_args_list
                                    )


class TestConstants:
    """Test module constants."""

    def test_notebooks_root_dir(self):
        """Test NOTEBOOKS_ROOT_DIR constant."""
        assert NOTEBOOKS_ROOT_DIR.name == "converted_notebooks"

    def test_output_csv_file(self):
        """Test OUTPUT_CSV_FILE constant."""
        assert OUTPUT_CSV_FILE.name == "quantum_concept_matches_with_patterns.csv"

    def test_unclassified_concepts_file(self):
        """Test UNCLASSIFIED_CONCEPTS_FILE constant."""
        assert UNCLASSIFIED_CONCEPTS_FILE.name == "unclassified_concepts.csv"

    def test_similarity_thresholds(self):
        """Test SIMILARITY_THRESHOLDS constant."""
        assert SIMILARITY_THRESHOLDS["name"] == 0.90
        assert SIMILARITY_THRESHOLDS["summary"] == 0.65

    def test_concept_files(self):
        """Test CONCEPT_FILES constant."""
        assert len(CONCEPT_FILES) == 3
        assert all("quantum_concepts.json" in str(f) for f in CONCEPT_FILES)

    def test_pattern_files(self):
        """Test PATTERN_FILES constant."""
        assert len(PATTERN_FILES) == 3
        assert all("enriched" in str(f) for f in PATTERN_FILES)

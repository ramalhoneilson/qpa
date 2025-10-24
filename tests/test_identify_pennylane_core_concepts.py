"""
Test cases for identify_pennylane_core_concepts.py module.

This module tests the PennyLane core concepts identification functionality.
"""

import ast
import tempfile
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

# Import the module under test
import sys

sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.core_concepts.identify_pennylane_core_concepts import (
    _PennylaneConceptVisitor,
    _find_concepts_in_file,
    extract_pennylane_concepts,
    _save_source_code_snippets,
    _save_concepts_to_csv,
    main,
    PENNYLANE_PROJECT_ROOT,
    SOURCE_SNIPPETS_DIR,
    OUTPUT_JSON_PATH,
    SEARCH_SUBDIRS,
)


class TestPennylaneConceptVisitor:
    """Test cases for _PennylaneConceptVisitor class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.source_text = '''
class QuantumTemplate:
    """A quantum template for quantum circuits."""
    def __init__(self):
        pass

class AnotherClass:
    pass
'''
        self.sdk_root = Path("/test/pennylane")
        self.file_path = self.sdk_root / "templates" / "test_file.py"

    def test_visitor_initialization(self):
        """Test visitor initialization."""
        visitor = _PennylaneConceptVisitor(
            self.source_text, self.file_path, self.sdk_root
        )
        assert visitor.source_text == self.source_text
        assert visitor.file_path == self.file_path
        assert visitor.sdk_root == self.sdk_root
        assert visitor.found_concepts == {}

    def test_visit_class_with_docstring(self):
        """Test visiting a class with docstring."""
        visitor = _PennylaneConceptVisitor(
            self.source_text, self.file_path, self.sdk_root
        )
        tree = ast.parse(self.source_text)
        visitor.visit(tree)

        assert len(visitor.found_concepts) == 1
        concept_name = list(visitor.found_concepts.keys())[0]
        assert "QuantumTemplate" in concept_name
        concept = visitor.found_concepts[concept_name]
        assert concept["docstring"] == "A quantum template for quantum circuits."
        assert "class QuantumTemplate:" in concept["source_code"]

    def test_visit_class_without_docstring(self):
        """Test that classes without docstrings are ignored."""
        visitor = _PennylaneConceptVisitor(
            self.source_text, self.file_path, self.sdk_root
        )
        tree = ast.parse(self.source_text)
        visitor.visit(tree)

        # Only the class with docstring should be found
        assert len(visitor.found_concepts) == 1
        concept_name = list(visitor.found_concepts.keys())[0]
        assert "AnotherClass" not in concept_name

    def test_concept_name_formatting(self):
        """Test that concept names are formatted correctly."""
        visitor = _PennylaneConceptVisitor(
            self.source_text, self.file_path, self.sdk_root
        )
        tree = ast.parse(self.source_text)
        visitor.visit(tree)

        concept_name = list(visitor.found_concepts.keys())[0]
        assert concept_name.startswith("/pennylane/")
        assert "templates.test_file.QuantumTemplate" in concept_name


class TestFindConceptsInFile:
    """Test cases for _find_concepts_in_file function."""

    def test_successful_parsing(self):
        """Test successful file parsing."""
        source_text = '''
class QuantumTemplate:
    """A quantum template for quantum circuits."""
    def __init__(self):
        pass
'''
        with tempfile.TemporaryDirectory() as temp_dir:
            sdk_root = Path(temp_dir) / "pennylane"
            file_path = sdk_root / "templates" / "test_file.py"
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(source_text)

            concepts = _find_concepts_in_file(file_path, sdk_root)
            assert len(concepts) == 1
            assert concepts[0]["name"].endswith("QuantumTemplate")

    def test_parse_error(self):
        """Test handling of parse errors."""
        source_text = "invalid python syntax {"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(source_text)
            f.flush()

            try:
                concepts = _find_concepts_in_file(Path(f.name), Path("/test/pennylane"))
                assert concepts == []
            finally:
                Path(f.name).unlink()

    def test_file_not_found(self):
        """Test handling of non-existent file."""
        concepts = _find_concepts_in_file(
            Path("/non/existent/file.py"), Path("/test/pennylane")
        )
        assert concepts == []


class TestExtractPennylaneConcepts:
    """Test cases for extract_pennylane_concepts function."""

    @patch("src.core_concepts.identify_pennylane_core_concepts.PENNYLANE_PROJECT_ROOT")
    def test_project_root_not_found(self, mock_root):
        """Test when project root doesn't exist."""
        mock_root.is_dir.return_value = False
        result = extract_pennylane_concepts()
        assert result == []

    @patch("src.core_concepts.identify_pennylane_core_concepts.PENNYLANE_PROJECT_ROOT")
    def test_no_python_files_found(self, mock_root):
        """Test when no Python files are found."""
        mock_root.is_dir.return_value = True
        mock_root.rglob.return_value = []

        with patch(
            "src.core_concepts.identify_pennylane_core_concepts.SEARCH_SUBDIRS",
            ["nonexistent/"],
        ):
            result = extract_pennylane_concepts()
            assert result == []


class TestSaveSourceCodeSnippets:
    """Test cases for _save_source_code_snippets function."""

    def test_empty_data(self):
        """Test with empty data."""
        with patch(
            "src.core_concepts.identify_pennylane_core_concepts.SOURCE_SNIPPETS_DIR"
        ) as mock_dir:
            _save_source_code_snippets([])
            # Should not raise any exceptions

    def test_save_snippets(self):
        """Test saving source code snippets."""
        concepts_data = [{"name": "test_concept", "source_code": "class Test: pass"}]

        with patch(
            "src.core_concepts.identify_pennylane_core_concepts.SOURCE_SNIPPETS_DIR"
        ) as mock_dir:
            mock_dir.mkdir.return_value = None
            with patch("builtins.open", MagicMock()) as mock_open:
                _save_source_code_snippets(concepts_data)
                mock_dir.mkdir.assert_called_once_with(parents=True, exist_ok=True)


class TestSaveConceptsToCsv:
    """Test cases for _save_concepts_to_csv function."""

    def test_empty_data(self):
        """Test with empty data."""
        with patch(
            "src.core_concepts.identify_pennylane_core_concepts.OUTPUT_JSON_PATH"
        ) as mock_path:
            _save_concepts_to_csv([])
            # Should not raise any exceptions

    def test_save_csv(self):
        """Test saving concepts to CSV."""
        concepts_data = [{"name": "test_concept", "summary": "Test summary"}]

        with patch(
            "src.core_concepts.identify_pennylane_core_concepts.OUTPUT_JSON_PATH"
        ) as mock_path:
            mock_csv_path = MagicMock()
            mock_path.with_suffix.return_value = mock_csv_path
            mock_csv_path.parent.mkdir.return_value = None
            with patch("builtins.open", MagicMock()) as mock_open:
                _save_concepts_to_csv(concepts_data)
                # The function should not call mkdir on the CSV path since it doesn't exist in the function


class TestMainFunction:
    """Test cases for main function."""

    @patch(
        "src.core_concepts.identify_pennylane_core_concepts.extract_pennylane_concepts"
    )
    def test_main_without_data(self, mock_extract):
        """Test main function without data."""
        mock_extract.return_value = []

        main()

        mock_extract.assert_called_once()


class TestConstants:
    """Test cases for module constants."""

    def test_search_subdirs(self):
        """Test SEARCH_SUBDIRS constant."""
        assert isinstance(SEARCH_SUBDIRS, list)
        assert len(SEARCH_SUBDIRS) > 0
        assert all(isinstance(s, str) for s in SEARCH_SUBDIRS)

    def test_path_constants(self):
        """Test path constants are Path objects."""
        assert isinstance(PENNYLANE_PROJECT_ROOT, Path)
        assert isinstance(SOURCE_SNIPPETS_DIR, Path)
        assert isinstance(OUTPUT_JSON_PATH, Path)

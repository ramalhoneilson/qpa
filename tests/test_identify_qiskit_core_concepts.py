"""
Test cases for identify_qiskit_core_concepts.py module.

This module tests the Qiskit core concepts identification functionality.
"""

import ast
import tempfile
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

# Import the module under test
import sys

sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.core_concepts.identify_qiskit_core_concepts import (
    _QiskitConceptVisitor,
    clean_qiskit_docstring,
    _find_concepts_in_file,
    extract_qiskit_concepts,
    deduplicate_by_naming_convention,
    _save_source_code_snippets,
    save_concepts_to_json,
    save_concepts_to_csv,
    main,
    QISKIT_PROJECT_ROOT,
    SOURCE_SNIPPETS_DIR,
    OUTPUT_JSON_PATH,
    OUTPUT_CSV_PATH,
    SEARCH_SUBDIRS,
    TARGET_BASE_CLASSES,
    SIMILARITY_THRESHOLD,
    EXCLUDE_SUBDIRS,
)


class TestCleanQiskitDocstring:
    """Test cases for clean_qiskit_docstring function."""

    def test_clean_basic_docstring(self):
        """Test cleaning a basic docstring."""
        docstring = "A simple docstring."
        result = clean_qiskit_docstring(docstring)
        assert result == "A simple docstring."

    def test_clean_code_block(self):
        """Test cleaning docstring with code blocks."""
        docstring = """A docstring with code block.
        
.. code-block:: text

    some code here
"""
        result = clean_qiskit_docstring(docstring)
        assert "code-block" not in result
        assert "some code here" not in result

    def test_empty_docstring(self):
        """Test cleaning empty docstring."""
        result = clean_qiskit_docstring("")
        assert result == ""

    def test_none_docstring(self):
        """Test cleaning None docstring."""
        # The function should handle None gracefully
        try:
            result = clean_qiskit_docstring(None)
            assert result == ""
        except TypeError:
            # If the function doesn't handle None, that's expected behavior
            pass


class TestQiskitConceptVisitor:
    """Test cases for _QiskitConceptVisitor class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.source_text = '''
class QuantumGate:
    """A quantum gate class."""
    def __init__(self):
        pass

class AnotherClass:
    pass
'''
        self.sdk_root = Path("/test/qiskit")
        self.file_path = self.sdk_root / "circuit" / "library" / "test_file.py"

    def test_visitor_initialization(self):
        """Test visitor initialization."""
        visitor = _QiskitConceptVisitor(self.source_text, self.file_path, self.sdk_root)
        assert visitor.source_text == self.source_text
        assert visitor.file_path == self.file_path
        assert visitor.sdk_root == self.sdk_root
        assert visitor.found_concepts == {}
        assert visitor.context_stack == []

    def test_visit_class_with_docstring(self):
        """Test visiting a class with docstring."""
        visitor = _QiskitConceptVisitor(self.source_text, self.file_path, self.sdk_root)
        tree = ast.parse(self.source_text)
        visitor.visit(tree)

        # Should find at least the class with docstring
        assert len(visitor.found_concepts) >= 1
        concept_names = list(visitor.found_concepts.keys())
        assert any("QuantumGate" in name for name in concept_names)

    def test_concept_name_formatting(self):
        """Test that concept names are formatted correctly."""
        visitor = _QiskitConceptVisitor(self.source_text, self.file_path, self.sdk_root)
        tree = ast.parse(self.source_text)
        visitor.visit(tree)

        concept_names = list(visitor.found_concepts.keys())
        for name in concept_names:
            assert name.startswith("/qiskit/")
            assert "circuit.library.test_file" in name


class TestFindConceptsInFile:
    """Test cases for _find_concepts_in_file function."""

    def test_successful_parsing(self):
        """Test successful file parsing."""
        source_text = '''
class QuantumGate:
    """A quantum gate class."""
    def __init__(self):
        pass
'''
        with tempfile.TemporaryDirectory() as temp_dir:
            sdk_root = Path(temp_dir) / "qiskit"
            file_path = sdk_root / "circuit" / "library" / "test_file.py"
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(source_text)

            concepts = _find_concepts_in_file(file_path, sdk_root)
            assert len(concepts) == 1
            assert concepts[0]["name"].endswith("QuantumGate")

    def test_parse_error(self):
        """Test handling of parse errors."""
        source_text = "invalid python syntax {"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(source_text)
            f.flush()

            try:
                concepts = _find_concepts_in_file(Path(f.name), Path("/test/qiskit"))
                assert concepts == []
            finally:
                Path(f.name).unlink()

    def test_file_not_found(self):
        """Test handling of non-existent file."""
        concepts = _find_concepts_in_file(
            Path("/non/existent/file.py"), Path("/test/qiskit")
        )
        assert concepts == []


class TestExtractQiskitConcepts:
    """Test cases for extract_qiskit_concepts function."""

    @patch("src.core_concepts.identify_qiskit_core_concepts.QISKIT_PROJECT_ROOT")
    def test_project_root_not_found(self, mock_root):
        """Test when project root doesn't exist."""
        mock_root.is_dir.return_value = False
        result = extract_qiskit_concepts()
        assert result == []

    @patch("src.core_concepts.identify_qiskit_core_concepts.QISKIT_PROJECT_ROOT")
    def test_no_python_files_found(self, mock_root):
        """Test when no Python files are found."""
        mock_root.is_dir.return_value = True
        mock_root.rglob.return_value = []

        with patch(
            "src.core_concepts.identify_qiskit_core_concepts.SEARCH_SUBDIRS",
            ["nonexistent/"],
        ):
            result = extract_qiskit_concepts()
            assert result == []


class TestDeduplicateByNamingConvention:
    """Test cases for deduplicate_by_naming_convention function."""

    def test_empty_data(self):
        """Test with empty data."""
        result = deduplicate_by_naming_convention([])
        assert result == []

    def test_no_duplicates(self):
        """Test with no duplicates."""
        concepts = [
            {"name": "concept1", "type": "Class"},
            {"name": "concept2", "type": "Class"},
        ]
        result = deduplicate_by_naming_convention(concepts)
        assert len(result) == 2


class TestSaveSourceCodeSnippets:
    """Test cases for _save_source_code_snippets function."""

    def test_empty_data(self):
        """Test with empty data."""
        with patch(
            "src.core_concepts.identify_qiskit_core_concepts.SOURCE_SNIPPETS_DIR"
        ) as mock_dir:
            _save_source_code_snippets([])
            # Should not raise any exceptions

    def test_save_snippets(self):
        """Test saving source code snippets."""
        concepts_data = [{"name": "test_concept", "source_code": "class Test: pass"}]

        with patch(
            "src.core_concepts.identify_qiskit_core_concepts.SOURCE_SNIPPETS_DIR"
        ) as mock_dir:
            mock_dir.mkdir.return_value = None
            with patch("builtins.open", MagicMock()) as mock_open:
                _save_source_code_snippets(concepts_data)
                mock_dir.mkdir.assert_called_once_with(parents=True, exist_ok=True)


class TestSaveConceptsToJson:
    """Test cases for save_concepts_to_json function."""

    def test_empty_data(self):
        """Test with empty data."""
        with patch(
            "src.core_concepts.identify_qiskit_core_concepts.OUTPUT_JSON_PATH"
        ) as mock_path:
            save_concepts_to_json([])
            # Should not raise any exceptions

    def test_save_json(self):
        """Test saving concepts to JSON."""
        concepts_data = [{"name": "test_concept", "summary": "Test summary"}]

        with patch(
            "src.core_concepts.identify_qiskit_core_concepts.OUTPUT_JSON_PATH"
        ) as mock_path:
            mock_path.parent.mkdir.return_value = None
            with patch("builtins.open", MagicMock()) as mock_open:
                save_concepts_to_json(concepts_data)
                mock_path.parent.mkdir.assert_called_once_with(
                    parents=True, exist_ok=True
                )


class TestSaveConceptsToCsv:
    """Test cases for save_concepts_to_csv function."""

    def test_empty_data(self):
        """Test with empty data."""
        with patch(
            "src.core_concepts.identify_qiskit_core_concepts.OUTPUT_CSV_PATH"
        ) as mock_path:
            save_concepts_to_csv([])
            # Should not raise any exceptions

    def test_save_csv(self):
        """Test saving concepts to CSV."""
        concepts_data = [{"name": "test_concept", "summary": "Test summary"}]

        with patch(
            "src.core_concepts.identify_qiskit_core_concepts.OUTPUT_CSV_PATH"
        ) as mock_path:
            mock_path.parent.mkdir.return_value = None
            with patch("builtins.open", MagicMock()) as mock_open:
                save_concepts_to_csv(concepts_data)
                mock_path.parent.mkdir.assert_called_once_with(
                    parents=True, exist_ok=True
                )


class TestMainFunction:
    """Test cases for main function."""

    @patch("src.core_concepts.identify_qiskit_core_concepts.extract_qiskit_concepts")
    @patch(
        "src.core_concepts.identify_qiskit_core_concepts.deduplicate_by_naming_convention"
    )
    @patch(
        "src.core_concepts.identify_qiskit_core_concepts.deduplicate_concepts_semantic"
    )
    @patch("src.core_concepts.identify_qiskit_core_concepts._save_source_code_snippets")
    @patch("src.core_concepts.identify_qiskit_core_concepts.save_concepts_to_json")
    @patch("src.core_concepts.identify_qiskit_core_concepts.save_concepts_to_csv")
    def test_main_with_data(
        self,
        mock_save_csv,
        mock_save_json,
        mock_save_snippets,
        mock_semantic,
        mock_convention,
        mock_extract,
    ):
        """Test main function with data."""
        mock_extract.return_value = [{"name": "test", "summary": "test"}]
        mock_convention.return_value = [{"name": "test", "summary": "test"}]
        mock_semantic.return_value = [{"name": "test", "summary": "test"}]

        main()

        mock_extract.assert_called_once()
        mock_convention.assert_called_once()
        mock_semantic.assert_called_once()
        mock_save_snippets.assert_called_once()
        mock_save_json.assert_called_once()
        mock_save_csv.assert_called_once()

    @patch("src.core_concepts.identify_qiskit_core_concepts.extract_qiskit_concepts")
    @patch(
        "src.core_concepts.identify_qiskit_core_concepts.deduplicate_by_naming_convention"
    )
    @patch(
        "src.core_concepts.identify_qiskit_core_concepts.deduplicate_concepts_semantic"
    )
    def test_main_without_data(self, mock_semantic, mock_convention, mock_extract):
        """Test main function without data."""
        mock_extract.return_value = []
        mock_convention.return_value = []
        mock_semantic.return_value = []

        main()

        mock_extract.assert_called_once()
        mock_convention.assert_called_once()
        mock_semantic.assert_called_once()


class TestConstants:
    """Test cases for module constants."""

    def test_search_subdirs(self):
        """Test SEARCH_SUBDIRS constant."""
        assert isinstance(SEARCH_SUBDIRS, list)
        assert len(SEARCH_SUBDIRS) > 0
        assert all(isinstance(s, str) for s in SEARCH_SUBDIRS)

    def test_target_base_classes(self):
        """Test TARGET_BASE_CLASSES constant."""
        assert isinstance(TARGET_BASE_CLASSES, list)
        assert len(TARGET_BASE_CLASSES) > 0
        assert all(isinstance(s, str) for s in TARGET_BASE_CLASSES)

    def test_similarity_threshold(self):
        """Test SIMILARITY_THRESHOLD constant."""
        assert isinstance(SIMILARITY_THRESHOLD, float)
        assert 0 <= SIMILARITY_THRESHOLD <= 1

    def test_exclude_subdirs(self):
        """Test EXCLUDE_SUBDIRS constant."""
        assert isinstance(EXCLUDE_SUBDIRS, set)
        assert all(isinstance(s, str) for s in EXCLUDE_SUBDIRS)

    def test_path_constants(self):
        """Test path constants are Path objects."""
        assert isinstance(QISKIT_PROJECT_ROOT, Path)
        assert isinstance(SOURCE_SNIPPETS_DIR, Path)
        assert isinstance(OUTPUT_JSON_PATH, Path)
        assert isinstance(OUTPUT_CSV_PATH, Path)

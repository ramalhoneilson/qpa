"""
Tests for src/core_concepts/extractor/extractors.py

This module tests the ConceptExtractor class functionality for scanning
directories and extracting code concepts from Python files.
"""

import ast
import logging
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch, mock_open

import pytest

from src.core_concepts.extractor.extractors import ConceptExtractor, Concept
from src.core_concepts.extractor.processors import BaseProcessor
from src.core_concepts.extractor.visitors import BaseConceptVisitor


class MockProcessor(BaseProcessor):
    """Mock processor for testing."""
    
    def clean_docstring(self, docstring: str) -> str:
        return docstring.strip()
    
    def create_summary(self, docstring: str) -> str:
        return docstring.split('.')[0] + '.' if '.' in docstring else docstring


class MockVisitor(BaseConceptVisitor):
    """Mock visitor for testing."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.found_concepts = {}
    
    def visit_FunctionDef(self, node):
        """Mock function visitor."""
        if node.name.startswith('test_'):
            self.found_concepts[node.name] = {
                "name": f"{self.file_path.stem}.{node.name}",
                "summary": "Test function",
                "docstring": "Test docstring",
                "source_code": f"def {node.name}(): pass"
            }
    
    def visit_ClassDef(self, node):
        """Mock class visitor."""
        if node.name.startswith('Test'):
            self.found_concepts[node.name] = {
                "name": f"{self.file_path.stem}.{node.name}",
                "summary": "Test class",
                "docstring": "Test docstring",
                "source_code": f"class {node.name}: pass"
            }


class TestConceptExtractor:
    """Test the ConceptExtractor class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_processor = MockProcessor()
        self.mock_visitor_class = MockVisitor
        self.extractor = ConceptExtractor(
            visitor_class=self.mock_visitor_class,
            processor=self.mock_processor
        )

    def test_initialization_valid_parameters(self):
        """Test initialization with valid parameters."""
        assert self.extractor.visitor_class == self.mock_visitor_class
        assert self.extractor.processor == self.mock_processor

    def test_initialization_invalid_visitor_class(self):
        """Test initialization with invalid visitor class."""
        with pytest.raises(TypeError, match="visitor_class must be a subclass of BaseConceptVisitor"):
            ConceptExtractor(visitor_class=dict, processor=self.mock_processor)

    def test_initialization_invalid_processor(self):
        """Test initialization with invalid processor."""
        with pytest.raises(TypeError, match="processor must be an instance of a BaseProcessor subclass"):
            ConceptExtractor(visitor_class=self.mock_visitor_class, processor=dict())

    def test_extract_from_package_empty_search_paths(self):
        """Test extraction with empty search paths."""
        package_root = Path("/test/package")
        search_paths = []
        
        with patch('logging.info') as mock_info:
            result = self.extractor.extract_from_package(package_root, search_paths)
            
            assert result == []
            mock_info.assert_not_called()

    def test_extract_from_package_nonexistent_directory(self):
        """Test extraction with nonexistent directory."""
        package_root = Path("/test/package")
        search_paths = ["nonexistent"]
        
        with patch('logging.warning') as mock_warning, \
             patch('logging.info') as mock_info:
            
            result = self.extractor.extract_from_package(package_root, search_paths)
            
            assert result == []
            mock_warning.assert_called_once_with(
                "Source directory not found, skipping: /test/package/nonexistent"
            )
            mock_info.assert_not_called()

    def test_extract_from_package_with_files(self):
        """Test extraction with actual files."""
        # Create temporary directory structure
        with patch('pathlib.Path.is_dir', return_value=True), \
             patch('pathlib.Path.rglob') as mock_rglob, \
             patch.object(self.extractor, '_find_concepts_in_file') as mock_find_concepts, \
             patch('logging.info') as mock_info:
            
            # Mock file paths
            mock_file1 = Path("/test/package/subdir/file1.py")
            mock_file2 = Path("/test/package/subdir/file2.py")
            mock_rglob.return_value = [mock_file1, mock_file2]
            
            # Mock found concepts
            mock_concepts1 = [
                {"name": "file1.test_function", "summary": "Test function 1"},
                {"name": "file1.test_function2", "summary": "Test function 2"}
            ]
            mock_concepts2 = [
                {"name": "file2.test_function", "summary": "Test function 3"}
            ]
            mock_find_concepts.side_effect = [mock_concepts1, mock_concepts2]
            
            package_root = Path("/test/package")
            search_paths = ["subdir"]
            
            result = self.extractor.extract_from_package(package_root, search_paths)
            
            # Should have found all concepts
            assert len(result) == 3
            concept_names = [c["name"] for c in result]
            assert "file1.test_function" in concept_names
            assert "file1.test_function2" in concept_names
            assert "file2.test_function" in concept_names
            
            # Check logging
            mock_info.assert_called_once_with("Scanning files in 'subdir'...")
            
            # Check that _find_concepts_in_file was called for each file
            assert mock_find_concepts.call_count == 2

    def test_extract_from_package_skips_init_files(self):
        """Test that __init__.py files are skipped."""
        with patch('pathlib.Path.is_dir', return_value=True), \
             patch('pathlib.Path.rglob') as mock_rglob, \
             patch.object(self.extractor, '_find_concepts_in_file') as mock_find_concepts, \
             patch('logging.info'):
            
            # Mock file paths including __init__.py
            mock_file1 = Path("/test/package/subdir/__init__.py")
            mock_file2 = Path("/test/package/subdir/file.py")
            mock_rglob.return_value = [mock_file1, mock_file2]
            
            mock_find_concepts.return_value = [{"name": "test", "summary": "test"}]
            
            package_root = Path("/test/package")
            search_paths = ["subdir"]
            
            result = self.extractor.extract_from_package(package_root, search_paths)
            
            # Should only have called _find_concepts_in_file once (for file.py, not __init__.py)
            assert mock_find_concepts.call_count == 1
            mock_find_concepts.assert_called_once_with(mock_file2, package_root)

    def test_extract_from_package_duplicate_prevention(self):
        """Test that duplicate concepts are prevented."""
        with patch('pathlib.Path.is_dir', return_value=True), \
             patch('pathlib.Path.rglob') as mock_rglob, \
             patch.object(self.extractor, '_find_concepts_in_file') as mock_find_concepts, \
             patch('logging.info'):
            
            mock_file1 = Path("/test/package/subdir/file1.py")
            mock_file2 = Path("/test/package/subdir/file2.py")
            mock_rglob.return_value = [mock_file1, mock_file2]
            
            # Both files return the same concept name
            mock_concepts = [{"name": "duplicate.concept", "summary": "Duplicate concept"}]
            mock_find_concepts.return_value = mock_concepts
            
            package_root = Path("/test/package")
            search_paths = ["subdir"]
            
            result = self.extractor.extract_from_package(package_root, search_paths)
            
            # Should only have one concept despite being found in both files
            assert len(result) == 1
            assert result[0]["name"] == "duplicate.concept"

    def test_find_concepts_in_file_success(self):
        """Test successful concept finding in a file."""
        mock_file = Path("/test/file.py")
        mock_sdk_root = Path("/test")
        
        source_text = """
def test_function():
    \"\"\"Test function docstring.\"\"\"
    pass

class TestClass:
    \"\"\"Test class docstring.\"\"\"
    pass
"""
        
        with patch('pathlib.Path.read_text', return_value=source_text), \
             patch('ast.parse') as mock_parse, \
             patch('logging.warning') as mock_warning:
            
            # Mock AST tree
            mock_tree = MagicMock()
            mock_parse.return_value = mock_tree
            
            # Mock visitor
            mock_visitor = MagicMock()
            mock_visitor.found_concepts = {
                "test_function": {"name": "file.test_function", "summary": "Test function"},
                "TestClass": {"name": "file.TestClass", "summary": "Test class"}
            }
            
            with patch.object(self.extractor, 'visitor_class', return_value=mock_visitor):
                result = self.extractor._find_concepts_in_file(mock_file, mock_sdk_root)
                
                # Should have found concepts
                assert len(result) == 2
                concept_names = [c["name"] for c in result]
                assert "file.test_function" in concept_names
                assert "file.TestClass" in concept_names
                
                # Should not have logged any warnings
                mock_warning.assert_not_called()

    def test_find_concepts_in_file_empty_file(self):
        """Test handling of empty files."""
        mock_file = Path("/test/file.py")
        mock_sdk_root = Path("/test")
        
        with patch('pathlib.Path.read_text', return_value="   \n\n   "), \
             patch('logging.warning') as mock_warning:
            
            result = self.extractor._find_concepts_in_file(mock_file, mock_sdk_root)
            
            # Should return empty list for empty file
            assert result == []
            mock_warning.assert_not_called()

    def test_find_concepts_in_file_very_small_file(self):
        """Test handling of very small files."""
        mock_file = Path("/test/file.py")
        mock_sdk_root = Path("/test")
        
        with patch('pathlib.Path.read_text', return_value="pass"), \
             patch('logging.warning') as mock_warning:
            
            result = self.extractor._find_concepts_in_file(mock_file, mock_sdk_root)
            
            # Should return empty list for very small file
            assert result == []
            mock_warning.assert_not_called()

    def test_find_concepts_in_file_parse_error(self):
        """Test handling of file parsing errors."""
        mock_file = Path("/test/file.py")
        mock_sdk_root = Path("/test")
        
        source_text = "def test_function():"
        
        with patch('pathlib.Path.read_text', return_value=source_text), \
             patch('ast.parse', side_effect=SyntaxError("Invalid syntax")), \
             patch('logging.warning') as mock_warning:
            
            result = self.extractor._find_concepts_in_file(mock_file, mock_sdk_root)
            
            # Should return empty list on parse error
            assert result == []
            mock_warning.assert_called_once_with(
                f"Could not parse file {mock_file.name}: Invalid syntax"
            )

    def test_find_concepts_in_file_read_error(self):
        """Test handling of file reading errors."""
        mock_file = Path("/test/file.py")
        mock_sdk_root = Path("/test")
        
        with patch('pathlib.Path.read_text', side_effect=IOError("Permission denied")), \
             patch('logging.warning') as mock_warning:
            
            result = self.extractor._find_concepts_in_file(mock_file, mock_sdk_root)
            
            # Should return empty list on read error
            assert result == []
            mock_warning.assert_called_once_with(
                f"Could not parse file {mock_file.name}: Permission denied"
            )

    def test_find_concepts_in_file_visitor_kwargs(self):
        """Test that visitor kwargs are passed correctly."""
        mock_file = Path("/test/file.py")
        mock_sdk_root = Path("/test")
        
        source_text = "def test_function(): pass"
        
        with patch('pathlib.Path.read_text', return_value=source_text), \
             patch('ast.parse') as mock_parse:
            
            mock_tree = MagicMock()
            mock_parse.return_value = mock_tree
            
            mock_visitor = MagicMock()
            mock_visitor.found_concepts = {}
            
            with patch.object(self.extractor, 'visitor_class', return_value=mock_visitor) as mock_visitor_class:
                # Call with extra kwargs
                extra_kwargs = {"public_api_names": {"test_function"}, "extra_param": "value"}
                self.extractor._find_concepts_in_file(mock_file, mock_sdk_root, **extra_kwargs)
                
                # Check that visitor was instantiated with correct parameters
                mock_visitor_class.assert_called_once_with(
                    source_text=source_text,
                    file_path=mock_file,
                    sdk_root=mock_sdk_root,
                    processor=self.mock_processor,
                    public_api_names={"test_function"},
                    extra_param="value"
                )

    def test_concept_type_alias(self):
        """Test that Concept type alias is properly defined."""
        # This is more of a type checking test
        concept: Concept = {
            "name": "test_concept",
            "summary": "test summary",
            "docstring": "test docstring",
            "source_code": "def test(): pass"
        }
        assert isinstance(concept, dict)
        assert "name" in concept
        assert "summary" in concept
        assert "docstring" in concept
        assert "source_code" in concept


class TestConceptExtractorIntegration:
    """Integration tests for ConceptExtractor."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_processor = MockProcessor()
        self.mock_visitor_class = MockVisitor
        self.extractor = ConceptExtractor(
            visitor_class=self.mock_visitor_class,
            processor=self.mock_processor
        )

    def test_full_extraction_workflow(self):
        """Test the complete extraction workflow."""
        # Create a temporary directory structure
        with patch('pathlib.Path.is_dir', return_value=True), \
             patch('pathlib.Path.rglob') as mock_rglob, \
             patch('pathlib.Path.read_text') as mock_read_text, \
             patch('ast.parse') as mock_parse, \
             patch('logging.info') as mock_info:
            
            # Mock file paths
            mock_file1 = Path("/test/package/subdir/file1.py")
            mock_file2 = Path("/test/package/subdir/file2.py")
            mock_rglob.return_value = [mock_file1, mock_file2]
            
            # Mock source text
            source_text = """
def test_function():
    \"\"\"Test function.\"\"\"
    pass
"""
            mock_read_text.return_value = source_text
            
            # Mock AST tree
            mock_tree = MagicMock()
            mock_parse.return_value = mock_tree
            
            # Mock visitor behavior
            mock_visitor = MagicMock()
            mock_visitor.found_concepts = {
                "test_function": {
                    "name": "file1.test_function",
                    "summary": "Test function",
                    "docstring": "Test function.",
                    "source_code": "def test_function(): pass"
                }
            }
            
            with patch.object(self.extractor, 'visitor_class', return_value=mock_visitor):
                package_root = Path("/test/package")
                search_paths = ["subdir"]
                
                result = self.extractor.extract_from_package(package_root, search_paths)
                
                # Should have found concepts
                assert len(result) == 1
                assert result[0]["name"] == "file1.test_function"
                
                # Check logging
                mock_info.assert_called_once_with("Scanning files in 'subdir'...")

    def test_multiple_search_paths(self):
        """Test extraction from multiple search paths."""
        with patch('pathlib.Path.is_dir', return_value=True), \
             patch('pathlib.Path.rglob') as mock_rglob, \
             patch.object(self.extractor, '_find_concepts_in_file') as mock_find_concepts, \
             patch('logging.info') as mock_info:
            
            # Mock different files in different directories
            mock_file1 = Path("/test/package/subdir1/file1.py")
            mock_file2 = Path("/test/package/subdir2/file2.py")
            mock_rglob.side_effect = [[mock_file1], [mock_file2]]
            
            mock_concepts1 = [{"name": "file1.concept1", "summary": "Concept 1"}]
            mock_concepts2 = [{"name": "file2.concept2", "summary": "Concept 2"}]
            mock_find_concepts.side_effect = [mock_concepts1, mock_concepts2]
            
            package_root = Path("/test/package")
            search_paths = ["subdir1", "subdir2"]
            
            result = self.extractor.extract_from_package(package_root, search_paths)
            
            # Should have found concepts from both directories
            assert len(result) == 2
            concept_names = [c["name"] for c in result]
            assert "file1.concept1" in concept_names
            assert "file2.concept2" in concept_names
            
            # Check logging calls
            assert mock_info.call_count == 2
            mock_info.assert_any_call("Scanning files in 'subdir1'...")
            mock_info.assert_any_call("Scanning files in 'subdir2'...")

    def test_mixed_directory_conditions(self):
        """Test extraction with mix of existing and non-existing directories."""
        with patch('pathlib.Path.is_dir') as mock_is_dir, \
             patch('pathlib.Path.rglob') as mock_rglob, \
             patch.object(self.extractor, '_find_concepts_in_file') as mock_find_concepts, \
             patch('logging.info') as mock_info, \
             patch('logging.warning') as mock_warning:
            
            # First directory exists, second doesn't
            mock_is_dir.side_effect = [True, False]
            mock_file = Path("/test/package/existing/file.py")
            mock_rglob.return_value = [mock_file]
            
            mock_concepts = [{"name": "file.concept", "summary": "Concept"}]
            mock_find_concepts.return_value = mock_concepts
            
            package_root = Path("/test/package")
            search_paths = ["existing", "nonexistent"]
            
            result = self.extractor.extract_from_package(package_root, search_paths)
            
            # Should have found concepts only from existing directory
            assert len(result) == 1
            assert result[0]["name"] == "file.concept"
            
            # Check logging
            mock_info.assert_called_once_with("Scanning files in 'existing'...")
            mock_warning.assert_called_once_with(
                "Source directory not found, skipping: /test/package/nonexistent"
            )

    def test_edge_case_empty_concepts(self):
        """Test extraction when no concepts are found."""
        with patch('pathlib.Path.is_dir', return_value=True), \
             patch('pathlib.Path.rglob') as mock_rglob, \
             patch.object(self.extractor, '_find_concepts_in_file') as mock_find_concepts, \
             patch('logging.info'):
            
            mock_file = Path("/test/package/subdir/file.py")
            mock_rglob.return_value = [mock_file]
            mock_find_concepts.return_value = []  # No concepts found
            
            package_root = Path("/test/package")
            search_paths = ["subdir"]
            
            result = self.extractor.extract_from_package(package_root, search_paths)
            
            # Should return empty list
            assert result == []

    def test_visitor_kwargs_propagation(self):
        """Test that visitor kwargs are properly propagated through the workflow."""
        with patch('pathlib.Path.is_dir', return_value=True), \
             patch('pathlib.Path.rglob') as mock_rglob, \
             patch.object(self.extractor, '_find_concepts_in_file') as mock_find_concepts, \
             patch('logging.info'):
            
            mock_file = Path("/test/package/subdir/file.py")
            mock_rglob.return_value = [mock_file]
            mock_find_concepts.return_value = []
            
            package_root = Path("/test/package")
            search_paths = ["subdir"]
            visitor_kwargs = {"public_api_names": {"test"}, "extra_param": "value"}
            
            self.extractor.extract_from_package(package_root, search_paths, **visitor_kwargs)
            
            # Check that kwargs were passed to _find_concepts_in_file
            mock_find_concepts.assert_called_once_with(mock_file, package_root, **visitor_kwargs)


class TestConceptExtractorErrorHandling:
    """Test error handling in ConceptExtractor."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_processor = MockProcessor()
        self.mock_visitor_class = MockVisitor
        self.extractor = ConceptExtractor(
            visitor_class=self.mock_visitor_class,
            processor=self.mock_processor
        )

    def test_visitor_instantiation_error(self):
        """Test handling of visitor instantiation errors."""
        mock_file = Path("/test/file.py")
        mock_sdk_root = Path("/test")
        
        source_text = "def test_function(): pass"
        
        with patch('pathlib.Path.read_text', return_value=source_text), \
             patch('ast.parse') as mock_parse, \
             patch('logging.warning') as mock_warning:
            
            mock_tree = MagicMock()
            mock_parse.return_value = mock_tree
            
            # Mock visitor class that raises an error
            with patch.object(self.extractor, 'visitor_class', side_effect=TypeError("Invalid visitor")):
                result = self.extractor._find_concepts_in_file(mock_file, mock_sdk_root)
                
                # Should return empty list on visitor instantiation error
                assert result == []
                mock_warning.assert_called_once_with(
                    f"Could not parse file {mock_file.name}: Invalid visitor"
                )

    def test_visitor_visit_error(self):
        """Test handling of visitor visit errors."""
        mock_file = Path("/test/file.py")
        mock_sdk_root = Path("/test")
        
        source_text = "def test_function(): pass"
        
        with patch('pathlib.Path.read_text', return_value=source_text), \
             patch('ast.parse') as mock_parse, \
             patch('logging.warning') as mock_warning:
            
            mock_tree = MagicMock()
            mock_parse.return_value = mock_tree
            
            # Mock visitor that raises an error during visit
            mock_visitor = MagicMock()
            mock_visitor.visit.side_effect = RuntimeError("Visit error")
            
            with patch.object(self.extractor, 'visitor_class', return_value=mock_visitor):
                result = self.extractor._find_concepts_in_file(mock_file, mock_sdk_root)
                
                # Should return empty list on visit error
                assert result == []
                mock_warning.assert_called_once_with(
                    f"Could not parse file {mock_file.name}: Visit error"
                )

    def test_file_permission_error(self):
        """Test handling of file permission errors."""
        mock_file = Path("/test/file.py")
        mock_sdk_root = Path("/test")
        
        with patch('pathlib.Path.read_text', side_effect=PermissionError("Permission denied")), \
             patch('logging.warning') as mock_warning:
            
            result = self.extractor._find_concepts_in_file(mock_file, mock_sdk_root)
            
            # Should return empty list on permission error
            assert result == []
            mock_warning.assert_called_once_with(
                f"Could not parse file {mock_file.name}: Permission denied"
            )

    def test_ast_parse_syntax_error(self):
        """Test handling of AST parsing syntax errors."""
        mock_file = Path("/test/file.py")
        mock_sdk_root = Path("/test")
        
        source_text = "def test_function(:"  # Invalid syntax
        
        with patch('pathlib.Path.read_text', return_value=source_text), \
             patch('ast.parse', side_effect=SyntaxError("Invalid syntax")), \
             patch('logging.warning') as mock_warning:
            
            result = self.extractor._find_concepts_in_file(mock_file, mock_sdk_root)
            
            # Should return empty list on syntax error
            assert result == []
            mock_warning.assert_called_once_with(
                f"Could not parse file {mock_file.name}: Invalid syntax"
            )

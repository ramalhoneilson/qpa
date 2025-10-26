"""
Tests for src/core_concepts/pipelines/extract_classiq.py

This module tests the Classiq-specific concept extraction pipeline functionality,
focusing on the project's implementation rather than external framework features.
"""

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.core_concepts.pipelines.extract_classiq import (
    get_sdk_root_path,
    get_public_api_names,
    main,
    TARGET_MODULES,
    SOURCE_CODE_SEARCH_PATHS,
)
from src.core_concepts.extractor.visitors import ClassiqFunctionVisitor


class TestGetSdkRootPath:
    """Test the get_sdk_root_path function."""

    def test_successful_sdk_path_finding(self):
        """Test successful finding of SDK path."""
        mock_path = ["/path/to/classiq"]
        
        with patch('src.core_concepts.pipelines.extract_classiq.classiq.__path__', mock_path), \
             patch('logging.info') as mock_info:
            
            result = get_sdk_root_path()
            
            assert result == Path("/path/to/classiq")
            mock_info.assert_called_once_with("Found installed Classiq SDK at: /path/to/classiq")

    def test_index_error(self):
        """Test handling of IndexError."""
        with patch('src.core_concepts.pipelines.extract_classiq.classiq.__path__', []), \
             patch('logging.error') as mock_error:
            
            result = get_sdk_root_path()
            
            assert result is None
            mock_error.assert_called_once_with(
                "Could not find the installed 'classiq' package. Please ensure it is installed in the environment."
            )

    def test_multiple_paths(self):
        """Test handling of multiple paths in __path__."""
        mock_paths = ["/path/to/classiq", "/another/path"]
        
        with patch('src.core_concepts.pipelines.extract_classiq.classiq.__path__', mock_paths), \
             patch('logging.info') as mock_info:
            
            result = get_sdk_root_path()
            
            # Should use the first path
            assert result == Path("/path/to/classiq")
            mock_info.assert_called_once_with("Found installed Classiq SDK at: /path/to/classiq")


class TestGetPublicApiNames:
    """Test the get_public_api_names function."""

    def test_empty_modules_list(self):
        """Test handling of empty modules list."""
        with patch('logging.info') as mock_info:
            result = get_public_api_names([])
            
            assert result == set()
            mock_info.assert_not_called()

    def test_successful_api_extraction(self):
        """Test successful extraction of public API names."""
        mock_modules = ["classiq.open_library.functions"]
        mock_module = MagicMock()
        mock_module.__all__ = ["function1", "function2", "function3"]
        
        with patch('src.core_concepts.pipelines.extract_classiq.importlib.import_module', return_value=mock_module):
            
            result = get_public_api_names(mock_modules)
            
            assert result == {"function1", "function2", "function3"}


class TestMainFunction:
    """Test the main orchestration function."""

    def setup_method(self):
        """Set up test fixtures."""
        # Mock the config module
        self.mock_config = MagicMock()
        self.mock_config.RESULTS_DIR = Path("/test/results")

    def test_successful_extraction_pipeline(self):
        """Test successful execution of the complete pipeline."""
        mock_sdk_root = Path("/test/classiq")
        mock_public_apis = {"function1", "function2"}
        mock_concepts = [
            {"name": "classiq.function1", "summary": "Test function 1"},
            {"name": "classiq.function2", "summary": "Test function 2"}
        ]
        
        with patch('src.core_concepts.pipelines.extract_classiq.get_sdk_root_path', return_value=mock_sdk_root), \
             patch('src.core_concepts.pipelines.extract_classiq.get_public_api_names', return_value=mock_public_apis), \
             patch('src.core_concepts.pipelines.extract_classiq.ClassiqProcessor') as mock_processor_class, \
             patch('src.core_concepts.pipelines.extract_classiq.ConceptExtractor') as mock_extractor_class, \
             patch('src.core_concepts.pipelines.extract_classiq.ConceptStorage') as mock_storage_class, \
             patch('src.core_concepts.pipelines.extract_classiq.config', self.mock_config), \
             patch('logging.info') as mock_info:
            
            # Mock the extractor instance
            mock_extractor = MagicMock()
            mock_extractor.extract_from_package.return_value = mock_concepts
            mock_extractor_class.return_value = mock_extractor
            
            # Mock the storage instance
            mock_storage = MagicMock()
            mock_storage_class.return_value = mock_storage
            
            main()
            
            # Check that components were instantiated correctly
            mock_processor_class.assert_called_once()
            mock_extractor_class.assert_called_once_with(
                visitor_class=ClassiqFunctionVisitor,
                processor=mock_processor_class.return_value
            )
            
            # Check that extraction was called with correct parameters
            mock_extractor.extract_from_package.assert_called_once_with(
                package_root=mock_sdk_root,
                search_paths=SOURCE_CODE_SEARCH_PATHS,
                public_api_names=mock_public_apis
            )
            
            # Check that storage was instantiated with correct paths
            mock_storage_class.assert_called_once_with(
                json_path=Path("/test/results/classiq_quantum_concepts.json"),
                csv_path=Path("/test/results/classiq_quantum_concepts.csv")
            )
            
            # Check that save_all was called
            mock_storage.save_all.assert_called_once_with(mock_concepts)
            
            # Check logging
            assert mock_info.call_count >= 3  # Multiple info messages

    def test_no_sdk_root_found(self):
        """Test handling when SDK root is not found."""
        with patch('src.core_concepts.pipelines.extract_classiq.get_sdk_root_path', return_value=None), \
             patch('logging.info') as mock_info:
            
            main()
            
            # Should return early without further processing
            mock_info.assert_called_once_with("--- Starting Classiq Core Concepts Generation ---")

    def test_no_public_apis_found(self):
        """Test handling when no public APIs are found."""
        mock_sdk_root = Path("/test/classiq")
        
        with patch('src.core_concepts.pipelines.extract_classiq.get_sdk_root_path', return_value=mock_sdk_root), \
             patch('src.core_concepts.pipelines.extract_classiq.get_public_api_names', return_value=set()), \
             patch('logging.error') as mock_error, \
             patch('logging.info') as mock_info:
            
            main()
            
            mock_error.assert_called_once_with("No public API functions found to target. Exiting.")
            # Should not proceed to extraction

    def test_no_concepts_found(self):
        """Test handling when no concepts are extracted."""
        mock_sdk_root = Path("/test/classiq")
        mock_public_apis = {"function1", "function2"}
        
        with patch('src.core_concepts.pipelines.extract_classiq.get_sdk_root_path', return_value=mock_sdk_root), \
             patch('src.core_concepts.pipelines.extract_classiq.get_public_api_names', return_value=mock_public_apis), \
             patch('src.core_concepts.pipelines.extract_classiq.ClassiqProcessor') as mock_processor_class, \
             patch('src.core_concepts.pipelines.extract_classiq.ConceptExtractor') as mock_extractor_class, \
             patch('src.core_concepts.pipelines.extract_classiq.config', self.mock_config), \
             patch('logging.warning') as mock_warning, \
             patch('logging.info') as mock_info:
            
            # Mock the extractor instance
            mock_extractor = MagicMock()
            mock_extractor.extract_from_package.return_value = []  # No concepts
            mock_extractor_class.return_value = mock_extractor
            
            main()
            
            # Check that extraction was called
            mock_extractor.extract_from_package.assert_called_once()
            
            # Check warning and info messages
            mock_warning.assert_called_once_with("Extraction complete, but no concepts were found.")
            mock_info.assert_any_call("--- Classiq Generation Complete ---")


class TestConstants:
    """Test module constants."""

    def test_target_modules_constant(self):
        """Test that TARGET_MODULES is defined correctly."""
        assert TARGET_MODULES == ["classiq.open_library.functions"]
        assert isinstance(TARGET_MODULES, list)
        assert len(TARGET_MODULES) == 1

    def test_source_code_search_paths_constant(self):
        """Test that SOURCE_CODE_SEARCH_PATHS is defined correctly."""
        expected_paths = [
            "open_library/functions",
            "qmod/builtins/functions",
        ]
        assert SOURCE_CODE_SEARCH_PATHS == expected_paths
        assert isinstance(SOURCE_CODE_SEARCH_PATHS, list)
        assert len(SOURCE_CODE_SEARCH_PATHS) == 2


class TestIntegration:
    """Integration tests for the Classiq extraction pipeline."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_config = MagicMock()
        self.mock_config.RESULTS_DIR = Path("/test/results")

    def test_complete_pipeline_integration(self):
        """Test the complete pipeline integration."""
        mock_sdk_root = Path("/test/classiq")
        mock_public_apis = {"test_function", "another_function"}
        mock_concepts = [
            {"name": "classiq.test_function", "summary": "Test function"},
            {"name": "classiq.another_function", "summary": "Another function"}
        ]
        
        with patch('src.core_concepts.pipelines.extract_classiq.get_sdk_root_path', return_value=mock_sdk_root), \
             patch('src.core_concepts.pipelines.extract_classiq.get_public_api_names', return_value=mock_public_apis), \
             patch('src.core_concepts.pipelines.extract_classiq.ClassiqProcessor') as mock_processor_class, \
             patch('src.core_concepts.pipelines.extract_classiq.ConceptExtractor') as mock_extractor_class, \
             patch('src.core_concepts.pipelines.extract_classiq.ConceptStorage') as mock_storage_class, \
             patch('src.core_concepts.pipelines.extract_classiq.config', self.mock_config), \
             patch('logging.info') as mock_info:
            
            # Mock processor
            mock_processor = MagicMock()
            mock_processor_class.return_value = mock_processor
            
            # Mock extractor
            mock_extractor = MagicMock()
            mock_extractor.extract_from_package.return_value = mock_concepts
            mock_extractor_class.return_value = mock_extractor
            
            # Mock storage
            mock_storage = MagicMock()
            mock_storage_class.return_value = mock_storage
            
            main()
            
            # Verify the complete workflow
            assert mock_processor_class.call_count == 1
            assert mock_extractor_class.call_count == 1
            assert mock_storage_class.call_count == 1
            
            # Verify extraction parameters
            mock_extractor.extract_from_package.assert_called_once_with(
                package_root=mock_sdk_root,
                search_paths=SOURCE_CODE_SEARCH_PATHS,
                public_api_names=mock_public_apis
            )
            
            # Verify storage parameters
            mock_storage_class.assert_called_once_with(
                json_path=Path("/test/results/classiq_quantum_concepts.json"),
                csv_path=Path("/test/results/classiq_quantum_concepts.csv")
            )
            
            # Verify save operation
            mock_storage.save_all.assert_called_once_with(mock_concepts)

    def test_pipeline_with_realistic_data(self):
        """Test pipeline with realistic concept data."""
        mock_sdk_root = Path("/test/classiq")
        mock_public_apis = {"quantum_function", "classical_function"}
        mock_concepts = [
            {
                "name": "/classiq/open_library/functions.quantum_function",
                "summary": "A quantum function for processing qubits",
                "docstring": "This function processes quantum states.",
                "source_code": "def quantum_function(): pass"
            },
            {
                "name": "/classiq/open_library/functions.classical_function", 
                "summary": "A classical function for data processing",
                "docstring": "This function processes classical data.",
                "source_code": "def classical_function(): pass"
            }
        ]
        
        with patch('src.core_concepts.pipelines.extract_classiq.get_sdk_root_path', return_value=mock_sdk_root), \
             patch('src.core_concepts.pipelines.extract_classiq.get_public_api_names', return_value=mock_public_apis), \
             patch('src.core_concepts.pipelines.extract_classiq.ClassiqProcessor') as mock_processor_class, \
             patch('src.core_concepts.pipelines.extract_classiq.ConceptExtractor') as mock_extractor_class, \
             patch('src.core_concepts.pipelines.extract_classiq.ConceptStorage') as mock_storage_class, \
             patch('src.core_concepts.pipelines.extract_classiq.config', self.mock_config), \
             patch('logging.info') as mock_info:
            
            # Mock components
            mock_processor = MagicMock()
            mock_processor_class.return_value = mock_processor
            
            mock_extractor = MagicMock()
            mock_extractor.extract_from_package.return_value = mock_concepts
            mock_extractor_class.return_value = mock_extractor
            
            mock_storage = MagicMock()
            mock_storage_class.return_value = mock_storage
            
            main()
            
            # Verify that concepts were processed correctly
            mock_storage.save_all.assert_called_once_with(mock_concepts)
            
            # Verify logging includes success message
            log_messages = [call[0][0] for call in mock_info.call_args_list]
            assert any("Successfully extracted 2 unique Classiq concepts." in msg for msg in log_messages)

    def test_edge_case_empty_results(self):
        """Test edge case with empty results."""
        mock_sdk_root = Path("/test/classiq")
        mock_public_apis = {"function1"}
        
        with patch('src.core_concepts.pipelines.extract_classiq.get_sdk_root_path', return_value=mock_sdk_root), \
             patch('src.core_concepts.pipelines.extract_classiq.get_public_api_names', return_value=mock_public_apis), \
             patch('src.core_concepts.pipelines.extract_classiq.ClassiqProcessor') as mock_processor_class, \
             patch('src.core_concepts.pipelines.extract_classiq.ConceptExtractor') as mock_extractor_class, \
             patch('src.core_concepts.pipelines.extract_classiq.config', self.mock_config), \
             patch('logging.warning') as mock_warning, \
             patch('logging.info') as mock_info:
            
            # Mock components
            mock_processor = MagicMock()
            mock_processor_class.return_value = mock_processor
            
            mock_extractor = MagicMock()
            mock_extractor.extract_from_package.return_value = []  # Empty results
            mock_extractor_class.return_value = mock_extractor
            
            main()
            
            # Verify warning was logged
            mock_warning.assert_called_once_with("Extraction complete, but no concepts were found.")
            
            # Verify completion message
            log_messages = [call[0][0] for call in mock_info.call_args_list]
            assert any("--- Classiq Generation Complete ---" in msg for msg in log_messages)
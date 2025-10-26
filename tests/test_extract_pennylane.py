"""
Tests for src/core_concepts/pipelines/extract_pennylane.py

This module tests the PennyLane-specific concept extraction pipeline functionality,
focusing on the project's implementation rather than external framework features.
"""

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.core_concepts.pipelines.extract_pennylane import (
    main,
    PENNYLANE_PROJECT_ROOT,
    SEARCH_SUBDIRS,
)
from src.core_concepts.extractor.visitors import PennylaneClassVisitor
from src.core_concepts.extractor.processors import PennylaneProcessor


class TestMainFunction:
    """Test the main orchestration function."""

    def setup_method(self):
        """Set up test fixtures."""
        # Mock the config module
        self.mock_config = MagicMock()
        self.mock_config.PROJECT_ROOT = Path("/test/project")
        self.mock_config.RESULTS_DIR = Path("/test/results")

    def test_successful_extraction_pipeline(self):
        """Test successful execution of the complete pipeline."""
        mock_concepts = [
            {"name": "pennylane.template1", "summary": "Test template 1"},
            {"name": "pennylane.template2", "summary": "Test template 2"}
        ]
        
        with patch('src.core_concepts.pipelines.extract_pennylane.PENNYLANE_PROJECT_ROOT') as mock_root, \
             patch('pathlib.Path.is_dir', return_value=True), \
             patch('src.core_concepts.pipelines.extract_pennylane.PennylaneProcessor') as mock_processor_class, \
             patch('src.core_concepts.pipelines.extract_pennylane.ConceptExtractor') as mock_extractor_class, \
             patch('src.core_concepts.pipelines.extract_pennylane.ConceptStorage') as mock_storage_class, \
             patch('src.core_concepts.pipelines.extract_pennylane.config', self.mock_config), \
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
                visitor_class=PennylaneClassVisitor,
                processor=mock_processor_class.return_value
            )
            
            # Check that extraction was called with correct parameters
            mock_extractor.extract_from_package.assert_called_once_with(
                package_root=mock_root,
                search_paths=SEARCH_SUBDIRS,
            )
            
            # Check that storage was instantiated with correct paths
            mock_storage_class.assert_called_once_with(
                json_path=Path("/test/results/pennylane_quantum_concepts.json"),
                csv_path=Path("/test/results/pennylane_quantum_concepts.csv")
            )
            
            # Check that save_all was called
            mock_storage.save_all.assert_called_once_with(mock_concepts)
            
            # Check logging
            assert mock_info.call_count >= 3  # Multiple info messages

    def test_project_root_not_found(self):
        """Test handling when PennyLane project root is not found."""
        with patch('src.core_concepts.pipelines.extract_pennylane.PENNYLANE_PROJECT_ROOT') as mock_root, \
             patch.object(mock_root, 'is_dir', return_value=False), \
             patch('logging.error') as mock_error, \
             patch('logging.info') as mock_info:
            
            main()
            
            # Check error messages
            assert mock_error.call_count == 2
            mock_error.assert_any_call(
                f"PennyLane project root not found at '{mock_root.resolve()}'"
            )
            mock_error.assert_any_call("Please ensure the repository is cloned at that location.")
            
            # Should return early without further processing
            mock_info.assert_called_once_with("--- Starting PennyLane Core Concepts Generation ---")

    def test_no_concepts_found(self):
        """Test handling when no concepts are extracted."""
        with patch('src.core_concepts.pipelines.extract_pennylane.PENNYLANE_PROJECT_ROOT') as mock_root, \
             patch('pathlib.Path.is_dir', return_value=True), \
             patch('src.core_concepts.pipelines.extract_pennylane.PennylaneProcessor') as mock_processor_class, \
             patch('src.core_concepts.pipelines.extract_pennylane.ConceptExtractor') as mock_extractor_class, \
             patch('src.core_concepts.pipelines.extract_pennylane.config', self.mock_config), \
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
            mock_info.assert_any_call("--- PennyLane Generation Complete ---")

    def test_extraction_error_handling(self):
        """Test handling of extraction errors."""
        with patch('src.core_concepts.pipelines.extract_pennylane.PENNYLANE_PROJECT_ROOT') as mock_root, \
             patch('pathlib.Path.is_dir', return_value=True), \
             patch('src.core_concepts.pipelines.extract_pennylane.PennylaneProcessor') as mock_processor_class, \
             patch('src.core_concepts.pipelines.extract_pennylane.ConceptExtractor') as mock_extractor_class, \
             patch('src.core_concepts.pipelines.extract_pennylane.config', self.mock_config), \
             patch('logging.info') as mock_info:
            
            # Mock the extractor instance to raise an error
            mock_extractor = MagicMock()
            mock_extractor.extract_from_package.side_effect = Exception("Extraction failed")
            mock_extractor_class.return_value = mock_extractor
            
            # Should not raise an exception, but handle it gracefully
            with pytest.raises(Exception, match="Extraction failed"):
                main()

    def test_storage_error_handling(self):
        """Test handling of storage errors."""
        mock_concepts = [{"name": "pennylane.template1", "summary": "Test template 1"}]
        
        with patch('src.core_concepts.pipelines.extract_pennylane.PENNYLANE_PROJECT_ROOT') as mock_root, \
             patch('pathlib.Path.is_dir', return_value=True), \
             patch('src.core_concepts.pipelines.extract_pennylane.PennylaneProcessor') as mock_processor_class, \
             patch('src.core_concepts.pipelines.extract_pennylane.ConceptExtractor') as mock_extractor_class, \
             patch('src.core_concepts.pipelines.extract_pennylane.ConceptStorage') as mock_storage_class, \
             patch('src.core_concepts.pipelines.extract_pennylane.config', self.mock_config), \
             patch('logging.info') as mock_info:
            
            # Mock the extractor instance
            mock_extractor = MagicMock()
            mock_extractor.extract_from_package.return_value = mock_concepts
            mock_extractor_class.return_value = mock_extractor
            
            # Mock the storage instance to raise an error
            mock_storage = MagicMock()
            mock_storage.save_all.side_effect = Exception("Storage failed")
            mock_storage_class.return_value = mock_storage
            
            # Should not raise an exception, but handle it gracefully
            with pytest.raises(Exception, match="Storage failed"):
                main()


class TestConstants:
    """Test module constants."""

    def test_pennylane_project_root_constant(self):
        """Test that PENNYLANE_PROJECT_ROOT is defined correctly."""
        # This tests the structure, not the exact path since it depends on config
        assert isinstance(PENNYLANE_PROJECT_ROOT, Path)
        assert "pennylane" in str(PENNYLANE_PROJECT_ROOT)

    def test_search_subdirs_constant(self):
        """Test that SEARCH_SUBDIRS is defined correctly."""
        expected_paths = [
            "pennylane/templates/",
        ]
        assert SEARCH_SUBDIRS == expected_paths
        assert isinstance(SEARCH_SUBDIRS, list)
        assert len(SEARCH_SUBDIRS) == 1


class TestIntegration:
    """Integration tests for the PennyLane extraction pipeline."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_config = MagicMock()
        self.mock_config.PROJECT_ROOT = Path("/test/project")
        self.mock_config.RESULTS_DIR = Path("/test/results")

    def test_complete_pipeline_integration(self):
        """Test the complete pipeline integration."""
        mock_concepts = [
            {"name": "pennylane.template1", "summary": "Test template 1"},
            {"name": "pennylane.template2", "summary": "Test template 2"}
        ]
        
        with patch('src.core_concepts.pipelines.extract_pennylane.PENNYLANE_PROJECT_ROOT') as mock_root, \
             patch('pathlib.Path.is_dir', return_value=True), \
             patch('src.core_concepts.pipelines.extract_pennylane.PennylaneProcessor') as mock_processor_class, \
             patch('src.core_concepts.pipelines.extract_pennylane.ConceptExtractor') as mock_extractor_class, \
             patch('src.core_concepts.pipelines.extract_pennylane.ConceptStorage') as mock_storage_class, \
             patch('src.core_concepts.pipelines.extract_pennylane.config', self.mock_config), \
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
                package_root=mock_root,
                search_paths=SEARCH_SUBDIRS,
            )
            
            # Verify storage parameters
            mock_storage_class.assert_called_once_with(
                json_path=Path("/test/results/pennylane_quantum_concepts.json"),
                csv_path=Path("/test/results/pennylane_quantum_concepts.csv")
            )
            
            # Verify save operation
            mock_storage.save_all.assert_called_once_with(mock_concepts)

    def test_pipeline_with_realistic_data(self):
        """Test pipeline with realistic concept data."""
        mock_concepts = [
            {
                "name": "/pennylane/templates/quantum_template",
                "summary": "A quantum template for circuit construction",
                "docstring": "This template constructs quantum circuits.",
                "source_code": "class QuantumTemplate: pass"
            },
            {
                "name": "/pennylane/templates/classical_template", 
                "summary": "A classical template for data processing",
                "docstring": "This template processes classical data.",
                "source_code": "class ClassicalTemplate: pass"
            }
        ]
        
        with patch('src.core_concepts.pipelines.extract_pennylane.PENNYLANE_PROJECT_ROOT') as mock_root, \
             patch('pathlib.Path.is_dir', return_value=True), \
             patch('src.core_concepts.pipelines.extract_pennylane.PennylaneProcessor') as mock_processor_class, \
             patch('src.core_concepts.pipelines.extract_pennylane.ConceptExtractor') as mock_extractor_class, \
             patch('src.core_concepts.pipelines.extract_pennylane.ConceptStorage') as mock_storage_class, \
             patch('src.core_concepts.pipelines.extract_pennylane.config', self.mock_config), \
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
            assert any("Successfully extracted 2 unique PennyLane concepts." in msg for msg in log_messages)

    def test_edge_case_empty_results(self):
        """Test edge case with empty results."""
        with patch('src.core_concepts.pipelines.extract_pennylane.PENNYLANE_PROJECT_ROOT') as mock_root, \
             patch('pathlib.Path.is_dir', return_value=True), \
             patch('src.core_concepts.pipelines.extract_pennylane.PennylaneProcessor') as mock_processor_class, \
             patch('src.core_concepts.pipelines.extract_pennylane.ConceptExtractor') as mock_extractor_class, \
             patch('src.core_concepts.pipelines.extract_pennylane.config', self.mock_config), \
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
            assert any("--- PennyLane Generation Complete ---" in msg for msg in log_messages)

    def test_path_validation_integration(self):
        """Test path validation integration."""
        with patch('src.core_concepts.pipelines.extract_pennylane.PENNYLANE_PROJECT_ROOT') as mock_root, \
             patch.object(mock_root, 'is_dir', return_value=False), \
             patch('logging.error') as mock_error, \
             patch('logging.info') as mock_info:
            
            main()
            
            # Verify path validation
            mock_root.is_dir.assert_called_once()
            
            # Verify error logging
            assert mock_error.call_count == 2
            mock_error.assert_any_call(
                f"PennyLane project root not found at '{mock_root.resolve()}'"
            )
            mock_error.assert_any_call("Please ensure the repository is cloned at that location.")

    def test_component_instantiation_order(self):
        """Test that components are instantiated in the correct order."""
        mock_concepts = [{"name": "test", "summary": "test"}]
        
        with patch('src.core_concepts.pipelines.extract_pennylane.PENNYLANE_PROJECT_ROOT') as mock_root, \
             patch('pathlib.Path.is_dir', return_value=True), \
             patch('src.core_concepts.pipelines.extract_pennylane.PennylaneProcessor') as mock_processor_class, \
             patch('src.core_concepts.pipelines.extract_pennylane.ConceptExtractor') as mock_extractor_class, \
             patch('src.core_concepts.pipelines.extract_pennylane.ConceptStorage') as mock_storage_class, \
             patch('src.core_concepts.pipelines.extract_pennylane.config', self.mock_config), \
             patch('logging.info'):
            
            # Mock components
            mock_processor = MagicMock()
            mock_processor_class.return_value = mock_processor
            
            mock_extractor = MagicMock()
            mock_extractor.extract_from_package.return_value = mock_concepts
            mock_extractor_class.return_value = mock_extractor
            
            mock_storage = MagicMock()
            mock_storage_class.return_value = mock_storage
            
            main()
            
            # Verify instantiation order
            assert mock_processor_class.call_count == 1
            assert mock_extractor_class.call_count == 1
            assert mock_storage_class.call_count == 1
            
            # Verify that extractor was called with processor
            mock_extractor_class.assert_called_once_with(
                visitor_class=PennylaneClassVisitor,
                processor=mock_processor
            )

    def test_no_extra_kwargs_passed(self):
        """Test that no extra keyword arguments are passed to extractor."""
        mock_concepts = [{"name": "test", "summary": "test"}]
        
        with patch('src.core_concepts.pipelines.extract_pennylane.PENNYLANE_PROJECT_ROOT') as mock_root, \
             patch('pathlib.Path.is_dir', return_value=True), \
             patch('src.core_concepts.pipelines.extract_pennylane.PennylaneProcessor') as mock_processor_class, \
             patch('src.core_concepts.pipelines.extract_pennylane.ConceptExtractor') as mock_extractor_class, \
             patch('src.core_concepts.pipelines.extract_pennylane.ConceptStorage') as mock_storage_class, \
             patch('src.core_concepts.pipelines.extract_pennylane.config', self.mock_config), \
             patch('logging.info'):
            
            # Mock components
            mock_processor = MagicMock()
            mock_processor_class.return_value = mock_processor
            
            mock_extractor = MagicMock()
            mock_extractor.extract_from_package.return_value = mock_concepts
            mock_extractor_class.return_value = mock_extractor
            
            mock_storage = MagicMock()
            mock_storage_class.return_value = mock_storage
            
            main()
            
            # Verify that extract_from_package was called with only the required parameters
            mock_extractor.extract_from_package.assert_called_once_with(
                package_root=mock_root,
                search_paths=SEARCH_SUBDIRS,
            )
            
            # Verify no extra kwargs were passed (unlike Classiq which passes public_api_names)
            # The function does pass keyword arguments, but they are the standard ones
            call_args = mock_extractor.extract_from_package.call_args
            assert 'package_root' in call_args.kwargs
            assert 'search_paths' in call_args.kwargs
            # Verify no extra kwargs like 'public_api_names' are passed
            assert 'public_api_names' not in call_args.kwargs

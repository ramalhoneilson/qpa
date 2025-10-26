"""
Tests for src/core_concepts/pipelines/extract_qiskit.py

This module tests the Qiskit-specific concept extraction pipeline functionality,
focusing on the project's implementation rather than external framework features.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any
from unittest.mock import MagicMock, patch

import pytest

from src.core_concepts.pipelines.extract_qiskit import (
    main,
    _gather_qiskit_source_files,
    QISKIT_PROJECT_ROOT,
    SEARCH_SUBDIRS,
    EXCLUDE_SUBDIRS,
)
from src.core_concepts.extractor.visitors import QiskitVisitor
from src.core_concepts.extractor.processors import QiskitProcessor
from src.core_concepts.pipelines.qiskit_filters import (
    deduplicate_by_naming_convention,
    deduplicate_by_semantic_similarity,
)


class TestGatherQiskitSourceFiles:
    """Test the _gather_qiskit_source_files function."""

    def test_function_exists_and_callable(self):
        """Test that the function exists and is callable."""
        assert callable(_gather_qiskit_source_files)
        
    def test_function_returns_list(self):
        """Test that the function returns a list."""
        with patch('src.core_concepts.pipelines.extract_qiskit.QISKIT_PROJECT_ROOT') as mock_root, \
             patch.object(mock_root, 'is_dir', return_value=False):
            
            result = _gather_qiskit_source_files()
            assert isinstance(result, list)


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
        mock_files = [
            Path("/test/qiskit/circuit/library/gate1.py"),
            Path("/test/qiskit/circuit/library/gate2.py"),
        ]
        mock_concepts = [
            {"name": "qiskit.gate1", "summary": "Test gate 1"},
            {"name": "qiskit.gate2", "summary": "Test gate 2"}
        ]
        
        with patch('src.core_concepts.pipelines.extract_qiskit.QISKIT_PROJECT_ROOT') as mock_root, \
             patch.object(mock_root, 'is_dir', return_value=True), \
             patch('src.core_concepts.pipelines.extract_qiskit._gather_qiskit_source_files', return_value=mock_files), \
             patch('src.core_concepts.pipelines.extract_qiskit.QiskitProcessor') as mock_processor_class, \
             patch('src.core_concepts.pipelines.extract_qiskit.ConceptExtractor') as mock_extractor_class, \
             patch('src.core_concepts.pipelines.extract_qiskit.deduplicate_by_naming_convention', return_value=mock_concepts), \
             patch('src.core_concepts.pipelines.extract_qiskit.deduplicate_by_semantic_similarity', return_value=mock_concepts), \
             patch('src.core_concepts.pipelines.extract_qiskit.ConceptStorage') as mock_storage_class, \
             patch('src.core_concepts.pipelines.extract_qiskit.config', self.mock_config), \
             patch('logging.info') as mock_info:
            
            # Mock the extractor instance
            mock_extractor = MagicMock()
            mock_extractor._find_concepts_in_file.return_value = mock_concepts
            mock_extractor_class.return_value = mock_extractor
            
            # Mock the storage instance
            mock_storage = MagicMock()
            mock_storage_class.return_value = mock_storage
            
            main()
            
            # Check that components were instantiated correctly
            mock_processor_class.assert_called_once()
            mock_extractor_class.assert_called_once_with(
                visitor_class=QiskitVisitor,
                processor=mock_processor_class.return_value
            )
            
            # Check that file gathering was called
            assert mock_extractor._find_concepts_in_file.call_count == 2
            
            # Check that filtering was called
            assert mock_storage_class.call_count == 1
            
            # Check that save_all was called
            mock_storage.save_all.assert_called_once_with(mock_concepts)
            
            # Check logging
            assert mock_info.call_count >= 5  # Multiple info messages

    def test_project_root_not_found(self):
        """Test handling when Qiskit project root is not found."""
        with patch('src.core_concepts.pipelines.extract_qiskit.QISKIT_PROJECT_ROOT') as mock_root, \
             patch.object(mock_root, 'is_dir', return_value=False), \
             patch('logging.error') as mock_error, \
             patch('logging.info') as mock_info:
            
            main()
            
            # Check error message
            mock_error.assert_called_once_with(
                f"Qiskit project root not found at '{mock_root.resolve()}'"
            )
            
            # Should return early without further processing
            mock_info.assert_called_once_with("--- Starting Qiskit Core Quantum Concepts Generation ---")

    def test_no_source_files_found(self):
        """Test handling when no source files are found."""
        with patch('src.core_concepts.pipelines.extract_qiskit.QISKIT_PROJECT_ROOT') as mock_root, \
             patch.object(mock_root, 'is_dir', return_value=True), \
             patch('src.core_concepts.pipelines.extract_qiskit._gather_qiskit_source_files', return_value=[]), \
             patch('logging.warning') as mock_warning, \
             patch('logging.info') as mock_info:
            
            main()
            
            # Check warning message
            mock_warning.assert_called_once_with(
                "No Python files found in the Qiskit search directories after filtering."
            )

    def test_no_concepts_found_after_filtering(self):
        """Test handling when no concepts remain after filtering."""
        mock_files = [Path("/test/qiskit/circuit/library/gate1.py")]
        
        with patch('src.core_concepts.pipelines.extract_qiskit.QISKIT_PROJECT_ROOT') as mock_root, \
             patch.object(mock_root, 'is_dir', return_value=True), \
             patch('src.core_concepts.pipelines.extract_qiskit._gather_qiskit_source_files', return_value=mock_files), \
             patch('src.core_concepts.pipelines.extract_qiskit.QiskitProcessor') as mock_processor_class, \
             patch('src.core_concepts.pipelines.extract_qiskit.ConceptExtractor') as mock_extractor_class, \
             patch('src.core_concepts.pipelines.extract_qiskit.deduplicate_by_naming_convention', return_value=[]), \
             patch('src.core_concepts.pipelines.extract_qiskit.deduplicate_by_semantic_similarity', return_value=[]), \
             patch('src.core_concepts.pipelines.extract_qiskit.config', self.mock_config), \
             patch('logging.warning') as mock_warning, \
             patch('logging.info') as mock_info:
            
            # Mock the extractor instance
            mock_extractor = MagicMock()
            mock_extractor._find_concepts_in_file.return_value = []
            mock_extractor_class.return_value = mock_extractor
            
            main()
            
            # Check warning message
            mock_warning.assert_called_once_with("No quantum concepts remained after filtering.")
            
            # Check completion message
            log_messages = [call[0][0] for call in mock_info.call_args_list]
            assert any("--- Qiskit Generation Complete ---" in msg for msg in log_messages)

    def test_concept_deduplication(self):
        """Test that duplicate concepts are properly handled."""
        mock_files = [Path("/test/qiskit/circuit/library/gate1.py")]
        raw_concepts = [
            {"name": "qiskit.gate1", "summary": "Test gate 1"},
            {"name": "qiskit.gate1", "summary": "Test gate 1"},  # Duplicate
            {"name": "qiskit.gate2", "summary": "Test gate 2"}
        ]
        filtered_concepts = [
            {"name": "qiskit.gate1", "summary": "Test gate 1"},
            {"name": "qiskit.gate2", "summary": "Test gate 2"}
        ]
        
        with patch('src.core_concepts.pipelines.extract_qiskit.QISKIT_PROJECT_ROOT') as mock_root, \
             patch.object(mock_root, 'is_dir', return_value=True), \
             patch('src.core_concepts.pipelines.extract_qiskit._gather_qiskit_source_files', return_value=mock_files), \
             patch('src.core_concepts.pipelines.extract_qiskit.QiskitProcessor') as mock_processor_class, \
             patch('src.core_concepts.pipelines.extract_qiskit.ConceptExtractor') as mock_extractor_class, \
             patch('src.core_concepts.pipelines.extract_qiskit.deduplicate_by_naming_convention', return_value=filtered_concepts), \
             patch('src.core_concepts.pipelines.extract_qiskit.deduplicate_by_semantic_similarity', return_value=filtered_concepts), \
             patch('src.core_concepts.pipelines.extract_qiskit.ConceptStorage') as mock_storage_class, \
             patch('src.core_concepts.pipelines.extract_qiskit.config', self.mock_config), \
             patch('logging.info') as mock_info:
            
            # Mock the extractor instance
            mock_extractor = MagicMock()
            mock_extractor._find_concepts_in_file.return_value = raw_concepts
            mock_extractor_class.return_value = mock_extractor
            
            # Mock the storage instance
            mock_storage = MagicMock()
            mock_storage_class.return_value = mock_storage
            
            main()
            
            # Check that deduplication worked
            mock_storage.save_all.assert_called_once_with(filtered_concepts)


class TestConstants:
    """Test module constants."""

    def test_qiskit_project_root_constant(self):
        """Test that QISKIT_PROJECT_ROOT is defined correctly."""
        # This tests the structure, not the exact path since it depends on config
        assert isinstance(QISKIT_PROJECT_ROOT, Path)
        assert "qiskit" in str(QISKIT_PROJECT_ROOT)

    def test_search_subdirs_constant(self):
        """Test that SEARCH_SUBDIRS is defined correctly."""
        expected_paths = ["qiskit/circuit/library/"]
        assert SEARCH_SUBDIRS == expected_paths
        assert isinstance(SEARCH_SUBDIRS, list)
        assert len(SEARCH_SUBDIRS) == 1

    def test_exclude_subdirs_constant(self):
        """Test that EXCLUDE_SUBDIRS is defined correctly."""
        expected_exclusions = {"standard_gates", "templates"}
        assert EXCLUDE_SUBDIRS == expected_exclusions
        assert isinstance(EXCLUDE_SUBDIRS, set)
        assert len(EXCLUDE_SUBDIRS) == 2


class TestIntegration:
    """Integration tests for the Qiskit extraction pipeline."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_config = MagicMock()
        self.mock_config.PROJECT_ROOT = Path("/test/project")
        self.mock_config.RESULTS_DIR = Path("/test/results")

    def test_complete_pipeline_integration(self):
        """Test the complete pipeline integration."""
        mock_files = [
            Path("/test/qiskit/circuit/library/gate1.py"),
            Path("/test/qiskit/circuit/library/gate2.py"),
        ]
        raw_concepts = [
            {"name": "qiskit.gate1", "summary": "Test gate 1"},
            {"name": "qiskit.gate2", "summary": "Test gate 2"}
        ]
        filtered_concepts = [
            {"name": "qiskit.gate1", "summary": "Test gate 1"},
            {"name": "qiskit.gate2", "summary": "Test gate 2"}
        ]
        
        with patch('src.core_concepts.pipelines.extract_qiskit.QISKIT_PROJECT_ROOT') as mock_root, \
             patch.object(mock_root, 'is_dir', return_value=True), \
             patch('src.core_concepts.pipelines.extract_qiskit._gather_qiskit_source_files', return_value=mock_files), \
             patch('src.core_concepts.pipelines.extract_qiskit.QiskitProcessor') as mock_processor_class, \
             patch('src.core_concepts.pipelines.extract_qiskit.ConceptExtractor') as mock_extractor_class, \
             patch('src.core_concepts.pipelines.extract_qiskit.deduplicate_by_naming_convention', return_value=filtered_concepts), \
             patch('src.core_concepts.pipelines.extract_qiskit.deduplicate_by_semantic_similarity', return_value=filtered_concepts), \
             patch('src.core_concepts.pipelines.extract_qiskit.ConceptStorage') as mock_storage_class, \
             patch('src.core_concepts.pipelines.extract_qiskit.config', self.mock_config), \
             patch('logging.info') as mock_info:
            
            # Mock processor
            mock_processor = MagicMock()
            mock_processor_class.return_value = mock_processor
            
            # Mock extractor
            mock_extractor = MagicMock()
            mock_extractor._find_concepts_in_file.return_value = raw_concepts
            mock_extractor_class.return_value = mock_extractor
            
            # Mock storage
            mock_storage = MagicMock()
            mock_storage_class.return_value = mock_storage
            
            main()
            
            # Verify the complete workflow
            assert mock_processor_class.call_count == 1
            assert mock_extractor_class.call_count == 1
            assert mock_storage_class.call_count == 1
            
            # Verify file processing
            assert mock_extractor._find_concepts_in_file.call_count == 2
            
            # Verify filtering pipeline
            assert mock_storage_class.call_count == 1
            
            # Verify save operation
            mock_storage.save_all.assert_called_once_with(filtered_concepts)

    def test_pipeline_with_realistic_data(self):
        """Test pipeline with realistic concept data."""
        mock_files = [
            Path("/test/qiskit/circuit/library/h_gate.py"),
            Path("/test/qiskit/circuit/library/cx_gate.py"),
        ]
        raw_concepts = [
            {
                "name": "/qiskit/circuit/library/h_gate.HGate",
                "summary": "Hadamard gate for quantum circuits",
                "docstring": "This gate applies a Hadamard transformation.",
                "source_code": "class HGate: pass"
            },
            {
                "name": "/qiskit/circuit/library/cx_gate.CXGate", 
                "summary": "Controlled-X gate for quantum circuits",
                "docstring": "This gate applies a controlled X transformation.",
                "source_code": "class CXGate: pass"
            }
        ]
        filtered_concepts = raw_concepts  # No filtering in this test
        
        with patch('src.core_concepts.pipelines.extract_qiskit.QISKIT_PROJECT_ROOT') as mock_root, \
             patch.object(mock_root, 'is_dir', return_value=True), \
             patch('src.core_concepts.pipelines.extract_qiskit._gather_qiskit_source_files', return_value=mock_files), \
             patch('src.core_concepts.pipelines.extract_qiskit.QiskitProcessor') as mock_processor_class, \
             patch('src.core_concepts.pipelines.extract_qiskit.ConceptExtractor') as mock_extractor_class, \
             patch('src.core_concepts.pipelines.extract_qiskit.deduplicate_by_naming_convention', return_value=filtered_concepts), \
             patch('src.core_concepts.pipelines.extract_qiskit.deduplicate_by_semantic_similarity', return_value=filtered_concepts), \
             patch('src.core_concepts.pipelines.extract_qiskit.ConceptStorage') as mock_storage_class, \
             patch('src.core_concepts.pipelines.extract_qiskit.config', self.mock_config), \
             patch('logging.info') as mock_info:
            
            # Mock components
            mock_processor = MagicMock()
            mock_processor_class.return_value = mock_processor
            
            mock_extractor = MagicMock()
            mock_extractor._find_concepts_in_file.return_value = raw_concepts
            mock_extractor_class.return_value = mock_extractor
            
            mock_storage = MagicMock()
            mock_storage_class.return_value = mock_storage
            
            main()
            
            # Verify that concepts were processed correctly
            mock_storage.save_all.assert_called_once_with(filtered_concepts)
            
            # Verify logging includes success message
            log_messages = [call[0][0] for call in mock_info.call_args_list]
            assert any("Successfully identified 2 unique, filtered quantum concepts." in msg for msg in log_messages)

    def test_filtering_pipeline_steps(self):
        """Test that the filtering pipeline steps are executed in order."""
        mock_files = [Path("/test/qiskit/circuit/library/gate1.py")]
        raw_concepts = [{"name": "qiskit.gate1", "summary": "Test gate 1"}]
        convention_filtered = [{"name": "qiskit.gate1", "summary": "Test gate 1"}]
        final_concepts = [{"name": "qiskit.gate1", "summary": "Test gate 1"}]
        
        with patch('src.core_concepts.pipelines.extract_qiskit.QISKIT_PROJECT_ROOT') as mock_root, \
             patch.object(mock_root, 'is_dir', return_value=True), \
             patch('src.core_concepts.pipelines.extract_qiskit._gather_qiskit_source_files', return_value=mock_files), \
             patch('src.core_concepts.pipelines.extract_qiskit.QiskitProcessor') as mock_processor_class, \
             patch('src.core_concepts.pipelines.extract_qiskit.ConceptExtractor') as mock_extractor_class, \
             patch('src.core_concepts.pipelines.extract_qiskit.deduplicate_by_naming_convention', return_value=convention_filtered) as mock_naming, \
             patch('src.core_concepts.pipelines.extract_qiskit.deduplicate_by_semantic_similarity', return_value=final_concepts) as mock_semantic, \
             patch('src.core_concepts.pipelines.extract_qiskit.ConceptStorage') as mock_storage_class, \
             patch('src.core_concepts.pipelines.extract_qiskit.config', self.mock_config), \
             patch('logging.info') as mock_info:
            
            # Mock components
            mock_processor = MagicMock()
            mock_processor_class.return_value = mock_processor
            
            mock_extractor = MagicMock()
            mock_extractor._find_concepts_in_file.return_value = raw_concepts
            mock_extractor_class.return_value = mock_extractor
            
            mock_storage = MagicMock()
            mock_storage_class.return_value = mock_storage
            
            main()
            
            # Verify filtering steps were called in order
            mock_naming.assert_called_once_with(raw_concepts)
            mock_semantic.assert_called_once_with(convention_filtered)
            
            # Verify final storage
            mock_storage.save_all.assert_called_once_with(final_concepts)

    def test_edge_case_empty_results(self):
        """Test edge case with empty results."""
        mock_files = [Path("/test/qiskit/circuit/library/gate1.py")]
        
        with patch('src.core_concepts.pipelines.extract_qiskit.QISKIT_PROJECT_ROOT') as mock_root, \
             patch.object(mock_root, 'is_dir', return_value=True), \
             patch('src.core_concepts.pipelines.extract_qiskit._gather_qiskit_source_files', return_value=mock_files), \
             patch('src.core_concepts.pipelines.extract_qiskit.QiskitProcessor') as mock_processor_class, \
             patch('src.core_concepts.pipelines.extract_qiskit.ConceptExtractor') as mock_extractor_class, \
             patch('src.core_concepts.pipelines.extract_qiskit.deduplicate_by_naming_convention', return_value=[]), \
             patch('src.core_concepts.pipelines.extract_qiskit.deduplicate_by_semantic_similarity', return_value=[]), \
             patch('src.core_concepts.pipelines.extract_qiskit.config', self.mock_config), \
             patch('logging.warning') as mock_warning, \
             patch('logging.info') as mock_info:
            
            # Mock components
            mock_processor = MagicMock()
            mock_processor_class.return_value = mock_processor
            
            mock_extractor = MagicMock()
            mock_extractor._find_concepts_in_file.return_value = []
            mock_extractor_class.return_value = mock_extractor
            
            main()
            
            # Verify warning was logged
            mock_warning.assert_called_once_with("No quantum concepts remained after filtering.")
            
            # Verify completion message
            log_messages = [call[0][0] for call in mock_info.call_args_list]
            assert any("--- Qiskit Generation Complete ---" in msg for msg in log_messages)

    def test_file_iteration_logic(self):
        """Test that files are processed individually."""
        mock_files = [
            Path("/test/qiskit/circuit/library/gate1.py"),
            Path("/test/qiskit/circuit/library/gate2.py"),
        ]
        concepts_per_file = [
            [{"name": "qiskit.gate1", "summary": "Test gate 1"}],
            [{"name": "qiskit.gate2", "summary": "Test gate 2"}]
        ]
        
        with patch('src.core_concepts.pipelines.extract_qiskit.QISKIT_PROJECT_ROOT') as mock_root, \
             patch.object(mock_root, 'is_dir', return_value=True), \
             patch('src.core_concepts.pipelines.extract_qiskit._gather_qiskit_source_files', return_value=mock_files), \
             patch('src.core_concepts.pipelines.extract_qiskit.QiskitProcessor') as mock_processor_class, \
             patch('src.core_concepts.pipelines.extract_qiskit.ConceptExtractor') as mock_extractor_class, \
             patch('src.core_concepts.pipelines.extract_qiskit.deduplicate_by_naming_convention', return_value=[]), \
             patch('src.core_concepts.pipelines.extract_qiskit.deduplicate_by_semantic_similarity', return_value=[]), \
             patch('src.core_concepts.pipelines.extract_qiskit.config', self.mock_config), \
             patch('logging.info') as mock_info:
            
            # Mock components
            mock_processor = MagicMock()
            mock_processor_class.return_value = mock_processor
            
            mock_extractor = MagicMock()
            mock_extractor._find_concepts_in_file.side_effect = concepts_per_file
            mock_extractor_class.return_value = mock_extractor
            
            main()
            
            # Verify that _find_concepts_in_file was called for each file
            assert mock_extractor._find_concepts_in_file.call_count == 2
            mock_extractor._find_concepts_in_file.assert_any_call(mock_files[0], mock_root)
            mock_extractor._find_concepts_in_file.assert_any_call(mock_files[1], mock_root)

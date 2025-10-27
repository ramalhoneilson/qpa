"""
Tests for src/preprocessing/knowledge_base_consolidator.py

This module tests the knowledge base consolidation functionality,
including CSV file processing, DataFrame concatenation, and output generation.
"""

from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pandas as pd
import pytest

from src.preprocessing.knowledge_base_consolidator import (
    KnowledgeBaseConsolidator,
    main,
)


class TestKnowledgeBaseConsolidator:
    """Test the KnowledgeBaseConsolidator class."""

    def test_init(self):
        """Test KnowledgeBaseConsolidator initialization."""
        consolidator = KnowledgeBaseConsolidator()
        
        assert consolidator.kb_dir is not None
        assert consolidator.output_file is not None
        assert len(consolidator.input_files) == 3
        assert "classiq" in consolidator.input_files
        assert "pennylane" in consolidator.input_files
        assert "qiskit" in consolidator.input_files

    def test_consolidate_knowledge_base_success(self):
        """Test successful knowledge base consolidation."""
        consolidator = KnowledgeBaseConsolidator()
        
        # Mock dataframes for each framework
        mock_dataframes = {
            "classiq": pd.DataFrame({
                "concept": ["classiq.concept1", "classiq.concept2"],
                "summary": ["Summary 1", "Summary 2"]
            }),
            "pennylane": pd.DataFrame({
                "concept": ["pennylane.concept1"],
                "summary": ["Summary 3"]
            }),
            "qiskit": pd.DataFrame({
                "concept": ["qiskit.concept1", "qiskit.concept2", "qiskit.concept3"],
                "summary": ["Summary 4", "Summary 5", "Summary 6"]
            })
        }
        
        with patch.object(consolidator, 'input_files', {
            "classiq": Path("/test/classiq.csv"),
            "pennylane": Path("/test/pennylane.csv"),
            "qiskit": Path("/test/qiskit.csv")
        }), \
        patch('pandas.read_csv', side_effect=lambda path, **kwargs: mock_dataframes[path.stem]), \
        patch.object(consolidator.kb_dir, 'mkdir'), \
        patch.object(consolidator.output_file, 'parent', consolidator.kb_dir), \
        patch('pandas.DataFrame.to_csv') as mock_to_csv:
            
            consolidator.consolidate_knowledge_base()
            
            # Verify to_csv was called
            mock_to_csv.assert_called_once()

    def test_consolidate_knowledge_base_missing_files(self):
        """Test consolidation when some input files are missing."""
        consolidator = KnowledgeBaseConsolidator()
        
        with patch.object(consolidator, 'input_files', {
            "classiq": Path("/test/classiq.csv"),
            "pennylane": Path("/test/pennylane.csv"),
            "qiskit": Path("/test/qiskit.csv")
        }), \
        patch('pathlib.Path.exists', side_effect=lambda path: path.name != "pennylane.csv"), \
        patch('pandas.read_csv') as mock_read_csv, \
        patch.object(consolidator.kb_dir, 'mkdir'), \
        patch('pandas.DataFrame.to_csv') as mock_to_csv:
            
            # Mock successful reads for existing files
            mock_read_csv.return_value = pd.DataFrame({
                "concept": ["test.concept"],
                "summary": ["Test summary"]
            })
            
            consolidator.consolidate_knowledge_base()
            
            # Should still call to_csv with available data
            mock_to_csv.assert_called_once()

    def test_consolidate_knowledge_base_no_files(self):
        """Test consolidation when no input files exist."""
        consolidator = KnowledgeBaseConsolidator()
        
        with patch.object(consolidator, 'input_files', {
            "classiq": Path("/test/classiq.csv"),
            "pennylane": Path("/test/pennylane.csv"),
            "qiskit": Path("/test/qiskit.csv")
        }), \
        patch('pathlib.Path.exists', return_value=False):
            
            consolidator.consolidate_knowledge_base()
            # Should not raise any exceptions

    def test_consolidate_knowledge_base_read_error(self):
        """Test consolidation when CSV reading fails."""
        consolidator = KnowledgeBaseConsolidator()
        
        with patch.object(consolidator, 'input_files', {
            "classiq": Path("/test/classiq.csv")
        }), \
        patch('pathlib.Path.exists', return_value=True), \
        patch('pandas.read_csv', side_effect=pd.errors.EmptyDataError("Empty file")):
            
            consolidator.consolidate_knowledge_base()
            # Should not raise any exceptions

    def test_get_consolidated_data_success(self):
        """Test successful loading of consolidated data."""
        consolidator = KnowledgeBaseConsolidator()
        mock_df = pd.DataFrame({
            "concept": ["test.concept"],
            "summary": ["Test summary"],
            "framework": ["test"]
        })
        
        with patch('pandas.read_csv', return_value=mock_df), \
        patch('pathlib.Path.exists', return_value=True):
            
            result = consolidator.get_consolidated_data()
            
            assert isinstance(result, pd.DataFrame)
            assert len(result) == 1
            assert "framework" in result.columns

    def test_get_consolidated_data_file_not_found(self):
        """Test loading when consolidated file doesn't exist."""
        consolidator = KnowledgeBaseConsolidator()
        
        with patch('pathlib.Path.exists', return_value=False):
            result = consolidator.get_consolidated_data()
            
            assert isinstance(result, pd.DataFrame)
            assert result.empty

    def test_get_consolidated_data_read_error(self):
        """Test loading when CSV reading fails."""
        consolidator = KnowledgeBaseConsolidator()
        
        with patch('pathlib.Path.exists', return_value=True), \
        patch('pandas.read_csv', side_effect=pd.errors.EmptyDataError("Empty file")):
            
            result = consolidator.get_consolidated_data()
            
            assert isinstance(result, pd.DataFrame)
            assert result.empty

    def test_get_framework_data_success(self):
        """Test getting data for specific framework."""
        consolidator = KnowledgeBaseConsolidator()
        mock_df = pd.DataFrame({
            "concept": ["test.concept1", "test.concept2"],
            "summary": ["Summary 1", "Summary 2"],
            "framework": ["test", "test"]
        })
        
        with patch.object(consolidator, 'get_consolidated_data', return_value=mock_df):
            result = consolidator.get_framework_data("test")
            
            assert isinstance(result, pd.DataFrame)
            assert len(result) == 2
            assert all(result["framework"] == "test")

    def test_get_framework_data_empty_data(self):
        """Test getting framework data when consolidated data is empty."""
        consolidator = KnowledgeBaseConsolidator()
        
        with patch.object(consolidator, 'get_consolidated_data', return_value=pd.DataFrame()):
            result = consolidator.get_framework_data("test")
            
            assert isinstance(result, pd.DataFrame)
            assert result.empty

    def test_get_framework_data_no_framework_column(self):
        """Test getting framework data when framework column doesn't exist."""
        consolidator = KnowledgeBaseConsolidator()
        mock_df = pd.DataFrame({
            "concept": ["test.concept"],
            "summary": ["Test summary"]
        })
        
        with patch.object(consolidator, 'get_consolidated_data', return_value=mock_df):
            result = consolidator.get_framework_data("test")
            
            assert isinstance(result, pd.DataFrame)
            assert result.empty

    def test_get_pattern_counts_success(self):
        """Test getting pattern counts by framework."""
        consolidator = KnowledgeBaseConsolidator()
        mock_df = pd.DataFrame({
            "concept": ["test.concept1", "test.concept2", "other.concept"],
            "summary": ["Summary 1", "Summary 2", "Summary 3"],
            "framework": ["test", "test", "other"]
        })
        
        with patch.object(consolidator, 'get_consolidated_data', return_value=mock_df):
            result = consolidator.get_pattern_counts()
            
            assert isinstance(result, dict)
            assert result["test"] == 2
            assert result["other"] == 1

    def test_get_pattern_counts_empty_data(self):
        """Test getting pattern counts when data is empty."""
        consolidator = KnowledgeBaseConsolidator()
        
        with patch.object(consolidator, 'get_consolidated_data', return_value=pd.DataFrame()):
            result = consolidator.get_pattern_counts()
            
            assert isinstance(result, dict)
            assert result == {}

    def test_get_pattern_counts_no_framework_column(self):
        """Test getting pattern counts when framework column doesn't exist."""
        consolidator = KnowledgeBaseConsolidator()
        mock_df = pd.DataFrame({
            "concept": ["test.concept"],
            "summary": ["Test summary"]
        })
        
        with patch.object(consolidator, 'get_consolidated_data', return_value=mock_df):
            result = consolidator.get_pattern_counts()
            
            assert isinstance(result, dict)
            assert result == {}


class TestMainFunction:
    """Test the main function."""

    def test_main_function(self):
        """Test the main function execution."""
        with patch('src.preprocessing.knowledge_base_consolidator.KnowledgeBaseConsolidator') as mock_class:
            mock_instance = MagicMock()
            mock_class.return_value = mock_instance
            
            main()
            
            mock_instance.consolidate_knowledge_base.assert_called_once()


class TestIntegration:
    """Integration tests for knowledge base consolidation."""

    def test_complete_workflow_integration(self):
        """Test the complete consolidation workflow."""
        consolidator = KnowledgeBaseConsolidator()
        
        # Mock all external dependencies
        with patch.object(consolidator, 'input_files', {
            "classiq": Path("/test/classiq.csv"),
            "pennylane": Path("/test/pennylane.csv"),
            "qiskit": Path("/test/qiskit.csv")
        }), \
        patch('pathlib.Path.exists', return_value=True), \
        patch('pandas.read_csv') as mock_read_csv, \
        patch.object(consolidator.kb_dir, 'mkdir'), \
        patch('pandas.DataFrame.to_csv') as mock_to_csv:
            
            # Mock successful reads
            mock_read_csv.return_value = pd.DataFrame({
                "concept": ["test.concept"],
                "summary": ["Test summary"]
            })
            
            consolidator.consolidate_knowledge_base()
            
            # Verify the workflow was executed
            mock_to_csv.assert_called_once()

    def test_error_handling_integration(self):
        """Test error handling in the complete workflow."""
        consolidator = KnowledgeBaseConsolidator()
        
        with patch.object(consolidator, 'input_files', {
            "classiq": Path("/test/classiq.csv")
        }), \
        patch('pathlib.Path.exists', return_value=True), \
        patch('pandas.read_csv', side_effect=Exception("Read error")):
            
            consolidator.consolidate_knowledge_base()
            # Should not raise any exceptions
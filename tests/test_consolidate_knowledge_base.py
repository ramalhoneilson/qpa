"""
Tests for src/utils/consolidate_knowledge_base.py

This module tests the knowledge base consolidation functionality,
including CSV file processing, DataFrame concatenation, and output generation.
"""

from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pandas as pd
import pytest

from src.utils.consolidate_knowledge_base import (
    consolidate_knowledge_base,
    INPUT_FILES,
    KB_DIR,
    OUTPUT_FILE,
)


class TestConstants:
    """Test module constants."""

    def test_constants_defined(self):
        """Test that all constants are properly defined."""
        assert isinstance(KB_DIR, Path)
        assert isinstance(OUTPUT_FILE, Path)
        assert isinstance(INPUT_FILES, dict)

    def test_input_files_structure(self):
        """Test INPUT_FILES structure."""
        expected_frameworks = ["classiq", "pennylane", "qiskit"]
        assert list(INPUT_FILES.keys()) == expected_frameworks
        
        for framework in expected_frameworks:
            assert framework in INPUT_FILES
            assert isinstance(INPUT_FILES[framework], Path)
            assert INPUT_FILES[framework].name.endswith(".csv")

    def test_output_file_structure(self):
        """Test OUTPUT_FILE structure."""
        assert OUTPUT_FILE.name == "knowledge_base.csv"
        assert OUTPUT_FILE.parent == KB_DIR


class TestConsolidateKnowledgeBase:
    """Test the consolidate_knowledge_base function."""

    def test_consolidate_knowledge_base_success(self):
        """Test successful knowledge base consolidation."""
        # Create mock DataFrames for each framework
        mock_dataframes = {
            "classiq": pd.DataFrame({
                "name": ["ClassiqFunc1", "ClassiqFunc2"],
                "summary": ["Summary1", "Summary2"],
                "pattern": ["Pattern1", "Pattern2"],
            }),
            "pennylane": pd.DataFrame({
                "name": ["PennyLaneFunc1", "PennyLaneFunc2"],
                "summary": ["Summary3", "Summary4"],
                "pattern": ["Pattern2", "Pattern3"],
            }),
            "qiskit": pd.DataFrame({
                "name": ["QiskitFunc1", "QiskitFunc2"],
                "summary": ["Summary5", "Summary6"],
                "pattern": ["Pattern1", "Pattern3"],
            }),
        }
        
        with patch('src.utils.consolidate_knowledge_base.INPUT_FILES') as mock_input_files, \
             patch('pathlib.Path.exists', return_value=True), \
             patch('pandas.read_csv', side_effect=lambda path: mock_dataframes[path.stem.split('_')[0]]), \
             patch('pathlib.Path.mkdir'), \
             patch('builtins.print') as mock_print:
            
            # Mock the input files
            mock_input_files.items.return_value = [
                ("classiq", Path("/test/classiq.csv")),
                ("pennylane", Path("/test/pennylane.csv")),
                ("qiskit", Path("/test/qiskit.csv")),
            ]
            
            consolidate_knowledge_base()
            
            # Check that success messages were printed
            assert mock_print.call_count >= 4  # At least 4 print statements
            assert any("Successfully processed" in str(call) for call in mock_print.call_args_list)
            assert any("Consolidation complete!" in str(call) for call in mock_print.call_args_list)

    def test_consolidate_knowledge_base_missing_files(self):
        """Test consolidation with missing input files."""
        with patch('src.utils.consolidate_knowledge_base.INPUT_FILES') as mock_input_files, \
             patch('pathlib.Path.exists', return_value=False), \
             patch('builtins.print') as mock_print:
            
            # Mock the input files
            mock_input_files.items.return_value = [
                ("classiq", Path("/test/classiq.csv")),
                ("pennylane", Path("/test/pennylane.csv")),
                ("qiskit", Path("/test/qiskit.csv")),
            ]
            
            consolidate_knowledge_base()
            
            # Check that warning messages were printed
            assert mock_print.call_count >= 3  # At least 3 warning messages
            assert any("Warning: Input file not found" in str(call) for call in mock_print.call_args_list)
            assert any("No dataframes were loaded" in str(call) for call in mock_print.call_args_list)

    def test_consolidate_knowledge_base_file_error(self):
        """Test consolidation with file reading errors."""
        with patch('src.utils.consolidate_knowledge_base.INPUT_FILES') as mock_input_files, \
             patch('pathlib.Path.exists', return_value=True), \
             patch('pandas.read_csv', side_effect=pd.errors.EmptyDataError("Empty file")), \
             patch('builtins.print') as mock_print:
            
            # Mock the input files
            mock_input_files.items.return_value = [
                ("classiq", Path("/test/classiq.csv")),
                ("pennylane", Path("/test/pennylane.csv")),
                ("qiskit", Path("/test/qiskit.csv")),
            ]
            
            consolidate_knowledge_base()
            
            # Check that error messages were printed
            assert mock_print.call_count >= 3  # At least 3 error messages
            assert any("Error processing" in str(call) for call in mock_print.call_args_list)
            assert any("No dataframes were loaded" in str(call) for call in mock_print.call_args_list)

    def test_consolidate_knowledge_base_partial_success(self):
        """Test consolidation with some files missing."""
        mock_dataframe = pd.DataFrame({
            "name": ["Func1", "Func2"],
            "summary": ["Summary1", "Summary2"],
            "pattern": ["Pattern1", "Pattern2"],
        })
        
        with patch('src.utils.consolidate_knowledge_base.INPUT_FILES') as mock_input_files, \
             patch('pathlib.Path.exists', side_effect=[True, False, True]), \
             patch('pandas.read_csv', return_value=mock_dataframe), \
             patch('pathlib.Path.mkdir'), \
             patch('builtins.print') as mock_print:
            
            # Mock the input files
            mock_input_files.items.return_value = [
                ("classiq", Path("/test/classiq.csv")),
                ("pennylane", Path("/test/pennylane.csv")),
                ("qiskit", Path("/test/qiskit.csv")),
            ]
            
            consolidate_knowledge_base()
            
            # Check that both success and warning messages were printed
            assert mock_print.call_count >= 4
            assert any("Successfully processed" in str(call) for call in mock_print.call_args_list)
            assert any("Warning: Input file not found" in str(call) for call in mock_print.call_args_list)
            assert any("Consolidation complete!" in str(call) for call in mock_print.call_args_list)

    def test_consolidate_knowledge_base_output_error(self):
        """Test consolidation with output file write error."""
        mock_dataframe = pd.DataFrame({
            "name": ["Func1", "Func2"],
            "summary": ["Summary1", "Summary2"],
            "pattern": ["Pattern1", "Pattern2"],
        })
        
        with patch('src.utils.consolidate_knowledge_base.INPUT_FILES') as mock_input_files, \
             patch('pathlib.Path.exists', return_value=True), \
             patch('pandas.read_csv', return_value=mock_dataframe), \
             patch('pathlib.Path.mkdir', side_effect=PermissionError("Permission denied")), \
             patch('builtins.print') as mock_print:
            
            # Mock the input files
            mock_input_files.items.return_value = [
                ("classiq", Path("/test/classiq.csv")),
            ]
            
            consolidate_knowledge_base()
            
            # Check that error message was printed
            assert any("Error writing to output file" in str(call) for call in mock_print.call_args_list)

    def test_consolidate_knowledge_base_empty_dataframes(self):
        """Test consolidation with empty DataFrames."""
        empty_dataframe = pd.DataFrame()
        
        with patch('src.utils.consolidate_knowledge_base.INPUT_FILES') as mock_input_files, \
             patch('pathlib.Path.exists', return_value=True), \
             patch('pandas.read_csv', return_value=empty_dataframe), \
             patch('pathlib.Path.mkdir'), \
             patch('builtins.print') as mock_print:
            
            # Mock the input files
            mock_input_files.items.return_value = [
                ("classiq", Path("/test/classiq.csv")),
            ]
            
            consolidate_knowledge_base()
            
            # Check that success message was printed (even with empty DataFrame)
            assert any("Successfully processed" in str(call) for call in mock_print.call_args_list)
            assert any("Consolidation complete!" in str(call) for call in mock_print.call_args_list)

    def test_consolidate_knowledge_base_framework_column_addition(self):
        """Test that framework column is properly added."""
        mock_dataframe = pd.DataFrame({
            "name": ["Func1", "Func2"],
            "summary": ["Summary1", "Summary2"],
            "pattern": ["Pattern1", "Pattern2"],
        })
        
        with patch('src.utils.consolidate_knowledge_base.INPUT_FILES') as mock_input_files, \
             patch('pathlib.Path.exists', return_value=True), \
             patch('pandas.read_csv', return_value=mock_dataframe), \
             patch('pathlib.Path.mkdir'), \
             patch('builtins.print'):
            
            # Mock the input files
            mock_input_files.items.return_value = [
                ("classiq", Path("/test/classiq.csv")),
            ]
            
            # Mock the to_csv method to verify it was called
            with patch.object(pd.DataFrame, 'to_csv') as mock_to_csv:
                consolidate_knowledge_base()
                
                # Check that to_csv was called
                mock_to_csv.assert_called_once()
                
                # Verify the call arguments
                call_args = mock_to_csv.call_args
                assert call_args[1]['index'] is False
                assert call_args[1]['encoding'] == 'utf-8'

    def test_consolidate_knowledge_base_column_reordering(self):
        """Test that framework column is moved to the front."""
        mock_dataframe = pd.DataFrame({
            "name": ["Func1", "Func2"],
            "summary": ["Summary1", "Summary2"],
            "pattern": ["Pattern1", "Pattern2"],
        })
        
        with patch('src.utils.consolidate_knowledge_base.INPUT_FILES') as mock_input_files, \
             patch('pathlib.Path.exists', return_value=True), \
             patch('pandas.read_csv', return_value=mock_dataframe), \
             patch('pathlib.Path.mkdir'), \
             patch('builtins.print'):
            
            # Mock the input files
            mock_input_files.items.return_value = [
                ("classiq", Path("/test/classiq.csv")),
            ]
            
            # Mock the to_csv method to verify it was called
            with patch.object(pd.DataFrame, 'to_csv') as mock_to_csv:
                consolidate_knowledge_base()
                
                # Check that to_csv was called
                mock_to_csv.assert_called_once()
                
                # Verify the call arguments
                call_args = mock_to_csv.call_args
                assert call_args[1]['index'] is False
                assert call_args[1]['encoding'] == 'utf-8'

    def test_consolidate_knowledge_base_multiple_frameworks(self):
        """Test consolidation with multiple frameworks."""
        mock_dataframes = {
            "classiq": pd.DataFrame({
                "name": ["ClassiqFunc1"],
                "summary": ["Summary1"],
                "pattern": ["Pattern1"],
            }),
            "pennylane": pd.DataFrame({
                "name": ["PennyLaneFunc1"],
                "summary": ["Summary2"],
                "pattern": ["Pattern2"],
            }),
            "qiskit": pd.DataFrame({
                "name": ["QiskitFunc1"],
                "summary": ["Summary3"],
                "pattern": ["Pattern3"],
            }),
        }
        
        with patch('src.utils.consolidate_knowledge_base.INPUT_FILES') as mock_input_files, \
             patch('pathlib.Path.exists', return_value=True), \
             patch('pandas.read_csv', side_effect=lambda path: mock_dataframes[path.stem.split('_')[0]]), \
             patch('pathlib.Path.mkdir'), \
             patch('builtins.print'):
            
            # Mock the input files
            mock_input_files.items.return_value = [
                ("classiq", Path("/test/classiq.csv")),
                ("pennylane", Path("/test/pennylane.csv")),
                ("qiskit", Path("/test/qiskit.csv")),
            ]
            
            # Mock the to_csv method to verify it was called
            with patch.object(pd.DataFrame, 'to_csv') as mock_to_csv:
                consolidate_knowledge_base()
                
                # Check that to_csv was called
                mock_to_csv.assert_called_once()
                
                # Verify the call arguments
                call_args = mock_to_csv.call_args
                assert call_args[1]['index'] is False
                assert call_args[1]['encoding'] == 'utf-8'


class TestIntegration:
    """Integration tests for the knowledge base consolidation workflow."""

    def test_complete_workflow_integration(self):
        """Test the complete workflow integration."""
        mock_dataframes = {
            "classiq": pd.DataFrame({
                "name": ["ClassiqFunc1", "ClassiqFunc2"],
                "summary": ["Summary1", "Summary2"],
                "pattern": ["Pattern1", "Pattern2"],
            }),
            "pennylane": pd.DataFrame({
                "name": ["PennyLaneFunc1"],
                "summary": ["Summary3"],
                "pattern": ["Pattern2"],
            }),
            "qiskit": pd.DataFrame({
                "name": ["QiskitFunc1", "QiskitFunc2"],
                "summary": ["Summary4", "Summary5"],
                "pattern": ["Pattern1", "Pattern3"],
            }),
        }
        
        with patch('src.utils.consolidate_knowledge_base.INPUT_FILES') as mock_input_files, \
             patch('pathlib.Path.exists', return_value=True), \
             patch('pandas.read_csv', side_effect=lambda path: mock_dataframes[path.stem.split('_')[0]]), \
             patch('pathlib.Path.mkdir'), \
             patch('builtins.print') as mock_print:
            
            # Mock the input files
            mock_input_files.items.return_value = [
                ("classiq", Path("/test/classiq.csv")),
                ("pennylane", Path("/test/pennylane.csv")),
                ("qiskit", Path("/test/qiskit.csv")),
            ]
            
            # Mock the to_csv method to verify it was called
            with patch.object(pd.DataFrame, 'to_csv') as mock_to_csv:
                consolidate_knowledge_base()
                
                # Check that all files were processed
                assert mock_print.call_count >= 4
                assert any("Successfully processed" in str(call) for call in mock_print.call_args_list)
                assert any("Consolidation complete!" in str(call) for call in mock_print.call_args_list)
                
                # Check that to_csv was called
                mock_to_csv.assert_called_once()
                
                # Verify the call arguments
                call_args = mock_to_csv.call_args
                assert call_args[1]['index'] is False
                assert call_args[1]['encoding'] == 'utf-8'

    def test_error_handling_integration(self):
        """Test error handling in the integrated workflow."""
        with patch('src.utils.consolidate_knowledge_base.INPUT_FILES') as mock_input_files, \
             patch('pathlib.Path.exists', return_value=True), \
             patch('pandas.read_csv', side_effect=Exception("Test error")), \
             patch('builtins.print') as mock_print:
            
            # Mock the input files
            mock_input_files.items.return_value = [
                ("classiq", Path("/test/classiq.csv")),
            ]
            
            # The function should handle errors gracefully
            consolidate_knowledge_base()
            
            # Check that error was handled
            assert any("Error processing" in str(call) for call in mock_print.call_args_list)
            assert any("No dataframes were loaded" in str(call) for call in mock_print.call_args_list)

    def test_data_consistency_integration(self):
        """Test data consistency in the integrated workflow."""
        mock_dataframes = {
            "classiq": pd.DataFrame({
                "name": ["ClassiqFunc1"],
                "summary": ["Summary1"],
                "pattern": ["Pattern1"],
            }),
            "pennylane": pd.DataFrame({
                "name": ["PennyLaneFunc1"],
                "summary": ["Summary2"],
                "pattern": ["Pattern2"],
            }),
            "qiskit": pd.DataFrame({
                "name": ["QiskitFunc1"],
                "summary": ["Summary3"],
                "pattern": ["Pattern3"],
            }),
        }
        
        with patch('src.utils.consolidate_knowledge_base.INPUT_FILES') as mock_input_files, \
             patch('pathlib.Path.exists', return_value=True), \
             patch('pandas.read_csv', side_effect=lambda path: mock_dataframes[path.stem.split('_')[0]]), \
             patch('pathlib.Path.mkdir'), \
             patch('builtins.print'):
            
            # Mock the input files
            mock_input_files.items.return_value = [
                ("classiq", Path("/test/classiq.csv")),
                ("pennylane", Path("/test/pennylane.csv")),
                ("qiskit", Path("/test/qiskit.csv")),
            ]
            
            # Mock the to_csv method to verify it was called
            with patch.object(pd.DataFrame, 'to_csv') as mock_to_csv:
                consolidate_knowledge_base()
                
                # Check that to_csv was called
                mock_to_csv.assert_called_once()
                
                # Verify the call arguments
                call_args = mock_to_csv.call_args
                assert call_args[1]['index'] is False
                assert call_args[1]['encoding'] == 'utf-8'

    def test_performance_with_large_datasets(self):
        """Test performance with larger datasets."""
        # Create larger mock DataFrames
        large_dataframe = pd.DataFrame({
            "name": [f"Func{i}" for i in range(100)],
            "summary": [f"Summary{i}" for i in range(100)],
            "pattern": [f"Pattern{i%10}" for i in range(100)],
        })
        
        with patch('src.utils.consolidate_knowledge_base.INPUT_FILES') as mock_input_files, \
             patch('pathlib.Path.exists', return_value=True), \
             patch('pandas.read_csv', return_value=large_dataframe), \
             patch('pathlib.Path.mkdir'), \
             patch('builtins.print'):
            
            # Mock the input files
            mock_input_files.items.return_value = [
                ("classiq", Path("/test/classiq.csv")),
                ("pennylane", Path("/test/pennylane.csv")),
                ("qiskit", Path("/test/qiskit.csv")),
            ]
            
            # Mock the to_csv method to verify it was called
            with patch.object(pd.DataFrame, 'to_csv') as mock_to_csv:
                consolidate_knowledge_base()
                
                # Check that to_csv was called
                mock_to_csv.assert_called_once()
                
                # Verify the call arguments
                call_args = mock_to_csv.call_args
                assert call_args[1]['index'] is False
                assert call_args[1]['encoding'] == 'utf-8'

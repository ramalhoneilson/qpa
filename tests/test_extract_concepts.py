"""
Tests for src/extraction/extract_concepts.py

This module tests the workflow orchestration functionality for running
all core concept extraction pipelines in sequence.
"""

import logging
import time
from unittest.mock import MagicMock, patch

import pytest

from src.extraction.extract_concepts import (
    main,
    run_pipeline,
)


class TestRunPipeline:
    """Test the run_pipeline helper function."""

    def test_successful_pipeline_execution(self):
        """Test successful execution of a pipeline."""
        mock_main_function = MagicMock()
        pipeline_name = "TestPipeline"
        
        with patch('time.time', side_effect=[1000.0, 1005.0]), \
             patch('logging.info') as mock_info, \
             patch('logging.error') as mock_error:
            
            run_pipeline(pipeline_name, mock_main_function)
            
            # Check that the main function was called
            mock_main_function.assert_called_once()
            
            # Check logging messages
            mock_info.assert_any_call(f"--- Starting {pipeline_name} Concept Extraction Pipeline ---")
            mock_info.assert_any_call(f"--- Finished {pipeline_name} Pipeline successfully in 5.00 seconds ---\n")
            
            # Check that no error was logged
            mock_error.assert_not_called()

    def test_pipeline_execution_with_exception(self):
        """Test pipeline execution when an exception occurs."""
        mock_main_function = MagicMock()
        mock_main_function.side_effect = Exception("Test error")
        pipeline_name = "TestPipeline"
        
        with patch('time.time', side_effect=[1000.0, 1005.0]), \
             patch('logging.info') as mock_info, \
             patch('logging.error') as mock_error:
            
            run_pipeline(pipeline_name, mock_main_function)
            
            # Check that the main function was called
            mock_main_function.assert_called_once()
            
            # Check logging messages
            mock_info.assert_any_call(f"--- Starting {pipeline_name} Concept Extraction Pipeline ---")
            
            # Check error logging
            mock_error.assert_any_call(
                f"!!! The {pipeline_name} Pipeline failed after 5.00 seconds: Test error",
                exc_info=True
            )
            mock_error.assert_any_call(f"--- Aborting {pipeline_name} Pipeline due to error ---\n")

    def test_timing_accuracy(self):
        """Test that timing is calculated correctly."""
        mock_main_function = MagicMock()
        pipeline_name = "TestPipeline"
        
        with patch('time.time', side_effect=[1000.0, 1003.5]), \
             patch('logging.info') as mock_info, \
             patch('logging.error') as mock_error:
            
            run_pipeline(pipeline_name, mock_main_function)
            
            # Check that timing is calculated correctly (3.5 seconds)
            mock_info.assert_any_call(f"--- Finished {pipeline_name} Pipeline successfully in 3.50 seconds ---\n")

    def test_timing_with_exception(self):
        """Test that timing is calculated correctly even when an exception occurs."""
        mock_main_function = MagicMock()
        mock_main_function.side_effect = Exception("Test error")
        pipeline_name = "TestPipeline"
        
        with patch('time.time', side_effect=[1000.0, 1002.7]), \
             patch('logging.info') as mock_info, \
             patch('logging.error') as mock_error:
            
            run_pipeline(pipeline_name, mock_main_function)
            
            # Check that timing is calculated correctly (2.7 seconds)
            mock_error.assert_any_call(
                f"!!! The {pipeline_name} Pipeline failed after 2.70 seconds: Test error",
                exc_info=True
            )

    def test_different_pipeline_names(self):
        """Test that different pipeline names are handled correctly."""
        mock_main_function = MagicMock()
        
        test_names = ["Classiq", "PennyLane", "Qiskit", "CustomPipeline"]
        
        for name in test_names:
            with patch('time.time', side_effect=[1000.0, 1001.0]), \
                 patch('logging.info') as mock_info, \
                 patch('logging.error') as mock_error:
                
                run_pipeline(name, mock_main_function)
                
                # Check that the correct name is used in logging
                mock_info.assert_any_call(f"--- Starting {name} Concept Extraction Pipeline ---")
                mock_info.assert_any_call(f"--- Finished {name} Pipeline successfully in 1.00 seconds ---\n")
                
                # Reset mock for next iteration
                mock_main_function.reset_mock()

    def test_exception_info_logging(self):
        """Test that exception info is logged when exc_info=True."""
        mock_main_function = MagicMock()
        mock_main_function.side_effect = ValueError("Test ValueError")
        pipeline_name = "TestPipeline"
        
        with patch('time.time', side_effect=[1000.0, 1001.0]), \
             patch('logging.info') as mock_info, \
             patch('logging.error') as mock_error:
            
            run_pipeline(pipeline_name, mock_main_function)
            
            # Check that exc_info=True is passed to logging.error
            error_calls = mock_error.call_args_list
            assert len(error_calls) == 2
            
            # First call should have exc_info=True
            first_call = error_calls[0]
            assert first_call[1]['exc_info'] is True
            
            # Second call should not have exc_info
            second_call = error_calls[1]
            assert 'exc_info' not in second_call[1]


class TestMainFunction:
    """Test the main workflow function."""

    def test_successful_workflow_execution(self):
        """Test successful execution of the complete workflow."""
        mock_classiq_main = MagicMock()
        mock_pennylane_main = MagicMock()
        mock_qiskit_main = MagicMock()
        
        with patch('src.extraction.extract_concepts.extract_classiq.main', mock_classiq_main), \
             patch('src.extraction.extract_concepts.extract_pennylane.main', mock_pennylane_main), \
             patch('src.extraction.extract_concepts.extract_qiskit.main', mock_qiskit_main), \
             patch('src.extraction.extract_concepts.run_pipeline') as mock_run_pipeline, \
             patch('logging.info') as mock_info:
            
            main()
            
            # Check that run_pipeline was called for each pipeline
            assert mock_run_pipeline.call_count == 3
            
            # Check that pipelines were called in the correct order
            expected_calls = [
                (("Classiq", mock_classiq_main),),
                (("PennyLane", mock_pennylane_main),),
                (("Qiskit", mock_qiskit_main),),
            ]
            assert mock_run_pipeline.call_args_list == expected_calls
            
            # Check workflow logging
            mock_info.assert_any_call("=========================================================")
            mock_info.assert_any_call("=== STARTING COMPLETE CORE CONCEPTS EXTRACTION WORKFLOW ===")
            mock_info.assert_any_call("=========================================================\n")
            mock_info.assert_any_call("=======================================================")
            mock_info.assert_any_call("=== ALL EXTRACTION WORKFLOWS HAVE BEEN EXECUTED ===")
            mock_info.assert_any_call("=======================================================")

    def test_workflow_with_pipeline_failure(self):
        """Test workflow execution when one pipeline fails."""
        mock_classiq_main = MagicMock()
        mock_pennylane_main = MagicMock()
        mock_pennylane_main.side_effect = Exception("PennyLane failed")
        mock_qiskit_main = MagicMock()
        
        with patch('src.extraction.extract_concepts.extract_classiq.main', mock_classiq_main), \
             patch('src.extraction.extract_concepts.extract_pennylane.main', mock_pennylane_main), \
             patch('src.extraction.extract_concepts.extract_qiskit.main', mock_qiskit_main), \
             patch('src.extraction.extract_concepts.run_pipeline') as mock_run_pipeline, \
             patch('logging.info') as mock_info:
            
            main()
            
            # Check that run_pipeline was called for all pipelines (including failed one)
            assert mock_run_pipeline.call_count == 3
            
            # Check that all pipelines were attempted
            expected_calls = [
                (("Classiq", mock_classiq_main),),
                (("PennyLane", mock_pennylane_main),),
                (("Qiskit", mock_qiskit_main),),
            ]
            assert mock_run_pipeline.call_args_list == expected_calls
            
            # Check workflow completion logging
            mock_info.assert_any_call("=== ALL EXTRACTION WORKFLOWS HAVE BEEN EXECUTED ===")

    def test_workflow_pipeline_order(self):
        """Test that pipelines are executed in the correct order."""
        mock_classiq_main = MagicMock()
        mock_pennylane_main = MagicMock()
        mock_qiskit_main = MagicMock()
        
        with patch('src.extraction.extract_concepts.extract_classiq.main', mock_classiq_main), \
             patch('src.extraction.extract_concepts.extract_pennylane.main', mock_pennylane_main), \
             patch('src.extraction.extract_concepts.extract_qiskit.main', mock_qiskit_main), \
             patch('src.extraction.extract_concepts.run_pipeline') as mock_run_pipeline, \
             patch('logging.info'):
            
            main()
            
            # Check that pipelines were called in the correct order
            call_args_list = mock_run_pipeline.call_args_list
            
            # First call should be Classiq
            assert call_args_list[0][0][0] == "Classiq"
            assert call_args_list[0][0][1] == mock_classiq_main
            
            # Second call should be PennyLane
            assert call_args_list[1][0][0] == "PennyLane"
            assert call_args_list[1][0][1] == mock_pennylane_main
            
            # Third call should be Qiskit
            assert call_args_list[2][0][0] == "Qiskit"
            assert call_args_list[2][0][1] == mock_qiskit_main

    def test_workflow_logging_structure(self):
        """Test that the workflow logging has the correct structure."""
        mock_classiq_main = MagicMock()
        mock_pennylane_main = MagicMock()
        mock_qiskit_main = MagicMock()
        
        with patch('src.extraction.extract_concepts.extract_classiq.main', mock_classiq_main), \
             patch('src.extraction.extract_concepts.extract_pennylane.main', mock_pennylane_main), \
             patch('src.extraction.extract_concepts.extract_qiskit.main', mock_qiskit_main), \
             patch('src.extraction.extract_concepts.run_pipeline'), \
             patch('logging.info') as mock_info:
            
            main()
            
            # Check that the workflow logging messages are present
            info_calls = [call[0][0] for call in mock_info.call_args_list]
            
            # Check start messages
            assert "=========================================================" in info_calls
            assert "=== STARTING COMPLETE CORE CONCEPTS EXTRACTION WORKFLOW ===" in info_calls
            
            # Check end messages
            assert "=======================================================" in info_calls
            assert "=== ALL EXTRACTION WORKFLOWS HAVE BEEN EXECUTED ===" in info_calls

    def test_workflow_continues_after_failure(self):
        """Test that the workflow continues even if one pipeline fails."""
        mock_classiq_main = MagicMock()
        mock_pennylane_main = MagicMock()
        mock_pennylane_main.side_effect = Exception("PennyLane failed")
        mock_qiskit_main = MagicMock()
        
        with patch('src.extraction.extract_concepts.extract_classiq.main', mock_classiq_main), \
             patch('src.extraction.extract_concepts.extract_pennylane.main', mock_pennylane_main), \
             patch('src.extraction.extract_concepts.extract_qiskit.main', mock_qiskit_main), \
             patch('src.extraction.extract_concepts.run_pipeline') as mock_run_pipeline, \
             patch('logging.info') as mock_info:
            
            main()
            
            # Check that all three pipelines were attempted
            assert mock_run_pipeline.call_count == 3
            
            # Check that the workflow completed successfully
            info_calls = [call[0][0] for call in mock_info.call_args_list]
            assert "=== ALL EXTRACTION WORKFLOWS HAVE BEEN EXECUTED ===" in info_calls

    def test_workflow_with_all_failures(self):
        """Test workflow execution when all pipelines fail."""
        mock_classiq_main = MagicMock()
        mock_classiq_main.side_effect = Exception("Classiq failed")
        mock_pennylane_main = MagicMock()
        mock_pennylane_main.side_effect = Exception("PennyLane failed")
        mock_qiskit_main = MagicMock()
        mock_qiskit_main.side_effect = Exception("Qiskit failed")
        
        with patch('src.extraction.extract_concepts.extract_classiq.main', mock_classiq_main), \
             patch('src.extraction.extract_concepts.extract_pennylane.main', mock_pennylane_main), \
             patch('src.extraction.extract_concepts.extract_qiskit.main', mock_qiskit_main), \
             patch('src.extraction.extract_concepts.run_pipeline') as mock_run_pipeline, \
             patch('logging.info') as mock_info:
            
            main()
            
            # Check that all three pipelines were attempted
            assert mock_run_pipeline.call_count == 3
            
            # Check that the workflow still completed
            info_calls = [call[0][0] for call in mock_info.call_args_list]
            assert "=== ALL EXTRACTION WORKFLOWS HAVE BEEN EXECUTED ===" in info_calls


class TestIntegration:
    """Integration tests for the workflow orchestration."""

    def test_workflow_with_real_pipeline_functions(self):
        """Test workflow with actual pipeline function references."""
        # This test verifies that the pipeline functions are correctly referenced
        with patch('src.extraction.extract_concepts.extract_classiq.main') as mock_classiq, \
             patch('src.extraction.extract_concepts.extract_pennylane.main') as mock_pennylane, \
             patch('src.extraction.extract_concepts.extract_qiskit.main') as mock_qiskit, \
             patch('src.extraction.extract_concepts.run_pipeline') as mock_run_pipeline, \
             patch('logging.info'):
            
            main()
            
            # Verify that the correct functions are passed to run_pipeline
            call_args_list = mock_run_pipeline.call_args_list
            
            # Check that the actual main functions are passed
            assert call_args_list[0][0][1] == mock_classiq
            assert call_args_list[1][0][1] == mock_pennylane
            assert call_args_list[2][0][1] == mock_qiskit

    def test_workflow_error_handling_integration(self):
        """Test that errors in individual pipelines don't stop the workflow."""
        mock_classiq_main = MagicMock()
        mock_pennylane_main = MagicMock()
        mock_pennylane_main.side_effect = RuntimeError("PennyLane runtime error")
        mock_qiskit_main = MagicMock()
        
        with patch('src.extraction.extract_concepts.extract_classiq.main', mock_classiq_main), \
             patch('src.extraction.extract_concepts.extract_pennylane.main', mock_pennylane_main), \
             patch('src.extraction.extract_concepts.extract_qiskit.main', mock_qiskit_main), \
             patch('src.extraction.extract_concepts.run_pipeline') as mock_run_pipeline, \
             patch('logging.info') as mock_info:
            
            # The workflow should not raise an exception even if individual pipelines fail
            main()
            
            # Verify all pipelines were attempted
            assert mock_run_pipeline.call_count == 3
            
            # Verify workflow completion
            info_calls = [call[0][0] for call in mock_info.call_args_list]
            assert "=== ALL EXTRACTION WORKFLOWS HAVE BEEN EXECUTED ===" in info_calls

    def test_workflow_timing_integration(self):
        """Test that timing works correctly in the integrated workflow."""
        mock_classiq_main = MagicMock()
        mock_pennylane_main = MagicMock()
        mock_qiskit_main = MagicMock()
        
        with patch('src.extraction.extract_concepts.extract_classiq.main', mock_classiq_main), \
             patch('src.extraction.extract_concepts.extract_pennylane.main', mock_pennylane_main), \
             patch('src.extraction.extract_concepts.extract_qiskit.main', mock_qiskit_main), \
             patch('src.extraction.extract_concepts.run_pipeline') as mock_run_pipeline, \
             patch('logging.info'):
            
            main()
            
            # Verify that run_pipeline was called for each pipeline
            assert mock_run_pipeline.call_count == 3
            
            # Each call should have the correct pipeline name and function
            for i, (name, func) in enumerate([("Classiq", mock_classiq_main), 
                                           ("PennyLane", mock_pennylane_main), 
                                           ("Qiskit", mock_qiskit_main)]):
                call_args = mock_run_pipeline.call_args_list[i][0]
                assert call_args[0] == name
                assert call_args[1] == func

    def test_workflow_logging_format(self):
        """Test that the logging format is consistent throughout the workflow."""
        mock_classiq_main = MagicMock()
        mock_pennylane_main = MagicMock()
        mock_qiskit_main = MagicMock()
        
        with patch('src.extraction.extract_concepts.extract_classiq.main', mock_classiq_main), \
             patch('src.extraction.extract_concepts.extract_pennylane.main', mock_pennylane_main), \
             patch('src.extraction.extract_concepts.extract_qiskit.main', mock_qiskit_main), \
             patch('src.extraction.extract_concepts.run_pipeline'), \
             patch('logging.info') as mock_info:
            
            main()
            
            # Check that the workflow uses consistent logging format
            info_calls = [call[0][0] for call in mock_info.call_args_list]
            
            # Check for the specific workflow messages
            assert any("STARTING COMPLETE CORE CONCEPTS EXTRACTION WORKFLOW" in msg for msg in info_calls)
            assert any("ALL EXTRACTION WORKFLOWS HAVE BEEN EXECUTED" in msg for msg in info_calls)
            
            # Check that the separator lines are present
            assert any("=" * 57 in msg for msg in info_calls)  # Start separator
            assert any("=" * 55 in msg for msg in info_calls)  # End separator


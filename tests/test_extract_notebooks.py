"""
Tests for src/preprocessing/extract_notebooks.py

This module tests the notebook archiving functionality,
including project discovery, notebook copying, and error handling.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.preprocessing.extract_notebooks import (
    TARGET_PROJECTS,
    TARGET_PROJECTS_BASE_PATH,
    NOTEBOOKS_DEST_ROOT,
    archive_notebooks,
)


class TestConstants:
    """Test module constants."""

    def test_constants_defined(self):
        """Test that all constants are properly defined."""
        assert TARGET_PROJECTS_BASE_PATH is not None
        assert TARGET_PROJECTS is not None
        assert NOTEBOOKS_DEST_ROOT is not None

    def test_constants_types(self):
        """Test that constants have correct types."""
        assert isinstance(TARGET_PROJECTS_BASE_PATH, Path)
        assert isinstance(TARGET_PROJECTS, list)
        assert isinstance(NOTEBOOKS_DEST_ROOT, Path)

    def test_target_projects_structure(self):
        """Test TARGET_PROJECTS structure."""
        assert len(TARGET_PROJECTS) > 0
        for project in TARGET_PROJECTS:
            assert isinstance(project, str)
            # TARGET_PROJECTS contains directory names, not paths starting with /
            assert len(project) > 0


class TestArchiveNotebooks:
    """Test the archive_notebooks function."""

    def test_archive_notebooks_success(self):
        """Test successful notebook archiving."""
        mock_notebooks = [
            Path("/test/project1/notebook1.ipynb"),
            Path("/test/project1/subdir/notebook2.ipynb"),
        ]
        
        with patch('src.preprocessing.extract_notebooks.TARGET_PROJECTS', ["/project1"]), \
             patch('src.preprocessing.extract_notebooks.TARGET_PROJECTS_BASE_PATH', Path("/test")), \
             patch('src.preprocessing.extract_notebooks.NOTEBOOKS_DEST_ROOT', Path("/dest")), \
             patch('pathlib.Path.is_dir', return_value=True), \
             patch('pathlib.Path.rglob', return_value=mock_notebooks), \
             patch('pathlib.Path.relative_to', side_effect=lambda x: Path("notebook1.ipynb") if "notebook1" in str(x) else Path("subdir/notebook2.ipynb")), \
             patch('pathlib.Path.mkdir'), \
             patch('shutil.copy2') as mock_copy, \
             patch('builtins.print') as mock_print:
            
            archive_notebooks()
            
            # Check that notebooks were copied
            assert mock_copy.call_count == 2
            mock_print.assert_any_call("--- Starting Notebook Archiving Step ---")
            mock_print.assert_any_call("Successfully archived: 2 notebooks.")

    def test_archive_notebooks_project_not_found(self):
        """Test handling when project directory doesn't exist."""
        with patch('src.preprocessing.extract_notebooks.TARGET_PROJECTS', ["/nonexistent"]), \
             patch('src.preprocessing.extract_notebooks.TARGET_PROJECTS_BASE_PATH', Path("/test")), \
             patch('src.preprocessing.extract_notebooks.NOTEBOOKS_DEST_ROOT', Path("/dest")), \
             patch('pathlib.Path.is_dir', return_value=False), \
             patch('builtins.print') as mock_print:
            
            archive_notebooks()
            
            # Check that warning was printed
            mock_print.assert_any_call("--- Starting Notebook Archiving Step ---")
            # Check that warning message was printed (exact format may vary)
            call_strings = [str(call) for call in mock_print.call_args_list]
            combined_calls = " ".join(call_strings)
            assert "WARNING" in combined_calls
            assert "skipping" in combined_calls
            mock_print.assert_any_call("Successfully archived: 0 notebooks.")

    def test_archive_notebooks_no_notebooks(self):
        """Test handling when no notebooks are found."""
        with patch('src.preprocessing.extract_notebooks.TARGET_PROJECTS', ["/project1"]), \
             patch('src.preprocessing.extract_notebooks.TARGET_PROJECTS_BASE_PATH', Path("/test")), \
             patch('src.preprocessing.extract_notebooks.NOTEBOOKS_DEST_ROOT', Path("/dest")), \
             patch('pathlib.Path.is_dir', return_value=True), \
             patch('pathlib.Path.rglob', return_value=[]), \
             patch('builtins.print') as mock_print:
            
            archive_notebooks()
            
            # Check that appropriate message was printed
            mock_print.assert_any_call("Successfully archived: 0 notebooks.")

    def test_archive_notebooks_copy_error(self):
        """Test handling of copy errors."""
        mock_notebooks = [Path("/test/project1/notebook1.ipynb")]
        
        with patch('src.preprocessing.extract_notebooks.TARGET_PROJECTS', ["/project1"]), \
             patch('src.preprocessing.extract_notebooks.TARGET_PROJECTS_BASE_PATH', Path("/test")), \
             patch('src.preprocessing.extract_notebooks.NOTEBOOKS_DEST_ROOT', Path("/dest")), \
             patch('pathlib.Path.is_dir', return_value=True), \
             patch('pathlib.Path.rglob', return_value=mock_notebooks), \
             patch('pathlib.Path.relative_to', return_value=Path("notebook1.ipynb")), \
             patch('pathlib.Path.mkdir'), \
             patch('shutil.copy2', side_effect=PermissionError("Permission denied")), \
             patch('builtins.print') as mock_print:
            
            archive_notebooks()
            
            # Check that error was handled
            mock_print.assert_any_call("[ERROR] Failed processing /project1: Permission denied")
            mock_print.assert_any_call("Successfully archived: 0 notebooks.")
            mock_print.assert_any_call("Errors encountered:    1 projects.")

    def test_archive_notebooks_multiple_projects(self):
        """Test archiving from multiple projects."""
        mock_notebooks_project1 = [Path("/test/project1/notebook1.ipynb")]
        mock_notebooks_project2 = [Path("/test/project2/notebook2.ipynb")]
        
        with patch('src.preprocessing.extract_notebooks.TARGET_PROJECTS', ["/project1", "/project2"]), \
             patch('src.preprocessing.extract_notebooks.TARGET_PROJECTS_BASE_PATH', Path("/test")), \
             patch('src.preprocessing.extract_notebooks.NOTEBOOKS_DEST_ROOT', Path("/dest")), \
             patch('pathlib.Path.is_dir', return_value=True), \
             patch('pathlib.Path.rglob', side_effect=[mock_notebooks_project1, mock_notebooks_project2]), \
             patch('pathlib.Path.relative_to', side_effect=lambda x: Path("notebook1.ipynb") if "project1" in str(x) else Path("notebook2.ipynb")), \
             patch('pathlib.Path.mkdir'), \
             patch('shutil.copy2') as mock_copy, \
             patch('builtins.print') as mock_print:
            
            archive_notebooks()
            
            # Check that both projects were processed
            assert mock_copy.call_count == 2
            # Check that processing messages were printed (exact format may vary)
            call_strings = [str(call) for call in mock_print.call_args_list]
            combined_calls = " ".join(call_strings)
            assert "Processing project" in combined_calls
            assert "project1" in combined_calls
            assert "project2" in combined_calls
            mock_print.assert_any_call("Successfully archived: 2 notebooks.")

    def test_archive_notebooks_nested_directory_structure(self):
        """Test archiving with nested directory structure."""
        mock_notebooks = [
            Path("/test/project1/level1/level2/notebook.ipynb"),
        ]
        
        with patch('src.preprocessing.extract_notebooks.TARGET_PROJECTS', ["/project1"]), \
             patch('src.preprocessing.extract_notebooks.TARGET_PROJECTS_BASE_PATH', Path("/test")), \
             patch('src.preprocessing.extract_notebooks.NOTEBOOKS_DEST_ROOT', Path("/dest")), \
             patch('pathlib.Path.is_dir', return_value=True), \
             patch('pathlib.Path.rglob', return_value=mock_notebooks), \
             patch('pathlib.Path.relative_to', return_value=Path("level1/level2/notebook.ipynb")), \
             patch('pathlib.Path.mkdir') as mock_mkdir, \
             patch('shutil.copy2') as mock_copy, \
             patch('builtins.print') as mock_print:
            
            archive_notebooks()
            
            # Check that nested directories were created
            mock_mkdir.assert_called_with(parents=True, exist_ok=True)
            mock_copy.assert_called_once()
            mock_print.assert_any_call("Successfully archived: 1 notebooks.")

    def test_archive_notebooks_project_name_conversion(self):
        """Test project name conversion for directory names."""
        mock_notebooks = [Path("/test/project/subdir/notebook.ipynb")]
        
        with patch('src.preprocessing.extract_notebooks.TARGET_PROJECTS', ["/project/subdir"]), \
             patch('src.preprocessing.extract_notebooks.TARGET_PROJECTS_BASE_PATH', Path("/test")), \
             patch('src.preprocessing.extract_notebooks.NOTEBOOKS_DEST_ROOT', Path("/dest")), \
             patch('pathlib.Path.is_dir', return_value=True), \
             patch('pathlib.Path.rglob', return_value=mock_notebooks), \
             patch('pathlib.Path.relative_to', return_value=Path("notebook.ipynb")), \
             patch('pathlib.Path.mkdir'), \
             patch('shutil.copy2') as mock_copy, \
             patch('builtins.print') as mock_print:
            
            archive_notebooks()
            
            # Check that project name was converted correctly
            mock_copy.assert_called_once()
            call_args = mock_copy.call_args
            dest_path = call_args[0][1]
            assert "project_subdir" in str(dest_path)

    def test_archive_notebooks_mixed_success_failure(self):
        """Test archiving with mixed success and failure."""
        mock_notebooks_project1 = [Path("/test/project1/notebook1.ipynb")]
        
        with patch('src.preprocessing.extract_notebooks.TARGET_PROJECTS', ["/project1", "/nonexistent"]), \
             patch('src.preprocessing.extract_notebooks.TARGET_PROJECTS_BASE_PATH', Path("/test")), \
             patch('src.preprocessing.extract_notebooks.NOTEBOOKS_DEST_ROOT', Path("/dest")), \
             patch('pathlib.Path.is_dir', side_effect=[True, False]), \
             patch('pathlib.Path.rglob', return_value=mock_notebooks_project1), \
             patch('pathlib.Path.relative_to', return_value=Path("notebook1.ipynb")), \
             patch('pathlib.Path.mkdir'), \
             patch('shutil.copy2') as mock_copy, \
             patch('builtins.print') as mock_print:
            
            archive_notebooks()
            
            # Check that one project succeeded and one failed
            mock_copy.assert_called_once()
            mock_print.assert_any_call("Successfully archived: 1 notebooks.")
            # Check that warning message was printed (exact format may vary)
            call_strings = [str(call) for call in mock_print.call_args_list]
            combined_calls = " ".join(call_strings)
            assert "WARNING" in combined_calls
            assert "skipping" in combined_calls

    def test_archive_notebooks_empty_project_list(self):
        """Test archiving with empty project list."""
        with patch('src.preprocessing.extract_notebooks.TARGET_PROJECTS', []), \
             patch('src.preprocessing.extract_notebooks.TARGET_PROJECTS_BASE_PATH', Path("/test")), \
             patch('src.preprocessing.extract_notebooks.NOTEBOOKS_DEST_ROOT', Path("/dest")), \
             patch('builtins.print') as mock_print:
            
            archive_notebooks()
            
            # Check that appropriate message was printed
            mock_print.assert_any_call("Successfully archived: 0 notebooks.")

    def test_archive_notebooks_destination_directory_creation(self):
        """Test that destination directories are created."""
        mock_notebooks = [Path("/test/project1/notebook.ipynb")]
        
        with patch('src.preprocessing.extract_notebooks.TARGET_PROJECTS', ["/project1"]), \
             patch('src.preprocessing.extract_notebooks.TARGET_PROJECTS_BASE_PATH', Path("/test")), \
             patch('src.preprocessing.extract_notebooks.NOTEBOOKS_DEST_ROOT', Path("/dest")), \
             patch('pathlib.Path.is_dir', return_value=True), \
             patch('pathlib.Path.rglob', return_value=mock_notebooks), \
             patch('pathlib.Path.relative_to', return_value=Path("notebook.ipynb")), \
             patch('pathlib.Path.mkdir') as mock_mkdir, \
             patch('shutil.copy2'), \
             patch('builtins.print'):
            
            archive_notebooks()
            
            # Check that mkdir was called with correct parameters
            mock_mkdir.assert_called_with(parents=True, exist_ok=True)


class TestIntegration:
    """Integration tests for the notebook archiving workflow."""

    def test_complete_workflow_integration(self):
        """Test the complete workflow integration."""
        mock_notebooks = [
            Path("/test/project1/notebook1.ipynb"),
            Path("/test/project1/subdir/notebook2.ipynb"),
            Path("/test/project2/notebook3.ipynb"),
        ]
        
        with patch('src.preprocessing.extract_notebooks.TARGET_PROJECTS', ["/project1", "/project2"]), \
             patch('src.preprocessing.extract_notebooks.TARGET_PROJECTS_BASE_PATH', Path("/test")), \
             patch('src.preprocessing.extract_notebooks.NOTEBOOKS_DEST_ROOT', Path("/dest")), \
             patch('pathlib.Path.is_dir', return_value=True), \
             patch('pathlib.Path.rglob', side_effect=[
                 [mock_notebooks[0], mock_notebooks[1]],  # project1
                 [mock_notebooks[2]]  # project2
             ]), \
             patch('pathlib.Path.relative_to', side_effect=lambda x: Path("notebook1.ipynb") if "notebook1" in str(x) else Path("subdir/notebook2.ipynb") if "notebook2" in str(x) else Path("notebook3.ipynb")), \
             patch('pathlib.Path.mkdir'), \
             patch('shutil.copy2') as mock_copy, \
             patch('builtins.print') as mock_print:
            
            archive_notebooks()
            
            # Check that all notebooks were processed
            assert mock_copy.call_count == 3
            mock_print.assert_any_call("--- Starting Notebook Archiving Step ---")
            # Check that processing messages were printed (exact format may vary)
            call_strings = [str(call) for call in mock_print.call_args_list]
            combined_calls = " ".join(call_strings)
            assert "Processing project" in combined_calls
            assert "project1" in combined_calls
            assert "project2" in combined_calls
            mock_print.assert_any_call("Successfully archived: 3 notebooks.")

    def test_error_handling_integration(self):
        """Test error handling in the integrated workflow."""
        mock_notebooks = [Path("/test/project1/notebook1.ipynb")]
        
        with patch('src.preprocessing.extract_notebooks.TARGET_PROJECTS', ["/project1"]), \
             patch('src.preprocessing.extract_notebooks.TARGET_PROJECTS_BASE_PATH', Path("/test")), \
             patch('src.preprocessing.extract_notebooks.NOTEBOOKS_DEST_ROOT', Path("/dest")), \
             patch('pathlib.Path.is_dir', return_value=True), \
             patch('pathlib.Path.rglob', return_value=mock_notebooks), \
             patch('pathlib.Path.relative_to', return_value=Path("notebook1.ipynb")), \
             patch('pathlib.Path.mkdir'), \
             patch('shutil.copy2', side_effect=Exception("Test error")), \
             patch('builtins.print') as mock_print:
            
            archive_notebooks()
            
            # Check that error was handled gracefully
            mock_print.assert_any_call("[ERROR] Failed processing /project1: Test error")
            mock_print.assert_any_call("Successfully archived: 0 notebooks.")
            mock_print.assert_any_call("Errors encountered:    1 projects.")

    def test_data_consistency_integration(self):
        """Test data consistency in the integrated workflow."""
        mock_notebooks = [
            Path("/test/project1/notebook1.ipynb"),
            Path("/test/project1/notebook2.ipynb"),
        ]
        
        with patch('src.preprocessing.extract_notebooks.TARGET_PROJECTS', ["/project1"]), \
             patch('src.preprocessing.extract_notebooks.TARGET_PROJECTS_BASE_PATH', Path("/test")), \
             patch('src.preprocessing.extract_notebooks.NOTEBOOKS_DEST_ROOT', Path("/dest")), \
             patch('pathlib.Path.is_dir', return_value=True), \
             patch('pathlib.Path.rglob', return_value=mock_notebooks), \
             patch('pathlib.Path.relative_to', side_effect=lambda x: Path("notebook1.ipynb") if "notebook1" in str(x) else Path("notebook2.ipynb")), \
             patch('pathlib.Path.mkdir'), \
             patch('shutil.copy2') as mock_copy, \
             patch('builtins.print') as mock_print:
            
            archive_notebooks()
            
            # Verify data consistency
            assert mock_copy.call_count == 2
            mock_print.assert_any_call("Successfully archived: 2 notebooks.")
            
            # Check that all copy calls were made
            call_args_list = mock_copy.call_args_list
            assert len(call_args_list) == 2

    def test_performance_with_large_dataset(self):
        """Test performance with larger datasets."""
        # Create a larger number of mock notebooks
        mock_notebooks = [Path(f"/test/project1/notebook{i}.ipynb") for i in range(50)]
        
        with patch('src.preprocessing.extract_notebooks.TARGET_PROJECTS', ["/project1"]), \
             patch('src.preprocessing.extract_notebooks.TARGET_PROJECTS_BASE_PATH', Path("/test")), \
             patch('src.preprocessing.extract_notebooks.NOTEBOOKS_DEST_ROOT', Path("/dest")), \
             patch('pathlib.Path.is_dir', return_value=True), \
             patch('pathlib.Path.rglob', return_value=mock_notebooks), \
             patch('pathlib.Path.relative_to', side_effect=[Path(f"notebook{i}.ipynb") for i in range(50)]), \
             patch('pathlib.Path.mkdir'), \
             patch('shutil.copy2') as mock_copy, \
             patch('builtins.print') as mock_print:
            
            archive_notebooks()
            
            # Verify that all notebooks were processed
            assert mock_copy.call_count == 50
            mock_print.assert_any_call("Successfully archived: 50 notebooks.")

    def test_path_handling_integration(self):
        """Test path handling in the integrated workflow."""
        mock_notebooks = [Path("/test/project1/subdir/notebook.ipynb")]
        
        with patch('src.preprocessing.extract_notebooks.TARGET_PROJECTS', ["/project1"]), \
             patch('src.preprocessing.extract_notebooks.TARGET_PROJECTS_BASE_PATH', Path("/test")), \
             patch('src.preprocessing.extract_notebooks.NOTEBOOKS_DEST_ROOT', Path("/dest")), \
             patch('pathlib.Path.is_dir', return_value=True), \
             patch('pathlib.Path.rglob', return_value=mock_notebooks), \
             patch('pathlib.Path.relative_to', return_value=Path("subdir/notebook.ipynb")), \
             patch('pathlib.Path.mkdir') as mock_mkdir, \
             patch('shutil.copy2') as mock_copy, \
             patch('builtins.print') as mock_print:
            
            archive_notebooks()
            
            # Check that paths were handled correctly
            mock_copy.assert_called_once()
            call_args = mock_copy.call_args
            source_path = call_args[0][0]
            dest_path = call_args[0][1]
            
            assert source_path == Path("/test/project1/subdir/notebook.ipynb")
            assert "project1" in str(dest_path)
            assert "subdir" in str(dest_path)
            assert "notebook.ipynb" in str(dest_path)
            
            # Check that mkdir was called on the parent directory
            mock_mkdir.assert_called_with(parents=True, exist_ok=True)

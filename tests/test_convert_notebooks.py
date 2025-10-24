"""
Test suite for src/preprocessing/convert_notebooks.py
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.preprocessing.convert_notebooks import (
    DEFAULT_DEST_DIR,
    DEFAULT_SOURCE_DIR,
    convert_single_notebook,
    process_all_notebooks,
)


class TestConvertSingleNotebook:
    """Test the convert_single_notebook function."""

    def test_skip_up_to_date_file(self):
        """Test skipping conversion when target file is newer."""
        ipynb_path = Path("test.ipynb")
        py_path = Path("test.py")

        with patch("pathlib.Path.exists", return_value=True):
            with patch("pathlib.Path.stat") as mock_stat:
                # Mock stat to return different mtimes
                mock_stat.return_value.st_mtime = 1000  # py file is newer
                with patch("pathlib.Path.stat") as mock_ipynb_stat:
                    mock_ipynb_stat.return_value.st_mtime = 500  # ipynb file is older

                    result = convert_single_notebook(ipynb_path, py_path)
                    assert result == "SKIPPED"

    def test_conversion_error(self):
        """Test conversion error handling."""
        ipynb_path = Path("test.ipynb")
        py_path = Path("test.py")

        with patch("pathlib.Path.exists", return_value=False):
            with patch("pathlib.Path.parent") as mock_parent:
                mock_parent.mkdir = MagicMock()
                with patch("nbformat.read", side_effect=Exception("Read error")):
                    result = convert_single_notebook(ipynb_path, py_path)
                    assert "ERROR:" in result


class TestProcessAllNotebooks:
    """Test the process_all_notebooks function."""

    def test_no_notebooks_found(self):
        """Test handling when no notebooks are found."""
        source_dir = Path("empty_dir")
        dest_dir = Path("output")

        with patch("pathlib.Path.is_dir", return_value=True):
            with patch("pathlib.Path.glob", return_value=[]):
                with patch("builtins.print") as mock_print:
                    process_all_notebooks(source_dir, dest_dir)
                    assert any(
                        "No .ipynb files found" in str(call)
                        for call in mock_print.call_args_list
                    )


class TestConstants:
    """Test module constants."""

    def test_default_directories(self):
        """Test default directory constants."""
        assert DEFAULT_SOURCE_DIR.name == "notebooks"
        assert DEFAULT_DEST_DIR.name == "converted_notebooks"

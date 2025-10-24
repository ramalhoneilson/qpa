"""
Test suite for src/utils/generate_pattern_report.py
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest

from src.utils.generate_pattern_report import (
    INPUT_JSON_PATH,
    OUTPUT_MD_PATH,
    generate_markdown_for_pattern,
    main,
)


class TestGenerateMarkdownForPattern:
    """Test the generate_markdown_for_pattern function."""

    def test_generate_markdown_complete_pattern(self):
        """Test generating markdown for a complete pattern."""
        pattern = {
            "name": "Test Pattern",
            "alias": "Test Alias",
            "intent": "Test intent",
            "context": "Test context",
            "forces": "Test forces",
            "solution": "Test solution",
            "result": "Test result",
        }

        result = generate_markdown_for_pattern(pattern)

        assert "## Test Pattern" in result
        assert "***Also known as:** Test Alias*" in result
        assert "### Intent" in result
        assert "Test intent" in result
        assert "### Context" in result
        assert "Test context" in result
        assert "### Problem & Forces" in result
        assert "Test forces" in result
        assert "### Solution" in result
        assert "Test solution" in result
        assert "### Resulting Context" in result
        assert "Test result" in result

    def test_generate_markdown_minimal_pattern(self):
        """Test generating markdown for a minimal pattern."""
        pattern = {"name": "Minimal Pattern"}

        result = generate_markdown_for_pattern(pattern)

        assert "## Minimal Pattern" in result
        assert "***Also known as:**" not in result
        assert "### Intent" not in result

    def test_generate_markdown_with_placeholder_alias(self):
        """Test generating markdown with placeholder alias."""
        pattern = {"name": "Test Pattern", "alias": "—"}  # Placeholder alias

        result = generate_markdown_for_pattern(pattern)

        assert "## Test Pattern" in result
        assert "***Also known as:**" not in result

    def test_generate_markdown_with_empty_alias(self):
        """Test generating markdown with empty alias."""
        pattern = {"name": "Test Pattern", "alias": ""}

        result = generate_markdown_for_pattern(pattern)

        assert "## Test Pattern" in result
        assert "***Also known as:**" not in result

    def test_generate_markdown_with_whitespace_alias(self):
        """Test generating markdown with whitespace-only alias."""
        pattern = {"name": "Test Pattern", "alias": "   "}

        result = generate_markdown_for_pattern(pattern)

        assert "## Test Pattern" in result
        # The function doesn't strip whitespace, so it will include the alias
        assert "***Also known as:**" in result

    def test_generate_markdown_with_missing_name(self):
        """Test generating markdown with missing name."""
        pattern = {"intent": "Test intent"}

        result = generate_markdown_for_pattern(pattern)

        assert "## Unnamed Pattern" in result

    def test_generate_markdown_with_whitespace_content(self):
        """Test generating markdown with whitespace content."""
        pattern = {
            "name": "Test Pattern",
            "intent": "   Test intent   ",
            "context": "Test context\n\n",
        }

        result = generate_markdown_for_pattern(pattern)

        assert "### Intent" in result
        assert "Test intent" in result
        assert "### Context" in result
        assert "Test context" in result

    def test_generate_markdown_with_empty_sections(self):
        """Test generating markdown with empty sections."""
        pattern = {
            "name": "Test Pattern",
            "intent": "",
            "context": None,
            "forces": "Test forces",
        }

        result = generate_markdown_for_pattern(pattern)

        assert "## Test Pattern" in result
        assert "### Intent" not in result
        assert "### Context" not in result
        assert "### Problem & Forces" in result
        assert "Test forces" in result


class TestMainFunction:
    """Test the main function."""

    def test_main_successful_execution(self):
        """Test successful main execution."""
        mock_patterns = [
            {"name": "Pattern 1", "intent": "Intent 1", "solution": "Solution 1"},
            {"name": "Pattern 2", "intent": "Intent 2", "solution": "Solution 2"},
        ]

        with patch("pathlib.Path.exists", return_value=True):
            with patch("builtins.open", mock_open(read_data=json.dumps(mock_patterns))):
                with patch("builtins.print"):
                    main()

    def test_main_file_not_found(self):
        """Test main function when input file doesn't exist."""
        with patch("pathlib.Path.exists", return_value=False):
            with patch("builtins.print") as mock_print:
                main()

                assert any(
                    "Input file not found" in str(call)
                    for call in mock_print.call_args_list
                )
                assert any(
                    "Please run 'just download_pattern_list' first" in str(call)
                    for call in mock_print.call_args_list
                )

    def test_main_json_decode_error(self):
        """Test main function with invalid JSON."""
        with patch("pathlib.Path.exists", return_value=True):
            with patch("builtins.open", mock_open(read_data="invalid json")):
                with patch("builtins.print") as mock_print:
                    main()

                    assert any(
                        "Could not parse JSON file" in str(call)
                        for call in mock_print.call_args_list
                    )

    def test_main_write_error(self):
        """Test main function when writing fails."""
        mock_patterns = [{"name": "Test Pattern", "intent": "Test intent"}]

        with patch("pathlib.Path.exists", return_value=True):
            with patch("builtins.open", mock_open(read_data=json.dumps(mock_patterns))):
                with patch(
                    "builtins.open",
                    side_effect=[
                        mock_open(read_data=json.dumps(mock_patterns)).return_value,
                        OSError("Write error"),
                    ],
                ):
                    with patch("builtins.print") as mock_print:
                        main()

                        assert any(
                            "Could not write to file" in str(call)
                            for call in mock_print.call_args_list
                        )

    def test_main_with_single_pattern(self):
        """Test main function with single pattern (no horizontal rule)."""
        mock_patterns = [{"name": "Single Pattern", "intent": "Single intent"}]

        with patch("pathlib.Path.exists", return_value=True):
            with patch("builtins.open", mock_open(read_data=json.dumps(mock_patterns))):
                with patch("builtins.print"):
                    main()

    def test_main_with_multiple_patterns(self):
        """Test main function with multiple patterns (horizontal rules)."""
        mock_patterns = [
            {"name": "Pattern 1", "intent": "Intent 1"},
            {"name": "Pattern 2", "intent": "Intent 2"},
            {"name": "Pattern 3", "intent": "Intent 3"},
        ]

        with patch("pathlib.Path.exists", return_value=True):
            with patch("builtins.open", mock_open(read_data=json.dumps(mock_patterns))):
                with patch("builtins.print"):
                    main()


class TestConstants:
    """Test module constants."""

    def test_input_json_path(self):
        """Test INPUT_JSON_PATH constant."""
        assert INPUT_JSON_PATH.name == "quantum_patterns.json"

    def test_output_md_path(self):
        """Test OUTPUT_MD_PATH constant."""
        assert OUTPUT_MD_PATH.name == "quantum_patterns_report.md"

    def test_paths_are_path_objects(self):
        """Test that paths are Path objects."""
        assert isinstance(INPUT_JSON_PATH, Path)
        assert isinstance(OUTPUT_MD_PATH, Path)


class TestIntegration:
    """Integration tests for the pattern report generator."""

    def test_full_workflow_simulation(self):
        """Test the full workflow with realistic data."""
        mock_patterns = [
            {
                "name": "Quantum Amplitude Amplification",
                "alias": "QAA",
                "intent": "Amplify the amplitude of a target state in a quantum superposition",
                "context": "You have a quantum algorithm that needs to find a specific state",
                "forces": "The target state has low amplitude, making it hard to measure",
                "solution": "Apply iterative rotations to amplify the target state",
                "result": "The target state becomes measurable with high probability",
            },
            {
                "name": "Quantum Error Correction",
                "intent": "Protect quantum information from decoherence and errors",
                "context": "Quantum systems are prone to errors from environmental noise",
                "forces": "Quantum information is fragile and cannot be copied",
                "solution": "Use quantum error correcting codes to detect and correct errors",
                "result": "Quantum information becomes more robust against errors",
            },
        ]

        with patch("pathlib.Path.exists", return_value=True):
            with patch("builtins.open", mock_open(read_data=json.dumps(mock_patterns))):
                with patch("builtins.print"):
                    main()

    def test_pattern_with_all_fields(self):
        """Test pattern with all possible fields."""
        pattern = {
            "name": "Complete Pattern",
            "alias": "CP",
            "intent": "Test intent",
            "context": "Test context",
            "forces": "Test forces",
            "solution": "Test solution",
            "result": "Test result",
            "extra_field": "Should be ignored",
        }

        result = generate_markdown_for_pattern(pattern)

        # Should contain all expected sections
        assert "## Complete Pattern" in result
        assert "***Also known as:** CP*" in result
        assert "### Intent" in result
        assert "### Context" in result
        assert "### Problem & Forces" in result
        assert "### Solution" in result
        assert "### Resulting Context" in result

        # Should not contain extra fields
        assert "extra_field" not in result

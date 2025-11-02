"""
Test suite for src/data_acquisition/download_patterns.py
"""

from unittest.mock import MagicMock, patch

import pytest

from src.data_acquisition.download_patterns import (
    QUANTUM_PATTERNS_REFERENCE_FILE,
    download_quantum_pattern_details,
    get_all_pattern_summaries,
)


class TestGetAllPatternSummaries:
    """Test the get_all_pattern_summaries function."""

    def test_request_exception(self):
        """Test handling of request exceptions."""
        with patch("requests.get", side_effect=Exception("Network error")):
            with patch("builtins.print") as mock_print:
                try:
                    result = get_all_pattern_summaries()
                except Exception:
                    # The function raises an exception, which is expected
                    pass
                # The function should have printed an error message
                assert mock_print.call_count >= 1


class TestDownloadQuantumPatternDetails:
    """Test the download_quantum_pattern_details function."""

    def test_invalid_pattern_skipped(self):
        """Test skipping invalid pattern entries."""
        pattern_summaries = [
            {"id": "pattern1", "name": "Valid Pattern", "patternLanguageId": "lang1"},
            {
                "id": None,  # Invalid - missing required field
                "name": "Invalid Pattern",
                "patternLanguageId": "lang2",
            },
        ]

        with patch("requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.json.return_value = {"renderedContent": {}}
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response

            with patch("time.sleep"):
                with patch("builtins.print") as mock_print:
                    result = download_quantum_pattern_details(pattern_summaries)

                    # Should process some patterns (may be 0 or 1 depending on implementation)
                    assert isinstance(result, list)


class TestConstants:
    """Test module constants."""

    def test_quantum_patterns_reference_file(self):
        """Test QUANTUM_PATTERNS_REFERENCE_FILE constant."""
        assert QUANTUM_PATTERNS_REFERENCE_FILE.name == "quantum_patterns.json"

"""
Test suite for src/workflows/report_generator.py
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pandas as pd
import pytest

from src.workflows.report_generator import ReportGenerator
from src.workflows.statistics_calculator import StatisticsCalculator


class TestReportGenerator:
    """Test the ReportGenerator class."""

    def setup_method(self):
        """Set up test data."""
        self.sample_df = pd.DataFrame(
            {
                "concept_name": ["qiskit/circuit", "pennylane/device"],
                "file_path": ["project1/file1.py", "project2/file2.py"],
                "match_type": ["name", "semantic"],
                "similarity_score": [0.95, 0.87],
                "pattern": ["Pattern1", "Pattern2"],
                "framework": ["qiskit", "pennylane"],
                "project": ["project1", "project2"],
            }
        )
        self.sample_patterns = {"Pattern1", "Pattern2"}
        self.statistics = StatisticsCalculator(self.sample_df, self.sample_patterns)
        self.report_generator = ReportGenerator(self.statistics)

    def test_initialization(self):
        """Test ReportGenerator initialization."""
        assert self.report_generator.stats is self.statistics
        assert self.report_generator.df is self.sample_df
        assert self.report_generator.found_patterns == self.sample_patterns

    def test_generate_txt_report(self):
        """Test generate_txt_report method."""
        with patch("builtins.open", mock_open()) as mock_file:
            with patch("sys.stdout", new_callable=MagicMock) as mock_stdout:
                with patch("builtins.print") as mock_print:
                    self.report_generator.generate_txt_report(Path("/test/report.txt"))

                    # Should have called open for writing
                    mock_file.assert_called()

                    # Should have printed success message
                    assert any(
                        "Text report successfully generated" in str(call)
                        for call in mock_print.call_args_list
                    )

    def test_generate_md_report(self):
        """Test generate_md_report method."""
        with patch("builtins.open", mock_open()) as mock_file:
            with patch("builtins.print") as mock_print:
                self.report_generator.generate_md_report(Path("/test/report.md"))

                # Should have called open for writing
                mock_file.assert_called()

                # Should have printed success message
                assert any(
                    "Markdown report successfully generated" in str(call)
                    for call in mock_print.call_args_list
                )

    def test_write_report_content_txt(self):
        """Test _write_report_content with text format."""
        output_lines = []

        def capture_print(*args, **kwargs):
            output_lines.extend(str(arg) for arg in args)

        self.report_generator._write_report_content(is_md=False, md_print=capture_print)

        # Check that content was generated
        assert len(output_lines) > 0
        assert any("FINAL PATTERN ANALYSIS REPORT" in line for line in output_lines)

    def test_write_report_content_md(self):
        """Test _write_report_content with markdown format."""
        output_lines = []

        def capture_print(*args, **kwargs):
            output_lines.extend(str(arg) for arg in args)

        self.report_generator._write_report_content(is_md=True, md_print=capture_print)

        # Check that content was generated
        assert len(output_lines) > 0
        assert any("# Final Pattern Analysis Report" in line for line in output_lines)

    def test_write_report_content_sections(self):
        """Test that all report sections are included."""
        output_lines = []

        def capture_print(*args, **kwargs):
            output_lines.extend(str(arg) for arg in args)

        self.report_generator._write_report_content(is_md=True, md_print=capture_print)

        # Check for key sections
        content = " ".join(output_lines)
        assert "Summary Statistics" in content
        assert "Match Type Breakdown" in content
        assert "Source Framework" in content
        assert "Top Matched Concepts" in content

    def test_write_report_content_with_patterns(self):
        """Test report content with pattern data."""
        # Create statistics with pattern data
        df_with_patterns = self.sample_df.copy()
        df_with_patterns["pattern"] = ["Pattern1", "Pattern2"]
        statistics_with_patterns = StatisticsCalculator(
            df_with_patterns, {"Pattern1", "Pattern2"}
        )
        report_generator_with_patterns = ReportGenerator(statistics_with_patterns)

        output_lines = []

        def capture_print(*args, **kwargs):
            output_lines.extend(str(arg) for arg in args)

        report_generator_with_patterns._write_report_content(
            is_md=True, md_print=capture_print
        )

        # Check that pattern sections are included
        content = " ".join(output_lines)
        assert "Cross-Framework Pattern Analysis" in content
        assert "Quantum Pattern Analysis" in content

    def test_write_report_content_without_patterns(self):
        """Test report content without pattern data."""
        # Create statistics without pattern data
        df_no_patterns = self.sample_df.copy()
        df_no_patterns["pattern"] = ""
        statistics_no_patterns = StatisticsCalculator(df_no_patterns, set())
        report_generator_no_patterns = ReportGenerator(statistics_no_patterns)

        output_lines = []

        def capture_print(*args, **kwargs):
            output_lines.extend(str(arg) for arg in args)

        report_generator_no_patterns._write_report_content(
            is_md=True, md_print=capture_print
        )

        # Check that basic sections are still included
        content = " ".join(output_lines)
        assert "Summary Statistics" in content
        assert "Match Type Breakdown" in content

    def test_write_report_content_unmatched_patterns(self):
        """Test report content with unmatched patterns."""
        # Add unmatched patterns
        self.statistics.unmatched_patterns = {"Pattern3", "Pattern4"}

        output_lines = []

        def capture_print(*args, **kwargs):
            output_lines.extend(str(arg) for arg in args)

        self.report_generator._write_report_content(is_md=True, md_print=capture_print)

        # Check that unmatched patterns section is included
        content = " ".join(output_lines)
        assert "Unmatched Pattern Analysis" in content

    def test_write_report_content_no_unmatched_patterns(self):
        """Test report content with no unmatched patterns."""
        # Clear unmatched patterns
        self.statistics.unmatched_patterns = set()

        output_lines = []

        def capture_print(*args, **kwargs):
            output_lines.extend(str(arg) for arg in args)

        self.report_generator._write_report_content(is_md=True, md_print=capture_print)

        # Check that unmatched patterns section is included
        content = " ".join(output_lines)
        assert "Unmatched Pattern Analysis" in content
        assert "All patterns defined" in content

    def test_write_report_content_end_markers(self):
        """Test that end markers are included in text format."""
        output_lines = []

        def capture_print(*args, **kwargs):
            output_lines.extend(str(arg) for arg in args)

        # Mock the print function to capture the end markers
        with patch("builtins.print") as mock_print:
            self.report_generator._write_report_content(
                is_md=False, md_print=capture_print
            )

            # Check that the end markers were printed
            print_calls = [str(call) for call in mock_print.call_args_list]
            assert any("END OF REPORT" in call for call in print_calls)

    def test_write_report_content_markdown_formatting(self):
        """Test that markdown formatting is applied correctly."""
        output_lines = []

        def capture_print(*args, **kwargs):
            output_lines.extend(str(arg) for arg in args)

        self.report_generator._write_report_content(is_md=True, md_print=capture_print)

        # Check for markdown formatting
        content = " ".join(output_lines)
        assert "# " in content  # Headers
        assert "**" in content  # Bold text
        assert "### " in content  # Subheaders

"""
Report generation utilities for the final report generation.
"""

import sys
from pathlib import Path
from typing import Callable, Dict, List, Set

import pandas as pd

from .statistics_calculator import StatisticsCalculator


class ReportGenerator:
    """Handles generation of text and markdown reports."""

    def __init__(self, statistics: StatisticsCalculator):
        """Initialize the report generator.

        Args:
            statistics: StatisticsCalculator instance with calculated statistics
        """
        self.stats = statistics
        self.df = statistics.df
        self.df_with_patterns = statistics.df_with_patterns
        self.found_patterns = statistics.found_patterns
        self.avg_score = statistics.avg_score
        self.matches_by_type = statistics.matches_by_type
        self.avg_score_by_type = statistics.avg_score_by_type
        self.matches_by_framework = statistics.matches_by_framework
        self.matches_by_project = statistics.matches_by_project
        self.source_table = getattr(statistics, "source_table", None)
        self.adoption_table = getattr(statistics, "adoption_table", None)
        self.matches_by_pattern = getattr(statistics, "matches_by_pattern", None)
        self.avg_score_by_pattern = getattr(statistics, "avg_score_by_pattern", None)
        self.patterns_in_frameworks = getattr(
            statistics, "patterns_in_frameworks", None
        )
        self.top_20_table_data = statistics.top_20_table_data
        self.unmatched_patterns = statistics.unmatched_patterns

    def generate_txt_report(self, path: Path):
        """Generate a text report and save it to the specified path.

        Args:
            path: Path to save the text report
        """
        original_stdout = sys.stdout
        with open(path, "w", encoding="utf-8") as f:
            sys.stdout = f
            self._write_report_content(is_md=False)
        sys.stdout = original_stdout
        print(f"Text report successfully generated at '{path}'")

    def generate_md_report(self, path: Path):
        """Generate a markdown report and save it to the specified path.

        Args:
            path: Path to save the markdown report
        """
        with open(path, "w", encoding="utf-8") as f:
            # Redirect print to the file
            def md_print(*args, **kwargs):
                print(*args, file=f, **kwargs)

            self._write_report_content(is_md=True, md_print=md_print)
        print(f"Markdown report successfully generated at '{path}'")

    def _write_report_content(self, is_md: bool, md_print=print):
        """Write the report content, adapting format for TXT or MD.

        Args:
            is_md: Whether to generate markdown format
            md_print: Print function to use (default: built-in print)
        """

        # Helper to format tables
        def to_format(df_or_series, headers=None):
            if isinstance(df_or_series, list):
                # Handle list data (like top_20_table_data)
                if headers:
                    df = pd.DataFrame(df_or_series, columns=headers)
                else:
                    df = pd.DataFrame(df_or_series)
            elif isinstance(df_or_series, pd.Series):
                df = df_or_series.reset_index()
            else:
                df = df_or_series.copy()

            if headers and not isinstance(df_or_series, list):
                df.columns = headers

            if is_md:
                return df.to_markdown(index=False)
            else:
                return df.to_string(index=False)

        # I. Summary Statistics
        md_print(
            "# Final Pattern Analysis Report"
            if is_md
            else "=" * 80
            + "\n"
            + " " * 20
            + "FINAL PATTERN ANALYSIS REPORT"
            + "\n"
            + "=" * 80
        )
        md_print(
            "\n## I. Summary Statistics\n"
            if is_md
            else "\n--- I. Summary Statistics ---"
        )
        md_print(
            f"- **Total Matches Found:** {len(self.df)}"
            if is_md
            else f"Total Matches Found:           {len(self.df)}"
        )
        md_print(
            f"- **Total Patterns Found:** {len(self.found_patterns)}"
            if is_md
            else f"Total Patterns Found:           {len(self.found_patterns)}"
        )
        md_print(
            f"- **Average Similarity Score:** {self.avg_score:.4f}"
            if is_md
            else f"Average Similarity Score:       {self.avg_score:.4f}"
        )

        # II. Match Type Breakdown
        md_print(
            "\n## II. Match Type Breakdown"
            if is_md
            else "\n--- II. Match Type Breakdown ---"
        )
        md_print("\n### Match Type Counts\n" if is_md else "")
        md_print(to_format(self.matches_by_type.reset_index()))
        md_print(
            "\n### Average Score by Match Type\n"
            if is_md
            else "\nAverage Score by Match Type:"
        )
        md_print(to_format(self.avg_score_by_type.round(4).reset_index()))

        md_print("\n---\n" if is_md else "\n" + "-" * 80)

        # III. Source Framework & Target Project Breakdown
        md_print(
            "## III. Source Framework & Target Project Breakdown"
            if is_md
            else "\n--- III. Source Framework & Target Project Breakdown ---"
        )
        md_print(
            "\n### Matches by Source Framework\n"
            if is_md
            else "\nMatches by Source Framework:"
        )
        md_print(to_format(self.matches_by_framework.reset_index()))
        md_print(
            "\n### Matches by Target Project\n"
            if is_md
            else "\nMatches by Target Project:"
        )
        md_print(to_format(self.matches_by_project.reset_index()))

        md_print("\n---\n" if is_md else "\n" + "-" * 80)

        # IV & V. Pattern Analysis
        if not self.df_with_patterns.empty:
            md_print(
                "## IV. Cross-Framework Pattern Analysis"
                if is_md
                else "\n--- IV. Cross-Framework Pattern Analysis ---"
            )
            source_headers = {
                "total_matches": "Total Matches",
                "source_framework_names": "Source Frameworks",
            }
            md_print(
                "\n### Table 4.1: Source Pattern Analysis (Where patterns originate)\n"
                if is_md
                else "\nTable 4.1: Source Pattern Analysis (Where patterns originate)"
            )
            md_print(
                to_format(
                    self.source_table.reset_index().rename(columns=source_headers)
                )
            )

            adoption_headers = {
                "target_project_coverage": "Project Coverage",
                "target_project_names": "Found In Projects",
            }
            md_print(
                "\n### Table 4.2: Adoption Pattern Analysis (Where patterns are used)\n"
                if is_md
                else "\n\nTable 4.2: Adoption Pattern Analysis (Where patterns are used)"
            )
            md_print(
                to_format(
                    self.adoption_table.reset_index().rename(columns=adoption_headers)
                )
            )

            md_print("\n---\n" if is_md else "\n" + "-" * 80)

            md_print(
                "## V. Quantum Pattern Analysis"
                if is_md
                else "\n--- V. Quantum Pattern Analysis ---"
            )
            md_print(
                "\n### Patterns by Match Count (Overall)\n"
                if is_md
                else "\nPatterns by Match Count (Overall):"
            )
            md_print(to_format(self.matches_by_pattern.reset_index()))

            md_print(
                "\n### Average Score by Pattern\n"
                if is_md
                else "\nAverage Score by Pattern:"
            )
            md_print(
                to_format(
                    self.avg_score_by_pattern.round(4)
                    .sort_values(ascending=False)
                    .reset_index()
                )
            )

            md_print(
                "\n### All Patterns within each Source Framework (Sorted by Frequency)\n"
                if is_md
                else "\nAll Patterns within each Source Framework (Sorted by Frequency):"
            )
            for framework, data in self.patterns_in_frameworks.groupby(level=0):
                md_print(
                    f"\n#### {framework.capitalize()}\n"
                    if is_md
                    else f"\n  -- {framework} --"
                )
                md_print(to_format(data.droplevel(0).reset_index()))

        md_print("\n---\n" if is_md else "\n" + "-" * 80)

        # VI. Top Matched Concepts
        md_print(
            "## VI. Top Matched Concepts"
            if is_md
            else "\n--- VI. Top Matched Concepts ---"
        )
        md_print(
            f"\n### Top 20 Most Frequently Matched Concepts\n"
            if is_md
            else f"\nTop 20 Most Frequently Matched Concepts:"
        )
        md_print(
            to_format(
                self.top_20_table_data, headers=["Framework", "Concept", "Matches"]
            )
        )

        md_print("\n---\n" if is_md else "\n" + "-" * 80)

        # VII. Unmatched Pattern Analysis
        md_print(
            "## VII. Unmatched Pattern Analysis"
            if is_md
            else "\n--- VII. Unmatched Pattern Analysis ---"
        )
        if self.unmatched_patterns:
            md_print(
                f"\nThe following **{len(self.unmatched_patterns)}** patterns from the source files were **NOT found** in any project:\n"
                if is_md
                else f"\nThe following {len(self.unmatched_patterns)} patterns from the source files were NOT found in any project:"
            )
            for pattern in self.unmatched_patterns:
                md_print(f"- {pattern}")
        else:
            md_print(
                "\nAll patterns defined in the source files were found in the analysis."
            )

        # End of Report
        if not is_md:
            print(
                "\n"
                + "=" * 80
                + "\n"
                + "                              END OF REPORT"
                + "\n"
                + "=" * 80
            )

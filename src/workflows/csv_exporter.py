"""
CSV export utilities for the final report generation.
"""

from pathlib import Path
from typing import Dict, List, Set

import pandas as pd


class CSVExporter:
    """Handles exporting statistics tables to CSV files."""

    def __init__(self, output_dir: Path):
        """Initialize the CSV exporter.

        Args:
            output_dir: Directory to save CSV files
        """
        self.output_dir = output_dir

    def export_all_tables(self, statistics: "StatisticsCalculator"):
        """Export all statistics tables as CSV files.

        Args:
            statistics: StatisticsCalculator instance with calculated statistics
        """
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Exporting tables to CSV files in '{self.output_dir}'...")

        # Export basic statistics tables
        self._export_basic_statistics(statistics)

        # Export pattern analysis tables if they exist
        if not statistics.df_with_patterns.empty:
            self._export_pattern_statistics(statistics)

        # Export additional tables
        self._export_additional_tables(statistics)

        print(
            f"Successfully exported {len(list(self.output_dir.glob('*.csv')))} CSV files to '{self.output_dir}'"
        )

    def _export_basic_statistics(self, statistics: "StatisticsCalculator"):
        """Export basic statistics tables."""
        # Match type statistics
        statistics.matches_by_type.reset_index().to_csv(
            self.output_dir / "match_type_counts.csv", index=False
        )
        statistics.avg_score_by_type.round(4).reset_index().to_csv(
            self.output_dir / "avg_score_by_type.csv", index=False
        )

        # Framework and project statistics
        statistics.matches_by_framework.reset_index().to_csv(
            self.output_dir / "matches_by_framework.csv", index=False
        )
        statistics.matches_by_project.reset_index().to_csv(
            self.output_dir / "matches_by_project.csv", index=False
        )

    def _export_pattern_statistics(self, statistics: "StatisticsCalculator"):
        """Export pattern-specific statistics tables."""
        # Source pattern analysis
        source_headers = {
            "total_matches": "Total Matches",
            "source_framework_names": "Source Frameworks",
        }
        statistics.source_table.reset_index().rename(columns=source_headers).to_csv(
            self.output_dir / "source_pattern_analysis.csv", index=False
        )

        # Adoption pattern analysis
        adoption_headers = {
            "target_project_coverage": "Project Coverage",
            "target_project_names": "Found In Projects",
        }
        statistics.adoption_table.reset_index().rename(columns=adoption_headers).to_csv(
            self.output_dir / "adoption_pattern_analysis.csv", index=False
        )

        # Pattern analysis tables
        statistics.matches_by_pattern.reset_index().to_csv(
            self.output_dir / "patterns_by_match_count.csv", index=False
        )
        statistics.avg_score_by_pattern.round(4).sort_values(
            ascending=False
        ).reset_index().to_csv(
            self.output_dir / "avg_score_by_pattern.csv", index=False
        )

        # Patterns by framework
        for framework, data in statistics.patterns_in_frameworks.groupby(level=0):
            data.droplevel(0).reset_index().to_csv(
                self.output_dir / f"patterns_in_{framework.lower()}.csv", index=False
            )

    def _export_additional_tables(self, statistics: "StatisticsCalculator"):
        """Export additional tables."""
        # Top concepts table
        top_concepts_df = pd.DataFrame(
            statistics.top_20_table_data, columns=["Framework", "Concept", "Matches"]
        )
        top_concepts_df.to_csv(
            self.output_dir / "top_matched_concepts.csv", index=False
        )

        # Unmatched patterns
        if statistics.unmatched_patterns:
            unmatched_df = pd.DataFrame(
                {"unmatched_patterns": list(statistics.unmatched_patterns)}
            )
            unmatched_df.to_csv(self.output_dir / "unmatched_patterns.csv", index=False)




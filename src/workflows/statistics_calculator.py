"""
Statistics calculation utilities for the final report generation.
"""

from typing import Dict, List, Tuple

import pandas as pd


class StatisticsCalculator:
    """Calculates various statistics from the processed data."""

    def __init__(self, df: pd.DataFrame, all_patterns: set):
        """Initialize the statistics calculator.

        Args:
            df: Processed DataFrame with concept matches
            all_patterns: Set of all available patterns
        """
        self.df = df
        self.all_patterns = all_patterns
        self._prepare_data()

    def _prepare_data(self):
        """Prepare data for statistics calculation."""
        # Filter data with patterns
        self.df_with_patterns = self.df[
            self.df["pattern"].notna() & (self.df["pattern"] != "")
        ]

        # Calculate basic statistics
        self.total_matches = len(self.df)
        self.found_patterns = set(self.df_with_patterns["pattern"].unique())
        self.avg_score = (
            self.df["similarity_score"].mean() if not self.df.empty else 0.0
        )

        # Calculate match type statistics
        self.matches_by_type = self.df["match_type"].value_counts()
        self.avg_score_by_type = self.df.groupby("match_type")[
            "similarity_score"
        ].mean()

        # Calculate framework and project statistics
        self.matches_by_framework = self.df["framework"].value_counts()
        self.matches_by_project = self.df["project"].value_counts()

        # Calculate pattern statistics if patterns exist
        if not self.df_with_patterns.empty:
            self._calculate_pattern_statistics()

        # Calculate top concepts
        self._calculate_top_concepts()

        # Find unmatched patterns
        self.unmatched_patterns = self.all_patterns - self.found_patterns

    def _calculate_pattern_statistics(self):
        """Calculate pattern-specific statistics."""
        # Pattern frequency
        self.matches_by_pattern = self.df_with_patterns["pattern"].value_counts()
        self.avg_score_by_pattern = self.df_with_patterns.groupby("pattern")[
            "similarity_score"
        ].mean()

        # Source pattern analysis
        source_analysis = (
            self.df_with_patterns.groupby("pattern")
            .agg({"framework": lambda x: list(x.unique()), "similarity_score": "count"})
            .rename(columns={"similarity_score": "total_matches"})
        )

        source_analysis["source_framework_names"] = source_analysis["framework"].apply(
            lambda x: ", ".join(sorted(x))
        )
        self.source_table = source_analysis.drop(columns=["framework"])

        # Adoption pattern analysis
        adoption_analysis = (
            self.df_with_patterns.groupby("pattern")
            .agg({"project": lambda x: list(x.unique()), "similarity_score": "count"})
            .rename(columns={"similarity_score": "total_matches"})
        )

        adoption_analysis["target_project_coverage"] = adoption_analysis[
            "project"
        ].apply(len)
        adoption_analysis["target_project_names"] = adoption_analysis["project"].apply(
            lambda x: ", ".join(sorted(x))
        )
        self.adoption_table = adoption_analysis.drop(columns=["project"])

        # Patterns by framework
        self.patterns_in_frameworks = self.df_with_patterns.groupby(
            ["framework", "pattern"]
        ).size()

    def _calculate_top_concepts(self, top_n: int = 20):
        """Calculate top matched concepts.

        Args:
            top_n: Number of top concepts to include
        """
        concept_counts = (
            self.df.groupby(["framework", "concept_name"]).size().reset_index()
        )
        concept_counts.columns = ["Framework", "Concept", "Matches"]

        # Sort by matches and take top N
        top_concepts = concept_counts.sort_values("Matches", ascending=False).head(
            top_n
        )

        # Create table data
        self.top_20_table_data = []
        for _, row in top_concepts.iterrows():
            self.top_20_table_data.append(
                [row["Framework"], row["Concept"], row["Matches"]]
            )

    def get_basic_statistics(self) -> Dict:
        """Get basic statistics summary.

        Returns:
            Dictionary with basic statistics
        """
        return {
            "total_matches": self.total_matches,
            "found_patterns": len(self.found_patterns),
            "avg_score": self.avg_score,
            "unmatched_patterns": len(self.unmatched_patterns),
        }

    def get_match_type_statistics(self) -> Tuple[pd.Series, pd.Series]:
        """Get match type statistics.

        Returns:
            Tuple of (matches_by_type, avg_score_by_type)
        """
        return self.matches_by_type, self.avg_score_by_type

    def get_framework_project_statistics(self) -> Tuple[pd.Series, pd.Series]:
        """Get framework and project statistics.

        Returns:
            Tuple of (matches_by_framework, matches_by_project)
        """
        return self.matches_by_framework, self.matches_by_project

    def get_pattern_statistics(self) -> Dict:
        """Get pattern-specific statistics.

        Returns:
            Dictionary with pattern statistics
        """
        if self.df_with_patterns.empty:
            return {}

        return {
            "matches_by_pattern": self.matches_by_pattern,
            "avg_score_by_pattern": self.avg_score_by_pattern,
            "source_table": self.source_table,
            "adoption_table": self.adoption_table,
            "patterns_in_frameworks": self.patterns_in_frameworks,
        }

    def get_top_concepts(self) -> List[List]:
        """Get top matched concepts.

        Returns:
            List of [framework, concept, matches] lists
        """
        return self.top_20_table_data

    def get_unmatched_patterns(self) -> set:
        """Get unmatched patterns.

        Returns:
            Set of unmatched patterns
        """
        return self.unmatched_patterns




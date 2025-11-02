"""
Test suite for src/workflows/statistics_calculator.py
"""

import pandas as pd
import pytest

from src.workflows.statistics_calculator import StatisticsCalculator


class TestStatisticsCalculator:
    """Test the StatisticsCalculator class."""

    def setup_method(self):
        """Set up test data."""
        self.sample_df = pd.DataFrame(
            {
                "concept_name": [
                    "qiskit/circuit",
                    "pennylane/device",
                    "classiq/function",
                ],
                "file_path": [
                    "project1/file1.py",
                    "project2/file2.py",
                    "project1/file3.py",
                ],
                "match_type": ["name", "semantic", "name"],
                "similarity_score": [0.95, 0.87, 0.92],
                "pattern": ["Pattern1", "Pattern2", "Pattern1"],
                "framework": ["qiskit", "pennylane", "classiq"],
                "project": ["project1", "project2", "project1"],
            }
        )
        self.sample_patterns = {"Pattern1", "Pattern2", "Pattern3"}

    def test_initialization(self):
        """Test StatisticsCalculator initialization."""
        calculator = StatisticsCalculator(self.sample_df, self.sample_patterns)

        assert calculator.df is self.sample_df
        assert calculator.all_patterns == self.sample_patterns
        assert calculator.total_matches == 3
        assert calculator.avg_score == (0.95 + 0.87 + 0.92) / 3

    def test_prepare_data(self):
        """Test data preparation."""
        calculator = StatisticsCalculator(self.sample_df, self.sample_patterns)

        # Check basic statistics
        assert calculator.total_matches == 3
        assert len(calculator.found_patterns) == 2  # Pattern1 and Pattern2
        assert calculator.avg_score == pytest.approx(0.9133, abs=0.001)

        # Check match type statistics
        assert len(calculator.matches_by_type) == 2
        assert calculator.matches_by_type["name"] == 2
        assert calculator.matches_by_type["semantic"] == 1

        # Check framework statistics
        assert len(calculator.matches_by_framework) == 3
        assert calculator.matches_by_framework["qiskit"] == 1
        assert calculator.matches_by_framework["pennylane"] == 1
        assert calculator.matches_by_framework["classiq"] == 1

    def test_calculate_pattern_statistics(self):
        """Test pattern statistics calculation."""
        calculator = StatisticsCalculator(self.sample_df, self.sample_patterns)

        # Check pattern frequency
        assert len(calculator.matches_by_pattern) == 2
        assert calculator.matches_by_pattern["Pattern1"] == 2
        assert calculator.matches_by_pattern["Pattern2"] == 1

        # Check source table
        assert "Pattern1" in calculator.source_table.index
        assert "Pattern2" in calculator.source_table.index

        # Check adoption table
        assert "Pattern1" in calculator.adoption_table.index
        assert "Pattern2" in calculator.adoption_table.index

    def test_calculate_top_concepts(self):
        """Test top concepts calculation."""
        calculator = StatisticsCalculator(self.sample_df, self.sample_patterns)

        # Check top concepts data
        assert len(calculator.top_20_table_data) == 3
        assert all(
            len(row) == 3 for row in calculator.top_20_table_data
        )  # [framework, concept, matches]

        # Check that concepts are sorted by matches
        matches = [row[2] for row in calculator.top_20_table_data]
        assert matches == sorted(matches, reverse=True)

    def test_get_basic_statistics(self):
        """Test get_basic_statistics method."""
        calculator = StatisticsCalculator(self.sample_df, self.sample_patterns)
        stats = calculator.get_basic_statistics()

        assert stats["total_matches"] == 3
        assert stats["found_patterns"] == 2
        assert stats["avg_score"] == pytest.approx(0.9133, abs=0.001)
        assert stats["unmatched_patterns"] == 1  # Pattern3 not found

    def test_get_match_type_statistics(self):
        """Test get_match_type_statistics method."""
        calculator = StatisticsCalculator(self.sample_df, self.sample_patterns)
        matches_by_type, avg_score_by_type = calculator.get_match_type_statistics()

        assert len(matches_by_type) == 2
        assert len(avg_score_by_type) == 2
        assert matches_by_type["name"] == 2
        assert matches_by_type["semantic"] == 1

    def test_get_framework_project_statistics(self):
        """Test get_framework_project_statistics method."""
        calculator = StatisticsCalculator(self.sample_df, self.sample_patterns)
        matches_by_framework, matches_by_project = (
            calculator.get_framework_project_statistics()
        )

        assert len(matches_by_framework) == 3
        assert len(matches_by_project) == 2
        assert matches_by_framework["qiskit"] == 1
        assert matches_by_project["project1"] == 2

    def test_get_pattern_statistics(self):
        """Test get_pattern_statistics method."""
        calculator = StatisticsCalculator(self.sample_df, self.sample_patterns)
        pattern_stats = calculator.get_pattern_statistics()

        assert "matches_by_pattern" in pattern_stats
        assert "avg_score_by_pattern" in pattern_stats
        assert "source_table" in pattern_stats
        assert "adoption_table" in pattern_stats
        assert "patterns_in_frameworks" in pattern_stats

    def test_get_top_concepts(self):
        """Test get_top_concepts method."""
        calculator = StatisticsCalculator(self.sample_df, self.sample_patterns)
        top_concepts = calculator.get_top_concepts()

        assert len(top_concepts) == 3
        assert all(len(row) == 3 for row in top_concepts)

    def test_get_unmatched_patterns(self):
        """Test get_unmatched_patterns method."""
        calculator = StatisticsCalculator(self.sample_df, self.sample_patterns)
        unmatched = calculator.get_unmatched_patterns()

        assert "Pattern3" in unmatched
        assert "Pattern1" not in unmatched
        assert "Pattern2" not in unmatched

    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        empty_df = pd.DataFrame(
            columns=[
                "concept_name",
                "file_path",
                "match_type",
                "similarity_score",
                "pattern",
                "framework",
                "project",
            ]
        )
        calculator = StatisticsCalculator(empty_df, set())

        assert calculator.total_matches == 0
        assert calculator.avg_score == 0.0
        assert len(calculator.found_patterns) == 0
        assert calculator.df_with_patterns.empty

    def test_no_patterns(self):
        """Test with no patterns."""
        df_no_patterns = self.sample_df.copy()
        df_no_patterns["pattern"] = ""

        calculator = StatisticsCalculator(df_no_patterns, set())

        assert calculator.df_with_patterns.empty
        assert len(calculator.found_patterns) == 0




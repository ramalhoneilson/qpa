"""
Data processing utilities for the final report generation.
"""

import csv
from pathlib import Path
from typing import List, Set

import pandas as pd


class DataProcessor:
    """Handles data loading and initial processing for report generation."""

    def __init__(self, csv_file: Path, pattern_files: List[Path]):
        """Initialize the data processor.

        Args:
            csv_file: Path to the main CSV file with concept matches
            pattern_files: List of paths to pattern CSV files
        """
        self.csv_file = csv_file
        self.pattern_files = pattern_files

    def load_main_data(self) -> pd.DataFrame:
        """Load and process the main CSV data.

        Returns:
            Processed DataFrame with framework and project columns added

        Raises:
            FileNotFoundError: If the CSV file doesn't exist
            pd.errors.EmptyDataError: If the CSV file is empty
        """
        if not self.csv_file.exists():
            raise FileNotFoundError(f"Input file '{self.csv_file}' not found.")

        try:
            df = pd.read_csv(self.csv_file, delimiter=";")
        except pd.errors.EmptyDataError:
            raise pd.errors.EmptyDataError(f"The file '{self.csv_file}' is empty.")

        # Add framework and project columns
        df["framework"] = df["concept_name"].apply(self._extract_framework)
        df["project"] = df["file_path"].apply(self._extract_project)

        return df

    def load_patterns(self) -> Set[str]:
        """Load all patterns from pattern files.

        Returns:
            Set of all unique patterns found in the files
        """
        all_patterns = set()

        for pattern_file in self.pattern_files:
            if pattern_file.exists():
                try:
                    with open(pattern_file, encoding="utf-8") as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            if "pattern" in row and row["pattern"]:
                                all_patterns.add(row["pattern"].strip())
                except Exception as e:
                    print(f"Warning: Could not read {pattern_file}: {e}")

        return all_patterns

    @staticmethod
    def _extract_framework(concept_name: str) -> str:
        """Extract framework name from concept name.

        Args:
            concept_name: Full concept name (e.g., "qiskit/circuit")

        Returns:
            Framework name or "unknown" if extraction fails
        """
        try:
            if not concept_name or concept_name.strip() == "":
                return "unknown"
            return concept_name.strip("/").split("/")[0]
        except (AttributeError, IndexError):
            return "unknown"

    @staticmethod
    def _extract_project(file_path: str) -> str:
        """Extract project name from file path.

        Args:
            file_path: Full file path (e.g., "project1/file1.py")

        Returns:
            Project name or original path if extraction fails
        """
        try:
            if not file_path or file_path is None:
                return str(file_path) if file_path is not None else ""
            return Path(file_path).parts[0]
        except (IndexError, TypeError):
            return str(file_path) if file_path is not None else ""

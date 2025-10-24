"""
Generates a comprehensive experimental data report from CSV files and JSON data.

This script reads various data files generated during the analysis and creates
a markdown report with complete datasets for research purposes.
"""

import json
from pathlib import Path
from typing import Dict, List

import pandas as pd

from src.conf import config


class ExperimentalDataReportGenerator:
    """Generates experimental data reports from analysis results."""

    def __init__(self):
        """Initialize the report generator with file paths."""
        self.data_dir = config.RESULTS_DIR
        self.report_dir = config.DOCS_DIR
        self.output_file = self.report_dir / "experimental_data.md"

        # Define input files
        self.concept_files = {
            "Classiq": self.data_dir / "classiq_quantum_concepts.csv",
            "PennyLane": self.data_dir / "pennylane_quantum_concepts.csv",
            "Qiskit": self.data_dir / "qiskit_quantum_concepts.csv",
        }

        self.patterns_file = self.data_dir / "quantum_patterns.json"
        self.report_files = {
            "top_matched_concepts": self.data_dir
            / "report"
            / "top_matched_concepts.csv",
            "match_type_counts": self.data_dir / "report" / "match_type_counts.csv",
            "matches_by_framework": self.data_dir
            / "report"
            / "matches_by_framework.csv",
            "patterns_by_match_count": self.data_dir
            / "report"
            / "patterns_by_match_count.csv",
        }

    def generate_report(self):
        """Generate the complete experimental data report."""
        print("Generating experimental data report...")

        # Create output directory if it doesn't exist
        self.report_dir.mkdir(parents=True, exist_ok=True)

        # Build the markdown content
        content = []

        # Title and introduction
        content.extend(self._generate_header())

        # Framework concept tables
        content.extend(self._generate_framework_tables())

        # Pattern analysis tables
        content.extend(self._generate_pattern_analysis_tables())

        # Pattern Atlas data
        content.extend(self._generate_pattern_atlas_section())

        # References
        content.extend(self._generate_references())

        # Write the report
        with open(self.output_file, "w", encoding="utf-8") as f:
            f.write("\n".join(content))

        print(f"Experimental data report generated: {self.output_file}")

    def _generate_header(self) -> List[str]:
        """Generate the report header and introduction."""
        return [
            "# Experimental Data",
            "",
            "This document contains the complete experimental datasets used in the quantum pattern analysis research. The data includes concept extractions from quantum computing frameworks, pattern matching results, and pattern definitions from the PlanQK Pattern Atlas.",
            "",
            "## Overview",
            "",
            "The experimental data consists of:",
            "- **Framework Concept Extractions**: Complete datasets of quantum concepts extracted from Classiq, PennyLane, and Qiskit frameworks",
            "- **Pattern Matching Results**: Analysis of how concepts match with known quantum patterns",
            "- **Pattern Atlas Data**: Complete set of quantum patterns from the PlanQK Pattern Atlas",
            "",
            "All datasets are provided in their entirety to ensure reproducibility and enable further analysis.",
            "",
        ]

    def _generate_framework_tables(self) -> List[str]:
        """Generate tables for framework concept data."""
        content = []

        for framework, file_path in self.concept_files.items():
            if file_path.exists():
                content.extend(self._generate_framework_table(framework, file_path))
            else:
                content.extend(
                    [
                        f"## {framework} Quantum Concepts",
                        "",
                        f"⚠️ **Note**: The {framework} concepts file was not found at `{file_path}`",
                        "",
                    ]
                )

        return content

    def _generate_framework_table(self, framework: str, file_path: Path) -> List[str]:
        """Generate a table for a specific framework's concepts."""
        try:
            # Try different delimiters based on framework
            delimiter = ";" if framework == "Qiskit" else ","

            # Read the CSV file with appropriate delimiter
            df = pd.read_csv(file_path, delimiter=delimiter)

            content = [
                f"## {framework} Quantum Concepts",
                "",
                f"The complete dataset of quantum concepts extracted from the {framework} framework.",
                "",
                f"**File**: `{file_path.name}`",
                f"**Total Concepts**: {len(df)}",
                "",
            ]

            # Add the complete table
            if not df.empty:
                # Clean column names for display
                display_df = df.copy()
                if "name" in display_df.columns:
                    display_df["name"] = display_df["name"].str.replace("/", ".")

                # Add row numbers
                display_df.insert(0, "Row", range(1, len(display_df) + 1))

                # Convert to markdown table
                table_md = display_df.to_markdown(index=False)
                content.append(table_md)
            else:
                content.append("*No concepts found in the dataset.*")

            content.extend(["", "---", ""])

        except Exception as e:
            content = [
                f"## {framework} Quantum Concepts",
                "",
                f"❌ **Error**: Could not read {framework} concepts file: {e}",
                "",
            ]

        return content

    def _generate_pattern_analysis_tables(self) -> List[str]:
        """Generate tables for pattern analysis data."""
        content = [
            "## Pattern Analysis Results",
            "",
            "The following tables contain the complete results of the pattern matching analysis.",
            "",
        ]

        # Top matched concepts
        if self.report_files["top_matched_concepts"].exists():
            content.extend(self._generate_top_concepts_table())

        # Match type analysis
        if self.report_files["match_type_counts"].exists():
            content.extend(self._generate_match_type_table())

        # Framework analysis
        if self.report_files["matches_by_framework"].exists():
            content.extend(self._generate_framework_analysis_table())

        # Pattern frequency analysis
        if self.report_files["patterns_by_match_count"].exists():
            content.extend(self._generate_pattern_frequency_table())

        return content

    def _generate_top_concepts_table(self) -> List[str]:
        """Generate the top matched concepts table."""
        try:
            df = pd.read_csv(self.report_files["top_matched_concepts"])

            content = [
                "### Top Matched Quantum Concepts",
                "",
                "The most frequently matched quantum concepts across all frameworks and projects.",
                "",
                f"**Total Concepts**: {len(df)}",
                "",
            ]

            if not df.empty:
                # Add row numbers
                display_df = df.copy()
                display_df.insert(0, "Rank", range(1, len(display_df) + 1))

                table_md = display_df.to_markdown(index=False)
                content.append(table_md)
            else:
                content.append("*No matched concepts found.*")

            content.extend(["", ""])

        except Exception as e:
            content = [
                "### Top Matched Quantum Concepts",
                "",
                f"❌ **Error**: Could not read top concepts file: {e}",
                "",
            ]

        return content

    def _generate_match_type_table(self) -> List[str]:
        """Generate the match type analysis table."""
        try:
            df = pd.read_csv(self.report_files["match_type_counts"])

            content = [
                "### Match Type Analysis",
                "",
                "Distribution of matches by type (name-based, semantic, etc.).",
                "",
                f"**Total Match Types**: {len(df)}",
                "",
            ]

            if not df.empty:
                # Add row numbers
                display_df = df.copy()
                display_df.insert(0, "Row", range(1, len(display_df) + 1))

                table_md = display_df.to_markdown(index=False)
                content.append(table_md)
            else:
                content.append("*No match type data found.*")

            content.extend(["", ""])

        except Exception as e:
            content = [
                "### Match Type Analysis",
                "",
                f"❌ **Error**: Could not read match type file: {e}",
                "",
            ]

        return content

    def _generate_framework_analysis_table(self) -> List[str]:
        """Generate the framework analysis table."""
        try:
            df = pd.read_csv(self.report_files["matches_by_framework"])

            content = [
                "### Framework Analysis",
                "",
                "Distribution of matches by source framework.",
                "",
                f"**Frameworks**: {len(df)}",
                "",
            ]

            if not df.empty:
                # Add row numbers
                display_df = df.copy()
                display_df.insert(0, "Row", range(1, len(display_df) + 1))

                table_md = display_df.to_markdown(index=False)
                content.append(table_md)
            else:
                content.append("*No framework data found.*")

            content.extend(["", ""])

        except Exception as e:
            content = [
                "### Framework Analysis",
                "",
                f"❌ **Error**: Could not read framework analysis file: {e}",
                "",
            ]

        return content

    def _generate_pattern_frequency_table(self) -> List[str]:
        """Generate the pattern frequency table."""
        try:
            df = pd.read_csv(self.report_files["patterns_by_match_count"])

            content = [
                "### Pattern Frequency Analysis",
                "",
                "Frequency of quantum patterns in the analysis.",
                "",
                f"**Total Patterns**: {len(df)}",
                "",
            ]

            if not df.empty:
                # Add row numbers
                display_df = df.copy()
                display_df.insert(0, "Row", range(1, len(display_df) + 1))

                table_md = display_df.to_markdown(index=False)
                content.append(table_md)
            else:
                content.append("*No pattern frequency data found.*")

            content.extend(["", ""])

        except Exception as e:
            content = [
                "### Pattern Frequency Analysis",
                "",
                f"❌ **Error**: Could not read pattern frequency file: {e}",
                "",
            ]

        return content

    def _generate_pattern_atlas_section(self) -> List[str]:
        """Generate the Pattern Atlas section."""
        content = [
            "## Quantum Patterns from PlanQK Pattern Atlas",
            "",
            "This section contains the complete dataset of quantum patterns downloaded from the PlanQK Pattern Atlas. These patterns serve as the reference set for pattern matching analysis.",
            "",
        ]

        if self.patterns_file.exists():
            try:
                with open(self.patterns_file, "r", encoding="utf-8") as f:
                    patterns_data = json.load(f)

                content.extend(
                    [
                        f"**Total Patterns**: {len(patterns_data)}",
                        f"**Source**: [PlanQK Pattern Atlas](https://patternatlas.planqk.de/pattern-languages/af7780d5-1f97-4536-8da7-4194b093ab1d)",
                        f"**File**: `{self.patterns_file.name}`",
                        "",
                        "### Pattern Details",
                        "",
                        "The following table contains all patterns with their metadata:",
                        "",
                    ]
                )

                # Create a summary table of patterns
                pattern_summary = []
                for i, pattern in enumerate(patterns_data, 1):
                    pattern_summary.append(
                        {
                            "ID": i,
                            "Name": pattern.get("name", "N/A"),
                            "Alias": pattern.get("alias", "N/A"),
                            "Intent": (
                                pattern.get("intent", "N/A")[:100] + "..."
                                if len(pattern.get("intent", "")) > 100
                                else pattern.get("intent", "N/A")
                            ),
                        }
                    )

                if pattern_summary:
                    summary_df = pd.DataFrame(pattern_summary)
                    # Add row numbers (ID is already there, but let's add a Row column for consistency)
                    summary_df.insert(0, "Row", range(1, len(summary_df) + 1))
                    table_md = summary_df.to_markdown(index=False)
                    content.append(table_md)
                else:
                    content.append("*No pattern data found.*")

                content.extend(["", ""])

            except Exception as e:
                content.extend([f"❌ **Error**: Could not read patterns file: {e}", ""])
        else:
            content.extend(
                [
                    f"⚠️ **Note**: The patterns file was not found at `{self.patterns_file}`",
                    "",
                ]
            )

        return content

    def _generate_references(self) -> List[str]:
        """Generate the references section."""
        return [
            "## References",
            "",
            "@online{PlanQK_QuantumPatterns_2024,",
            "  author       = {{PlanQK}},",
            "  title        = {Quantum Computing Patterns},",
            "  year         = {2025},",
            "  url          = {https://patternatlas.planqk.de/pattern-languages/af7780d5-1f97-4536-8da7-4194b093ab1d},",
            "  urldate      = {2025-09-28}",
            "}",
            "",
            "---",
            "",
            "*This document was automatically generated from the experimental data files.*",
            "",
        ]


def main():
    """Main function to generate the experimental data report."""
    generator = ExperimentalDataReportGenerator()
    generator.generate_report()


if __name__ == "__main__":
    main()

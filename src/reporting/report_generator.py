"""
Unified report generation system for QPA: Quantum Patterns Analyser.

This module provides a comprehensive reporting system that can generate
different types of reports from the analysis data, including:
- Experimental data reports (complete datasets)
- Base concept reports (framework summaries)
- Pattern reports (PlanQK Pattern Atlas)
- Extended pattern analysis reports
"""

import csv
import json
from pathlib import Path
from typing import Dict, List, Set, Optional

import pandas as pd

from src.conf import config


class ReportGenerator:
    """Unified report generator for all analysis outputs."""
    
    def __init__(self):
        """Initialize the report generator with file paths."""
        self.data_dir = config.RESULTS_DIR
        self.report_dir = config.DOCS_DIR
        
        # Define input files
        self.concept_files = {
            "Classiq": self.data_dir / "classiq_quantum_concepts.csv",
            "PennyLane": self.data_dir / "pennylane_quantum_concepts.csv",
            "Qiskit": self.data_dir / "qiskit_quantum_concepts.csv",
        }
        
        self.patterns_file = self.data_dir / "quantum_patterns.json"
        self.extended_patterns_file = self.data_dir / "patterns_used_in_categorization.csv"
        
        self.report_files = {
            "top_matched_concepts": self.data_dir / "report" / "top_matched_concepts.csv",
            "match_type_counts": self.data_dir / "report" / "match_type_counts.csv",
            "matches_by_framework": self.data_dir / "report" / "matches_by_framework.csv",
            "patterns_by_match_count": self.data_dir / "report" / "patterns_by_match_count.csv",
        }

    def generate_experimental_data_report(self) -> Path:
        """Generate comprehensive experimental data report."""
        output_file = self.report_dir / "experimental_data.md"
        print("Generating experimental data report...")
        
        # Create output directory
        self.report_dir.mkdir(parents=True, exist_ok=True)
        
        content = []
        content.extend(self._generate_experimental_header())
        content.extend(self._generate_framework_tables())
        content.extend(self._generate_pattern_analysis_tables())
        content.extend(self._generate_pattern_atlas_section())
        content.extend(self._generate_references())
        
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("\n".join(content))
        
        print(f"Experimental data report generated: {output_file}")
        return output_file

    def generate_base_concept_report(self) -> Path:
        """Generate base concept extraction report."""
        output_file = self.report_dir / "extracted_concepts_summary.md"
        print("Generating base concept report...")
        
        # Create output directory
        self.report_dir.mkdir(parents=True, exist_ok=True)
        
        content = []
        content.extend(self._generate_base_concept_header())
        content.extend(self._generate_framework_concept_tables())
        content.extend(self._generate_pattern_coverage_section())
        
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("\n".join(content))
        
        print(f"Base concept report generated: {output_file}")
        return output_file

    def generate_pattern_report(self) -> Path:
        """Generate PlanQK Pattern Atlas report."""
        output_file = self.report_dir / "quantum_patterns_report.md"
        print("Generating pattern report...")
        
        if not self.patterns_file.exists():
            print(f"WARNING: Pattern file not found: {self.patterns_file}")
            return None
        
        # Create output directory
        self.report_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(self.patterns_file, "r", encoding="utf-8") as f:
                patterns_data = json.load(f)
            
            content = []
            content.extend(self._generate_pattern_header(len(patterns_data)))
            content.extend(self._generate_pattern_details(patterns_data))
            
            with open(output_file, "w", encoding="utf-8") as f:
                f.write("\n".join(content))
            
            print(f"Pattern report generated: {output_file}")
            return output_file
            
        except Exception as e:
            print(f"ERROR: Error generating pattern report: {e}")
            return None

    def generate_extended_pattern_analysis(self) -> Path:
        """Generate extended pattern coverage analysis."""
        output_file = self.report_dir / "extended_pattern_coverage_analysis.md"
        print("Generating extended pattern analysis...")
        
        # Create output directory
        self.report_dir.mkdir(parents=True, exist_ok=True)
        
        # Load data
        extended_patterns = self._load_extended_patterns()
        framework_patterns = self._load_framework_patterns()
        target_patterns = self._load_target_project_patterns()
        
        # Perform analysis
        analysis_results = self._perform_extended_analysis(
            extended_patterns, framework_patterns, target_patterns
        )
        
        # Generate report
        content = []
        content.extend(self._generate_extended_header(analysis_results["summary"]))
        content.extend(self._generate_framework_coverage_analysis(analysis_results["framework_coverage"]))
        content.extend(self._generate_target_coverage_analysis(analysis_results["target_coverage"]))
        content.extend(self._generate_detailed_pattern_lists(analysis_results))
        
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("\n".join(content))
        
        print(f"Extended pattern analysis generated: {output_file}")
        self._print_extended_summary(analysis_results["summary"])
        return output_file

    def generate_all_reports(self) -> Dict[str, Path]:
        """Generate all available reports."""
        print("Generating all reports...")
        
        reports = {}
        
        # Generate each report
        reports["experimental_data"] = self.generate_experimental_data_report()
        reports["base_concept"] = self.generate_base_concept_report()
        reports["pattern"] = self.generate_pattern_report()
        reports["extended_pattern"] = self.generate_extended_pattern_analysis()
        
        print(f"\nGenerated {len(reports)} reports:")
        for report_type, path in reports.items():
            if path:
                print(f"  SUCCESS: {report_type}: {path.name}")
            else:
                print(f"  FAILED: {report_type}: Failed to generate")
        
        return reports

    # === Private Methods ===

    def _generate_experimental_header(self) -> List[str]:
        """Generate experimental data report header."""
        return [
            "# Experimental Data",
            "",
            "This document contains the complete experimental datasets used in the quantum pattern analysis research.",
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

    def _generate_base_concept_header(self) -> List[str]:
        """Generate base concept report header."""
        return [
            "# Extracted Quantum Concepts Summary",
            "",
            "This document provides a comprehensive summary of quantum concepts extracted from three major quantum computing frameworks.",
            "",
            "## Overview",
            "",
            "The concept extraction process identified quantum computing concepts from:",
            "- **Qiskit**: IBM's quantum computing framework",
            "- **PennyLane**: Xanadu's quantum machine learning framework", 
            "- **Classiq**: High-level quantum algorithm design platform",
            "",
            "Each framework's concepts are presented in separate tables below, showing the concept names and their summaries.",
            "",
        ]

    def _generate_pattern_header(self, pattern_count: int) -> List[str]:
        """Generate pattern report header."""
        return [
            "# Quantum Patterns Report",
            "",
            "This document provides a comprehensive overview of quantum computing patterns from the PlanQK Pattern Atlas.",
            "",
            "## Overview",
            "",
            f"**Total Patterns**: {pattern_count}",
            f"**Source**: [PlanQK Pattern Atlas](https://patternatlas.planqk.de/pattern-languages/af7780d5-1f97-4536-8da7-4194b093ab1d)",
            f"**Generated from**: `quantum_patterns.json`",
            "",
            "Each pattern includes its intent, context, solution, and related information as provided by the PlanQK Pattern Atlas.",
            "",
            "---",
            "",
        ]

    def _generate_extended_header(self, summary: Dict) -> List[str]:
        """Generate extended pattern analysis header."""
        return [
            "# Extended Pattern Coverage Analysis",
            "",
            "This report analyzes the coverage of extended quantum patterns across sourcing frameworks and target projects.",
            "",
            "## Summary",
            "",
            f"- **Total Extended Patterns**: {summary['total_extended_patterns']}",
            f"- **Found in Sourcing Frameworks**: {summary['found_in_frameworks']} ({summary['framework_coverage_rate']:.1f}%)",
            f"- **Found in Target Projects**: {summary['found_in_targets']} ({summary['target_coverage_rate']:.1f}%)",
            "",
            "---",
            "",
        ]

    def _generate_framework_tables(self) -> List[str]:
        """Generate framework concept tables for experimental data."""
        content = []
        
        for framework, file_path in self.concept_files.items():
            if file_path.exists():
                content.extend(self._generate_framework_table(framework, file_path))
            else:
                content.extend([
                    f"## {framework} Quantum Concepts",
                    "",
                    f"**Note**: The {framework} concepts file was not found at `{file_path}`",
                    "",
                ])
        
        return content

    def _generate_framework_concept_tables(self) -> List[str]:
        """Generate framework concept tables for base concept report."""
        content = []
        
        for framework, file_path in self.concept_files.items():
            if file_path.exists():
                content.extend(self._generate_framework_table(framework, file_path))
            else:
                content.extend([
                    f"## {framework} Quantum Concepts",
                    "",
                    f"**Note**: The {framework} concepts file was not found at `{file_path}`",
                    "",
                ])
        
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
                f"**Error**: Could not read {framework} concepts file: {e}",
                "",
            ]
        
        return content

    def _generate_pattern_analysis_tables(self) -> List[str]:
        """Generate pattern analysis tables."""
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
                f"**Error**: Could not read top concepts file: {e}",
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
                f" **Error**: Could not read match type file: {e}",
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
                f"**Error**: Could not read framework analysis file: {e}",
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
                f"**Error**: Could not read pattern frequency file: {e}",
                "",
            ]
        
        return content

    def _generate_pattern_atlas_section(self) -> List[str]:
        """Generate the Pattern Atlas section."""
        content = [
            "## Quantum Patterns from PlanQK Pattern Atlas",
            "",
            "This section contains the complete dataset of quantum patterns downloaded from the PlanQK Pattern Atlas.",
            "",
        ]
        
        if self.patterns_file.exists():
            try:
                with open(self.patterns_file, "r", encoding="utf-8") as f:
                    patterns_data = json.load(f)
                
                content.extend([
                    f"**Total Patterns**: {len(patterns_data)}",
                    f"**Source**: [PlanQK Pattern Atlas](https://patternatlas.planqk.de/pattern-languages/af7780d5-1f97-4536-8da7-4194b093ab1d)",
                    f"**File**: `{self.patterns_file.name}`",
                    "",
                    "### Pattern Details",
                    "",
                    "The following table contains all patterns with their metadata:",
                    "",
                ])
                
                # Create a summary table of patterns
                pattern_summary = []
                for i, pattern in enumerate(patterns_data, 1):
                    pattern_summary.append({
                        "ID": i,
                        "Name": pattern.get("name", "N/A"),
                        "Alias": pattern.get("alias", "N/A"),
                        "Intent": (
                            pattern.get("intent", "N/A")[:100] + "..."
                            if len(pattern.get("intent", "")) > 100
                            else pattern.get("intent", "N/A")
                        ),
                    })
                
                if pattern_summary:
                    summary_df = pd.DataFrame(pattern_summary)
                    summary_df.insert(0, "Row", range(1, len(summary_df) + 1))
                    table_md = summary_df.to_markdown(index=False)
                    content.append(table_md)
                else:
                    content.append("*No pattern data found.*")
                
                content.extend(["", ""])
                
            except Exception as e:
                content.extend([f"**Error**: Could not read patterns file: {e}", ""])
        else:
            content.extend([
                f"**Note**: The patterns file was not found at `{self.patterns_file}`",
                "",
            ])
        
        return content

    def _generate_pattern_coverage_section(self) -> List[str]:
        """Generate pattern coverage analysis section."""
        return [
            "## Pattern Coverage Analysis",
            "",
            "This section analyzes how the extracted concepts relate to known quantum patterns.",
            "",
            "### Pattern Coverage Summary",
            "",
            "The pattern coverage analysis shows which quantum patterns from the PlanQK Pattern Atlas are represented in the extracted concepts.",
            "",
            "### Complete List of Patterns Found",
            "",
            "The following table shows all patterns that were found in the analysis:",
            "",
            "| Pattern | Framework | Concept Count |",
            "|---------|-----------|----------------|",
            "| *Pattern analysis not available* | - | - |",
            "",
            "### New Patterns Created",
            "",
            "The following patterns were created during the analysis and are not in the base pattern list:",
            "",
            "| Pattern | Observed In | Concept Count |",
            "|---------|-------------|----------------|",
            "| *New pattern analysis not available* | - | - |",
            "",
            "---",
            "",
            "*This document was automatically generated from the concept extraction files.*",
            "",
        ]

    def _generate_pattern_details(self, patterns_data: List[Dict]) -> List[str]:
        """Generate detailed sections for each pattern."""
        content = []
        
        for pattern in patterns_data:
            content.extend(self._generate_pattern_section(pattern))
            content.append("")  # Add spacing between patterns
        
        return content

    def _generate_pattern_section(self, pattern: Dict) -> List[str]:
        """Generate a detailed section for a single pattern."""
        content = []
        
        # Title
        name = pattern.get('name', 'Unnamed Pattern')
        content.append(f"## {name}")
        content.append("")
        
        # Alias
        alias = pattern.get("alias")
        if alias and alias.strip() != "—":
            content.append(f"**Alias**: {alias}")
            content.append("")
        
        # Intent
        intent = pattern.get("intent")
        if intent and intent.strip():
            content.append("### Intent")
            content.append("")
            content.append(intent)
            content.append("")
        
        # Context
        context = pattern.get("context")
        if context and context.strip():
            content.append("### Context")
            content.append("")
            content.append(context)
            content.append("")
        
        # Solution
        solution = pattern.get("solution")
        if solution and solution.strip():
            content.append("### Solution")
            content.append("")
            content.append(solution)
            content.append("")
        
        # Consequences
        consequences = pattern.get("consequences")
        if consequences and consequences.strip():
            content.append("### Consequences")
            content.append("")
            content.append(consequences)
            content.append("")
        
        # Related Patterns
        related_patterns = pattern.get("related_patterns")
        if related_patterns and related_patterns.strip():
            content.append("### Related Patterns")
            content.append("")
            content.append(related_patterns)
            content.append("")
        
        # Implementation Notes
        implementation_notes = pattern.get("implementation_notes")
        if implementation_notes and implementation_notes.strip():
            content.append("### Implementation Notes")
            content.append("")
            content.append(implementation_notes)
            content.append("")
        
        # Examples
        examples = pattern.get("examples")
        if examples and examples.strip():
            content.append("### Examples")
            content.append("")
            content.append(examples)
            content.append("")
        
        content.append("---")
        
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

    # === Extended Pattern Analysis Methods ===

    def _load_extended_patterns(self) -> Set[str]:
        """Load the extended pattern list from CSV file."""
        patterns = set()
        
        try:
            with open(self.extended_patterns_file, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    pattern_name = row.get("PatternName", "").strip()
                    if pattern_name:
                        patterns.add(pattern_name)
        except Exception as e:
            print(f"Error loading extended patterns: {e}")
        
        return patterns

    def _load_framework_patterns(self) -> Dict[str, Set[str]]:
        """Load patterns from the three sourcing frameworks."""
        framework_files = {
            "Classiq": self.data_dir / "knowledge_base/enriched_classiq_quantum_patterns.csv",
            "PennyLane": self.data_dir / "knowledge_base/enriched_pennylane_quantum_patterns.csv",
            "Qiskit": self.data_dir / "knowledge_base/enriched_qiskit_quantum_patterns.csv",
        }
        
        framework_patterns = {}
        
        for framework, file_path in framework_files.items():
            patterns = set()
            try:
                with open(file_path, encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        pattern = row.get("pattern", "").strip()
                        if pattern:
                            patterns.add(pattern)
            except Exception as e:
                print(f"Error loading {framework} patterns: {e}")
            
            framework_patterns[framework] = patterns
        
        return framework_patterns

    def _load_target_project_patterns(self) -> Set[str]:
        """Load patterns from the broader target project list."""
        patterns = set()
        target_file = self.data_dir / "quantum_concept_matches_with_patterns.csv"
        
        try:
            with open(target_file, encoding="utf-8") as f:
                reader = csv.DictReader(f, delimiter=";")
                for row in reader:
                    pattern = row.get("pattern", "").strip()
                    if pattern and pattern != "N/A":
                        patterns.add(pattern)
        except Exception as e:
            print(f"Error loading target project patterns: {e}")
        
        return patterns

    def _perform_extended_analysis(self, extended_patterns: Set[str], framework_patterns: Dict[str, Set[str]], target_patterns: Set[str]) -> Dict:
        """Perform the pattern coverage analysis."""
        results = {
            "extended_patterns": extended_patterns,
            "framework_patterns": framework_patterns,
            "target_patterns": target_patterns,
            "framework_coverage": {},
            "target_coverage": {},
            "summary": {}
        }
        
        # Analyze framework coverage
        for framework, patterns in framework_patterns.items():
            found_patterns = extended_patterns.intersection(patterns)
            results["framework_coverage"][framework] = {
                "found": found_patterns,
                "count": len(found_patterns),
                "total": len(extended_patterns),
                "percentage": (len(found_patterns) / len(extended_patterns)) * 100 if extended_patterns else 0
            }
        
        # Analyze target project coverage
        found_in_targets = extended_patterns.intersection(target_patterns)
        results["target_coverage"] = {
            "found": found_in_targets,
            "count": len(found_in_targets),
            "total": len(extended_patterns),
            "percentage": (len(found_in_targets) / len(extended_patterns)) * 100 if extended_patterns else 0
        }
        
        # Generate summary
        results["summary"] = {
            "total_extended_patterns": len(extended_patterns),
            "found_in_frameworks": len(extended_patterns.intersection(set().union(*framework_patterns.values()))),
            "found_in_targets": len(found_in_targets),
            "framework_coverage_rate": sum(r["percentage"] for r in results["framework_coverage"].values()) / len(framework_patterns) if framework_patterns else 0,
            "target_coverage_rate": results["target_coverage"]["percentage"]
        }
        
        return results

    def _generate_framework_coverage_analysis(self, framework_coverage: Dict) -> List[str]:
        """Generate framework coverage analysis section."""
        content = [
            "## Framework Coverage Analysis",
            "",
            "Analysis of how many extended patterns were found in each sourcing framework:",
            "",
        ]
        
        for framework, data in framework_coverage.items():
            content.extend([
                f"### {framework}",
                "",
                f"- **Patterns Found**: {data['count']} out of {data['total']} ({data['percentage']:.1f}%)",
                "",
            ])
        
        return content

    def _generate_target_coverage_analysis(self, target_coverage: Dict) -> List[str]:
        """Generate target project coverage analysis section."""
        return [
            "## Target Project Coverage Analysis",
            "",
            "Analysis of how many extended patterns were found in the broader target project list:",
            "",
            f"- **Patterns Found**: {target_coverage['count']} out of {target_coverage['total']} ({target_coverage['percentage']:.1f}%)",
            "",
            "---",
            "",
        ]

    def _generate_detailed_pattern_lists(self, results: Dict) -> List[str]:
        """Generate detailed pattern lists."""
        content = [
            "## Detailed Pattern Lists",
            "",
            "### Extended Patterns Found in Frameworks",
            "",
        ]
        
        # Framework patterns
        for framework, data in results["framework_coverage"].items():
            if data["found"]:
                content.extend([
                    f"#### {framework}",
                    "",
                    "| Pattern |",
                    "|---------|",
                ])
                for pattern in sorted(data["found"]):
                    content.append(f"| {pattern} |")
                content.append("")
        
        # Target patterns
        if results["target_coverage"]["found"]:
            content.extend([
                "### Extended Patterns Found in Target Projects",
                "",
                "| Pattern |",
                "|---------|",
            ])
            for pattern in sorted(results["target_coverage"]["found"]):
                content.append(f"| {pattern} |")
            content.append("")
        
        return content

    def _print_extended_summary(self, summary: Dict):
        """Print analysis summary to console."""
        print("\n" + "="*60)
        print("EXTENDED PATTERN COVERAGE ANALYSIS SUMMARY")
        print("="*60)
        print(f"Total Extended Patterns: {summary['total_extended_patterns']}")
        print(f"Found in Sourcing Frameworks: {summary['found_in_frameworks']} ({summary['framework_coverage_rate']:.1f}%)")
        print(f"Found in Target Projects: {summary['found_in_targets']} ({summary['target_coverage_rate']:.1f}%)")
        print("="*60)


# === Convenience Functions ===

def generate_experimental_data_report():
    """Convenience function for experimental data report."""
    generator = ReportGenerator()
    return generator.generate_experimental_data_report()


def generate_base_concept_report():
    """Convenience function for base concept report."""
    generator = ReportGenerator()
    return generator.generate_base_concept_report()


def generate_pattern_report():
    """Convenience function for pattern report."""
    generator = ReportGenerator()
    return generator.generate_pattern_report()


def generate_extended_pattern_analysis():
    """Convenience function for extended pattern analysis."""
    generator = ReportGenerator()
    return generator.generate_extended_pattern_analysis()


def generate_all_reports():
    """Convenience function to generate all reports."""
    generator = ReportGenerator()
    return generator.generate_all_reports()


def main():
    """Main function to generate all reports."""
    generator = ReportGenerator()
    generator.generate_all_reports()


if __name__ == "__main__":
    main()

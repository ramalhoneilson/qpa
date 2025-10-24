"""
Generates a Markdown summary of the raw quantum concepts extracted from each framework.

This script reads the output CSV files from the 'identify-concepts' step and
compiles them into a single, human-readable Markdown document. Each framework's
concepts are presented in a separate table.
"""

import csv
import json
from pathlib import Path

from src.conf import config

# Define the input CSV files and their specific delimiters
INPUT_FILES = {
    "Qiskit": {
        "path": config.RESULTS_DIR / "qiskit_quantum_concepts.csv",
        "delimiter": ";",
    },
    "PennyLane": {
        "path": config.RESULTS_DIR / "pennylane_quantum_concepts.csv",
        "delimiter": ",",
    },
    "Classiq": {
        "path": config.RESULTS_DIR / "classiq_quantum_concepts.csv",
        "delimiter": ",",
    },
}

OUTPUT_MD_PATH = config.DOCS_DIR / "extracted_concepts_summary.md"


def read_concepts_from_csv(file_path: Path, delimiter: str) -> list[dict]:
    """Reads concept data from a CSV file with a specific delimiter."""
    if not file_path.exists():
        print(f"  - Warning: Input file not found: {file_path}")
        return []

    concepts = []
    try:
        with open(file_path, encoding="utf-8") as f:
            # DictReader automatically uses the first row as headers
            reader = csv.DictReader(f, delimiter=delimiter)
            for row in reader:
                concepts.append(row)
        return concepts
    except Exception as e:
        print(f"  - Error reading {file_path}: {e}")
        return []


def generate_markdown_table(concepts: list[dict]) -> str:
    """Formats a list of concepts into a Markdown table string."""
    if not concepts:
        return "*No concepts were extracted for this framework.*\n"

    # Use a list of strings for efficient building
    table_parts = []

    # --- Table Header ---
    table_parts.append("| Concept Name | Summary |")
    table_parts.append("|--------------|---------|")

    # --- Table Rows ---
    for concept in concepts:
        # Get data, providing default empty strings if keys are missing
        name = concept.get("name", "N/A").strip()
        summary = concept.get("summary", "").strip()

        # Clean up content for Markdown table cells
        # Replace pipe characters to prevent breaking table formatting
        name = name.replace("|", "\\|")
        # Remove newlines and extra whitespace from summary
        summary = " ".join(summary.split()).replace("|", "\\|")

        table_parts.append(f"| `{name}` | {summary} |")

    return "\n".join(table_parts) + "\n"


def load_base_patterns() -> list[dict]:
    """Load the base quantum patterns from the JSON file."""
    patterns_file = config.RESULTS_DIR / "quantum_patterns.json"
    try:
        with open(patterns_file, encoding="utf-8") as f:
            patterns = json.load(f)
        return patterns
    except Exception as e:
        print(f"  - Error loading base patterns: {e}")
        return []


def extract_framework_patterns(enriched_file_path: Path) -> set[str]:
    """Extract unique patterns from an enriched framework CSV file."""
    patterns = set()
    try:
        with open(enriched_file_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                pattern = row.get("pattern", "").strip()
                if pattern:
                    patterns.add(pattern)
    except Exception as e:
        print(f"  - Error reading {enriched_file_path}: {e}")
    return patterns


def analyze_pattern_coverage() -> dict:
    """Analyze pattern coverage across frameworks."""
    # Load base patterns
    base_patterns = load_base_patterns()
    base_pattern_names = {pattern["name"] for pattern in base_patterns}

    # Define enriched files for each framework
    enriched_files = {
        "Classiq": config.RESULTS_DIR
        / "knowledge_base/enriched_classiq_quantum_patterns.csv",
        "PennyLane": config.RESULTS_DIR
        / "knowledge_base/enriched_pennylane_quantum_patterns.csv",
        "Qiskit": config.RESULTS_DIR
        / "knowledge_base/enriched_qiskit_quantum_patterns.csv",
    }

    # Extract patterns from each framework
    framework_patterns = {}
    all_found_patterns = set()

    for framework, file_path in enriched_files.items():
        patterns = extract_framework_patterns(file_path)
        framework_patterns[framework] = patterns
        all_found_patterns.update(patterns)

    # Find missing patterns (patterns in base but not found in any framework)
    missing_patterns = base_pattern_names - all_found_patterns

    # Find patterns found in frameworks but not in base
    extra_patterns = all_found_patterns - base_pattern_names

    return {
        "base_patterns": base_pattern_names,
        "framework_patterns": framework_patterns,
        "all_found_patterns": all_found_patterns,
        "missing_patterns": missing_patterns,
        "extra_patterns": extra_patterns,
        "coverage_percentage": (
            len(all_found_patterns & base_pattern_names) / len(base_pattern_names) * 100
            if base_pattern_names
            else 0
        ),
    }


def extract_framework_patterns_with_sources(
    enriched_file_path: Path,
) -> dict[str, set[str]]:
    """Extract patterns from an enriched framework CSV file with their sources."""
    patterns_with_sources = {}
    try:
        with open(enriched_file_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                pattern = row.get("pattern", "").strip()
                concept_name = row.get("name", "").strip()
                if pattern and concept_name:
                    if pattern not in patterns_with_sources:
                        patterns_with_sources[pattern] = set()
                    patterns_with_sources[pattern].add(concept_name)
    except Exception as e:
        print(f"  - Error reading {enriched_file_path}: {e}")
    return patterns_with_sources


def analyze_pattern_coverage_with_sources() -> dict:
    """Analyze pattern coverage across frameworks with source information."""
    # Load base patterns
    base_patterns = load_base_patterns()
    base_pattern_names = {pattern["name"] for pattern in base_patterns}

    # Define enriched files for each framework
    enriched_files = {
        "Classiq": config.RESULTS_DIR
        / "knowledge_base/enriched_classiq_quantum_patterns.csv",
        "PennyLane": config.RESULTS_DIR
        / "knowledge_base/enriched_pennylane_quantum_patterns.csv",
        "Qiskit": config.RESULTS_DIR
        / "knowledge_base/enriched_qiskit_quantum_patterns.csv",
    }

    # Extract patterns from each framework with sources
    framework_patterns_with_sources = {}
    all_found_patterns = set()

    for framework, file_path in enriched_files.items():
        patterns_with_sources = extract_framework_patterns_with_sources(file_path)
        framework_patterns_with_sources[framework] = patterns_with_sources
        all_found_patterns.update(patterns_with_sources.keys())

    # Find missing patterns (patterns in base but not found in any framework)
    missing_patterns = base_pattern_names - all_found_patterns

    # Find patterns found in frameworks but not in base
    extra_patterns = all_found_patterns - base_pattern_names

    return {
        "base_patterns": base_pattern_names,
        "framework_patterns_with_sources": framework_patterns_with_sources,
        "all_found_patterns": all_found_patterns,
        "missing_patterns": missing_patterns,
        "extra_patterns": extra_patterns,
        "coverage_percentage": (
            len(all_found_patterns & base_pattern_names) / len(base_pattern_names) * 100
            if base_pattern_names
            else 0
        ),
    }


def generate_pattern_coverage_section() -> str:
    """Generate the pattern coverage analysis section."""
    print("Analyzing pattern coverage...")
    coverage_data = analyze_pattern_coverage_with_sources()

    sections = [
        "## Pattern Coverage Analysis\n",
        f"This analysis compares the quantum patterns found in the three frameworks against the base list of {len(coverage_data['base_patterns'])} patterns from `quantum_patterns.json`.\n",
        f"**Coverage: {coverage_data['coverage_percentage']:.1f}%** ({len(coverage_data['all_found_patterns'] & coverage_data['base_patterns'])}/{len(coverage_data['base_patterns'])} base patterns found)\n",
    ]

    # Framework-specific pattern counts
    sections.append("### Framework Pattern Distribution\n")
    sections.append("| Framework | Patterns Found |")
    sections.append("|-----------|----------------|")

    for framework, patterns in coverage_data["framework_patterns_with_sources"].items():
        sections.append(f"| {framework} | {len(patterns)} |")

    # Complete list of patterns found in each framework
    sections.append("\n### Complete List of Patterns Found\n")

    for framework, patterns_with_sources in coverage_data[
        "framework_patterns_with_sources"
    ].items():
        sections.append(f"#### {framework} Patterns\n")
        if patterns_with_sources:
            sections.append("| Pattern | Concepts |")
            sections.append("|---------|----------|")
            for pattern in sorted(patterns_with_sources.keys()):
                concepts = sorted(patterns_with_sources[pattern])
                concepts_str = ", ".join(
                    [f"`{c.split('/')[-1]}`" for c in concepts[:3]]
                )  # Show first 3 concepts
                if len(concepts) > 3:
                    concepts_str += f" (+{len(concepts) - 3} more)"
                sections.append(f"| {pattern} | {concepts_str} |")
        else:
            sections.append("*No patterns found.*")
        sections.append("")

    # Missing patterns
    if coverage_data["missing_patterns"]:
        sections.extend(
            [
                "### Missing Patterns\n",
                f"The following {len(coverage_data['missing_patterns'])} patterns from the base list were not found in any of the three frameworks:\n",
            ]
        )

        for pattern in sorted(coverage_data["missing_patterns"]):
            sections.append(f"- {pattern}")
        sections.append("")

    # New patterns (extra patterns) with their sources
    if coverage_data["extra_patterns"]:
        sections.extend(
            [
                "### New Patterns Created\n",
                f"The following {len(coverage_data['extra_patterns'])} patterns were found in the frameworks but are not in the base list:\n",
            ]
        )

        for pattern in sorted(coverage_data["extra_patterns"]):
            sections.append(f"#### {pattern}\n")
            sections.append("**Observed in:**\n")

            for framework, patterns_with_sources in coverage_data[
                "framework_patterns_with_sources"
            ].items():
                if pattern in patterns_with_sources:
                    concepts = sorted(patterns_with_sources[pattern])
                    sections.append(f"- **{framework}**: {len(concepts)} concepts")
                    for concept in concepts:
                        concept_name = concept.split("/")[-1]
                        sections.append(f"  - `{concept_name}`")
            sections.append("")

    return "\n".join(sections)


def main():
    """Main function to read all CSVs and write the final Markdown report."""
    print("--- Generating Summary Report for Extracted Quantum Concepts ---")

    # --- Build the full Markdown document ---
    all_md_content = [
        "# Summary of Extracted Quantum Concepts (Pre-Classification)\n",
        "This document summarizes the raw quantum concepts automatically extracted from the source code of the Qiskit, PennyLane, and Classiq frameworks. These concepts were identified by the `src/core_concepts/identify_*.py` scripts and serve as the input for the manual pattern classification step.\n",
    ]

    # Add pattern coverage analysis section
    all_md_content.append(generate_pattern_coverage_section())

    for framework_name, details in INPUT_FILES.items():
        print(f"\nProcessing concepts for {framework_name}...")

        # Add a section header for the framework
        all_md_content.append(f"## {framework_name} Concepts\n")

        # Read the concepts from the corresponding CSV file
        concepts = read_concepts_from_csv(details["path"], details["delimiter"])

        if concepts:
            print(f"  - Found {len(concepts)} concepts. Generating table...")
            # Generate and append the Markdown table for these concepts
            table_md = generate_markdown_table(concepts)
            all_md_content.append(table_md)
        else:
            all_md_content.append("*No concepts found or file was missing.*\n")

    final_md = "\n".join(all_md_content)

    # --- Write the final report to disk ---
    print(f"\nWriting Markdown report to: {OUTPUT_MD_PATH}")
    try:
        with open(OUTPUT_MD_PATH, "w", encoding="utf-8") as f:
            f.write(final_md)
        print("--- Report generation complete. ---")
    except OSError as e:
        print(f"Error: Could not write to file '{OUTPUT_MD_PATH}'. {e}")


if __name__ == "__main__":
    main()

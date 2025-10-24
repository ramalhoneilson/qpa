"""
Analyzes the coverage of extended patterns across sourcing frameworks and target projects.

This script reads the extended pattern list and analyzes:
1. How many patterns were found in the three sourcing frameworks (Classiq, PennyLane, Qiskit)
2. How many patterns were found in the broader target list of projects
"""

import csv
import json
from pathlib import Path
from collections import defaultdict

from src.conf import config


def load_extended_patterns() -> set[str]:
    """Load the extended pattern list from CSV file."""
    patterns = set()
    extended_patterns_file = config.RESULTS_DIR / "patterns_used_in_categorization.csv"

    try:
        with open(extended_patterns_file, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                pattern_name = row.get("PatternName", "").strip()
                if pattern_name:
                    patterns.add(pattern_name)
    except Exception as e:
        print(f"Error loading extended patterns: {e}")

    return patterns


def load_framework_patterns() -> dict[str, set[str]]:
    """Load patterns from the three sourcing frameworks."""
    framework_files = {
        "Classiq": config.RESULTS_DIR
        / "knowledge_base/enriched_classiq_quantum_patterns.csv",
        "PennyLane": config.RESULTS_DIR
        / "knowledge_base/enriched_pennylane_quantum_patterns.csv",
        "Qiskit": config.RESULTS_DIR
        / "knowledge_base/enriched_qiskit_quantum_patterns.csv",
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


def load_target_project_patterns() -> set[str]:
    """Load patterns from the target project matches."""
    patterns = set()
    target_file = config.RESULTS_DIR / "quantum_concept_matches_with_patterns.csv"

    try:
        with open(target_file, encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter=";")
            for row in reader:
                pattern = row.get("pattern", "").strip()
                if pattern:
                    patterns.add(pattern)
    except Exception as e:
        print(f"Error loading target project patterns: {e}")

    return patterns


def analyze_pattern_coverage():
    """Analyze the coverage of extended patterns."""
    print("Loading extended pattern list...")
    extended_patterns = load_extended_patterns()
    print(f"Found {len(extended_patterns)} extended patterns")

    print("\nLoading framework patterns...")
    framework_patterns = load_framework_patterns()

    print("\nLoading target project patterns...")
    target_patterns = load_target_project_patterns()

    # Analyze framework coverage
    framework_coverage = {}
    for framework, patterns in framework_patterns.items():
        found_patterns = extended_patterns.intersection(patterns)
        framework_coverage[framework] = {
            "total_found": len(found_patterns),
            "patterns": found_patterns,
            "missing": extended_patterns - patterns,
        }

    # Analyze target project coverage
    target_found = extended_patterns.intersection(target_patterns)
    target_missing = extended_patterns - target_patterns

    # Find patterns that appear in frameworks but not in target projects
    all_framework_patterns = set()
    for patterns in framework_patterns.values():
        all_framework_patterns.update(patterns)

    framework_only = (
        all_framework_patterns.intersection(extended_patterns) - target_patterns
    )

    return {
        "extended_patterns": extended_patterns,
        "framework_coverage": framework_coverage,
        "target_coverage": {
            "total_found": len(target_found),
            "patterns": target_found,
            "missing": target_missing,
        },
        "framework_only": framework_only,
        "target_patterns": target_patterns,
    }


def generate_report():
    """Generate the extended pattern coverage report."""
    print("=== Extended Pattern Coverage Analysis ===\n")

    coverage_data = analyze_pattern_coverage()

    # Report header
    report = [
        "# Extended Pattern Coverage Analysis",
        "",
        f"This report analyzes the coverage of {len(coverage_data['extended_patterns'])} extended patterns across:",
        "- Three sourcing frameworks (Classiq, PennyLane, Qiskit)",
        "- Broader target list of projects",
        "",
        "## Summary Statistics",
        "",
    ]

    # Framework coverage summary
    report.append("### Framework Coverage")
    report.append("| Framework | Patterns Found | Coverage % |")
    report.append("|-----------|----------------|------------|")

    total_extended = len(coverage_data["extended_patterns"])
    for framework, data in coverage_data["framework_coverage"].items():
        coverage_pct = (
            (data["total_found"] / total_extended * 100) if total_extended > 0 else 0
        )
        report.append(f"| {framework} | {data['total_found']} | {coverage_pct:.1f}% |")

    # Target project coverage
    target_coverage_pct = (
        (coverage_data["target_coverage"]["total_found"] / total_extended * 100)
        if total_extended > 0
        else 0
    )
    report.extend(
        [
            "",
            "### Target Project Coverage",
            f"**Patterns found in target projects: {coverage_data['target_coverage']['total_found']} ({target_coverage_pct:.1f}%)**",
            "",
        ]
    )

    # Detailed framework analysis
    report.append("## Detailed Framework Analysis")
    report.append("")

    for framework, data in coverage_data["framework_coverage"].items():
        report.append(f"### {framework}")
        report.append(f"**Found: {data['total_found']} patterns**")
        report.append("")

        if data["patterns"]:
            report.append("**Patterns found:**")
            for pattern in sorted(data["patterns"]):
                report.append(f"- {pattern}")
            report.append("")

        if data["missing"]:
            report.append(f"**Missing patterns ({len(data['missing'])}):**")
            for pattern in sorted(data["missing"]):
                report.append(f"- {pattern}")
            report.append("")

    # Target project analysis
    report.append("## Target Project Analysis")
    report.append("")
    report.append(
        f"**Patterns found in target projects: {coverage_data['target_coverage']['total_found']}**"
    )
    report.append("")

    if coverage_data["target_coverage"]["patterns"]:
        report.append("**Patterns found:**")
        for pattern in sorted(coverage_data["target_coverage"]["patterns"]):
            report.append(f"- {pattern}")
        report.append("")

    if coverage_data["target_coverage"]["missing"]:
        report.append(
            f"**Missing patterns ({len(coverage_data['target_coverage']['missing'])}):**"
        )
        for pattern in sorted(coverage_data["target_coverage"]["missing"]):
            report.append(f"- {pattern}")
        report.append("")

    # Framework-only patterns
    if coverage_data["framework_only"]:
        report.append("## Patterns Found Only in Frameworks (Not in Target Projects)")
        report.append("")
        report.append(
            f"**{len(coverage_data['framework_only'])} patterns found in frameworks but not in target projects:**"
        )
        for pattern in sorted(coverage_data["framework_only"]):
            report.append(f"- {pattern}")
        report.append("")

    # Cross-framework analysis
    report.append("## Cross-Framework Analysis")
    report.append("")

    # Find patterns that appear in multiple frameworks
    framework_names = list(coverage_data["framework_coverage"].keys())
    for i, framework1 in enumerate(framework_names):
        for framework2 in framework_names[i + 1 :]:
            common_patterns = coverage_data["framework_coverage"][framework1][
                "patterns"
            ].intersection(coverage_data["framework_coverage"][framework2]["patterns"])
            if common_patterns:
                report.append(
                    f"**Common patterns between {framework1} and {framework2} ({len(common_patterns)}):**"
                )
                for pattern in sorted(common_patterns):
                    report.append(f"- {pattern}")
                report.append("")

    # All frameworks common
    all_common = set.intersection(
        *[data["patterns"] for data in coverage_data["framework_coverage"].values()]
    )
    if all_common:
        report.append(
            f"**Patterns found in all three frameworks ({len(all_common)}):**"
        )
        for pattern in sorted(all_common):
            report.append(f"- {pattern}")
        report.append("")

    return "\n".join(report)


def main():
    """Main function to generate the extended pattern coverage report."""
    print("Generating extended pattern coverage analysis...")

    # Generate the report
    report_content = generate_report()

    # Write to file
    output_file = config.DOCS_DIR / "extended_pattern_coverage_analysis.md"
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(report_content)
        print(f"\nReport successfully generated: {output_file}")
    except Exception as e:
        print(f"Error writing report: {e}")

    # Also print summary to console
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    coverage_data = analyze_pattern_coverage()
    total_extended = len(coverage_data["extended_patterns"])

    print(f"Total extended patterns: {total_extended}")
    print()

    print("Framework coverage:")
    for framework, data in coverage_data["framework_coverage"].items():
        coverage_pct = (
            (data["total_found"] / total_extended * 100) if total_extended > 0 else 0
        )
        print(
            f"  {framework}: {data['total_found']}/{total_extended} ({coverage_pct:.1f}%)"
        )

    target_coverage_pct = (
        (coverage_data["target_coverage"]["total_found"] / total_extended * 100)
        if total_extended > 0
        else 0
    )
    print(
        f"\nTarget project coverage: {coverage_data['target_coverage']['total_found']}/{total_extended} ({target_coverage_pct:.1f}%)"
    )

    if coverage_data["framework_only"]:
        print(
            f"\nPatterns in frameworks but not in target projects: {len(coverage_data['framework_only'])}"
        )


if __name__ == "__main__":
    main()

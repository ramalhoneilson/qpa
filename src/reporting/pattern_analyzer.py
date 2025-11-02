"""
Pattern analysis utilities for QPA: Quantum Patterns Analyser.

This module provides pattern analysis functionality that was previously
scattered across different modules, now consolidated for better organization.
"""

from pathlib import Path
from typing import Dict, List, Set

import pandas as pd

from src.conf import config


class PatternAnalyzer:
    """Analyzes patterns across frameworks and projects."""
    
    def __init__(self):
        """Initialize the pattern analyzer with file paths."""
        self.data_dir = config.RESULTS_DIR
        
        # Define pattern files
        self.pattern_files = [
            self.data_dir / "knowledge_base/enriched_classiq_quantum_patterns.csv",
            self.data_dir / "knowledge_base/enriched_pennylane_quantum_patterns.csv",
            self.data_dir / "knowledge_base/enriched_qiskit_quantum_patterns.csv",
        ]
        
        self.extended_patterns_file = self.data_dir / "patterns_used_in_categorization.csv"
        self.target_patterns_file = self.data_dir / "quantum_concept_matches_with_patterns.csv"

    def analyze_pattern_coverage(self) -> Dict:
        """Analyze pattern coverage across frameworks and projects."""
        print("Analyzing pattern coverage...")
        
        # Load data
        extended_patterns = self.load_extended_patterns()
        framework_patterns = self.load_framework_patterns()
        target_patterns = self.load_target_project_patterns()
        
        # Perform analysis
        results = self._perform_coverage_analysis(
            extended_patterns, framework_patterns, target_patterns
        )
        
        # Print summary
        self._print_coverage_summary(results["summary"])
        
        return results

    def load_extended_patterns(self) -> Set[str]:
        """Load the extended pattern list from CSV file."""
        patterns = set()
        
        if not self.extended_patterns_file.exists():
            print(f"Extended patterns file not found: {self.extended_patterns_file}")
            return patterns
        
        try:
            with open(self.extended_patterns_file, encoding="utf-8") as f:
                reader = pd.read_csv(f)
                if "PatternName" in reader.columns:
                    patterns = set(reader["PatternName"].dropna().astype(str))
        except Exception as e:
            print(f"Error loading extended patterns: {e}")
        
        return patterns

    def load_framework_patterns(self) -> Dict[str, Set[str]]:
        """Load patterns from the three sourcing frameworks."""
        framework_patterns = {}
        
        framework_names = ["Classiq", "PennyLane", "Qiskit"]
        
        for i, file_path in enumerate(self.pattern_files):
            framework = framework_names[i]
            patterns = set()
            
            if file_path.exists():
                try:
                    df = pd.read_csv(file_path)
                    if "pattern" in df.columns:
                        patterns = set(df["pattern"].dropna().astype(str))
                except Exception as e:
                    print(f"Error loading {framework} patterns: {e}")
            else:
                print(f"Framework patterns file not found: {file_path}")
            
            framework_patterns[framework] = patterns
        
        return framework_patterns

    def load_target_project_patterns(self) -> Set[str]:
        """Load patterns from the broader target project list."""
        patterns = set()
        
        if not self.target_patterns_file.exists():
            print(f"Target patterns file not found: {self.target_patterns_file}")
            return patterns
        
        try:
            df = pd.read_csv(self.target_patterns_file, delimiter=";")
            if "pattern" in df.columns:
                patterns = set(df["pattern"].dropna().astype(str))
                # Remove N/A patterns
                patterns.discard("N/A")
        except Exception as e:
            print(f"Error loading target project patterns: {e}")
        
        return patterns

    def _perform_coverage_analysis(self, extended_patterns: Set[str], framework_patterns: Dict[str, Set[str]], target_patterns: Set[str]) -> Dict:
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
        all_framework_patterns = set().union(*framework_patterns.values())
        results["summary"] = {
            "total_extended_patterns": len(extended_patterns),
            "found_in_frameworks": len(extended_patterns.intersection(all_framework_patterns)),
            "found_in_targets": len(found_in_targets),
            "framework_coverage_rate": sum(r["percentage"] for r in results["framework_coverage"].values()) / len(framework_patterns) if framework_patterns else 0,
            "target_coverage_rate": results["target_coverage"]["percentage"]
        }
        
        return results

    def _print_coverage_summary(self, summary: Dict):
        """Print analysis summary to console."""
        print("\n" + "="*60)
        print("PATTERN COVERAGE ANALYSIS SUMMARY")
        print("="*60)
        print(f"Total Extended Patterns: {summary['total_extended_patterns']}")
        print(f"Found in Sourcing Frameworks: {summary['found_in_frameworks']} ({summary['framework_coverage_rate']:.1f}%)")
        print(f"Found in Target Projects: {summary['found_in_targets']} ({summary['target_coverage_rate']:.1f}%)")
        print("="*60)

    def get_pattern_statistics(self) -> Dict:
        """Get comprehensive pattern statistics."""
        print("Generating pattern statistics...")
        
        extended_patterns = self.load_extended_patterns()
        framework_patterns = self.load_framework_patterns()
        target_patterns = self.load_target_project_patterns()
        
        stats = {
            "extended_patterns": {
                "total": len(extended_patterns),
                "patterns": list(extended_patterns)
            },
            "framework_patterns": {},
            "target_patterns": {
                "total": len(target_patterns),
                "patterns": list(target_patterns)
            },
            "coverage": {}
        }
        
        # Framework statistics
        for framework, patterns in framework_patterns.items():
            stats["framework_patterns"][framework] = {
                "total": len(patterns),
                "patterns": list(patterns)
            }
            
            # Coverage statistics
            found_patterns = extended_patterns.intersection(patterns)
            stats["coverage"][framework] = {
                "found": len(found_patterns),
                "percentage": (len(found_patterns) / len(extended_patterns)) * 100 if extended_patterns else 0,
                "patterns": list(found_patterns)
            }
        
        # Overall target coverage
        found_in_targets = extended_patterns.intersection(target_patterns)
        stats["coverage"]["target_projects"] = {
            "found": len(found_in_targets),
            "percentage": (len(found_in_targets) / len(extended_patterns)) * 100 if extended_patterns else 0,
            "patterns": list(found_in_targets)
        }
        
        return stats

    def export_pattern_analysis(self, output_file: Path) -> None:
        """Export pattern analysis results to CSV."""
        print(f"Exporting pattern analysis to {output_file}")
        
        stats = self.get_pattern_statistics()
        
        # Create comprehensive analysis DataFrame
        analysis_data = []
        
        # Extended patterns
        for pattern in stats["extended_patterns"]["patterns"]:
            row = {
                "pattern": pattern,
                "type": "extended",
                "found_in_classiq": pattern in stats["framework_patterns"]["Classiq"]["patterns"],
                "found_in_pennylane": pattern in stats["framework_patterns"]["PennyLane"]["patterns"],
                "found_in_qiskit": pattern in stats["framework_patterns"]["Qiskit"]["patterns"],
                "found_in_targets": pattern in stats["target_patterns"]["patterns"],
                "framework_count": sum([
                    pattern in stats["framework_patterns"]["Classiq"]["patterns"],
                    pattern in stats["framework_patterns"]["PennyLane"]["patterns"],
                    pattern in stats["framework_patterns"]["Qiskit"]["patterns"]
                ])
            }
            analysis_data.append(row)
        
        # Create DataFrame and export
        df = pd.DataFrame(analysis_data)
        df.to_csv(output_file, index=False)
        
        print(f"Pattern analysis exported to {output_file}")


# === Convenience Functions ===

def analyze_pattern_coverage():
    """Convenience function for pattern coverage analysis."""
    analyzer = PatternAnalyzer()
    return analyzer.analyze_pattern_coverage()


def get_pattern_statistics():
    """Convenience function for pattern statistics."""
    analyzer = PatternAnalyzer()
    return analyzer.get_pattern_statistics()


def export_pattern_analysis(output_file: Path):
    """Convenience function for pattern analysis export."""
    analyzer = PatternAnalyzer()
    analyzer.export_pattern_analysis(output_file)


def main():
    """Main function for pattern analysis."""
    analyzer = PatternAnalyzer()
    analyzer.analyze_pattern_coverage()


if __name__ == "__main__":
    main()


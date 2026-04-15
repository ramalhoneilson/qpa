"""Multilanguage semantic analysis — C++ edition.

Mirrors src/analysis/run_analysis.py but scans C++ source files from
target_multilanguage_projects/ instead of converted Python notebooks.

Uses tree-sitter-cpp for call extraction and comment parsing.
The same concept knowledge base, embedding model, and similarity thresholds
as the Python analysis are used so results are directly comparable.

Usage:
    python -m src.multilanguage_analysis.run_analysis_cpp
    python -m src.multilanguage_analysis.run_analysis_cpp --target-dir path/to/cpp/files
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path

from scipy.spatial.distance import cdist
from sentence_transformers import SentenceTransformer

from src.conf import config
from src.multilanguage_analysis.parsers.cpp_parser import CppParser

# ---------------------------------------------------------------------------
# Scan targets — example/demo directories per C++ quantum framework.
# PECOS is intentionally excluded (error-correction only, not general QC).
# qulacs is excluded (C++ files are all internal implementation, not examples).
# ---------------------------------------------------------------------------
MULTILANG_ROOT = config.PROJECT_ROOT / "target_multilanguage_projects"

CPP_SCAN_DIRS: list[Path] = [
    MULTILANG_ROOT / "cuda-quantum" / "docs" / "sphinx" / "examples" / "cpp",
    MULTILANG_ROOT / "qpp" / "examples",
    MULTILANG_ROOT / "intel-qs" / "examples",
    MULTILANG_ROOT / "qrack" / "examples",
]

OUTPUT_DIR = config.PROJECT_ROOT / "data" / "multilanguage_analysis"
OUTPUT_CSV_FILE = OUTPUT_DIR / "cpp_concept_matches.csv"
UNCLASSIFIED_CONCEPTS_FILE = OUTPUT_DIR / "cpp_unclassified_concepts.csv"

# Re-use the Python analysis knowledge base — same concepts, same patterns.
CONCEPT_FILES = [
    config.RESULTS_DIR / "classiq_quantum_concepts.json",
    config.RESULTS_DIR / "pennylane_quantum_concepts.json",
    config.RESULTS_DIR / "qiskit_quantum_concepts.json",
    config.RESULTS_DIR / "qiskit_algorithms_quantum_concepts.json",
]

PATTERN_FILES = [
    config.RESULTS_DIR / "knowledge_base/enriched_classiq_quantum_patterns.csv",
    config.RESULTS_DIR / "knowledge_base/enriched_pennylane_quantum_patterns.csv",
    config.RESULTS_DIR / "knowledge_base/enriched_qiskit_quantum_patterns.csv",
    config.RESULTS_DIR
    / "knowledge_base/enriched_qiskit_algorithms_quantum_patterns.csv",
]

SIMILARITY_THRESHOLDS = {"name": 0.90, "summary": 0.65}


# ---------------------------------------------------------------------------
# Identifier normalisation (mirrors src/analysis/run_analysis.py)
# ---------------------------------------------------------------------------

def normalize_identifier(name: str) -> str:
    """Split snake_case and CamelCase into lowercase tokens before encoding."""
    name = name.replace("_", " ")
    name = re.sub(r"([a-z])([A-Z])", r"\1 \2", name)
    name = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", name)
    return name.lower().strip()


# ---------------------------------------------------------------------------
# Knowledge base helpers (same logic as run_analysis.py)
# ---------------------------------------------------------------------------

def extract_short_name(full_name: str) -> str:
    if not full_name:
        return ""
    return full_name.split("/")[-1].split(".")[-1]


def load_patterns_map(file_paths: list[Path]) -> dict[str, str]:
    pattern_map: dict[str, str] = {}
    for path in file_paths:
        if not path.exists():
            print(f"Warning: Pattern file not found: {path}")
            continue
        try:
            with open(path, encoding="utf-8") as f:
                reader = csv.reader(f, delimiter=",")
                next(reader)
                for row in reader:
                    if len(row) >= 3:
                        concept_name = row[0].strip()
                        pattern = row[2].strip()
                        if concept_name and pattern:
                            pattern_map[concept_name] = pattern
        except Exception as e:
            print(f"Error loading patterns from {path}: {e}")
    return pattern_map


def load_quantum_concepts(
    file_paths: list[Path], pattern_map: dict[str, str]
) -> list[dict]:
    concepts: list[dict] = []
    pattern_map_by_short_name = {
        extract_short_name(k): v for k, v in pattern_map.items()
    }
    for path in file_paths:
        if not path.exists():
            continue
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
                for item in data:
                    if "name" in item and "summary" in item:
                        full_name = item["name"]
                        short_name = extract_short_name(full_name)
                        found_pattern = "N/A"
                        if full_name in pattern_map:
                            found_pattern = pattern_map[full_name]
                        elif short_name in pattern_map_by_short_name:
                            found_pattern = pattern_map_by_short_name[short_name]
                        else:
                            for csv_key, pattern_value in pattern_map.items():
                                if full_name.endswith(csv_key):
                                    found_pattern = pattern_value
                                    break
                        concepts.append(
                            {
                                "name": full_name,
                                "summary": item["summary"],
                                "short_name": short_name,
                                "pattern": found_pattern,
                            }
                        )
        except Exception as e:
            print(f"Error loading {path}: {e}")
    return concepts


def _save_unclassified_concepts(concepts: list[dict], output_path: Path):
    unclassified = [
        {"name": c["name"], "summary": c["summary"]}
        for c in concepts
        if c["pattern"] == "N/A"
    ]
    if not unclassified:
        if output_path.exists():
            output_path.unlink()
        return
    print(f"WARNING: {len(unclassified)} unclassified concepts → '{output_path}'")
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["name", "summary"])
            writer.writeheader()
            writer.writerows(unclassified)
    except OSError as e:
        print(f"Error writing unclassified concepts file: {e}")


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def main(scan_dirs: list[Path] | None = None, output_file: Path | None = None):
    targets = scan_dirs if scan_dirs else CPP_SCAN_DIRS
    out_csv = output_file if output_file else OUTPUT_CSV_FILE

    # Validate scan dirs
    valid_targets = [d for d in targets if d.exists()]
    missing = [d for d in targets if not d.exists()]
    if missing:
        for d in missing:
            print(f"Warning: Scan directory not found, skipping: {d}", file=sys.stderr)
    if not valid_targets:
        print("Error: No valid scan directories found. Exiting.", file=sys.stderr)
        return

    print(f"\nLoading patterns from {len(PATTERN_FILES)} CSV files...")
    pattern_map = load_patterns_map(PATTERN_FILES)
    print(f"Loaded {len(pattern_map)} concept-to-pattern mappings.")

    quantum_concepts = load_quantum_concepts(CONCEPT_FILES, pattern_map)
    if not quantum_concepts:
        print("No quantum concepts loaded. Exiting.")
        return
    print(f"Loaded {len(quantum_concepts)} concepts across {len(CONCEPT_FILES)} files.")

    _save_unclassified_concepts(quantum_concepts, UNCLASSIFIED_CONCEPTS_FILE)

    found_patterns_count = sum(1 for c in quantum_concepts if c["pattern"] != "N/A")
    print(f"\n--- MAPPING SUMMARY ---")
    print(
        f"Matched {found_patterns_count} / {len(quantum_concepts)} concepts with a pattern."
    )
    print("-----------------------\n")

    print(f"Loading embedding model '{config.EMBEDDING_MODEL_NAME}'...")
    model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)

    concept_short_names = [c["short_name"] for c in quantum_concepts]
    concept_summaries = [c["summary"] for c in quantum_concepts]
    concept_name_embeddings = model.encode(
        [normalize_identifier(n) for n in concept_short_names], convert_to_tensor=True
    )
    concept_summary_embeddings = model.encode(
        concept_summaries, convert_to_tensor=True
    )

    parser = CppParser()
    extensions = tuple(parser.file_extensions)

    # Collect all C++ files from all scan directories
    cpp_files: list[Path] = []
    for scan_dir in valid_targets:
        found = [f for f in scan_dir.rglob("*") if f.suffix in extensions]
        print(f"  {scan_dir.relative_to(config.PROJECT_ROOT)}: {len(found)} files")
        cpp_files.extend(found)

    total_files = len(cpp_files)
    print(f"\nFound {total_files} C++ files to analyse across {len(valid_targets)} directories.")

    # Key: (relative_file_path, concept_name) → best-scoring row for that pair
    best_matches: dict[tuple[str, str], dict] = {}

    for i, file_path in enumerate(cpp_files):
        if (i + 1) % 50 == 0 or (i + 1) == total_files:
            print(f"Processing {i + 1}/{total_files}...", flush=True)

        try:
            source = file_path.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            print(f"Could not read {file_path}: {e}", file=sys.stderr)
            continue

        # Use path relative to MULTILANG_ROOT so it's self-describing in output
        try:
            relative_path = str(file_path.relative_to(MULTILANG_ROOT))
        except ValueError:
            relative_path = str(file_path)

        # ── Name-based matching ──────────────────────────────────────────────
        located_elements = parser.extract_calls(source)
        if located_elements:
            element_names = [e[0] for e in located_elements]
            code_element_embeddings = model.encode(
                [normalize_identifier(n) for n in element_names],
                convert_to_tensor=True,
            )
            cosine_sim_names = 1 - cdist(
                code_element_embeddings.cpu(),
                concept_name_embeddings.cpu(),
                "cosine",
            )
            for elem_idx, (element, lineno) in enumerate(located_elements):
                for concept_idx, concept in enumerate(quantum_concepts):
                    score = float(cosine_sim_names[elem_idx, concept_idx])
                    if score >= SIMILARITY_THRESHOLDS["name"]:
                        key = (relative_path, concept["name"])
                        if key not in best_matches or score > best_matches[key]["score"]:
                            best_matches[key] = {
                                "file_path": relative_path,
                                "concept_name": concept["name"],
                                "pattern": concept["pattern"],
                                "match_type": "name",
                                "matched_text": element,
                                "score": score,
                                "first_line": lineno,
                            }

        # ── Summary-based matching (comment blocks) ──────────────────────────
        comment_block = parser.extract_comments(source)
        if comment_block:
            comment_embedding = model.encode([comment_block], convert_to_tensor=True)
            cosine_sim_summaries = 1 - cdist(
                comment_embedding.cpu(), concept_summary_embeddings.cpu(), "cosine"
            )
            truncated_comment = (
                (comment_block[:150] + "...")
                if len(comment_block) > 150
                else comment_block
            )
            for concept_idx, concept in enumerate(quantum_concepts):
                score = float(cosine_sim_summaries[0, concept_idx])
                if score >= SIMILARITY_THRESHOLDS["summary"]:
                    key = (relative_path, concept["name"])
                    if key not in best_matches or score > best_matches[key]["score"]:
                        best_matches[key] = {
                            "file_path": relative_path,
                            "concept_name": concept["name"],
                            "pattern": concept["pattern"],
                            "match_type": "summary",
                            "matched_text": truncated_comment.replace(";", ","),
                            "score": score,
                            "first_line": 0,
                        }

    # Sort by file then position within file
    sorted_rows = sorted(
        best_matches.values(), key=lambda r: (r["file_path"], r["first_line"])
    )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(
            [
                "file_path",
                "concept_name",
                "pattern",
                "match_type",
                "matched_text",
                "similarity_score",
                "first_line",
            ]
        )
        for row in sorted_rows:
            writer.writerow(
                [
                    row["file_path"],
                    row["concept_name"],
                    row["pattern"],
                    row["match_type"],
                    row["matched_text"],
                    f"{row['score']:.4f}",
                    row["first_line"],
                ]
            )

    print(
        f"\nAnalysis complete. Results saved to '{out_csv}' "
        f"({len(best_matches)} unique file–concept pairs across {total_files} files)."
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run quantum concept semantic analysis on C++ source files."
    )
    parser.add_argument(
        "--target-dir",
        type=str,
        action="append",
        default=None,
        help=(
            "Directory of C++ files to analyse (repeatable). "
            "Defaults to the configured per-repo example directories."
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV path. Defaults to data/multilanguage_analysis/cpp_concept_matches.csv.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    dirs = [Path(d).resolve() for d in args.target_dir] if args.target_dir else None
    out = Path(args.output).resolve() if args.output else None
    main(scan_dirs=dirs, output_file=out)

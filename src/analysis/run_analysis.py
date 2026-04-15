import argparse
import ast
import csv
import json
import re
from pathlib import Path

from scipy.spatial.distance import cdist
from sentence_transformers import SentenceTransformer

from src.conf import config

NOTEBOOKS_ROOT_DIR = config.PROJECT_ROOT / "converted_notebooks"
OUTPUT_CSV_FILE = config.RESULTS_DIR / "quantum_concept_matches_with_patterns.csv"
UNCLASSIFIED_CONCEPTS_FILE = config.RESULTS_DIR / "unclassified_concepts.csv"

SIMILARITY_THRESHOLDS = {"name": 0.90, "summary": 0.65}

CONCEPT_FILES = [
    config.RESULTS_DIR / "classiq_quantum_concepts.json",
    config.RESULTS_DIR / "pennylane_quantum_concepts.json",
    config.RESULTS_DIR / "qiskit_quantum_concepts.json",
    config.RESULTS_DIR / "qiskit_algorithms_quantum_concepts.json",
    config.RESULTS_DIR / "qiskit_machine_learning_quantum_concepts.json",
]

PATTERN_FILES = [
    config.RESULTS_DIR / "knowledge_base/enriched_classiq_quantum_patterns.csv",
    config.RESULTS_DIR / "knowledge_base/enriched_pennylane_quantum_patterns.csv",
    config.RESULTS_DIR / "knowledge_base/enriched_qiskit_quantum_patterns.csv",
    config.RESULTS_DIR / "knowledge_base/enriched_qiskit_algorithms_quantum_patterns.csv",
    config.RESULTS_DIR / "knowledge_base/enriched_qiskit_machine_learning_quantum_patterns.csv",
]


def normalize_identifier(name: str) -> str:
    """Normalise a code identifier for embedding.

    Splits snake_case and CamelCase into space-separated lowercase tokens so
    that e.g. ``ApplyQuantumFourierTransform`` and ``apply_qft`` land close to
    ``QFT`` in embedding space.  Applied only to the strings passed to
    ``model.encode()``; original names are preserved in the output CSV.
    """
    # snake_case → tokens
    name = name.replace("_", " ")
    # CamelCase / PascalCase → tokens (insert space before each uppercase run)
    name = re.sub(r"([a-z])([A-Z])", r"\1 \2", name)
    name = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", name)
    return name.lower().strip()


class CodeElementVisitor(ast.NodeVisitor):
    def __init__(self):
        self.found_elements = set()

    def visit_Call(self, node):
        func = node.func
        if isinstance(func, ast.Name):
            self.found_elements.add(func.id)
        elif isinstance(func, ast.Attribute):
            self.found_elements.add(func.attr)
        self.generic_visit(node)


def get_code_elements_from_script(script_content: str) -> list[str]:
    try:
        tree = ast.parse(script_content)
        visitor = CodeElementVisitor()
        visitor.visit(tree)
        return list(visitor.found_elements)
    except SyntaxError:
        return []


def extract_comments_from_script(file_path: Path) -> str:
    """Concatenate all ``#``-comment lines in the script into a single string.

    Used for whole-file summary matching: the resulting text is embedded once
    and compared against all KB concept summaries.
    """
    comments = []
    try:
        with open(file_path, encoding="utf-8", errors="ignore") as f:
            for line in f:
                stripped_line = line.strip()
                if stripped_line.startswith("#"):
                    comment_text = stripped_line.lstrip("# ").strip()
                    if comment_text:
                        comments.append(comment_text)
    except Exception as e:
        print(f"Error reading {file_path} for comments: {e}")
    return " ".join(comments)


def extract_short_name(full_name: str) -> str:
    if not full_name:
        return ""
    return full_name.split("/")[-1].split(".")[-1]


def load_patterns_map(file_paths: list[Path]) -> dict[str, str]:
    pattern_map = {}
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
    concepts = []
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
        print("All concepts are classified. No 'unclassified_concepts.csv' needed.")
        return

    print(
        f"\nWARNING: Found {len(unclassified)} unclassified concepts. Saving to-do list to '{output_path}'..."
    )
    try:
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["name", "summary"])
            writer.writeheader()
            writer.writerows(unclassified)
    except OSError as e:
        print(f"  - Error writing unclassified concepts file: {e}")


def extract_and_save_unique_patterns(input_files: list[Path], output_file: Path):
    unique_patterns = set()
    for path in input_files:
        if not path.exists():
            print(
                f"Warning: Pattern file not found, skipping for unique pattern extraction: {path}"
            )
            continue
        try:
            with open(path, encoding="utf-8") as f:
                reader = csv.reader(f, delimiter=",")
                next(reader)
                for row in reader:
                    if len(row) >= 3:
                        pattern = row[2].strip()
                        if pattern:
                            unique_patterns.add(pattern)
        except Exception as e:
            print(f"Error reading patterns from {path}: {e}")

    try:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["PatternName"])
            for pattern in sorted(list(unique_patterns)):
                writer.writerow([pattern])
        print(f"Saved {len(unique_patterns)} unique patterns to '{output_file}'")
    except Exception as e:
        print(f"Error saving unique patterns to {output_file}: {e}")


def main(
    target_dir: Path | None = None,
    output_file: Path | None = None,
):
    scan_dir = target_dir if target_dir else NOTEBOOKS_ROOT_DIR
    out_csv = output_file if output_file else OUTPUT_CSV_FILE

    OUTPUT_PATTERN_FILE = config.RESULTS_DIR / "patterns_used_in_categorization.csv"
    extract_and_save_unique_patterns(PATTERN_FILES, OUTPUT_PATTERN_FILE)

    print(f"\nLoading patterns from {len(PATTERN_FILES)} CSV files...")
    pattern_map = load_patterns_map(PATTERN_FILES)
    print(f"Loaded a total of {len(pattern_map)} concept-to-pattern mappings.")

    quantum_concepts = load_quantum_concepts(CONCEPT_FILES, pattern_map)
    if not quantum_concepts:
        print("No quantum concepts loaded. Exiting.")
        return
    print(
        f"Loaded {len(quantum_concepts)} concepts defined across {len(CONCEPT_FILES)} files."
    )

    _save_unclassified_concepts(quantum_concepts, UNCLASSIFIED_CONCEPTS_FILE)

    found_patterns_count = sum(1 for c in quantum_concepts if c["pattern"] != "N/A")
    print("\n--- MAPPING SUMMARY ---")
    print(
        f"Successfully matched {found_patterns_count} / {len(quantum_concepts)} concepts with a pattern."
    )
    print("-----------------------\n")

    print(f"Scanning directory: '{scan_dir}'")
    print(f"Output file: '{out_csv}'")

    print(f"Loading embedding model '{config.EMBEDDING_MODEL_NAME}'...")
    model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)

    concept_short_names = [c["short_name"] for c in quantum_concepts]
    concept_summaries = [c["summary"] for c in quantum_concepts]
    concept_name_embeddings = model.encode(
        [normalize_identifier(n) for n in concept_short_names], convert_to_tensor=True
    )
    concept_summary_embeddings = model.encode(concept_summaries, convert_to_tensor=True)

    script_files = list(scan_dir.rglob("*.py"))
    total_files = len(script_files)
    print(f"Found {total_files} Python files to analyze.")

    # For each file, keep the best-scoring match per concept.
    # The `first_line` field records where in the file the winning match
    # appeared so that the output can later be sorted into call order —
    # useful for building sequence graphs.
    # Key: (relative_file_path, concept_name)
    # Value: dict with the highest-score row seen for that pair.
    best_matches: dict[tuple[str, str], dict] = {}

    for i, file_path in enumerate(script_files):
        if (i + 1) % 100 == 0 or (i + 1) == total_files:
            print(f"Processing {i + 1}/{total_files}...")

        try:
            script_content = file_path.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            print(f"Could not read file {file_path}: {e}")
            continue

        relative_path = str(file_path.relative_to(scan_dir))

        # ── Name-based matching ──────────────────────────────────────────────
        # Walk the AST to collect (element_name, first_line) pairs so we know
        # where in the file each call site appears.
        try:
            tree = ast.parse(script_content)
        except SyntaxError:
            tree = None

        located_elements: list[tuple[str, int]] = []
        if tree:
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    func = node.func
                    name = None
                    if isinstance(func, ast.Name):
                        name = func.id
                    elif isinstance(func, ast.Attribute):
                        name = func.attr
                    if name:
                        located_elements.append((name, node.lineno))

        if located_elements:
            element_names = [e[0] for e in located_elements]
            code_element_embeddings = model.encode(
                [normalize_identifier(n) for n in element_names], convert_to_tensor=True
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

        # ── Summary-based matching (whole-file comment block) ────────────────
        # All # -comment lines are concatenated into one string and embedded
        # as a single vector, then compared against all KB concept summaries.
        # This captures the dominant topic of the file (notebook title, section
        # headers, algorithm descriptions) and works well when the KB covers
        # the target vocabulary.  Comments are assigned line 0 so they sort
        # before call-site matches in sequence output.
        comment_block = extract_comments_from_script(file_path)
        if comment_block:
            comment_embedding = model.encode(
                [comment_block], convert_to_tensor=True
            )
            cosine_sim_summaries = 1 - cdist(
                comment_embedding.cpu(), concept_summary_embeddings.cpu(), "cosine"
            )
            truncated_comment = (
                (comment_block[:150] + "...") if len(comment_block) > 150 else comment_block
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

    # Sort by file then by position within the file so that the CSV naturally
    # reflects the order in which concepts appear — ready for sequence analysis.
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
        f"Analysis complete. Results saved to '{out_csv}' "
        f"({len(best_matches)} unique file–concept pairs across {total_files} files)."
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run quantum concept semantic analysis on Python files."
    )
    parser.add_argument(
        "--target-dir",
        type=str,
        default=None,
        help=(
            "Path to the directory of Python files to analyze. "
            "Defaults to 'converted_notebooks/' if not specified."
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Path for the output CSV file. "
            "Defaults to 'data/quantum_concept_matches_with_patterns.csv'."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    target = Path(args.target_dir).resolve() if args.target_dir else None
    output = Path(args.output).resolve() if args.output else None
    main(target_dir=target, output_file=output)

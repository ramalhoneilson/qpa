import ast
import csv
import json
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
]

PATTERN_FILES = [
    config.RESULTS_DIR / "knowledge_base/enriched_classiq_quantum_patterns.csv",
    config.RESULTS_DIR / "knowledge_base/enriched_pennylane_quantum_patterns.csv",
    config.RESULTS_DIR / "knowledge_base/enriched_qiskit_quantum_patterns.csv",
]


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


def main():
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

    print(f"Loading embedding model '{config.EMBEDDING_MODEL_NAME}'...")
    model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)

    concept_short_names = [c["short_name"] for c in quantum_concepts]
    concept_summaries = [c["summary"] for c in quantum_concepts]
    concept_name_embeddings = model.encode(concept_short_names, convert_to_tensor=True)
    concept_summary_embeddings = model.encode(concept_summaries, convert_to_tensor=True)

    with open(OUTPUT_CSV_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(
            [
                "file_path",
                "concept_name",
                "pattern",
                "match_type",
                "matched_text",
                "similarity_score",
            ]
        )

        script_files = list(NOTEBOOKS_ROOT_DIR.rglob("*.py"))
        total_files = len(script_files)
        print(f"Found {total_files} Python files to analyze.")

        for i, file_path in enumerate(script_files):
            if (i + 1) % 100 == 0 or (i + 1) == total_files:
                print(f"Processing {i + 1}/{total_files}...")

            try:
                script_content = file_path.read_text(encoding="utf-8", errors="ignore")
            except Exception as e:
                print(f"Could not read file {file_path}: {e}")
                continue

            code_elements = get_code_elements_from_script(script_content)
            if code_elements:
                code_element_embeddings = model.encode(
                    code_elements, convert_to_tensor=True
                )
                cosine_sim_names = 1 - cdist(
                    code_element_embeddings.cpu(),
                    concept_name_embeddings.cpu(),
                    "cosine",
                )
                for elem_idx, element in enumerate(code_elements):
                    for concept_idx, concept in enumerate(quantum_concepts):
                        score = cosine_sim_names[elem_idx, concept_idx]
                        if score >= SIMILARITY_THRESHOLDS["name"]:
                            writer.writerow(
                                [
                                    str(file_path.relative_to(NOTEBOOKS_ROOT_DIR)),
                                    concept["name"],
                                    concept["pattern"],
                                    "name",
                                    element,
                                    f"{score:.4f}",
                                ]
                            )

            comment_block = extract_comments_from_script(file_path)
            if comment_block:
                comment_embedding = model.encode(
                    [comment_block], convert_to_tensor=True
                )
                cosine_sim_summaries = 1 - cdist(
                    comment_embedding.cpu(), concept_summary_embeddings.cpu(), "cosine"
                )
                for concept_idx, concept in enumerate(quantum_concepts):
                    score = cosine_sim_summaries[0, concept_idx]
                    if score >= SIMILARITY_THRESHOLDS["summary"]:
                        truncated_comment = (
                            (comment_block[:150] + "...")
                            if len(comment_block) > 150
                            else comment_block
                        )
                        writer.writerow(
                            [
                                str(file_path.relative_to(NOTEBOOKS_ROOT_DIR)),
                                concept["name"],
                                concept["pattern"],
                                "summary",
                                truncated_comment.replace(";", ","),
                                f"{score:.4f}",
                            ]
                        )

    print(f"Analysis complete. Results saved to '{OUTPUT_CSV_FILE}'.")


if __name__ == "__main__":
    main()

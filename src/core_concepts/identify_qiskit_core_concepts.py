import ast
import csv
import json
import re
from pathlib import Path
from typing import Any

from sentence_transformers import SentenceTransformer, util

from src.conf import config

PROJECT_ROOT = config.PROJECT_ROOT
QISKIT_PROJECT_ROOT = PROJECT_ROOT / "target_github_projects" / "qiskit"

SOURCE_SNIPPETS_DIR = config.RESULTS_DIR / "qiskit_source_snippets"
OUTPUT_JSON_PATH = config.RESULTS_DIR / "qiskit_quantum_concepts.json"
OUTPUT_CSV_PATH = config.RESULTS_DIR / "qiskit_quantum_concepts.csv"

EMBEDDING_MODEL_NAME = "all-mpnet-base-v2"
SEARCH_SUBDIRS = ["qiskit/circuit/library/"]
TARGET_BASE_CLASSES = ["QuantumCircuit", "Gate"]
SIMILARITY_THRESHOLD = 0.95

EXCLUDE_SUBDIRS = {"standard_gates", "templates"}

# --- Core Functions ---


def clean_qiskit_docstring(docstring: str) -> str:
    pattern = r"(.. (code-block|parsed-literal):: text\n\n)(^\s+.*$\n?)+"
    cleaned = re.sub(pattern, "", docstring, flags=re.MULTILINE)
    circuit_symbol_pattern = r".*Circuit symbol:.*(?:\n\s*.. code-block:: text)?\n\n(^\s*.*[┌┐└┘├┤│─].*$\n?)+"
    cleaned = re.sub(circuit_symbol_pattern, "", cleaned, flags=re.MULTILINE)
    math_pattern = r"(.. math::\n\n)(^\s+.*$\n?)+"
    cleaned = re.sub(math_pattern, "", cleaned, flags=re.MULTILINE)
    return cleaned.strip()


class _QiskitConceptVisitor(ast.NodeVisitor):
    def __init__(self, source_text: str, file_path: Path, sdk_root: Path):
        self.found_concepts: dict[str, dict[str, Any]] = {}
        self.source_text = source_text
        self.file_path = file_path
        self.sdk_root = sdk_root
        self.context_stack: list[ast.AST] = []

    def _get_module_path_str(self) -> str:
        relative_path = self.file_path.relative_to(self.sdk_root)
        module_path_parts = list(relative_path.parts)
        module_path_parts[-1] = relative_path.stem
        return ".".join(module_path_parts)

    def _visit_context_node(self, node: ast.AST):
        self.context_stack.append(node)
        self.generic_visit(node)
        self.context_stack.pop()

    def visit_ClassDef(self, node: ast.ClassDef):
        raw_docstring = ast.get_docstring(node)
        if raw_docstring and (docstring := clean_qiskit_docstring(raw_docstring)):
            module_path_str = self._get_module_path_str()
            full_concept_name = f"/qiskit/{module_path_str}.{node.name}"
            if full_concept_name not in self.found_concepts:
                base_names = [b.id for b in node.bases if isinstance(b, ast.Name)]
                self.found_concepts[full_concept_name] = {
                    "name": full_concept_name,
                    "summary": docstring.strip()
                    .split("\n\n")[0]
                    .strip()
                    .replace("\n", " "),
                    "docstring": docstring.strip(),
                    "source_code": ast.get_source_segment(self.source_text, node),
                    "type": "Class",
                    "is_target_subclass": any(
                        base in TARGET_BASE_CLASSES for base in base_names
                    ),
                    "base_classes": base_names,
                }
        self._visit_context_node(node)

    def visit_FunctionDef(self, node: ast.FunctionDef):
        if not any(isinstance(p, ast.ClassDef) for p in self.context_stack):
            raw_docstring = ast.get_docstring(node)
            if (
                raw_docstring
                and not node.name.startswith("_")
                and not node.name.startswith("get_")
                and (docstring := clean_qiskit_docstring(raw_docstring))
            ):
                module_path_str = self._get_module_path_str()
                full_concept_name = f"/qiskit/{module_path_str}.{node.name}"
                if full_concept_name not in self.found_concepts:
                    self.found_concepts[full_concept_name] = {
                        "name": full_concept_name,
                        "summary": docstring.strip()
                        .split("\n\n")[0]
                        .strip()
                        .replace("\n", " "),
                        "docstring": docstring.strip(),
                        "source_code": ast.get_source_segment(self.source_text, node),
                        "type": "Function",
                    }
        self._visit_context_node(node)


def _find_concepts_in_file(py_path: Path, sdk_root: Path) -> list:
    try:
        source_text = py_path.read_text(encoding="utf-8")
        if len(source_text.strip()) < 50:
            return []
        tree = ast.parse(source_text, filename=str(py_path))
        visitor = _QiskitConceptVisitor(source_text, py_path, sdk_root)
        visitor.visit(tree)
        return list(visitor.found_concepts.values())
    except Exception as e:
        print(f"  - Warning: Could not parse file {py_path.name}: {e}")
        return []


def extract_qiskit_concepts() -> list[dict[str, Any]]:
    if not QISKIT_PROJECT_ROOT.is_dir():
        print(
            f"Error: Qiskit project root not found at '{QISKIT_PROJECT_ROOT.resolve()}'"
        )
        return []
    print(f"Scanning Qiskit project at: {QISKIT_PROJECT_ROOT.resolve()}")
    search_path = QISKIT_PROJECT_ROOT / SEARCH_SUBDIRS[0]
    if not search_path.is_dir():
        print(f"  - Warning: SDK subdirectory not found, skipping: {search_path}")
        return []
    all_py_files = [
        py_file
        for py_file in search_path.rglob("*.py")
        if not any(
            part in EXCLUDE_SUBDIRS for part in py_file.relative_to(search_path).parts
        )
    ]
    if not all_py_files:
        print("  - No Python files found in the Qiskit search directories.")
        return []
    all_concepts_data = {}
    print(f"\nProcessing {len(all_py_files)} total Python files from the SDK...")
    for py_file in sorted(all_py_files):
        if py_file.name in ("__init__.py",) or py_file.name.startswith("test_"):
            continue
        for concept in _find_concepts_in_file(py_file, QISKIT_PROJECT_ROOT):
            if concept["name"] not in all_concepts_data:
                all_concepts_data[concept["name"]] = concept
    return list(all_concepts_data.values())


def _to_snake_case(name: str) -> str:
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


def deduplicate_by_naming_convention(
    concepts_data: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    print("\n--- Deduplicating concepts using naming conventions ---")
    concepts_by_module: dict[str, list[dict[str, Any]]] = {}
    for concept in concepts_data:
        module_path = ".".join(concept["name"].split(".")[:-1]).replace("/qiskit/", "")
        if module_path not in concepts_by_module:
            concepts_by_module[module_path] = []
        concepts_by_module[module_path].append(concept)
    discard_full_names = set()
    removed_concepts = []
    for module, concepts_in_module in concepts_by_module.items():
        classes = {
            c["name"].split(".")[-1]: c
            for c in concepts_in_module
            if c["type"] == "Class"
        }
        functions = {
            f["name"].split(".")[-1]: f
            for f in concepts_in_module
            if f["type"] == "Function"
        }
        if not classes or not functions:
            continue
        for class_name, class_concept in classes.items():
            expected_func_name = _to_snake_case(class_name)
            if expected_func_name in functions:
                func_to_discard = functions[expected_func_name]
                discard_full_names.add(func_to_discard["name"])
                removed_concepts.append(
                    {
                        "removed": func_to_discard["name"],
                        "reason": f"Wrapper for class {class_concept['name']}",
                        "summary": func_to_discard["summary"],
                    }
                )
    if discard_full_names:
        print(
            f"\nIdentified and marked {len(discard_full_names)} functions that are wrappers for classes."
        )
        print("\n=== REMOVED BY NAMING CONVENTION ===")
        for item in removed_concepts:
            print(f"\n  Removed: {item['removed']}")
            print(f"  Reason:  {item['reason']}")
            print(f"  Summary: {item['summary'][:100]}...")
    final_concepts = [c for c in concepts_data if c["name"] not in discard_full_names]
    print(f"\nReduced concepts from {len(concepts_data)} to {len(final_concepts)}.")
    return final_concepts


def _is_deprecated(concept: dict[str, Any]) -> bool:
    source = concept.get("source_code", "")
    docstring = concept.get("docstring", "")
    return "@deprecate" in source or "deprecated" in docstring.lower()


def deduplicate_concepts_semantic(
    concepts_data: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    if not concepts_data:
        return []
    print("\n--- Deduplicating concepts using semantic similarity ---")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    print("Generating embeddings for concept summaries...")
    embeddings = model.encode(
        [c["summary"] for c in concepts_data],
        convert_to_tensor=True,
        show_progress_bar=True,
    )
    clusters = []
    processed_indices = set()
    for i in range(len(concepts_data)):
        if i in processed_indices:
            continue
        new_cluster_indices = [i]
        processed_indices.add(i)
        similarities = util.cos_sim(embeddings[i], embeddings)[0]
        for j in range(i + 1, len(concepts_data)):
            if j not in processed_indices and similarities[j] > SIMILARITY_THRESHOLD:
                new_cluster_indices.append(j)
                processed_indices.add(j)
        clusters.append([concepts_data[k] for k in new_cluster_indices])
    final_concepts = []
    removed_by_semantic = []
    for cluster in clusters:
        if len(cluster) == 1:
            final_concepts.append(cluster[0])
            continue
        best_concept = max(
            cluster,
            key=lambda c: (
                not _is_deprecated(c),
                c.get("is_target_subclass", False),
                c["type"] == "Class",
                "Gate" in c.get("base_classes", []),
                len(c["docstring"]),
            ),
        )
        final_concepts.append(best_concept)
        for concept in cluster:
            if concept["name"] != best_concept["name"]:
                removed_by_semantic.append(
                    {
                        "removed": concept["name"],
                        "kept": best_concept["name"],
                        "similarity": "cluster",
                        "removed_summary": concept["summary"],
                        "kept_summary": best_concept["summary"],
                    }
                )
    if removed_by_semantic:
        print(
            f"\n=== REMOVED BY SEMANTIC SIMILARITY (threshold: {SIMILARITY_THRESHOLD}) ==="
        )
        for item in removed_by_semantic:
            print(f"\n  Removed: {item['removed']}")
            print(f"  Kept:    {item['kept']}")
            print(f"  Removed summary: {item['removed_summary'][:100]}...")
            print(f"  Kept summary:    {item['kept_summary'][:100]}...")
    print(
        f"\nDeduplication complete. Reduced {len(concepts_data)} concepts to {len(final_concepts)}."
    )
    return final_concepts


def _save_source_code_snippets(concepts_data: list[dict[str, Any]]):
    if not concepts_data:
        return
    SOURCE_SNIPPETS_DIR.mkdir(parents=True, exist_ok=True)
    print(
        f"\nSaving {len(concepts_data)} source files to: {SOURCE_SNIPPETS_DIR.resolve()}"
    )
    count = 0
    for concept in concepts_data:
        if source_code := concept.get("source_code"):
            sanitized_name = (
                concept["name"].replace("/", "_").replace(".", "_").lstrip("_")
            )
            try:
                (SOURCE_SNIPPETS_DIR / f"{sanitized_name}.py").write_text(
                    source_code, encoding="utf-8"
                )
                count += 1
            except Exception as e:
                print(
                    f"  - Warning: Could not write source file for '{concept['name']}': {e}"
                )
    print(f"Successfully saved {count} source code files.")


def save_concepts_to_json(concepts_data: list[dict[str, Any]]):
    try:
        json_data = [
            {k: v for k, v in item.items() if k != "source_code"}
            for item in concepts_data
        ]
        OUTPUT_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2)
        print(f"Debug dataset saved successfully to: '{OUTPUT_JSON_PATH}'")
    except Exception as e:
        print(f"Error: Could not save the debug dataset to JSON. {e}")


def save_concepts_to_csv(concepts_data: list[dict[str, Any]]):
    try:
        OUTPUT_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT_CSV_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter=";")
            writer.writerow(["name", "summary"])
            for concept in concepts_data:
                writer.writerow([concept.get("name", ""), concept.get("summary", "")])
        print(f"Debug dataset saved successfully to: '{OUTPUT_CSV_PATH}'")
    except Exception as e:
        print(f"Error: Could not save the debug dataset to CSV. {e}")


def main():
    print(" Starting Qiskit Core Quantum Concepts Generation and Storage ")
    raw_data = extract_qiskit_concepts()
    print(f"\n>>> Step 1: Extracted {len(raw_data)} raw concepts.")
    convention_filtered_data = deduplicate_by_naming_convention(raw_data)
    print(
        f"\n>>> Step 2: After naming convention filter, {len(convention_filtered_data)} concepts remain."
    )
    final_data = deduplicate_concepts_semantic(convention_filtered_data)
    print(
        f"\n>>> Step 3: After semantic filter, {len(final_data)} final concepts remain."
    )
    if not final_data:
        print("\nNo quantum concepts were found.\n Generation Complete ")
        return
    print(f"\nSuccessfully identified {len(final_data)} unique quantum concepts.")
    _save_source_code_snippets(final_data)
    save_concepts_to_json(final_data)
    save_concepts_to_csv(final_data)
    print("\n Generation Complete ")


if __name__ == "__main__":
    main()

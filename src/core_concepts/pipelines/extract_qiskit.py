import logging
from pathlib import Path
from typing import List, Dict, Any

from src.conf import config
from src.core_concepts.extractor.extractors import ConceptExtractor
from src.core_concepts.extractor.storage import ConceptStorage
from src.core_concepts.extractor.visitors import QiskitVisitor
from src.core_concepts.extractor.processors import QiskitProcessor
from src.core_concepts.pipelines.qiskit_filters import (
    deduplicate_by_naming_convention,
    deduplicate_by_semantic_similarity,
)


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

QISKIT_PROJECT_ROOT: Path = config.PROJECT_ROOT / "target_github_projects" / "qiskit"
SEARCH_SUBDIRS = ["qiskit/circuit/library/"]
EXCLUDE_SUBDIRS = {"standard_gates", "templates"}


def _gather_qiskit_source_files() -> List[Path]:
    """
    Gathers all relevant Qiskit source files, applying custom exclusion logic.
    """
    all_py_files = []
    for sub_dir in SEARCH_SUBDIRS:
        search_path = QISKIT_PROJECT_ROOT / sub_dir
        if not search_path.is_dir():
            logging.warning(f"SDK subdirectory not found, skipping: {search_path}")
            continue

        # This list comprehension implements the original file filtering logic
        files_in_dir = [
            py_file
            for py_file in search_path.rglob("*.py")
            if not any(
                part in EXCLUDE_SUBDIRS
                for part in py_file.relative_to(search_path).parts
            )
            and py_file.name not in ("__init__.py",)
            and not py_file.name.startswith("test_")
        ]
        all_py_files.extend(files_in_dir)

    return sorted(all_py_files)


def main():
    """Orchestrates the full, complex Qiskit concept extraction and filtering pipeline."""
    logging.info("--- Starting Qiskit Core Quantum Concepts Generation ---")

    if not QISKIT_PROJECT_ROOT.is_dir():
        logging.error(
            f"Qiskit project root not found at '{QISKIT_PROJECT_ROOT.resolve()}'"
        )
        return

    # 1. Gather files using Qiskit-specific logic
    source_files = _gather_qiskit_source_files()
    if not source_files:
        logging.warning(
            "No Python files found in the Qiskit search directories after filtering."
        )
        return
    logging.info(f"Found {len(source_files)} Python files to process.")

    # 2. Instantiate the toolkit components
    processor = QiskitProcessor()
    extractor = ConceptExtractor(visitor_class=QiskitVisitor, processor=processor)

    # 3. Run raw extraction by iterating through the gathered files
    raw_concepts: List[Dict[str, Any]] = []
    all_concept_names = set()
    for py_file in source_files:
        concepts_in_file = extractor._find_concepts_in_file(
            py_file, QISKIT_PROJECT_ROOT
        )
        for concept in concepts_in_file:
            if concept["name"] not in all_concept_names:
                raw_concepts.append(concept)
                all_concept_names.add(concept["name"])

    logging.info(f"\n>>> Step 1: Extracted {len(raw_concepts)} raw concepts.")

    # 4. Apply the sequence of Qiskit-specific filters
    convention_filtered_data = deduplicate_by_naming_convention(raw_concepts)
    logging.info(
        f"\n>>> Step 2: After naming convention filter, {len(convention_filtered_data)} concepts remain."
    )

    final_data = deduplicate_by_semantic_similarity(convention_filtered_data)
    logging.info(
        f"\n>>> Step 3: After semantic filter, {len(final_data)} final concepts remain."
    )

    if not final_data:
        logging.warning("No quantum concepts remained after filtering.")
        logging.info("--- Qiskit Generation Complete ---")
        return

    # 5. Save the final, filtered data using the generic storage component
    logging.info(
        f"\nSuccessfully identified {len(final_data)} unique, filtered quantum concepts."
    )
    storage = ConceptStorage(
        json_path=config.RESULTS_DIR / "qiskit_quantum_concepts.json",
        csv_path=config.RESULTS_DIR / "qiskit_quantum_concepts.csv",
    )
    storage.save_all(final_data)

    logging.info("\n--- Qiskit Generation and Storage Complete ---")


if __name__ == "__main__":
    main()

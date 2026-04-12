import logging
from pathlib import Path
from typing import List, Dict, Any

from src.conf import config
from src.core_concepts.extractor.extractors import ConceptExtractor
from src.core_concepts.extractor.storage import ConceptStorage
from src.core_concepts.extractor.visitors import QiskitAlgorithmsVisitor
from src.core_concepts.extractor.processors import QiskitProcessor

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

QISKIT_ALGORITHMS_ROOT: Path = (
    config.PROJECT_ROOT / "target_github_projects" / "qiskit-algorithms"
)

# Only scan directories that contain quantum algorithm implementations.
# The optimizers/ directory is intentionally excluded — it holds classical
# optimizers (SPSA, COBYLA, L-BFGS-B, …) that happen to reference quantum
# algorithms in their docstrings but are not quantum concepts themselves.
SEARCH_SUBDIRS = [
    "qiskit_algorithms/amplitude_amplifiers/",
    "qiskit_algorithms/amplitude_estimators/",
    "qiskit_algorithms/eigensolvers/",
    "qiskit_algorithms/minimum_eigensolvers/",
    "qiskit_algorithms/phase_estimators/",
    "qiskit_algorithms/state_fidelities/",
    "qiskit_algorithms/time_evolvers/",
]

# Subdirectory names whose subtrees should be skipped entirely.
# classical_methods/  — SciPy-backed classical time-evolution wrappers.
# nlopts/             — NLOpt-backed classical optimizers.
# solvers/            — Internal ODE/linear-algebra solvers for VarQTE; they are
#                       numerical utilities, not quantum algorithm concepts.
EXCLUDE_SUBDIRS: set[str] = {"classical_methods", "nlopts", "solvers"}

# Individual files that implement classical (non-quantum) algorithms and would
# otherwise be swept up by the directory scan.
EXCLUDE_FILES: set[str] = {
    "numpy_eigensolver.py",         # classical full diagonalisation eigensolver
    "numpy_minimum_eigensolver.py", # classical full diagonalisation minimum eigensolver
    "diagonal_estimator.py",        # internal classical estimator helper
}

# The root-level variational_algorithm.py lives outside every sub-package but
# defines VariationalAlgorithm — a core quantum concept.  We handle it as a
# single-file "sub-directory" so the existing extractor machinery can process it.
ROOT_LEVEL_FILES = [
    "qiskit_algorithms/variational_algorithm.py",
]


def _gather_source_files() -> List[Path]:
    """
    Gathers all relevant qiskit-algorithms source files, applying exclusion rules.

    Strategy:
    - Walk only the whitelisted SEARCH_SUBDIRS.
    - Skip any path component that matches EXCLUDE_SUBDIRS.
    - Skip any file whose name is in EXCLUDE_FILES.
    - Skip __init__.py (they aggregate imports, not algorithm definitions).
    - Skip test files.
    """
    all_py_files: List[Path] = []

    for sub_dir in SEARCH_SUBDIRS:
        search_path = QISKIT_ALGORITHMS_ROOT / sub_dir
        if not search_path.is_dir():
            logging.warning(f"Subdirectory not found, skipping: {search_path}")
            continue

        files_in_dir = [
            py_file
            for py_file in search_path.rglob("*.py")
            if not any(
                part in EXCLUDE_SUBDIRS
                for part in py_file.relative_to(search_path).parts
            )
            and py_file.name not in EXCLUDE_FILES
            and py_file.name != "__init__.py"
            and not py_file.name.startswith("test_")
        ]
        all_py_files.extend(files_in_dir)

    # Add root-level files that sit outside every sub-package.
    for rel_path in ROOT_LEVEL_FILES:
        full_path = QISKIT_ALGORITHMS_ROOT / rel_path
        if full_path.is_file():
            all_py_files.append(full_path)
        else:
            logging.warning(f"Root-level file not found, skipping: {full_path}")

    return sorted(set(all_py_files))


def main():
    """Orchestrates the qiskit-algorithms quantum concept extraction pipeline."""
    logging.info("--- Starting qiskit-algorithms Core Quantum Concepts Generation ---")

    if not QISKIT_ALGORITHMS_ROOT.is_dir():
        logging.error(
            f"qiskit-algorithms project root not found at "
            f"'{QISKIT_ALGORITHMS_ROOT.resolve()}'. "
            "Please ensure the repository is cloned at that location."
        )
        return

    # 1. Gather source files using the whitelist + exclusion logic above.
    source_files = _gather_source_files()
    if not source_files:
        logging.warning("No Python files found after filtering.")
        return
    logging.info(f"Found {len(source_files)} Python files to process.")

    # 2. Instantiate the toolkit: reuse QiskitProcessor (same RST docstring style)
    #    with the new QiskitAlgorithmsVisitor that skips *Result/*Error classes.
    processor = QiskitProcessor()
    extractor = ConceptExtractor(
        visitor_class=QiskitAlgorithmsVisitor, processor=processor
    )

    # 3. Run extraction across all gathered files, deduplicating by concept name.
    raw_concepts: List[Dict[str, Any]] = []
    seen_names: set[str] = set()
    for py_file in source_files:
        for concept in extractor._find_concepts_in_file(
            py_file, QISKIT_ALGORITHMS_ROOT
        ):
            if concept["name"] not in seen_names:
                raw_concepts.append(concept)
                seen_names.add(concept["name"])

    logging.info(f"Extracted {len(raw_concepts)} unique quantum concepts.")

    if not raw_concepts:
        logging.warning("No quantum concepts found after extraction.")
        logging.info("--- qiskit-algorithms Generation Complete ---")
        return

    # 4. Save results.
    storage = ConceptStorage(
        json_path=config.RESULTS_DIR / "qiskit_algorithms_quantum_concepts.json",
        csv_path=config.RESULTS_DIR / "qiskit_algorithms_quantum_concepts.csv",
    )
    storage.save_all(raw_concepts)

    logging.info("--- qiskit-algorithms Generation and Storage Complete ---")


if __name__ == "__main__":
    main()

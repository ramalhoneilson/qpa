import logging
from pathlib import Path
from src.conf import config

from src.core_concepts.extractor.extractors import ConceptExtractor
from src.core_concepts.extractor.storage import ConceptStorage
from src.core_concepts.extractor.visitors import PennylaneClassVisitor
from src.core_concepts.extractor.processors import PennylaneProcessor

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

PENNYLANE_PROJECT_ROOT: Path = (
    config.PROJECT_ROOT / "target_github_projects" / "pennylane"
)

SEARCH_SUBDIRS = [
    "pennylane/templates/",
]


# Main Orchestration Function
def main():
    """Orchestrates the PennyLane concept extraction and storage pipeline."""
    logging.info("--- Starting PennyLane Core Concepts Generation ---")

    # 1. Perform PennyLane-specific setup: Verify the repo path exists.
    if not PENNYLANE_PROJECT_ROOT.is_dir():
        logging.error(
            f"PennyLane project root not found at '{PENNYLANE_PROJECT_ROOT.resolve()}'"
        )
        logging.error("Please ensure the repository is cloned at that location.")
        return

    # 2. Instantiate the toolkit components with PennyLane-specific strategies.
    processor = PennylaneProcessor()
    extractor = ConceptExtractor(
        visitor_class=PennylaneClassVisitor, processor=processor
    )

    # 3. Execute the extraction process.
    #    Note: Unlike Classiq, we don't need to pass any extra keyword arguments
    #    like 'public_api_names' because the visitor's logic is self-contained.
    concepts = extractor.extract_from_package(
        package_root=PENNYLANE_PROJECT_ROOT,
        search_paths=SEARCH_SUBDIRS,
    )

    if not concepts:
        logging.warning("Extraction complete, but no concepts were found.")
        logging.info("--- PennyLane Generation Complete ---")
        return

    logging.info(f"Successfully extracted {len(concepts)} unique PennyLane concepts.")

    # 4. Instantiate the storage handler with final output paths.
    storage = ConceptStorage(
        json_path=config.RESULTS_DIR / "pennylane_quantum_concepts.json",
        csv_path=config.RESULTS_DIR / "pennylane_quantum_concepts.csv",
    )

    # 5. Save the extracted data.
    storage.save_all(concepts)

    logging.info("--- PennyLane Generation and Storage Complete ---")


if __name__ == "__main__":
    main()

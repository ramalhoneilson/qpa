# src/core_concepts/pipelines/extract_classiq.py

import importlib
import logging
from pathlib import Path
from typing import List, Set

import classiq

from src.conf import config
from src.core_concepts.extractor.extractors import ConceptExtractor
from src.core_concepts.extractor.processors import ClassiqProcessor
from src.core_concepts.extractor.storage import ConceptStorage
from src.core_concepts.extractor.visitors import ClassiqFunctionVisitor

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


# --- Classiq-Specific Configuration ---
# The modules whose '__all__' attribute we want to inspect.
TARGET_MODULES = [
    "classiq.open_library.functions",
]

# The subdirectories within the classiq SDK to search for source files.
SOURCE_CODE_SEARCH_PATHS = [
    "open_library/functions",
    "qmod/builtins/functions",
]


#  Classiq-Specific Helper Functions
def get_sdk_root_path() -> Path | None:
    """Finds the installed Classiq SDK path from the imported package."""
    try:
        # __path__[0] gives the directory of the package
        sdk_path = Path(classiq.__path__[0])
        logging.info(f"Found installed Classiq SDK at: {sdk_path}")
        return sdk_path
    except (ImportError, AttributeError, IndexError):
        logging.error(
            "Could not find the installed 'classiq' package. Please ensure it is installed in the environment."
        )
        return None


def get_public_api_names(modules: List[str]) -> Set[str]:
    """Imports target modules and aggregates their `__all__` attributes."""
    all_public_apis = set()
    for module_name in modules:
        try:
            module = importlib.import_module(module_name)
            public_api_names = set(getattr(module, "__all__", []))
            logging.info(
                f"Found {len(public_api_names)} public API functions in '{module_name}'."
            )
            all_public_apis.update(public_api_names)
        except (ImportError, AttributeError) as e:
            logging.error(f"Could not load public API for module '{module_name}': {e}")
    return all_public_apis


#  Main Orchestration Function ---
def main():
    """Orchestrates the Classiq concept extraction and storage pipeline."""
    logging.info("--- Starting Classiq Core Concepts Generation ---")

    # 1. Perform Classiq-specific setup.
    sdk_root = get_sdk_root_path()
    if not sdk_root:
        return

    public_api_names = get_public_api_names(TARGET_MODULES)
    if not public_api_names:
        logging.error("No public API functions found to target. Exiting.")
        return

    # 2. Instantiate the toolkit components with Classiq-specific strategies.
    processor = ClassiqProcessor()
    extractor = ConceptExtractor(
        visitor_class=ClassiqFunctionVisitor, processor=processor
    )

    # 3. Execute the extraction process.
    #    The 'public_api_names' are passed as a keyword argument, which the
    #    ConceptExtractor will forward to the ClassiqFunctionVisitor.
    concepts = extractor.extract_from_package(
        package_root=sdk_root,
        search_paths=SOURCE_CODE_SEARCH_PATHS,
        public_api_names=public_api_names,
    )

    if not concepts:
        logging.warning("Extraction complete, but no concepts were found.")
        logging.info("--- Classiq Generation Complete ---")
        return

    logging.info(f"Successfully extracted {len(concepts)} unique Classiq concepts.")

    # 4. Instantiate the storage handler with final output paths.
    storage = ConceptStorage(
        json_path=config.RESULTS_DIR / "classiq_quantum_concepts.json",
        csv_path=config.RESULTS_DIR / "classiq_quantum_concepts.csv",
    )

    # 5. Save the extracted data.
    storage.save_all(concepts)

    logging.info("--- Classiq Generation and Storage Complete ---")


if __name__ == "__main__":
    main()

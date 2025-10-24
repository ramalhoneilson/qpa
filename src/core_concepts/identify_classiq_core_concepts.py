# src/main.py
import importlib
import logging
from pathlib import Path
from typing import List, Set, Any, Dict

import classiq

from src.conf import config
from src.concept_extractor.extractor import ConceptExtractor
from src.concept_extractor.storage import ConceptStorage

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Configuration remains separate, making it easy to see what drives the script
TARGET_MODULES = ["classiq.open_library.functions"]
SOURCE_CODE_SEARCH_PATHS = ["open_library/functions", "qmod/builtins/functions"]


def get_sdk_root_path() -> Path | None:
    """Finds the installed Classiq SDK path directly from the imported package."""
    try:
        sdk_path = Path(classiq.__path__[0])
        logging.info(f"Found installed Classiq SDK at: {sdk_path}")
        return sdk_path
    except (ImportError, AttributeError, IndexError):
        logging.error("Could not find the installed 'classiq' package.")
        return None


def get_public_api_names(modules: List[str]) -> Set[str]:
    """Imports modules and aggregates their `__all__` attributes."""
    all_public_apis = set()
    for module_name in modules:
        try:
            module = importlib.import_module(module_name)
            public_api_names = set(getattr(module, "__all__", []))
            logging.info(
                f"Found {len(public_api_names)} public functions in '{module_name}'."
            )
            all_public_apis.update(public_api_names)
        except (ImportError, AttributeError) as e:
            logging.error(f"Could not load public API for module '{module_name}': {e}")
    return all_public_apis


def run_final_analysis(
    found_concepts: List[Dict[str, Any]], expected_public_functions: Set[str]
):
    """Compares found concepts against the expected list and prints a report."""
    logging.info("\n--- Final Analysis Report ---")
    found_names = {item["name"].split(".")[-1] for item in found_concepts}
    missing_functions = expected_public_functions - found_names

    logging.info(f"Total concepts found and documented: {len(found_names)}")
    logging.info(
        f"Total functions in public API (`__all__`): {len(expected_public_functions)}"
    )

    if missing_functions:
        logging.warning("Public API Functions NOT FOUND or documented:")
        for i, func in enumerate(sorted(list(missing_functions)), 1):
            print(f"  {i}. {func}")
    else:
        logging.info("Success! All public functions were found and documented.")
    logging.info("--- End of Analysis Report ---\n")


def main():
    """Main function to orchestrate the extraction and storage of concepts."""
    logging.info("--- Starting Core Quantum Concepts Generation ---")

    sdk_root = get_sdk_root_path()
    if not sdk_root:
        return

    public_api_names = get_public_api_names(TARGET_MODULES)
    if not public_api_names:
        logging.error("No public API functions found to target. Exiting.")
        return

    # 1. Instantiate the extractor with its configuration
    extractor = ConceptExtractor(search_paths=SOURCE_CODE_SEARCH_PATHS)

    # 2. Run the core logic to get the data
    concepts_data = extractor.extract_from_package(sdk_root, public_api_names)

    if not concepts_data:
        logging.warning("Extraction complete, but no concepts were found.")
    else:
        logging.info(f"Successfully extracted {len(concepts_data)} unique concepts.")

    # 3. Run analysis on the results
    run_final_analysis(concepts_data, public_api_names)

    # 4. Instantiate the storage handler with its configuration
    storage = ConceptStorage(
        json_path=config.RESULTS_DIR / "classiq_quantum_concepts.json",
        csv_path=config.RESULTS_DIR / "classiq_quantum_concepts.csv",
        snippets_dir=config.RESULTS_DIR / "classiq_source_snippets",
    )

    # 5. Save the data
    storage.save_all(concepts_data)

    logging.info("--- Generation and Storage Complete ---")


if __name__ == "__main__":
    main()

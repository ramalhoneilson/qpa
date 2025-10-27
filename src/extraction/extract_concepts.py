import logging
import time

from src.core_concepts.pipelines import extract_classiq
from src.core_concepts.pipelines import extract_pennylane
from src.core_concepts.pipelines import extract_qiskit

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def run_pipeline(name: str, main_function):
    """
    A helper function to run a single pipeline with consistent logging
    and error handling.

    Args:
        name: The name of the pipeline for logging purposes.
        main_function: The main function of the pipeline to execute.
    """
    logging.info(f"--- Starting {name} Concept Extraction Pipeline ---")
    start_time = time.time()
    try:
        main_function()
        end_time = time.time()
        logging.info(
            f"--- Finished {name} Pipeline successfully in {end_time - start_time:.2f} seconds ---\n"
        )
    except Exception as e:
        end_time = time.time()
        logging.error(
            f"!!! The {name} Pipeline failed after {end_time - start_time:.2f} seconds: {e}",
            exc_info=True,
        )
        logging.error(f"--- Aborting {name} Pipeline due to error ---\n")


def main():
    """
    Main workflow to run all core concept extraction pipelines in sequence.
    """
    logging.info("=========================================================")
    logging.info("=== STARTING COMPLETE CORE CONCEPTS EXTRACTION WORKFLOW ===")
    logging.info("=========================================================\n")

    # Define the pipelines to run in order
    pipelines_to_run = {
        "Classiq": extract_classiq.main,
        "PennyLane": extract_pennylane.main,
        "Qiskit": extract_qiskit.main,
    }

    # Execute each pipeline
    for name, func in pipelines_to_run.items():
        run_pipeline(name, func)

    logging.info("=======================================================")
    logging.info("=== ALL EXTRACTION WORKFLOWS HAVE BEEN EXECUTED ===")
    logging.info("=======================================================")


if __name__ == "__main__":
    main()

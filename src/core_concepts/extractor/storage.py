import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

# A type alias for clarity, matching the one in extractor.py
Concept = Dict[str, Any]


class ConceptStorage:
    """
    Handles saving extracted concept data to various file formats.

    This class encapsulates all filesystem I/O operations for the output,
    including writing JSON and CSV files. It is initialized with the
    desired output paths.
    """

    def __init__(self, json_path: Path, csv_path: Path):
        """
        Initializes the ConceptStorage with all necessary output paths.

        Args:
            json_path: The full path for the output JSON file.
            csv_path: The full path for the output CSV file.
        """
        self.json_path = json_path
        self.csv_path = csv_path

    def save_all(self, concepts_data: List[Concept]):
        """
        A master method to save the provided data to all configured formats.

        It acts as a single entry point for the storage process.

        Args:
            concepts_data: A list of concept dictionaries to be saved.
        """
        if not concepts_data:
            logging.warning(
                "No concepts data was provided to save. Skipping all storage operations."
            )
            return

        # Create parent directories for all outputs upfront.
        self.json_path.parent.mkdir(parents=True, exist_ok=True)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)

        self.save_as_json(concepts_data)
        self.save_as_csv(concepts_data)

    def save_as_json(self, concepts_data: List[Concept]):
        """Saves concept data (excluding source code) to a JSON file."""
        logging.info(
            f"Saving {len(concepts_data)} concepts to JSON: '{self.json_path}'"
        )
        try:
            # Prepare data by removing the bulky 'source_code' field for a cleaner JSON file.
            json_data = [
                {k: v for k, v in item.items() if k != "source_code"}
                for item in concepts_data
            ]

            with self.json_path.open("w", encoding="utf-8") as f:
                json.dump(json_data, f, indent=2)
            logging.info("Successfully saved JSON file.")
        except Exception as e:
            logging.error(
                f"Failed to save data to JSON file at '{self.json_path}': {e}"
            )

    def save_as_csv(self, concepts_data: List[Concept]):
        """
        Saves the name and summary of each concept to a CSV file, using a
        semicolon (;) as the delimiter.
        """
        logging.info(f"Saving {len(concepts_data)} summaries to CSV: '{self.csv_path}'")
        headers = ["name", "summary"]
        try:
            with self.csv_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=headers, delimiter=";")
                writer.writeheader()
                for concept in concepts_data:
                    writer.writerow(
                        {
                            "name": concept.get("name", ""),
                            "summary": concept.get("summary", ""),
                        }
                    )
            logging.info("Successfully saved CSV file.")
        except Exception as e:
            logging.error(f"Failed to save data to CSV file at '{self.csv_path}': {e}")

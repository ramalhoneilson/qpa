# src/core_concepts/extractor/extractor.py

import ast
import logging
from pathlib import Path
from typing import Any, Dict, List, Type

# We import the base classes that our injected components will inherit from.
# This allows for type hinting and ensures a consistent interface.
from .processors import BaseProcessor
from .visitors import BaseConceptVisitor

# A type alias for clarity, representing a single extracted concept.
Concept = Dict[str, Any]


class ConceptExtractor:
    """
    A generic class to scan directories and extract code concepts from Python files.

    This class is designed to be reusable for different Python libraries (e.g.,
    Classiq, PennyLane). It is configured by injecting a specific AST visitor
    class and a docstring processor, which define what to look for and how
    to clean the extracted text.
    """

    def __init__(
        self, visitor_class: Type[BaseConceptVisitor], processor: BaseProcessor
    ):
        """
        Initializes the ConceptExtractor.

        Args:
            visitor_class: The specific ast.NodeVisitor class to use for finding
                           concepts in the source code's AST.
            processor: An instance of a class that handles docstring cleaning
                       and summary generation.
        """
        if not issubclass(visitor_class, BaseConceptVisitor):
            raise TypeError("visitor_class must be a subclass of BaseConceptVisitor")
        if not isinstance(processor, BaseProcessor):
            raise TypeError("processor must be an instance of a BaseProcessor subclass")

        self.visitor_class = visitor_class
        self.processor = processor

    def extract_from_package(
        self, package_root: Path, search_paths: List[str], **visitor_kwargs: Any
    ) -> List[Concept]:
        """
        Scans specified subdirectories of a package and extracts concepts.

        Args:
            package_root: The absolute path to the root of the package to scan.
            search_paths: A list of subdirectory paths (relative to package_root)
                          to search for Python files.
            **visitor_kwargs: Extra keyword arguments to be passed directly to the
                              constructor of the visitor class. This allows for
                              passing source-specific data, like a set of public
                              API names for Classiq.

        Returns:
            A list of dictionaries, where each dictionary is a found concept.
        """
        all_concepts: Dict[str, Concept] = {}

        for sub_dir in search_paths:
            search_path = package_root / sub_dir
            if not search_path.is_dir():
                logging.warning(f"Source directory not found, skipping: {search_path}")
                continue

            logging.info(f"Scanning files in '{sub_dir}'...")
            for py_file in sorted(search_path.rglob("*.py")):
                if py_file.name == "__init__.py":
                    continue

                concepts_in_file = self._find_concepts_in_file(
                    py_file, package_root, **visitor_kwargs
                )
                for concept in concepts_in_file:
                    # Use the concept's unique name as a key to avoid duplicates
                    if concept["name"] not in all_concepts:
                        all_concepts[concept["name"]] = concept

        return list(all_concepts.values())

    def _find_concepts_in_file(
        self, py_path: Path, sdk_root: Path, **visitor_kwargs: Any
    ) -> List[Concept]:
        """
        Parses a single Python file and returns a list of found concepts.

        This is a helper method that encapsulates the file reading and AST parsing logic.
        """
        try:
            source_text = py_path.read_text(encoding="utf-8")
            # Avoid parsing empty or very small files
            if len(source_text.strip()) < 10:
                return []

            tree = ast.parse(source_text, filename=str(py_path))

            # Instantiate the specific visitor passed during initialization
            visitor = self.visitor_class(
                source_text=source_text,
                file_path=py_path,
                sdk_root=sdk_root,
                processor=self.processor,
                **visitor_kwargs,  # Pass through any extra arguments
            )
            visitor.visit(tree)
            return list(visitor.found_concepts.values())
        except Exception as e:
            logging.warning(f"Could not parse file {py_path.name}: {e}")
            return []
